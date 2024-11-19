from math import cos, pi, sin, tan
from typing import Any, List, Tuple
import casadi as ca
import cvxpy as cp
import numpy as np

from models import AbstractVehicleModel
from obstacles import AbstractRoad
from utils import State

BIG_M = 1e6


def mccormick_envelopes(z, x, y, x_L, x_U, y_L, y_U):
    constraints = [
        z >= x_L * y + x * y_L - x_L * y_L,
        z >= x_U * y + x * y_U - x_U * y_U,
        z <= x_U * y + x * y_L - x_U * y_L,
        z <= x_L * y + x * y_U - x_L * y_U,
    ]
    return constraints


class OrientedRoadFollowingModelAbstract(AbstractVehicleModel):
    def __init__(self,
                 dt: float,
                 v_range: Tuple[float, float],
                 acc_range: Tuple[float, float],
                 steering_angle_range: Tuple[float, float],
                 steering_velocity_range: Tuple[float, float],
                 road: AbstractRoad,
                 initial_state: np.ndarray,
                 goal_state: np.ndarray=None,
                 l_wb: float = 1.8,
                 ):
        super().__init__(
            dim_state=5,
            dim_control_input=2,
            control_input_labels=['a_x,b', 'v_delta'],
            state_labels=['s', 'n', 'xi', 'v', 'delta'],
            initial_state=initial_state
        )
        # Params:
        self.dt = dt
        self.road = road
        self.L_wb = l_wb

        # Aliases
        self.C = road.get_curvature_at
        self.dC = road.get_curvature_derivative_at

        # Aliases for range access
        self.v_min, self.v_max = v_range
        self.a_min, self.a_max = acc_range
        self.n_min, self.n_max = -road.width/2, road.width/2
        self.steering_angle_min, self.steering_angle_max = steering_angle_range
        self.steering_velocity_min, self.steering_velocity_max = steering_velocity_range

        # First Order Taylor Approximation:
        self.n_0 = 0
        self.xi_0 = 10 / 180 * pi
        self.delta_0 = self.steering_angle_max / 4
        self.v_0 = self.v_max / 4

        # Artificial Variables Approximation:
        self.xi_abs_bound = 45/180 * pi
        self.artificial_variables = []

    def update(self, current_state, control_inputs) -> Tuple[np.ndarray, List[Any]]:
        self._validate__state_dimension(current_state)
        self._validate__control_dimension(control_inputs)

        s, n, xi, v, delta = [current_state[i] for i in range(self.dim_state)]
        a_x_b, v_delta = [control_inputs[i] for i in range(self.dim_control_input)]

        if np.isscalar(s):
            ds = np.cos(xi) * v / (1 - n * self.C(s))
            d_theta = self.C(s) * ds
            d_phi = (v / self.L_wb) * np.tan(delta)
            next_state = np.array([
                s + ds * self.dt,
                n + v * np.sin(xi) * self.dt,
                xi + (d_phi - d_theta) * self.dt,
                v + a_x_b * self.dt,
                delta + v_delta * self.dt,
            ])
            return next_state, []

        constraints = []
        if self.solver_type == 'casadi':
            ds = ca.cos(xi) * v / (1 - n * self.C(s))
            d_theta = self.C(s) * ds
            d_phi = (v / self.L_wb) * ca.tan(delta)
            next_state = ca.vertcat(*[
                s + ds * self.dt,
                n + v * ca.sin(xi) * self.dt,
                xi + (d_phi - d_theta) * self.dt,
                v + a_x_b * self.dt,
                delta + v_delta * self.dt,
            ])
        elif self.solver_type == 'cvxpy':
            # We have following non-linear terms:
            # 1. v * cos(xi) / (1 - n * self.C(s)) (ds-term)
            # 2. v * sin(xi) (dn-term)
            # 3. v * tan(delta) (dxi-term)
            ds_term, dn_term, dxi_term = self.artificial_variables_approximation(s, n, xi, v, delta, constraints)

            next_state = cp.vstack([
                s + ds_term * self.dt,
                n + dn_term * self.dt,
                xi + (1/self.L_wb * dxi_term - self.C(s) * ds_term) * self.dt,
                v + a_x_b * self.dt,
                delta + v_delta * self.dt,
            ]).flatten()
        else:
            self._raise_unsupported_solver()

        constraints += [
            0 <= s, s <= self.road.length,
            self.n_min <= n, n <= self.n_max,
            self.v_min <= v, v <= self.v_max,
            self.steering_angle_min <= delta, delta <= self.steering_angle_max,
            self.steering_velocity_min <= v_delta, v_delta <= self.steering_velocity_max,
            self.a_min <= a_x_b, a_x_b <= self.a_max,
        ]

        return next_state, constraints

    def taylor_approximation(self, s, n, xi, v, delta):
        ds_term = (
                (self.v_0 * cos(self.xi_0)) / (1 - self.C(s) * self.n_0) +
                cos(self.xi_0) / (1 - self.C(s) * self.n_0) * (v - self.v_0) +
                -self.v_0 * sin(self.xi_0) / (1 - self.C(s) * self.n_0) * (xi - self.xi_0) +
                -self.v_0 * cos(self.xi_0) * self.C(s) / (1 - self.C(s) * self.n_0)**2 * (n - self.n_0)
        )
        dn_term = (
                self.v_0 * sin(self.xi_0) +
                sin(self.xi_0) * (v - self.v_0) +
                self.v_0 * cos(self.xi_0) * (xi - self.xi_0)
        )
        dxi_term = (
                self.v_0 * tan(self.delta_0) +
                tan(self.delta_0) * (v - self.v_0) +
                self.v_0 / cos(self.delta_0) ** 2 * (delta - self.delta_0)
        )
        return ds_term, dn_term, dxi_term

    def artificial_variables_approximation(self, s, n, xi, v, delta, constraints):
        if self.solver_type != 'cvxpy':
            raise NotImplementedError('artificial_variables_approximation only for cvxpy solver')
        # ds_term = cp.Variable()
        dn_term = cp.Variable()
        dxi_term = cp.Variable()
        self.artificial_variables += [(dn_term, dxi_term)]

        # Define variable bounds
        v_min = self.v_min
        v_max = self.v_max
        xi_min = -self.xi_abs_bound
        xi_max = self.xi_abs_bound
        delta_min = self.steering_angle_min
        delta_max = self.steering_angle_max

        # McCormick envelopes for dn_term = v * xi
        constraints += mccormick_envelopes(dn_term, v, xi, v_min, v_max, xi_min, xi_max)

        # McCormick envelopes for dxi_term = v * delta
        constraints += mccormick_envelopes(dxi_term, v, delta, v_min, v_max, delta_min, delta_max)

        return v, dn_term, dxi_term

    def get_vehicle_polygon(self, state) -> List[Tuple[float, float]]:
        front_wheel_front = self._add_tuple(self._rotate((0.5, 0), float(state[-1])), (1, 0))
        front_wheel_back = self._add_tuple(self._rotate((-0.5, 0), float(state[-1])), (1, 0))
        return [
            (-1, 0.5), (1, 0.5),
            (1, 0),
            front_wheel_back, front_wheel_front,
            (1, 0),
            (1, -0.5),
            (-1, -0.5),
        ]

    def convert_vec_to_state(self, vec) -> State:
        # vec: s, n, xi, v, delta
        self._validate__state_dimension(vec)
        return State(
            vec=vec,
            get_velocity=lambda: vec[3],
            get_offset_from_reference_path=lambda: self._absolute(vec[1]),
            get_remaining_distance=lambda: self.road.length - vec[0],
            get_traveled_distance=lambda: vec[0] - self.initial_state[0],
            get_distance_between=lambda other_state: self._norm_squared(vec[:2] - other_state.as_vector()[:2]),
            get_position_orientation=lambda: (
                np.array(self.road.get_global_position(vec[0], vec[1])),
                self.road.get_tangent_angle_at(vec[0]) + vec[2]
            ),
            to_string=lambda: self.state_vec_to_string(vec),
        )

    @staticmethod
    def _add_tuple(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
        return (
            a[0] + b[0],
            a[1] + b[1],
        )

    @staticmethod
    def _rotate(point: Tuple[float, float], theta: float) -> Tuple[float, float]:
        x, y = point
        return (
            x * cos(theta) - y * sin(theta),
            y * cos(theta) + x * sin(theta),
        )
