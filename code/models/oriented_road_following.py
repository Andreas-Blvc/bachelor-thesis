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
                 ):
        self.solver_type = None
        self.dim_state = 5
        self.dim_control_input = 2
        self.dt = dt
        self.road = road
        self.C = road.get_curvature_at
        self.dC = road.get_curvature_derivative_at
        self.L_wb = 1.8

        if initial_state.shape != (self.dim_state,):
            raise ValueError(f"initial_state must have shape ({self.dim_state},), got {initial_state.shape}")
        if goal_state is not None and goal_state.shape != (self.dim_state,):
            raise ValueError(f"goal_state must have shape ({self.dim_state},), got {goal_state.shape}")

        self.initial_state = initial_state
        self.goal_state = goal_state
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
        if current_state.shape != (self.dim_state,) and current_state.shape != (self.dim_state, 1):
            raise ValueError(f"current_state must have shape ({self.dim_state},) or ({self.dim_state}, 1), got {current_state.shape}")
        if control_inputs.shape != (self.dim_control_input,) and control_inputs.shape != (self.dim_control_input, 1):
            raise ValueError(f"control_inputs must have shape ({self.dim_control_input},)  or ({self.dim_control_input}, 1), got {control_inputs.shape}")

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
            raise ValueError(f"solver_type {self.solver_type} not supported")

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


    def get_initial_state(self) -> np.ndarray:
        return self.initial_state

    def get_goal_state(self) -> np.ndarray:
        return self.goal_state

    def get_position_orientation(self, state) -> Tuple[np.ndarray, float]:
        s, n, xi, v, delta = state
        cur_pos = np.array(self.road.get_global_position(s, n))
        return (
            cur_pos,
            self.road.get_tangent_angle_at(s) + xi
        )

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

    def get_distance_between(self, state_a: State, state_b: State):
        if self.solver_type == 'casadi':
            norm = ca.sumsqr
        elif self.solver_type == 'cvxpy':
            norm = cp.sum_squares
        else:
            raise ValueError(f"solver_type {self.solver_type} not supported")
        return norm(state_a.as_vector()[:2] - state_b.as_vector()[:2])

    def get_traveled_distance(self, state: State):
        return state.as_vector()[0] - self.initial_state[0]

    def get_remaining_distance(self, state: State):
        return self.road.length - state.as_vector()[0]

    def get_offset_from_reference_path(self, state: State):
        if self.solver_type == 'casadi':
            absolute_val = ca.fabs
        elif self.solver_type == 'cvxpy':
            absolute_val = cp.abs
        else:
            raise ValueError(f"solver_type {self.solver_type} not supported")
        return absolute_val(state.as_vector()[1])

    def get_velocity(self, state: State):
        return state.as_vector()[3]

    def get_dim_state(self) -> int:
        return self.dim_state

    def get_dim_control_input(self) -> int:
        return self.dim_control_input

    def get_a_max(self) -> float:
        return self.a_max

    def to_string(self, state, control):
        # Unpack state and control for clarity
        s, n, xi, v, delta = state
        a, kap = control

        # Format the output string
        state_str = f"State: [s: {s:.2f}, n: {n:.2f}, v: {v:.2f}, xi: {xi:.2f}, delta: {delta:.2f}]"
        control_str = f"Control: [a: {a:.2f}, kap: {kap:.2f}]"

        return f"{state_str} | {control_str}"

    def get_control_input_labels(self) -> List[str]:
        return ['a_x,b', 'v_delta']

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
