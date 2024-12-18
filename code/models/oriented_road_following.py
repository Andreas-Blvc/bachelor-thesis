from math import cos, pi, sin, tan
from typing import Any, List, Tuple
import casadi as ca
import cvxpy as cp
import numpy as np

from visualizer import plot_with_bounds
from models import AbstractVehicleModel
from roads import Road
from utils import State

class McCormickConvexRelaxation:
    def __init__(self, x, y, x_L, x_U, y_L, y_U):
        self._z = cp.Variable()
        self._x = x
        self._y = y
        self._x_L = x_L
        self._x_U = x_U
        self._y_L = y_L
        self._y_U = y_U

    def get_envelopes(self):
        constraints = [
            self._z >= self._x_L * self._y + self._x * self._y_L - self._x_L * self._y_L,
            self._z >= self._x_U * self._y + self._x * self._y_U - self._x_U * self._y_U,
            self._z <= self._x_U * self._y + self._x * self._y_L - self._x_U * self._y_L,
            self._z <= self._x_L * self._y + self._x * self._y_U - self._x_L * self._y_U,
        ]
        return constraints

    def get_relaxation_variable(self):
        return self._z

    def get_lower_upper_bound(self) -> Tuple[float, float]:
        # Only Callable if prob.solve() was called
        self._check()
        l = max(
            self._x_L * self._y.value + self._x.value * self._y_L - self._x_L * self._y_L,
            self._x_U * self._y.value + self._x.value * self._y_U - self._x_U * self._y_U,
        )
        u = min(
            self._x_U * self._y.value + self._x.value * self._y_L - self._x_U * self._y_L,
            self._x_L * self._y.value + self._x.value * self._y_U - self._x_L * self._y_U,
        )
        return l, u

    def get_bilinear_value(self):
        # Only Callable if prob.solve() was called
        self._check()
        return self._x.value * self._y.value

    def _check(self):
        # Check if prob.solve() was called
        if self._x.value is None or self._y.value is None:
            raise RuntimeError("Solver must be called before accessing bilinear value.")


class OrientedRoadFollowingModel(AbstractVehicleModel):
    def __init__(self,
                 v_range: Tuple[float, float],
                 acc_range: Tuple[float, float],
                 steering_angle_range: Tuple[float, float],
                 steering_velocity_range: Tuple[float, float],
                 road: Road,
                 l_wb: float = 1.8,
                 ):
        """
            Initialize the RoadAlignedModel.

            state vector of shape (5,) representing [s, n, xi, v, delta].
        """
        super().__init__(
            dim_state=5,
            dim_control_input=2,
            control_input_labels=['a_x,b', 'v_delta'],
            state_labels=['s', 'n', 'xi', 'v', 'delta'],
        )
        # Params:
        self.road = road
        self.L_wb = l_wb

        # Aliases for range access
        self.v_min, self.v_max = v_range
        self.a_min, self.a_max = acc_range
        self.steering_angle_min, self.steering_angle_max = steering_angle_range
        self.steering_velocity_min, self.steering_velocity_max = steering_velocity_range

        # First Order Taylor Approximation:
        self.n_0 = 0
        self.xi_0 = 10 / 180 * pi
        self.delta_0 = self.steering_angle_max / 4
        self.v_0 = self.v_max / 4

        # Artificial Variables Approximation:
        self.xi_abs_bound = 45/180 * pi
        self._mccormick_relaxations: List[Tuple[McCormickConvexRelaxation]] = []

    def update(self, current_state, control_inputs, dt) -> Tuple[np.ndarray, List[Any]]:
        self._validate__state_dimension(current_state)
        self._validate__control_dimension(control_inputs)

        s, n, xi, v, delta = [current_state[i] for i in range(self.dim_state)]
        a_x_b, v_delta = [control_inputs[i] for i in range(self.dim_control_input)]

        variables = self.road.get_segment_dependent_variables(s, self.solver_type == 'casadi', self.road_segment_idx)
        C = variables.C
        n_min = variables.n_min
        n_max = variables.n_max

        if np.isscalar(s):
            ds = np.cos(xi) * v / (1 - n * C(s))
            d_theta = C(s) * ds
            d_phi = (v / self.L_wb) * np.tan(delta)
            next_state = np.array([
                s + ds * dt,
                n + v * np.sin(xi) * dt,
                xi + (d_phi - d_theta) * dt,
                v + a_x_b * dt,
                delta + v_delta * dt,
            ])
            return next_state, []

        constraints = []
        if self.solver_type == 'casadi':
            ds = ca.cos(xi) * v / (1 - n * C(s))
            d_theta = C(s) * ds
            d_phi = (v / self.L_wb) * ca.tan(delta)
            next_state = ca.vertcat(*[
                s + ds * dt,
                n + v * ca.sin(xi) * dt,
                xi + (d_phi - d_theta) * dt,
                v + a_x_b * dt,
                delta + v_delta * dt,
            ])
        elif self.solver_type == 'cvxpy':
            # We have following non-linear terms:
            # 1. v * cos(xi) / (1 - n * C(s)) (ds-term)
            # 2. v * sin(xi) (dn-term)
            # 3. v * tan(delta) (dxi-term)
            ds_term, dn_term, dxi_term = self._convex_relax_bilinear_terms(s, n, xi, v, delta, constraints)

            next_state = cp.vstack([
                s + ds_term * dt,
                n + dn_term * dt,
                xi + (1/self.L_wb * dxi_term - C(s) * ds_term) * dt,
                v + a_x_b * dt,
                delta + v_delta * dt,
            ]).flatten()
        else:
            self._raise_unsupported_solver()

        constraints += [
            0 <= s, s <= self.road.length,
            n_min <= n, n <= n_max,  # depends on current road segment
            self.v_min <= v, v <= self.v_max,
            self.steering_angle_min <= delta, delta <= self.steering_angle_max,
            self.steering_velocity_min <= v_delta, v_delta <= self.steering_velocity_max,
            self.a_min <= a_x_b, a_x_b <= self.a_max,
        ]

        return next_state, constraints

    def _taylor_approximate_terms(self, s, n, xi, v, delta, C):
        ds_term = (
                (self.v_0 * cos(self.xi_0)) / (1 - C(s) * self.n_0) +
                cos(self.xi_0) / (1 - C(s) * self.n_0) * (v - self.v_0) +
                -self.v_0 * sin(self.xi_0) / (1 - C(s) * self.n_0) * (xi - self.xi_0) +
                -self.v_0 * cos(self.xi_0) * C(s) / (1 - C(s) * self.n_0)**2 * (n - self.n_0)
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

    def _convex_relax_bilinear_terms(self, s, n, xi, v, delta, constraints):
        if self.solver_type != 'cvxpy':
            raise NotImplementedError('artificial_variables_approximation only for cvxpy solver')

        # Define variable bounds
        v_min = self.v_min
        v_max = self.v_max
        xi_min = -self.xi_abs_bound
        xi_max = self.xi_abs_bound
        delta_min = self.steering_angle_min
        delta_max = self.steering_angle_max

        ds_term_relaxation = v
        dn_term_relaxation = McCormickConvexRelaxation(v, xi, v_min, v_max, xi_min, xi_max)
        dxi_term_relaxation = McCormickConvexRelaxation(v, delta, v_min, v_max, delta_min, delta_max)

        self._mccormick_relaxations += [(dn_term_relaxation, dxi_term_relaxation)]

        # McCormick envelopes for dn_term = v * xi
        constraints += dn_term_relaxation.get_envelopes()

        # McCormick envelopes for dxi_term = v * delta
        constraints += dxi_term_relaxation.get_envelopes()

        return (
            ds_term_relaxation,
            dn_term_relaxation.get_relaxation_variable(),
            dxi_term_relaxation.get_relaxation_variable()
        )

    def plot_additional_information(self, states, controls):
        # We want to plot the values of the relaxation variables, their bounds, the bilinear actual value, and the exact term
        for idx, y_label in enumerate(['dn_term', 'dxi_term']):
            bounds = []
            y_values = []
            for mccormick_relaxation_tuple in self._mccormick_relaxations:
                mccormick_relaxation = mccormick_relaxation_tuple[idx]
                bounds.append(mccormick_relaxation.get_lower_upper_bound())
                y_values.append([
                    mccormick_relaxation.get_relaxation_variable().value,
                    mccormick_relaxation.get_bilinear_value()
                ])
            y_labels = [
                'Relaxation Variable Value',
                'Actual Bilinear Value',
                'Actual Bilinear Value',
            ]
            plot_with_bounds(
                bounds=bounds,
                y_values_list=y_values,
                y_labels=y_labels,
                y_label=y_label,
                # dt=dt,
            )

        plot_with_bounds(
            y_values_list=[
                [
                    v * cos(xi) / (1 - n * self.road.get_curvature_at(s)), v
                ] for s, n, xi, v, delta in states
            ],
            y_labels=['actual', 'approximation'],
            y_label='ds_term',
            # dt=dt,
            no_bounds=True,
        )
        plot_with_bounds(
            y_values_list=[
                [
                    v * sin(xi), v * xi
                ] for _, _, xi, v, _ in states
            ],
            y_labels=['actual', 'approximation'],
            y_label='dn_term',
            # dt=dt,
            no_bounds=True,
        )
        plot_with_bounds(
            y_values_list=[
                [
                    v * tan(delta), v * delta
                ] for _, _, _, v, delta in states
            ],
            y_labels=['actual', 'approximation'],
            y_label='dxi_term',
            # dt=dt,
            no_bounds=True,
        )

    def get_v_max(self):
        return self.v_max

    def convert_vec_to_state(self, vec) -> State:
        # vec: s, n, xi, v, delta
        self._validate__state_dimension(vec)
        return State(
            vec=vec,
            get_velocity=lambda: vec[3],
            get_offset_from_reference_path=lambda: self._absolute(vec[1]),
            get_remaining_distance=lambda: self.road.length - vec[0],
            get_traveled_distance=lambda: vec[0],
            get_distance_between=lambda other_state: self._norm_squared(vec[:2] - other_state.as_vector()[:2]),
            get_position_orientation=lambda: (
                np.array(self.road.get_global_position(vec[0], vec[1])),
                self.road.get_tangent_angle_at(vec[0]) + vec[2]
            ),
            get_lateral_offset= lambda: vec[1],
            get_alignment_error= lambda: vec[2],
            to_string=lambda: self._state_vec_to_string(vec),
        )


    def get_state_vec_from_dsm(self, vec) -> np.ndarray:
        x, y, delta, v, psi, dpsi, beta = vec
        # s, n, xi, v, delta
        s, n = self.road.get_road_position(x, y)
        xi = (psi - self.road.get_tangent_angle_at(s) + np.pi) % (2 * np.pi) - np.pi
        return np.array([
            s,
            n,
            xi,
            v,
            delta
        ])

    def get_dsm_control_from_vec(self, control_vec, state_vec):
        a_x, v_delta = control_vec
        return np.array([
            v_delta,
            a_x
        ])