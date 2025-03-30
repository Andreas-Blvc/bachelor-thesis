from typing import Any, List, Tuple, Callable
from scipy.optimize import fsolve
import casadi as ca
import cvxpy as cp
import numpy as np

from models import AbstractVehicleModel
from range_bounding import eliminate_quantifier
from range_bounding.sympy_to_cvxpy_constraints import sympy_to_cvxpy_constraints
from roads import Road
from utils import MainPaperConstraintsReduction, State, StateRanges, lazy_setdefault, optimal_range_bound


class PointMassModel(AbstractVehicleModel):

    def __init__(self,
                 road: Road,
                 v_x_range: Tuple[float, float],
                 v_y_range: Tuple[float, float],
                 acc_x_range: Tuple[float, float],
                 acc_y_range: Tuple[float, float],
                 yaw_rate_range: Tuple[float, float],
                 yaw_acc_range: Tuple[float, float],
                 a_max,
                 ):
        """
        Initialize the PointMassModel.

        state vector of shape (4,) representing [s, n, ds, dn].
        :raises ValueError: If initial_state or goal_state do not have the correct shape.
        """
        super().__init__(
            dim_state=4,
            dim_control_input=2,
            state_labels=['s', 'n', 'ds', 'dn'],
            control_input_labels=['u\_t', 'u\_n'],
        )
        # ParamsNone
        self.road = road
        self.v_x_range = v_x_range
        self.v_y_range = v_y_range
        self.acc_x_range = acc_x_range
        self.acc_y_range = acc_y_range
        self.yaw_rate_range = yaw_rate_range
        self.yaw_acc_range = yaw_acc_range

        # Aliases for range access
        self.v_x_min, self.v_x_max = v_x_range
        self.v_y_min, self.v_y_max = v_y_range
        self.acc_x_min, self.acc_x_max = acc_x_range
        self.acc_y_min, self.acc_y_max = acc_y_range
        self.yaw_rate_min, self.yaw_rate_max = yaw_rate_range
        self.yaw_acc_min, self.yaw_acc_max = yaw_acc_range
        self.a_max = a_max

        # helper:
        self.delta_cur = 0  # current assumption: initial steering is 0
        self._constructed_ranges: dict[int, Callable[[cp.Variable,cp.Variable,cp.Variable,cp.Variable,cp.Variable,cp.Variable], List[Any]]] = {}
        self._optimal_ranges: List[StateRanges] = []

    def _get_polytopic_constrain_set(self, C, c_min, c_max, s_min, s_max, n_min, n_max) -> Callable[
        [Any, Any, Any, Any, Any, Any], StateRanges]:
        ranges = StateRanges(
            n=(n_min, n_max),
            c=(c_min, c_max),
            ds=self.v_x_range,
            dn=self.v_y_range,
            u_n=None,
            u_t=None,
        )
        # updates ranges in place
        MainPaperConstraintsReduction.apply_all(
            state_ranges=ranges,
            v_x_range=self.v_x_range,
            acc_x_range=self.acc_x_range,
            acc_y_range=self.acc_y_range,
            yaw_rate_range=self.yaw_rate_range,
            yaw_acc_range=self.yaw_acc_range,
            curvature_derivative=0  # constant for all s
        )

        if ranges.u_t == (0,0):
            ranges.u_t = (0, -1)

        optimal_range = optimal_range_bound(
            road_width_range=(n_min, n_max),
            v_x_range=self.v_x_range,
            v_y_range=self.v_y_range,
            a_x_range=self.acc_x_range,
            a_y_range=self.acc_y_range,
            yaw_rate_range=self.yaw_rate_range,
            yaw_acc_range=self.yaw_acc_range,
            curvature=c_min
        )
        self._optimal_ranges.append(optimal_range)
        # return optimal_range
        # formulas, [x1, x2, x3, x4, u1, u2] = eliminate_quantifier(
        #     Curvature = c_min,
        #     sMin = s_min,
        #     sMax = s_max,
        #     nMin = n_min,
        #     nMax = n_max,
        #     vxMin = self.v_x_min,
        #     vxMax = self.v_x_max,
        #     vyMin = self.v_y_min,
        #     vyMax = self.v_y_max,
        #     axMin = self.acc_x_min,
        #     axMax = self.acc_x_max,
        #     ayMin = self.acc_y_min,
        #     ayMax = self.acc_y_max,
        #     dpsiMin = self.yaw_rate_min,
        #     dpsiMax = self.yaw_rate_max,
        #     apsiMin = self.yaw_acc_min,
        #     apsiMax = self.yaw_acc_max,
        # )

        def _constrain_var(cvx_u1, cvx_u2, cvx_x1, cvx_x2, cvx_x3, cvx_x4):
            # Mapping of SymPy to CVXPY variables
            variable_mapping = {u1: cvx_u1, u2: cvx_u2, x1: cvx_x1, x2: cvx_x2, x3: cvx_x3, x4: cvx_x4}
            constraints = sympy_to_cvxpy_constraints(formulas, variable_mapping)
            # print(list(str(c) for c in constraints))
            return constraints

        def _constrained_rectangle(cvx_u1, cvx_u2, cvx_x1, cvx_x2, cvx_x3, cvx_x4):
            return ranges

        return _constrained_rectangle


    def g(self, x_tn, u, C, dC):
        s, n, ds, dn = [x_tn[i] for i in range(self.dim_state)]
        u_t, u_n = [u[i] for i in range(self.dim_control_input)]
        return [
            (1 - n*C(s)) * u_t -
            (2*dn*C(s)*ds + n*dC(s)*ds**2),
            u_n + C(s) * ds**2 * (1-n*C(s)),
        ]

    def forward_euler_step(self, current_state, control_inputs, dt: float, convexify_ref_state=None, amount_prev_planning_states=None) -> Tuple[np.ndarray, List[Any], cp.Expression | ca.MX | int]:
        self._validate__state_dimension(current_state)
        self._validate__control_dimension(control_inputs)

        dx_dt = np.array([
            current_state[2],
            current_state[3],
            control_inputs[0],
            control_inputs[1],
        ])

        s, n, ds, dn = [current_state[i] for i in range(self.dim_state)]
        u_t, u_n = [control_inputs[i] for i in range(self.dim_control_input)]

        if np.isscalar(s):
            next_state = current_state + dx_dt * dt
            return next_state, [], 0

        variables = self.road.get_segment_dependent_variables(s, self.solver_type == 'casadi', self.road_segment_idx)
        C = variables.C
        dC = variables.dC
        c_min = variables.c_min
        c_max = variables.c_max
        n_min = variables.n_min
        n_max = variables.n_max

        if self.solver_type == 'casadi':
            # Compute the next state based on the input accelerations
            next_state = ca.vertcat(*[
                current_state[i] + dx_dt[i] * dt for i in range(self.dim_state)
            ])

            g = self.g(current_state, control_inputs, C, dC)

            # Define the constraint for acceleration within limits
            constraints = [
                #  the next constraint is replaced by 0<=s<=road.length:
                # self.c_min <= C(s), C(s) <= self.c_max,
                0 <= s, s <= self.road.length,
                n_min <= n, n <= n_max,
                self.v_x_min <= ds*(1 + n*C(s)),
                ds*(1 + n*C(s)) <= self.v_x_max,
                self.v_y_min <= dn, dn <= self.v_y_max,
                self.yaw_rate_min <= C(s) * ds,
                C(s) * ds <= self.yaw_rate_max,
                self.yaw_acc_min <= dC(s) * ds**2 + C(s) * u_t,
                dC(s) * ds ** 2 + C(s) * u_t <= self.yaw_acc_max,
                self.acc_x_min <= g[0], g[0] <= self.acc_x_max,
                self.acc_y_min <= g[1], g[1] <= self.acc_y_max,
                g[0]**2 + g[1]**2 <= self.a_max,
            ]
            model_objective = ca.MX(0)
        elif self.solver_type == 'cvxpy':
            prev_length = sum([segment.length for segment in self.road.segments[:self.road_segment_idx]])
            cur_segment_length = self.road.segments[self.road_segment_idx].length
            get_constraints = lazy_setdefault(
                self._constructed_ranges,
                self.road_segment_idx,
                lambda: self._get_polytopic_constrain_set(
                    C,
                    c_min,
                    c_max,
                    prev_length,
                    prev_length + cur_segment_length,
                    min(n_min(x) for x in np.linspace(prev_length, prev_length + cur_segment_length, 500)),
                    max(n_max(x) for x in np.linspace(prev_length, prev_length + cur_segment_length, 500)),
                )
            )
            next_state = cp.vstack([current_state[i] + dx_dt[i] * dt for i in range(self.dim_state)]).flatten()
            # constraints = get_constraints(u_t, u_n, s, n, ds, dn) + [
            #     0 <= s, s <= self.road.length,
            #     n_min(s) <= n, n <= n_max(s),
            # ]
            # + [
            #                     Le(sMin, x1),
            #                     Le(x1, sMax),
            #                     Le(nMin, x2),
            #                     Le(x2, nMax),
            #                     Le(vyMin, x4),
            #                     Le(x4, vyMax),
            #                 ],
            soft_constraint_var = [cp.Variable() for _ in range(4)]
            ranges = get_constraints(u_t, u_n, s, n, ds, dn)
            constraints = [
                0 <= s, s <= self.road.length,
                # ranges.c[0] <= C(s), C(s) <= ranges.c[1],
                n_min(s) <= n, n <= n_max(s),
                ranges.ds[0] - soft_constraint_var[0] <= ds, ds <= ranges.ds[1] + soft_constraint_var[1],
                ranges.dn[0] - soft_constraint_var[2] <= dn, dn <= ranges.dn[1] + soft_constraint_var[3],
                ranges.u_t[0] <= u_t, u_t <= ranges.u_t[1],
                ranges.u_n[0] <= u_n, u_n <= ranges.u_n[1],
                *[var >= 0 for var in soft_constraint_var],
            ]
            model_objective = cp.sum([var * 1000 for var in soft_constraint_var])
        else:
            raise ValueError(f"solver_type {self.solver_type} not supported")

        return next_state, constraints, model_objective

    def plot_additional_information(self, states, controls):
        for (key, value), optimal_range, state in zip(self._constructed_ranges.items(), self._optimal_ranges, states):
            x1, x2, x3, x4 = state
            print(key, value(0, 0, x1, x2, x3, x4))
            print('optimal_range', optimal_range)

    def convert_vec_to_state(self, vec, road_segment_idx=None) -> State:
        # vec: s, n, ds, dn
        self._validate__state_dimension(vec)
        return State(
            vec=vec,
            get_velocity=lambda: vec[2],
            get_negative_distance_to_closest_border=lambda: cp.maximum(
                (vec[1] - self.road.n_max(vec[0], road_segment_idx)),
                (self.road.n_min(vec[0], road_segment_idx) - vec[1])
            ),
            get_remaining_distance=lambda: self.road.length - vec[0],
            get_traveled_distance=lambda: vec[0],
            get_distance_between=lambda other_state: self._norm_squared(vec[:2] - other_state.as_vector()[:2]),
            to_string=lambda: self._state_vec_to_string(vec),
            get_lateral_offset=lambda: vec[1],
            get_alignment_error=lambda: 0,
            get_position_orientation=lambda: (
                np.array(self.road.get_global_position(vec[0], vec[1])),
                self.road.get_tangent_angle_at(vec[0]),
            )
        )

    def get_state_vec_from_dsm(self, vec) -> np.ndarray:
        x, y, delta, v, psi, dpsi, beta = vec
        # s, n, ds, dn
        s, n = self.road.get_road_position(x, y)
        xi = psi - self.road.get_tangent_angle_at(s)
        ds = v * np.cos(xi) / (1 - n * self.road.get_curvature_at(s))
        dn = v * np.sin(xi)
        return np.array([
            s, n, ds, dn
        ])


    def get_dsm_control_from_vec(
            self,
            control_vec,
            state_vec,
            dynamics,
            dt=None,
            remaining_predictive_model_states:List[np.ndarray]=None,
            car_cur_state: AbstractVehicleModel.CarState=None
    ) -> np.ndarray:
        if dt is None:
            raise ValueError("dt must be specified")
        # choose one of the four approaches here:
        return self._approach_1(state_vec, control_vec, dt, car_cur_state.steering_angle)

    def _approach_1(self, state_vec, control_vec, dt, cur_steering_angle):
        # =========
        # Approach 1, based on the equations of main paper
        # =========
        l_wb = 0.883 + 1.508
        if cur_steering_angle is None:
            raise ValueError("cur_steering_angle must be specified")
        s, n, ds, dn = state_vec
        a_x_tn, a_y_tn = control_vec
        xi = np.arctan(dn / (ds * (1 - n * self.road.get_curvature_at(s))))
        v_x = np.sqrt((ds * (1 - n * self.road.get_curvature_at(s))) **2 + dn ** 2)
        dpsi = (a_y_tn - np.tan(xi) * a_x_tn) / (v_x * (np.tan(xi) * np.sin(xi) + np.cos(xi)))
        a_x = (a_x_tn + v_x * dpsi * np.sin(xi)) / np.cos(xi)
        dC = (self.road.get_curvature_at(s + ds * dt) - self.road.get_curvature_at(s)) /dt
        dxi = 1 / (1 + (dn / (ds * (1 - n * self.road.get_curvature_at(s)))) ** 2) * (
                a_y_tn * ds * (1 - n * self.road.get_curvature_at(s)) - dn * (a_x_tn - self.road.get_curvature_at(s) * (a_x_tn * n + ds * dn) - dC * ds * n)
        ) / (ds * (1 - n * self.road.get_curvature_at(s))) ** 2
        delta = np.arctan((dxi + self.road.get_curvature_at(s) * ds) * l_wb / v_x)
        v_delta = max(min((delta - cur_steering_angle) / dt, 8), -8)
        return np.array([
            v_delta, a_x
        ])

    def _approach_2_initial(self, state_vec, control_vec, dt, cur_orientation):
        # =========
        # Approach 2, LAG
        # =========
        l_wb = 0.883 + 1.508

        # Aliases
        s_0, n_0, ds_0, dn_0 = state_vec
        u_t, u_n = control_vec
        s_1, n_1, ds_1, dn_1 = (
            s_0 + ds_0 * dt,
            n_0 + dn_0 * dt,
            ds_0 + u_t * dt,
            dn_0 + u_n * dt,
        )
        v_0 = np.sqrt(ds_0 ** 2 + dn_0 ** 2)
        v_1 = np.sqrt(ds_1 ** 2 + dn_1 ** 2)

        dpsi = (- cur_orientation + self.road.get_tangent_angle_at(s_0)) / dt
        # DSM-Controls
        a = (v_1 - v_0) / dt
        v_delta = np.arctan(dpsi * l_wb / v_0) / dt
        return np.array([
            v_delta, a
        ])

    @staticmethod
    def _approach_2(state_vec, control_vec, dt, cur_steering_angle, cur_psi, planned_psi_2):
        # =========
        # Approach 2, based on forward euler KST dynamics
        # =========
        l_wb = 0.883 + 1.508
        if cur_psi is None or planned_psi_2 is None:
            raise ValueError("cur_psi and planned_psi_2 must be specified")
        # Aliases
        s_0, n_0, ds_0, dn_0 = state_vec
        u_t, u_n = control_vec
        s_1, n_1, ds_1, dn_1 = (
            s_0 + ds_0 * dt,
            n_0 + dn_0 * dt,
            ds_0 + u_t * dt,
            dn_0 + u_n * dt,
        )
        v_0 = np.sqrt(ds_0 ** 2 + dn_0 ** 2)
        v_1 = np.sqrt(ds_1 ** 2 + dn_1 ** 2)

        # DSM-Controls
        a = (v_1 - v_0) /dt
        v_delta = np.arctan(l_wb * (planned_psi_2 - (cur_psi + 1/l_wb * v_0 * np.tan(cur_steering_angle) * dt)) / (v_1 * dt)) / dt
        return np.array([
            v_delta, a
        ])

    def _approach_3(self, state_vec, control_vec, dt, car_dynamics):
        # =========
        # Approach 3, solve system of equations (work in progress)
        # =========
        if car_dynamics is None:
            raise ValueError("dynamics must be specified")

        s_0, n_0, ds_0, dn_0 = state_vec
        u_t, u_n = control_vec
        s_1, n_1, ds_1, dn_1 = (
            s_0 + ds_0 * dt,
            n_0 + dn_0 * dt,
            ds_0 + u_t * dt,
            dn_0 + u_n * dt,
        )
        v_0 = np.sqrt(ds_0 ** 2 + dn_0 ** 2)
        v_1 = np.sqrt(ds_1 ** 2 + dn_1 ** 2)
        a = (v_1 - v_0) / dt

        xi_0 = np.arctan(dn_0 / ds_0)
        xi_1 = np.arctan(dn_1 / ds_1)
        psi_0 = xi_0 + self.road.get_tangent_angle_at(s_0)
        psi_1 = xi_1 + self.road.get_tangent_angle_at(s_1)
        dpsi = (psi_1 - psi_0) / dt

        # Define the system of equations
        def equations(u):
            u1 = u[0]  # Access the first element from the input array
            dyns = car_dynamics(u1, a)  # Ensure dynamics returns the correct structure
            return [
                dyns[2] - u1,
                dyns[3] - a,
                dyns[4] + dyns[5] * dt - dpsi,
            ]

        # Initial guess as an array
        initial_guess = np.array([dpsi, 0.0, 0.0])
        # Solve
        solution = fsolve(equations, initial_guess)

        # print("Solution:", solution)
        v_delta = solution[0]
        return np.array([
            v_delta, a
        ])

    def _approach_4(self, state_vec, control_vec, dt):
        # =========
        # Approach 4, dumbest one
        # =========
        l_wb = 0.883 + 1.508
        s_0, n_0, ds_0, dn_0 = state_vec
        u_t, u_n = control_vec
        s_1, n_1, ds_1, dn_1 = (
            s_0 + ds_0 * dt,
            n_0 + dn_0 * dt,
            ds_0 + u_t * dt,
            dn_0 + u_n * dt,
        )
        v_0 = np.sqrt(ds_0 ** 2 + dn_0 ** 2)
        v_1 = np.sqrt(ds_1 ** 2 + dn_1 ** 2)
        a = (v_1 - v_0) / dt

        xi_0 = np.arctan(dn_0 / ds_0)
        xi_1 = np.arctan(dn_1 / ds_1)
        psi_0 = xi_0 + self.road.get_tangent_angle_at(s_0)
        psi_1 = xi_1 + self.road.get_tangent_angle_at(s_1)
        dpsi = (psi_1 - psi_0) / dt

        v_delta = (np.arctan(l_wb * dpsi / v_1) - self.delta_cur) / dt
        self.delta_cur = np.arctan(l_wb * dpsi / v_1)

        return np.array([
            v_delta, a
        ])

    def get_name(self):
        return "Point Mass"