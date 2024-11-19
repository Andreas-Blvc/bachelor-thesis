from math import atan
from typing import Any, List, Tuple

import casadi as ca
import cvxpy as cp
import numpy as np

from models import AbstractVehicleModel
from obstacles import AbstractRoad
from utils import MainPaperConstraintsReduction, State, StateRanges
from visualizer import plot_with_bounds


class RoadAlignedModelAbstract(AbstractVehicleModel):

    def __init__(self, initial_state: np.ndarray, goal_state: np.ndarray, dt: float, road: AbstractRoad,
                 v_x_range: Tuple[float, float],
                 v_y_range: Tuple[float, float],
                 acc_x_range: Tuple[float, float],
                 acc_y_range: Tuple[float, float],
                 yaw_rate_range: Tuple[float, float],
                 yaw_acc_range: Tuple[float, float],
                 a_max,
                 ):
        """
        Initialize the RoadAlignedModel.

        :param initial_state: Initial state vector of shape (4,) representing [s, n, ds, dn].
        :param goal_state: Goal state vector of shape (4,) representing [s, n, ds, dn].
        :param dt: Time step for state updates.
        :raises ValueError: If initial_state or goal_state do not have the correct shape.
        """
        self.solver_type = None
        self.dim_state = 4
        self.dim_control_input = 2
        self.dt = dt
        self.road = road

        if initial_state.shape != (self.dim_state,):
            raise ValueError(f"initial_state must have shape ({self.dim_state},), got {initial_state.shape}")
        if goal_state.shape != (self.dim_state,):
            raise ValueError(f"goal_state must have shape ({self.dim_state},), got {goal_state.shape}")

        self.initial_state = initial_state
        self.goal_state = goal_state

        self.c_min = road.get_curvature_min(float(initial_state[0]), float(goal_state[0]))
        self.c_max = road.get_curvature_max(float(initial_state[0]), float(goal_state[0]))
        self.n_min = -road.width/2
        self.n_max = road.width/2
        self.v_x_min, self.v_x_max = v_x_range
        self.v_y_min, self.v_y_max = v_y_range
        self.acc_x_min, self.acc_x_max = acc_x_range
        self.acc_y_min, self.acc_y_max = acc_y_range
        self.yaw_rate_min, self.yaw_rate_max = yaw_rate_range
        self.yaw_acc_min, self.yaw_acc_max = yaw_acc_range
        self.a_max = a_max

        ranges = StateRanges(
            n=(self.n_min, self.n_max),
            c=(self.c_min, self.c_max),
            ds=(-0.01, 0.16), # tbd: should not be necessary to set here
            dn=v_y_range,
            u_n=None,
            u_t=None,
        )

        # self.v_x_min / (1 + self.nc_max) <= ds <=  self.v_x_max / (1 + self.nc_min)
        new_ranges = MainPaperConstraintsReduction.v_x_constraint_reduction(
            v_x_range=v_x_range,
            c_range=ranges.c,
            n_range=ranges.n,
        )
        ranges.update(new_ranges)

        # self.yaw_rate_min <= C(s) * ds <= self.yaw_rate_max,
        new_ranges = MainPaperConstraintsReduction.yaw_rate_constraint_reduction(
            yaw_rate_range=yaw_rate_range,
            c_range=ranges.c,
        )
        ranges.update(new_ranges)

        # self.yaw_acc_min <= C'(s) * ds**2 + C(s) * u_t <= self.yaw_acc_max,
        new_ranges = MainPaperConstraintsReduction.yaw_acceleration_constraint_reduction(
            yaw_acceleration_range=yaw_acc_range,
            c_range=ranges.c,
            ds_range=ranges.ds,
            dc_ds=road.get_curvature_derivative_at(0.5) # constant for all s
        )
        ranges.update(new_ranges)

        # self.acc_x_min <= g[0] <= self.acc_x_max,
        new_ranges = MainPaperConstraintsReduction.x_acceleration_constraint_reduction(
            x_acceleration_range=acc_x_range,
            c_range=ranges.c,
            n_range=ranges.n,
            dn_range=ranges.dn,
            ds_range=ranges.ds,
            dc_ds=road.get_curvature_derivative_at(0.5) # constant for all s
        )
        ranges.update(new_ranges)

        # self.acc_y_min <= g[1] <= self.acc_y_max,
        new_ranges = MainPaperConstraintsReduction.y_acceleration_constraint_reduction(
            y_acceleration_range=acc_y_range,
            c_range=ranges.c,
            n_range=ranges.n,
            ds_range=ranges.ds,
        )
        ranges.update(new_ranges)

        self.ranges = ranges
        print(ranges)
        # helper:
        self.last_orientation = 0
        self.counter = 0


    def to_body_fixed(self, x_tn, u):
        a_tn = self.g(x_tn, u)
        s, n, ds, dn = [x_tn[i] for i in range(self.dim_state)]
        # u_t, u_n = [u[i] for i in range(self.dim_control_input)]
        C = self.road.get_curvature_at(s)
        return (
            a_tn[0] + C*ds*dn,
            a_tn[1] - C*(ds**2)*(1-n*C),
        )

    def g(self, x_tn, u):
        s, n, ds, dn = [x_tn[i] for i in range(self.dim_state)]
        u_t, u_n = [u[i] for i in range(self.dim_control_input)]
        return [
            (1 - n*self.road.get_curvature_at(s)) * u_t - (2*dn*self.road.get_curvature_at(s)*ds + n*self.road.get_curvature_derivative_at(s)*ds**2),
            u_n + self.road.get_curvature_at(s) * ds**2 * (1-n*self.road.get_curvature_at(s)),
        ]

    def update(self, current_state, control_inputs) -> Tuple[np.ndarray, List[Any]]:
        if current_state.shape != (self.dim_state,) and current_state.shape != (self.dim_state, 1):
            raise ValueError(f"current_state must have shape ({self.dim_state},) or ({self.dim_state}, 1), got {current_state.shape}")
        if control_inputs.shape != (self.dim_control_input,) and control_inputs.shape != (self.dim_control_input, 1):
            raise ValueError(f"control_inputs must have shape ({self.dim_control_input},)  or ({self.dim_control_input}, 1), got {control_inputs.shape}")

        dx_dt = np.array([
            current_state[2],
            current_state[3],
            control_inputs[0],
            control_inputs[1],
        ])

        s, n, ds, dn = [current_state[i] for i in range(self.dim_state)]
        u_t, u_n = [control_inputs[i] for i in range(self.dim_control_input)]
        g = self.g(current_state, control_inputs)

        if self.solver_type == 'casadi':
            # Compute the next state based on the input accelerations
            next_state = ca.vertcat(*[
                current_state[i] + dx_dt[i] * self.dt for i in range(self.dim_state)
            ])

            # Define the constraint for acceleration within limits
            constraints = [
                # 0 <= s, s <= 1,
                # 0 <= ds,
                self.c_min <= self.road.get_curvature_at(s), self.road.get_curvature_at(s) <= self.c_max,
                self.n_min <= n, n <= self.n_max,
                self.v_x_min <= ds*(1 + n*self.road.get_curvature_at(s)), ds*(1 + n*self.road.get_curvature_at(s)) <= self.v_x_max,
                self.v_y_min <= dn, dn <= self.v_y_max,
                self.yaw_rate_min <= self.road.get_curvature_at(s) * ds,
                self.road.get_curvature_at(s) * ds <= self.yaw_rate_max,
                self.yaw_acc_min <= self.road.get_curvature_derivative_at(s) * ds**2 + self.road.get_curvature_at(s) * u_t,
                self.road.get_curvature_derivative_at(s) * ds ** 2 + self.road.get_curvature_at(s) * u_t <= self.yaw_acc_max,
                self.acc_x_min <= g[0], g[0] <= self.acc_x_max,
                self.acc_y_min <= g[1], g[1] <= self.acc_y_max,
                g[0]**2 + g[1]**2 <= self.a_max,
            ]
        elif self.solver_type == 'cvxpy':
            next_state = cp.vstack([current_state[i] + dx_dt[i] * self.dt for i in range(self.dim_state)]).flatten()
            constraints = [
                0 <= s, s <= 1,
                self.ranges.c[0] <= self.road.get_curvature_at(s), self.road.get_curvature_at(s) <= self.ranges.c[1],
                self.ranges.n[0] <= n, n <= self.ranges.n[1],
                self.ranges.ds[0] <= ds, ds <= self.ranges.ds[1],
                self.ranges.dn[0] <= dn, dn <= self.ranges.dn[1],
                self.ranges.u_t[0] <= u_t, u_t <= self.ranges.u_t[1],
                self.ranges.u_n[0] <= u_n, u_n <= self.ranges.u_n[1],
            ]
        else:
            raise ValueError(f"solver_type {self.solver_type} not supported")

        return next_state, constraints

    def visualize_constraints(self, current_states, control_inputs):
        current_states_s = [x[0] for x in current_states]
        current_states_n = [x[1] for x in current_states]
        current_states_ds = [x[2] for x in current_states]
        current_states_dn = [x[3] for x in current_states]

        v_x_s = [ds*(1 + n*self.road.get_curvature_at(s)) for s, ds, n in zip(current_states_s, current_states_ds, current_states_n)]
        plot_with_bounds(
            y_values=v_x_s,
            lower_bound=self.v_x_min,
            upper_bound=self.v_x_max,
            y_label='C(s)'
        )


    def get_initial_state(self) -> np.ndarray:
        return self.initial_state

    def get_goal_state(self) -> np.ndarray:
        return self.goal_state

    def get_position_orientation(self, state) -> Tuple[np.ndarray, float]:
        s, n, ds, dn = state
        cur_pos = np.array(self.road.get_global_position(s, n))
        next_pos = np.array(self.road.get_global_position(s + ds * self.dt, n + dn * self.dt))
        delta_y = next_pos[1] - cur_pos[1]
        delta_x = next_pos[0] - cur_pos[0]
        if delta_y != 0 and delta_x != 0:
            orientation = atan(delta_y / delta_x)
            self.last_orientation = orientation
        else:
            orientation = self.last_orientation
        return (
            cur_pos,
            orientation,
        )

    def get_vehicle_polygon(self, state) -> List[Tuple[float, float]]:
        return [
            (-1, 0.5), (1, 0.5),
                       (1, -0.5),
            (-1, -0.5)
        ]

    def get_dim_state(self) -> int:
        return self.dim_state

    def get_dim_control_input(self) -> int:
        return self.dim_control_input

    def get_a_max(self) -> float:
        return self.a_max

    def to_string(self, state, control):
        # Unpack state and control for clarity
        s, n, ds, dn = state
        u_t, u_n = control

        # Format the output string
        state_str = f"State: [s: {s:.2f}, n: {n:.2f}, ds: {ds:.2f}, dn: {dn:.2f}]"
        control_str = f"Control: [u_t: {u_t:.2f}, u_n: {u_n:.2f}]"

        return f"{state_str} | {control_str}"

    def get_control_input_labels(self) -> List[str]:
        return [
            'u_t',
            'u_n'
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
        if self.solver_type == 'casadi':
            sqrt = ca.sqrt
        elif self.solver_type == 'cvxpy':
            sqrt = cp.sqrt
        else:
            raise ValueError(f"solver_type {self.solver_type} not supported")
        return sqrt(state.as_vector()[2] ** 2 + state.as_vector()[3] ** 2)