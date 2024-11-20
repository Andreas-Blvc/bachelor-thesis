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
        super().__init__(
            dim_state=4,
            dim_control_input=2,
            state_labels=['s', 'n', 'ds', 'dn'],
            control_input_labels=['u_t', 'u_n'],
            initial_state=initial_state,
            goal_state=goal_state
        )
        # Params
        self.dt = dt
        self.road = road

        # Aliases for range access
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

        MainPaperConstraintsReduction.apply_all(
            state_ranges=ranges,
            v_x_range=v_x_range,
            acc_x_range=acc_x_range,
            acc_y_range=acc_y_range,
            yaw_rate_range=yaw_rate_range,
            yaw_acc_range=yaw_acc_range,
            curvature_derivative=road.get_curvature_derivative_at(0.5)  # constant for all s

        )

        self.ranges = ranges
        print(ranges)

        # helper:
        self.last_orientation = 0


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

    def get_vehicle_polygon(self, state) -> List[Tuple[float, float]]:
        return [
            (-1, 0.5), (1, 0.5),
                       (1, -0.5),
            (-1, -0.5)
        ]

    def convert_vec_to_state(self, vec) -> State:
        # vec: s, n, ds, dn
        self._validate__state_dimension(vec)
        return State(
            vec=vec,
            get_velocity=lambda: self._sqrt(self._norm_squared(vec[2:])),
            get_offset_from_reference_path= lambda: self._absolute(vec[1]),
            get_remaining_distance=lambda: self.road.length - vec[0],
            get_traveled_distance=lambda: vec[0] - self.initial_state[0],
            get_distance_between=lambda other_state: self._norm_squared(vec[:2] - other_state.as_vector()[:2]),
            to_string=lambda: self.state_vec_to_string(vec),
            get_position_orientation=lambda: (
                np.array(self.road.get_global_position(vec[0], vec[1])),
                self.road.get_tangent_angle_at(vec[0]),
            )
        )