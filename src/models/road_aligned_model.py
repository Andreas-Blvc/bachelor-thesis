from typing import List, Tuple, Any
import numpy as np
import casadi as ca
from math import atan

from models.vehicle_model import VehicleModel
from obstacles.road import Road

class RoadAlignedModel(VehicleModel):

    def __init__(self, initial_state: np.ndarray, goal_state: np.ndarray, dt: float, road: Road,
                 v_x_range: Tuple[float, float],
                 v_y_range: Tuple[float, float],
                 acc_x_range: Tuple[float, float],
                 acc_y_range: Tuple[float, float],
                 yaw_rate_range: Tuple[float, float],
                 yaw_acc_range: Tuple[float, float],
                 a_max
                 ):
        """
        Initialize the RoadAlignedModel.

        :param initial_state: Initial state vector of shape (4,) representing [s, n, ds, dn].
        :param goal_state: Goal state vector of shape (4,) representing [s, n, ds, dn].
        :param dt: Time step for state updates.
        :raises ValueError: If initial_state or goal_state do not have the correct shape.
        """
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

        self.c_min = road.get_curvature_min()
        self.c_max = road.get_curvature_max()
        self.n_min = -road.width
        self.n_max = road.width
        self.v_x_min, self.v_x_max = v_x_range
        self.v_y_min, self.v_y_max = v_y_range
        self.acc_x_min, self.acc_x_max = acc_x_range
        self.acc_y_min, self.acc_y_max = acc_y_range
        self.yaw_rate_min, self.yaw_rate_max = yaw_rate_range
        self.yaw_acc_min, self.yaw_acc_max = yaw_acc_range
        self.a_max = a_max
        self.last_orientation = 0

        self.counter = 0

    def _g(self, x_tn, u):
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

        # Compute the next state based on the input accelerations
        next_state = ca.vertcat(*[
            current_state[i] + dx_dt[i] * self.dt for i in range(self.dim_state)
        ])

        # Define the constraint for acceleration within limits
        s, n, ds, dn = [current_state[i] for i in range(self.dim_state)]
        u_t, u_n = [control_inputs[i] for i in range(self.dim_control_input)]
        g = self._g(current_state, control_inputs)
        constraints = [
            0 <= s, s <= 1,
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
        print(f'Update #{self.counter}', end='\r' if self.counter % 10 != 0 else '\n')
        self.counter += 1
        return next_state, constraints

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

