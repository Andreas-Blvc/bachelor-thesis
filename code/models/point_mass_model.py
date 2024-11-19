from typing import List

import casadi as ca
import cvxpy as cp
import numpy as np

from models import AbstractVehicleModel
from utils import State


class PointMassModelAbstract(AbstractVehicleModel):
    """
    A point mass vehicle model implementing the VehicleModel abstract base class.
    This model represents a simple point mass with position and velocity in 2D space.
    """
    def __init__(self, initial_state: np.ndarray, goal_state: np.ndarray, a_max: float, dt: float):
        """
        Initialize the PointMassModel.

        :param initial_state: Initial state vector of shape (4,) representing [x, y, vx, vy].
        :param goal_state: Goal state vector of shape (4,) representing [x_goal, y_goal, vx_goal, vy_goal].
        :param a_max: Maximum allowable acceleration.
        :param dt: Time step for state updates.
        :raises ValueError: If initial_state or goal_state do not have the correct shape.
        """
        self.solver_type = None
        self.dim_state = 4
        self.dim_control_input = 2
        self.dt = dt
        self.a_max = a_max

        if initial_state.shape != (self.dim_state,):
            raise ValueError(f"initial_state must have shape ({self.dim_state},), got {initial_state.shape}")
        if goal_state.shape != (self.dim_state,):
            raise ValueError(f"goal_state must have shape ({self.dim_state},), got {goal_state.shape}")

        self.initial_state = initial_state
        self.goal_state = goal_state

        # Define the A matrix (state transition matrix)
        self.A = np.zeros((self.dim_state, self.dim_state))
        self.A[0, 2] = 1  # dx/dt = vx
        self.A[1, 3] = 1  # dy/dt = vy

        # Define the B matrix (input matrix)
        self.B = np.zeros((self.dim_state, self.dim_control_input))
        self.B[2, 0] = 1  # dvx/dt = ax
        self.B[3, 1] = 1  # dvy/dt = ay

    def update(self, current_state: np.ndarray, control_inputs: np.ndarray):
        if current_state.shape != (self.dim_state,) and current_state.shape != (self.dim_state, 1):
            raise ValueError(f"current_state must have shape ({self.dim_state},) or ({self.dim_state}, 1), got {current_state.shape}")
        if control_inputs.shape != (self.dim_control_input,) and control_inputs.shape != (self.dim_control_input, 1):
            raise ValueError(f"control_inputs must have shape ({self.dim_control_input},)  or ({self.dim_control_input}, 1), got {control_inputs.shape}")

        # Compute the next state based on the input acceleration
        next_state = current_state + (self.A @ current_state + self.B @ control_inputs) * self.dt

        # Define the constraint for acceleration within limits
        constraints = [
            cp.norm(control_inputs, "inf") <= self.a_max  # Constrain acceleration inputs
        ]

        return next_state, constraints

    def get_initial_state(self):
        return self.initial_state

    def get_goal_state(self):
        return self.goal_state

    def get_position_orientation(self, state):
        if state.shape == (self.dim_state,):
            position = state[:2]
            orientation = 0.0  # Assuming fixed orientation for simplicity
            return position, orientation
        else:
            raise ValueError(f"state must have shape ({self.dim_state},) or ({self.dim_state},), got {state.shape}")

    def get_vehicle_polygon(self, state: np.ndarray):
        return [(0.0, 0.0)]  # Representing a point

    def get_dim_state(self):
        return self.dim_state

    def get_dim_control_input(self):
        return self.dim_control_input

    def get_a_max(self):
        return self.a_max

    def to_string(self, state, control):
        a_x, a_y = control

        return (
            f"a_x = {a_x:.5f}, "
            f"a_y = {a_y:.5f}"
        )

    def get_control_input_labels(self) -> List[str]:
        return ['Longitude Acceleration', 'Latitude Acceleration']

    def get_distance_between(self, state_a: State, state_b: State):
        x_a, y_a = state_a.as_vector()[2:]
        x_b, y_b = state_b.as_vector()[2:]
        if self.solver_type == 'casadi':
            sqrt = ca.sqrt
        elif self.solver_type == 'cvxpy':
            sqrt = cp.sqrt
        else:
            raise ValueError(f"solver_type {self.solver_type} not supported")
        return sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2)

    def get_traveled_distance(self, state: State):
        return 0

    def get_remaining_distance(self, state: State):
        return 0

    def get_offset_from_reference_path(self, state: State):
        return 0

    def get_velocity(self, state: State):
        if self.solver_type == 'casadi':
            sqrt = ca.sqrt
        elif self.solver_type == 'cvxpy':
            sqrt = cp.sqrt
        else:
            raise ValueError(f"solver_type {self.solver_type} not supported")
        return sqrt(state.as_vector()[2] ** 2 + state.as_vector()[3] ** 2)
