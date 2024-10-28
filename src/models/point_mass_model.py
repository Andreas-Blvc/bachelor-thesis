import numpy as np
import cvxpy as cp
from models.vehicle_model import VehicleModel


class PointMassModel(VehicleModel):
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
        if current_state.shape != (self.dim_state,):
            raise ValueError(f"current_state must have shape ({self.dim_state},), got {current_state.shape}")
        if control_inputs.shape != (self.dim_control_input,):
            raise ValueError(f"acceleration_inputs must have shape ({self.dim_control_input},), got {control_inputs.shape}")

        # Compute the next state based on the input acceleration
        next_state = current_state + (self.A @ current_state + self.B @ control_inputs) * self.dt

        # Define the constraint for acceleration within limits
        uneq_constraints = [
            cp.norm(control_inputs, "inf") <= self.a_max  # Constrain acceleration inputs
        ]

        return next_state, [], uneq_constraints

    def get_initial_state(self):
        return self.initial_state

    def get_goal_state(self):
        return self.goal_state

    def get_position_orientation(self, state):
        if state.shape == (self.dim_state,):
            position = state[:2, 0]
            orientation = 0.0  # Assuming fixed orientation for simplicity
            return position, orientation
        else:
            raise ValueError(f"state must have shape ({self.dim_state},) or ({self.dim_state},), got {state.shape}")

    def get_shape(self, state: np.ndarray):
        return [(0.0, 0.0)]  # Representing a point

    def get_dim_state(self):
        return self.dim_state

    def get_dim_control_input(self):
        return self.dim_control_input

    def get_a_max(self):
        return self.a_max
