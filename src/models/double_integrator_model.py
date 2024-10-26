import numpy as np
import cvxpy as cp
from models.vehicle_model import VehicleModel

class DoubleIntegrator(VehicleModel):
    def __init__(self, initial_state, goal_state, a_max: float, dt: float):
        self.dim_state = 4
        self.dim_control_input = 2
        self.dt = dt
        self.a_max = a_max
        self.initial_state = np.reshape(initial_state, (4, 1))
        self.goal_state = np.reshape(goal_state, (4, 1))

        # Define the A matrix (state transition matrix)
        self.A = np.zeros((4, 4))
        self.A[0, 2] = 1
        self.A[1, 3] = 1

        # Define the B matrix (input matrix)
        self.B = np.zeros((4, 2))
        self.B[2, 0] = 1
        self.B[3, 1] = 1

    def update(self, current_state, acceleration_inputs):
        """
        Compute the next state as a CVXPY expression without updating the internal state.
        """
        # Compute the next state based on the input acceleration
        next_state = current_state + (self.A @ current_state + self.B @ acceleration_inputs) * self.dt

        # Define the constraint for acceleration within limits
        constraints = [
            cp.norm(acceleration_inputs, "inf") <= self.a_max  # Constrain acceleration inputs
        ]
        return next_state, constraints

    def get_initial_state(self):
        return self.initial_state

    def get_goal_state(self):
        return self.goal_state

    import numpy as np

    def get_position_orientation(self, state):
        if isinstance(state, list):
            state = np.array(state)

        if state.shape == (4, 1):
            return state[:2, 0], 0
        else:
            return state[:2], 0

    def get_shape(self):
        return []

    def get_dim_state(self):
        return self.dim_state

    def get_dim_control_input(self):
        return self.dim_control_input
