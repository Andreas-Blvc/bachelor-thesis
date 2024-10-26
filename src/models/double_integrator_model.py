import numpy as np
import cvxpy as cp
from models.vehicle_model import VehicleModel

class DoubleIntegrator(VehicleModel):
    def __init__(self, initial_state, a_max: float, dt: float):
        self.dt = dt
        self.a_max = a_max
        # Initial state: concatenate initial position with zero velocity
        self.initial_state = np.reshape(initial_state, (4, 1))

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
