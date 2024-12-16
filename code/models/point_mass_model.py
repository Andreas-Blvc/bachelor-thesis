import cvxpy as cp
import numpy as np

from models import AbstractVehicleModel
from utils import State


class PointMassModel(AbstractVehicleModel):
    """
    A point mass vehicle model implementing the VehicleModel abstract base class.
    This model represents a simple point mass with position and velocity in 2D space.
    """
    def __init__(self, a_max: float):
        """
        Initialize the PointMassModel.

        state vector of shape (4,) representing [x, y, vx, vy].
        :param a_max: Maximum allowable acceleration.
        :raises ValueError: If initial_state or goal_state do not have the correct shape.
        """
        super().__init__(
            dim_state=4,
            dim_control_input=2,
            state_labels=['x', 'y', 'v_x', 'v_y'],
            control_input_labels=['Longitude Acceleration', 'Latitude Acceleration'],
        )
        # params
        self.a_max = a_max

        # Define the A matrix (state transition matrix)
        self.A = np.zeros((self.dim_state, self.dim_state))
        self.A[0, 2] = 1  # dx/dt = vx
        self.A[1, 3] = 1  # dy/dt = vy

        # Define the B matrix (input matrix)
        self.B = np.zeros((self.dim_state, self.dim_control_input))
        self.B[2, 0] = 1  # dvx/dt = ax
        self.B[3, 1] = 1  # dvy/dt = ay

    def update(self, current_state: np.ndarray, control_inputs: np.ndarray, dt):
        self._validate__state_dimension(current_state)
        self._validate__control_dimension(control_inputs)

        if self.solver_type == 'cvxpy':
            # Compute the next state based on the input acceleration
            next_state = current_state + (self.A @ current_state + self.B @ control_inputs) * dt

            # Define the constraint for acceleration within limits
            constraints = [
                cp.norm(control_inputs, "inf") <= self.a_max  # Constrain acceleration inputs
            ]
        else:
            self._raise_unsupported_solver()

        return next_state, constraints

    def convert_vec_to_state(self, vec) -> State:
        # vec:
        self._validate__state_dimension(vec)
        return State(
            vec=vec,
            get_distance_between=lambda other_state: self._norm_squared(vec[2:] - other_state.as_vector()[2:]),
            get_remaining_distance=lambda: 0,
            get_traveled_distance=lambda: 0,
            get_offset_from_reference_path=lambda: 0,
            get_velocity=lambda: self._sqrt(self._norm_squared(vec[2:])),
            get_position_orientation=lambda: (vec[:2], .0),
            to_string=lambda: self._state_vec_to_string(vec)
        )
