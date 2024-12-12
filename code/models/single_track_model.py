from typing import Any, List, Tuple, Union

import casadi as ca
import cvxpy as cp
import numpy as np

from models import AbstractVehicleModel
from utils import State

# Define a large constant for Big-M constraints
BIG_M = 1e6


class SingleTrackModel(AbstractVehicleModel):
    """
    SingleTrackModel implements a single-track (bicycle) vehicle model.
    It supports both CVXPY and CasADi solvers for optimization tasks.
    """
    def __init__(
        self,
        initial_state: Union[List[float], np.ndarray],
        goal_state: Union[List[float], np.ndarray],
        l_wb: float,
        v_s: float,
        steering_velocity_range: Tuple[float, float],
        steering_angle_range: Tuple[float, float],
        velocity_range: Tuple[float, float],
        acceleration_range: Tuple[float, float],
    ):
        """
        Initialize the SingleTrackModel.

        :param initial_state: Initial state vector [x_position, y_position, steering_angle, velocity, orientation].
        :param goal_state: Goal state vector [x_position, y_position, steering_angle, velocity, orientation].
        :param l_wb: Wheelbase length of the vehicle.
        :param v_s: Scaling velocity.
        :param steering_velocity_range: Tuple representing the lower and upper bounds of steering velocity.
        :param steering_angle_range: Tuple representing the lower and upper bounds of steering angle (in radians).
        :param velocity_range: Tuple representing the lower and upper bounds of velocity.
        :param acceleration_range: Tuple representing the lower and upper bounds of acceleration.
        :raises ValueError: If invalid solver_type is provided.
        """
        super().__init__(
            dim_state=5,
            dim_control_input=2,
            state_labels=['x_position', 'y_position', 'steering_angle', 'velocity', 'orientation'],
            control_input_labels=['Steering Velocity', 'Acceleration Input'],
            initial_state=initial_state,
            goal_state=goal_state,
        )
        # params
        self.l_wb = l_wb
        self.v_s = v_s

        # range aliases
        self.steering_velocity_lb, self.steering_velocity_ub = steering_velocity_range
        self.steering_angle_lb, self.steering_angle_ub = steering_angle_range
        self.velocity_lb, self.velocity_ub = velocity_range
        self.acceleration_lb, self.a_max = acceleration_range

    def acceleration_ub(self, v: Union[cp.Variable, ca.MX, float]) -> Tuple[Any, List[Any]]:
        if self.solver_type == 'casadi':
            # Use CasADi expressions directly
            acc_ub = self.a_max * (self.v_s / ca.fmax(v, self.v_s))
            constraints = []
        else:
            raise ValueError(f"solver_type {self.solver_type} not supported")

        return acc_ub, constraints

    def steer(
        self,
        steering_angle: Union[cp.Variable, ca.MX, float],
        steering_velocity: Union[cp.Variable, ca.MX, float]
    ) -> Tuple[Any, List[Any]]:
        """
        Compute the steering output and associated constraints.

        :param steering_angle: Current steering angle.
        :param steering_velocity: Current steering velocity input.
        :return: Tuple containing steering output and list of constraints.
        """
        if self.solver_type == 'casadi':
            # Implement logic using CasADi expressions
            # Conditions:
            # if (steering_angle <= steering_angle_lb and steering_velocity <= 0) or
            #    (steering_angle >= steering_angle_ub and steering_velocity >= 0):
            #     steering_output = 0
            # else:
            #     steering_output = steering_velocity (clipped to bounds)
            cond1 = ca.logic_or(
                ca.logic_and(steering_angle <= self.steering_angle_lb, steering_velocity <= 0),
                ca.logic_and(steering_angle >= self.steering_angle_ub, steering_velocity >= 0)
            )
            steering_output = ca.if_else(
                cond1,
                0,
                ca.fmin(ca.fmax(steering_velocity, self.steering_velocity_lb), self.steering_velocity_ub)
            )
            constraints = []
        else:
            raise ValueError(f"solver_type {self.solver_type} not supported")

        return steering_output, constraints

    def acc(
        self,
        velocity: Union[cp.Variable, ca.MX, float],
        acceleration: Union[cp.Variable, ca.MX, float]
    ) -> Tuple[Any, List[Any]]:
        """
        Compute the acceleration output and associated constraints.

        :param velocity: Current velocity.
        :param acceleration: Current acceleration input.
        :return: Tuple containing acceleration output and list of constraints.
        """
        if self.solver_type == 'casadi':
            # Implement logic using CasADi expressions
            # Conditions:
            # if (velocity <= velocity_lb and acceleration <= 0) or
            #    (velocity >= velocity_ub and acceleration >= 0):
            #     acc_output = 0
            # else:
            #     acc_output = acceleration (clipped to bounds)
            cond1 = ca.logic_or(
                ca.logic_and(velocity <= self.velocity_lb, acceleration <= 0),
                ca.logic_and(velocity >= self.velocity_ub, acceleration >= 0)
            )
            acc_ub, _ = self.acceleration_ub(velocity)
            acc_output = ca.if_else(
                cond1,
                0,
                ca.fmin(ca.fmax(acceleration, self.acceleration_lb), acc_ub)
            )
            constraints = []
        else:
            raise ValueError(f"solver_type {self.solver_type} not supported")

        return acc_output, constraints

    def update(
        self,
        current_state: np.ndarray,
        control_inputs: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, List[Any]]:
        """
        Update the vehicle's state based on current state and control inputs.

        :param current_state: Current state vector [x_position, y_position, steering_angle, velocity, orientation].
        :param control_inputs: Control inputs [steering_velocity, acceleration].
        :param dt: Time step for state updates.
        :return: Tuple containing the next state and list of constraints.
        :raises ValueError: If input shapes are incorrect.
        """
        if current_state.shape != (self.dim_state,) and current_state.shape != (self.dim_state, 1):
            raise ValueError(f"current_state must have shape ({self.dim_state},) "
                             f"or ({self.dim_state}, 1), got {current_state.shape}")
        if control_inputs.shape != (self.dim_control_input,) and control_inputs.shape != (self.dim_control_input, 1):
            raise ValueError(f"control_inputs must have shape ({self.dim_control_input},)  "
                             f"or ({self.dim_control_input}, 1), got {control_inputs.shape}")

        # Extract state variables
        if current_state.shape == (self.dim_state,):
            x_position, y_position, steering_angle, velocity, orientation = current_state
        else:
            x_position, y_position, steering_angle, velocity, orientation = [current_state[i, 0] for i in range(self.dim_state)]

        # Extract control inputs
        if control_inputs.shape == (self.dim_control_input,):
            steering_velocity, acc_input = control_inputs
        else:
            steering_velocity, acc_input = [control_inputs[i, 0] for i in range(self.dim_control_input)]

        x1, x2, x3, x4, x5 = x_position, y_position, steering_angle, velocity, orientation
        u1, u2 = steering_velocity, acc_input

        if self.solver_type == 'casadi':
            # Use CasADi expressions

            # Steering and acceleration outputs
            steering_output, steering_constraints = self.steer(x3, u1)
            acceleration_output, acceleration_constraints = self.acc(x4, u2)
            acc_ub, _ = self.acceleration_ub(velocity)

            dx_dt = [
                x4 * ca.cos(x5),
                x4 * ca.sin(x5),
                steering_output,
                acceleration_output,
                (x4 / self.l_wb) * ca.tan(x3)
            ]

            # Next state
            next_state = ca.vertcat(*[
                current_state[i] + dx_dt[i] * dt for i in range(self.dim_state)
            ])

            constraints = steering_constraints + acceleration_constraints + [
                u2**2 + (x4*dx_dt[4])**2 <= self.a_max**2,
                self.steering_angle_lb <= steering_angle, steering_angle <= self.steering_angle_ub,
                self.velocity_lb <= velocity, velocity <= self.velocity_ub,
                self.steering_velocity_lb <= steering_velocity, steering_velocity <= self.steering_velocity_ub,
                self.acceleration_lb <= acc_input, acc_input <= acc_ub
            ]
        else:
            raise ValueError(f"solver_type {self.solver_type} not supported")

        return next_state, constraints

    def convert_vec_to_state(self, vec) -> State:
        # vec: x_position, y_position, steering_angle, velocity, orientation
        self._validate__state_dimension(vec)
        return State(
            vec=vec,
            get_velocity=lambda: float(vec[3]),
            get_offset_from_reference_path=lambda: 0,
            get_remaining_distance=lambda: 0,
            get_traveled_distance=lambda: 0,
            get_distance_between=lambda other_state: self._sqrt(self._norm_squared(vec[:2] - other_state.as_vector()[:2])),
            get_position_orientation=lambda: (
                vec[:2], float(vec[4])
            ),
            to_string=self._state_vec_to_string(vec)
        )