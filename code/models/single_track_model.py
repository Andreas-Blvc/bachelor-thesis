import numpy as np
from math import pi
import casadi as ca
import cvxpy as cp
from math import cos, sin
from typing import Tuple, Any, List, Union

from models.vehicle_model import VehicleModel
from utils.state_space import State

# Define a large constant for Big-M constraints
BIG_M = 1e6


class SingleTrackModel(VehicleModel):
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
        dt: float,
        solver_type: str = 'cvxpy'
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
        :param dt: Time step for state updates.
        :param solver_type: Type of solver to use ('cvxpy' or 'casadi').
        :raises ValueError: If invalid solver_type is provided.
        """
        self.dim_state = 5
        self.dim_control_input = 2
        self.dt = dt
        self.solver_type = solver_type.lower()

        if self.solver_type not in ['cvxpy', 'casadi']:
            raise ValueError("solver_type must be either 'cvxpy' or 'casadi'.")

        self.l_wb = l_wb
        self.v_s = v_s
        self.steering_velocity_lb, self.steering_velocity_ub = steering_velocity_range
        self.steering_angle_lb, self.steering_angle_ub = steering_angle_range
        self.velocity_lb, self.velocity_ub = velocity_range
        self.acceleration_lb, self.a_max = acceleration_range

        if initial_state.shape != (self.dim_state,):
            raise ValueError(f"initial_state must have shape ({self.dim_state},), got {initial_state.shape}")
        if goal_state.shape != (self.dim_state,):
            raise ValueError(f"goal_state must have shape ({self.dim_state},), got {goal_state.shape}")

        self.initial_state = initial_state
        self.goal_state = goal_state

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
        control_inputs: np.ndarray
    ) -> Tuple[np.ndarray, List[Any]]:
        """
        Update the vehicle's state based on current state and control inputs.

        :param current_state: Current state vector [x_position, y_position, steering_angle, velocity, orientation].
        :param control_inputs: Control inputs [steering_velocity, acceleration].
        :return: Tuple containing the next state and list of constraints.
        :raises ValueError: If input shapes are incorrect.
        """
        if current_state.shape != (self.dim_state,) and current_state.shape != (self.dim_state, 1):
            raise ValueError(f"current_state must have shape ({self.dim_state},) or ({self.dim_state}, 1), got {current_state.shape}")
        if control_inputs.shape != (self.dim_control_input,) and control_inputs.shape != (self.dim_control_input, 1):
            raise ValueError(f"control_inputs must have shape ({self.dim_control_input},)  or ({self.dim_control_input}, 1), got {control_inputs.shape}")

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
                current_state[i] + dx_dt[i] * self.dt for i in range(self.dim_state)
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

    def accurate_acceleration_ub(self, v):
        return self.a_max * self.v_s / np.max((v, self.v_s))

    def accurate_steer(self, steering_angle, steering_velocity):
        if (steering_angle <= self.steering_angle_lb and steering_velocity <= 0) or (steering_angle >= self.steering_angle_ub and steering_velocity >= 0):
            return 0
        else:
            return np.min((np.max((steering_velocity, self.steering_velocity_lb)), self.steering_velocity_ub))

    def accurate_acc(self, velocity, acc_input):
        if (velocity <= self.velocity_lb and acc_input <= 0) or (velocity >= self.velocity_ub and acc_input >= 0):
            return 0
        else:
            return np.min((np.max((acc_input, self.acceleration_lb)), self.accurate_acceleration_ub(velocity)))

    def accurate_update(
            self,
            current_state: np.ndarray,
            control_inputs: np.ndarray
    ):
        _, _, x3, x4, x5 = current_state.flatten()
        u1, u2 = control_inputs.flatten()
        dx_dt = np.reshape([
            x4 * np.cos(x5),
            x4 * np.sin(x5),
            self.accurate_steer(x3, u1),
            self.accurate_acc(x4, u2),
            (x4/self.l_wb) * np.tan(x3)
        ], (self.dim_state,))

        return current_state + dx_dt * self.dt


    def get_initial_state(self) -> np.ndarray:
        return self.initial_state

    def get_goal_state(self) -> np.ndarray:
        return self.goal_state

    def get_position_orientation(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        if state.shape != (self.dim_state,):
            raise ValueError(f"state must have shape ({self.dim_state},), got {state.shape}")
        return state[:2], float(state[4])

    def get_vehicle_polygon(self, state: np.ndarray) -> List[Tuple[float, float]]:
        front_wheel_front = self._add_tuple(self._rotate((0.5, 0), float(state[2])), (1, 0))
        front_wheel_back = self._add_tuple(self._rotate((-0.5, 0), float(state[2])), (1, 0))
        return [
            (-1, 0.5), (1, 0.5),
                                (1, 0),
                    front_wheel_back, front_wheel_front,
                                (1, 0),
                       (1, -0.5),
            (-1, -0.5),
        ]

    def get_dim_state(self) -> int:
        return self.dim_state

    def get_dim_control_input(self) -> int:
        return self.dim_control_input

    def get_a_max(self) -> float:
        return self.a_max

    def to_string(self, state, control):
        _, _, steering_angle, velocity, orientation = state
        steering_velocity, acc_input = control

        orientation_deg = orientation * 180 / pi  # Convert radians to degrees
        steering_angle_deg = steering_angle * 180 / pi  # Convert radians to degrees
        return (
            f"Orientation = {orientation_deg:.2f}°, "
            f"Steering Angle = {steering_angle_deg:.2f}°, "
            f"Velocity = {velocity:.2f}m/s, "
            f"Control Inputs = [Steering Velocity: {steering_velocity:.5f}, "
            f"Acceleration Input: {acc_input:.5f}]"
        )

    @staticmethod
    def _add_tuple(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
        return (
            a[0] + b[0],
            a[1] + b[1],
        )

    @staticmethod
    def _rotate(point: Tuple[float, float], theta: float) -> Tuple[float, float]:
        x, y = point
        return (
            x * cos(theta) - y * sin(theta),
            y * cos(theta) + x * sin(theta),
        )

    def get_control_input_labels(self):
        return ['Steering Velocity', 'Acceleration Input']

    def get_distance_between(self, state_a: State, state_b: State):
        x_a, y_a = state_a.as_vector()[:2]
        x_b, y_b = state_b.as_vector()[:2]
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
        return state.as_vector()[3]