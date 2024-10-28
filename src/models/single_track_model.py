from xmlrpc.client import Binary

import numpy as np
import casadi as ca
import cvxpy as cp
from math import cos, sin, tan
from typing import Tuple, Any, List, Union

from models.vehicle_model import VehicleModel

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
        a_max: float,
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
        :param a_max: Maximum allowable acceleration.
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

        self.a_max = a_max
        self.l_wb = l_wb
        self.v_s = v_s
        self.steering_velocity_lb, self.steering_velocity_ub = steering_velocity_range
        self.steering_angle_lb, self.steering_angle_ub = steering_angle_range
        self.velocity_lb, self.velocity_ub = velocity_range
        self.acceleration_lb, self.acceleration_ub_val = acceleration_range

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
            # Use CVXPY variables and constraints
            acc_ub = cp.Variable()
            constraints = [
                acc_ub == self.a_max # * (self.v_s / cp.maximum(v, self.v_s))
            ]
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
            # Use CVXPY variables and constraints
            steering_output = cp.Variable()
            # Define approximate Boolean variables
            z = cp.Variable()
            b1 = cp.Variable()
            b2 = cp.Variable()
            b3 = cp.Variable()
            c1 = cp.Variable()
            c2 = cp.Variable()

            d1, d2, d3, d4 = [cp.Variable() for _ in range(4)]

            constraints = [
              0 <= z, z <= 1,
              0 <= b1, b1 <= 1,
              0 <= b2, b2 <= 1,
              0 <= b3, b3 <= 1,
              0 <= c1, c1 <= 1,
              0 <= c2, c2 <= 1,
              0 <= d1, d1 <= 1,
              0 <= d2, d2 <= 1,
              0 <= d3, d3 <= 1,
              0 <= d4, d4 <= 1,
            ] + [
                self.steering_angle_lb - steering_angle <= BIG_M * b1,
                self.steering_angle_lb - steering_angle >= -BIG_M * (1 - b1),
                steering_angle - self.steering_angle_ub <= BIG_M * b2,
                steering_angle - self.steering_angle_ub >= -BIG_M * (1 - b2),
                steering_velocity >= -BIG_M * b3,
                steering_velocity <= BIG_M * (1 - b3),
                b1 + b3 <= 1 + c1,
                b1 + b3 >= 2 * c1,
                b1 + (1 - b3) <= 1 + c2,
                b1 + (1 - b3) >= 2 * c2,
                c1 + c2 >= z,
                c1 + c2 <= 2 * z,
                steering_output <= BIG_M * (1 - z),
                steering_output >= -BIG_M * (1 - z),
                steering_output <= self.steering_velocity_ub,
                steering_output >= self.steering_velocity_lb,

                d1 + d2 + d3 + d4 == 1,

                steering_output - 0 <= BIG_M * (1 - d1),
                0 - steering_output <= BIG_M * (1 - d1),
                steering_output - self.steering_velocity_lb <= BIG_M * (1 - d2),
                self.steering_velocity_lb - steering_output <= BIG_M * (1 - d2),
                steering_output - self.steering_velocity_ub <= BIG_M * (1 - d3),
                self.steering_velocity_ub - steering_output <= BIG_M * (1 - d3),
                steering_output - steering_velocity <= BIG_M * (1 - d4),
                steering_velocity - steering_output <= BIG_M * (1 - d4),
            ]
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
            # Use CVXPY variables and constraints
            acc_output = cp.Variable()
            # Define approximate Boolean variables
            z = cp.Variable()
            b1 = cp.Variable()
            b2 = cp.Variable()
            b3 = cp.Variable()
            c1 = cp.Variable()
            c2 = cp.Variable()

            d1, d2, d3, d4 = [cp.Variable() for _ in range(4)]

            constraints = [
              0 <= z, z <= 1,
              0 <= b1, b1 <= 1,
              0 <= b2, b2 <= 1,
              0 <= b3, b3 <= 1,
              0 <= c1, c1 <= 1,
              0 <= c2, c2 <= 1,
              0 <= d1, d1 <= 1,
              0 <= d2, d2 <= 1,
              0 <= d3, d3 <= 1,
              0 <= d4, d4 <= 1,
            ] + [
              self.velocity_lb - velocity <= BIG_M * b1,
              self.velocity_lb - velocity >= -BIG_M * (1 - b1),
              velocity - self.velocity_ub <= BIG_M * b2,
              velocity - self.velocity_ub >= -BIG_M * (1 - b2),
              acceleration >= -BIG_M * b3,
              acceleration <= BIG_M * (1 - b3),
              b1 + b3 <= 1 + c1,
              b1 + b3 >= 2 * c1,
              b1 + (1 - b3) <= 1 + c2,
              b1 + (1 - b3) >= 2 * c2,
              c1 + c2 >= z,
              c1 + c2 <= 2 * z,
              acc_output <= BIG_M * (1 - z),
              acc_output >= -BIG_M * (1 - z),
            ]
            acc_ub, acc_constraints = self.acceleration_ub(velocity)
            constraints += [
                acc_output >= self.acceleration_lb,
                acc_output <= acc_ub,

                d1 + d2 + d3 + d4 == 1,

                acc_output - 0 <= BIG_M * (1-d1),
                0 - acc_output <= BIG_M * (1-d1),
                acc_output - self.acceleration_lb <= BIG_M * (1-d2),
                self.acceleration_lb - acc_output <= BIG_M * (1-d2),
                acc_output - acc_ub <= BIG_M * (1-d3),
                acc_ub - acc_output <= BIG_M * (1-d3),
                acc_output - acceleration <= BIG_M * (1-d4),
                acceleration - acc_output <= BIG_M * (1-d4),
            ] + acc_constraints
        return acc_output, constraints

    def update(
        self,
        current_state: np.ndarray,
        control_inputs: np.ndarray
    ) -> Tuple[np.ndarray, List[Any], List[Any]]:
        """
        Update the vehicle's state based on current state and control inputs.

        :param current_state: Current state vector [x_position, y_position, steering_angle, velocity, orientation].
        :param control_inputs: Control inputs [steering_velocity, acceleration].
        :return: Tuple containing the next state and list of constraints.
        :raises ValueError: If input shapes are incorrect.
        """
        if current_state.shape != (self.dim_state,):
            raise ValueError(f"current_state must have shape ({self.dim_state},), got {current_state.shape}")
        if control_inputs.shape != (self.dim_control_input,):
            raise ValueError(f"control_inputs must have shape ({self.dim_control_input},), got {control_inputs.shape}")

        # Extract state variables
        x_position, y_position, steering_angle, velocity, orientation = current_state.T

        # Extract control inputs
        steering_velocity, acc_input = control_inputs

        if self.solver_type == 'casadi':
            # Use CasADi expressions
            steering_angle_cas = ca.MX.sym('steering_angle')
            # y_pos_cas = ca.MX.sym('y_pos')
            velocity_cas = ca.MX.sym('velocity')
            # acceleration_cas = ca.MX.sym('acceleration')
            orientation_cas = ca.MX.sym('orientation')

            # Steering and acceleration outputs
            steering_output, steering_constraints = self.steer(steering_angle_cas, steering_velocity)
            acceleration_output, acceleration_constraints = self.acc(velocity_cas, acc_input)

            # Dynamics equations
            cos_theta = ca.cos(orientation_cas)
            sin_theta = ca.sin(orientation_cas)
            tan_theta = ca.tan(steering_angle_cas)

            dx_dt = [
                velocity_cas * cos_theta,
                velocity_cas * sin_theta,
                steering_output,
                acceleration_output,
                (velocity_cas / self.l_wb) * tan_theta
            ]

            # Next state
            next_state = ca.vertcat(*[
                current_state[i] + dx_dt[i] * self.dt for i in range(self.dim_state)
            ])

            uneq_constraints = steering_constraints + acceleration_constraints

        else:
            # Use CVXPY expressions
            # For CVXPY, linearize both bilinear terms
            x3 = steering_angle
            x4 = velocity
            x5 = orientation
            u1 = steering_velocity
            u2 = acc_input

            # Set nominal operating points
            x3_0 = 0.0  # e.g., 0.0 radians
            x4_0 = 0.5  # e.g., 5.0 m/s
            x5_0 = 0  # e.g., 0.0 radians

            l_wb = self.l_wb  # Wheelbase length, assumed constant

            # Constants for small-angle approximations
            sin_theta_0 = x5_0  # sin(x5_0) ≈ x5_0
            cos_theta_0 = 1  # cos(x5_0) ≈ 1

            # Partial derivatives for x4 * sin_theta
            df_sin_dx4 = sin_theta_0
            df_sin_dx5 = x4_0 * cos_theta_0

            # Linearized x4 * sin_theta
            x4_sin_theta = (
                    x4_0 * sin_theta_0
                    + df_sin_dx4 * (x4 - x4_0)
                    + df_sin_dx5 * (x5 - x5_0)
            )

            # Partial derivatives for (x4 / l_wb) * tan_theta
            df_tan_dx3 = x4_0 / l_wb
            df_tan_dx4 = x3_0 / l_wb

            # Linearized (x4 / l_wb) * tan_theta
            x4_tan_theta = (
                    (x4_0 * x3_0) / l_wb
                    + df_tan_dx3 * (x3 - x3_0)
                    + df_tan_dx4 * (x4 - x4_0)
            )

            # Update other approximations
            cos_theta = cos_theta_0  # cos(θ) ≈ 1
            # x4 * cos_theta remains x4, since cos_theta ≈ 1

            steering_output, steering_constraints = self.steer(x3, u1)
            acceleration_output, acceleration_constraints = self.acc(x4, u2)

            dx_dt = cp.vstack([
                x4 * cos_theta,
                x4_sin_theta,
                steering_output,
                acceleration_output,
                x4_tan_theta
            ]).flatten()

            next_state = current_state + dx_dt * self.dt
            uneq_constraints = steering_constraints + acceleration_constraints

            # Nominal operating point
            x40 = 1.0  # Example nominal value for x4
            x4_tan_theta0 = 1.0  # Example nominal value for x4_tan_theta
            z0 = x40 * x4_tan_theta0
            dz_dx4 = x4_tan_theta0
            dz_dx4_tan_theta = x40

            # Linear approximation of z = x4 * x4_tan_theta
            z_linear = z0 + dz_dx4 * (x4 - x40) + dz_dx4_tan_theta * (x4_tan_theta - x4_tan_theta0)

            uneq_constraints += [
                self.steering_velocity_lb <= u1, u1 <= self.steering_velocity_ub,
                self.acceleration_lb <= u2, u2 <= self.acceleration_ub(x4)[0],
                self.steering_angle_lb <= x3, x3 <= self.steering_angle_ub,
                self.velocity_lb <= x4, x4 <= self.velocity_ub,
                u2**2 + z_linear**2 <= self.a_max
            ]

        return next_state, [], uneq_constraints

    def get_initial_state(self) -> np.ndarray:
        return self.initial_state

    def get_goal_state(self) -> np.ndarray:
        return self.goal_state

    def get_position_orientation(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        if state.shape != (self.dim_state,):
            raise ValueError(f"state must have shape ({self.dim_state},), got {state.shape}")
        return state[:2], float(state[2])

    def get_shape(self, state: Union[List[float], np.ndarray]) -> List[Tuple[float, float]]:
        _, orientation = self.get_position_orientation(state)
        front_wheel_front = self._add_tuple(self._rotate((0.5, 0), orientation), (1, 0))
        front_wheel_back = self._add_tuple(self._rotate((-0.5, 0), orientation), (1, 0))
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
