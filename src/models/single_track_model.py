import numpy as np
import casadi as ca
import cvxpy as cp
from math import cos, sin
from setuptools.command.rotate import rotate

from models.vehicle_model import VehicleModel

BIG_M = 1e6


def add_tuple(a, b):
    return (
        a[0] + b[0],
        a[1] + b[1],
    )


def rotate(point, theta):
    x, y = point
    return (
        x * cos(theta) - y * sin(theta),
        y * cos(theta) + x * sin(theta),
    )


class SingleTrackModel(VehicleModel):
    def __init__(self, initial_state, goal_state, a_max: float, l_wb: float, v_s: float,
                 steering_velocity_range, steering_angle_range, velocity_range, acceleration_range,
                 dt: float, solver_type='cvxpy'):
        self.dim_state = 5
        self.dim_control_input = 2
        self.dt = dt
        self.solver_type = solver_type  # 'cvxpy' or 'casadi'

        self.a_max = a_max
        self.l_wb = l_wb
        self.v_s = v_s
        self.steering_velocity_lb, self.steering_velocity_ub = steering_velocity_range
        self.steering_angle_lb, self.steering_angle_ub = steering_angle_range
        self.velocity_lb, self.velocity_ub = velocity_range
        self.acceleration_lb = acceleration_range[0]

        # Store numeric initial and goal states
        self.initial_state = np.reshape(initial_state, (self.dim_state, 1))
        self.goal_state = np.reshape(goal_state, (self.dim_state, 1))

    def acceleration_ub(self, v):
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

    def steer(self, steering_angle, steering_velocity):
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

            constraints = [
              0 <= z, z <= 1,
              0 <= b1, b1 <= 1,
              0 <= b2, b2 <= 1,
              0 <= b3, b3 <= 1,
              0 <= c1, c1 <= 1,
              0 <= c2, c2 <= 1,
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
            ]
        return steering_output, constraints

    def acc(self, velocity, acceleration):
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

            constraints = [
              0 <= z, z <= 1,
              0 <= b1, b1 <= 1,
              0 <= b2, b2 <= 1,
              0 <= b3, b3 <= 1,
              0 <= c1, c1 <= 1,
              0 <= c2, c2 <= 1,
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
                           ] + acc_constraints
        return acc_output, constraints

    def update(self, current_state, control_inputs):
        """Update state based on current state and control inputs."""
        x3 = current_state[2]
        x4 = current_state[3]
        x5 = current_state[4]

        u1 = control_inputs[0]
        u2 = control_inputs[1]

        if self.solver_type == 'casadi':
            # Use direct expressions without introducing new variables
            cos_theta = ca.cos(x5)
            sin_theta = ca.sin(x5)
            tan_theta = ca.tan(x3)
        else:
            # For CVXPY, linearize both bilinear terms

            # Set nominal operating points
            x3_0 = 0.0  # e.g., 0.0 radians
            x4_0 = 2  # e.g., 5.0 m/s
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

        dx_dt = [
            x4 * cos_theta,
            x4 * sin_theta if not self.solver_type == 'cvxpy' else x4_sin_theta,
            steering_output,
            acceleration_output,
            (x4 / self.l_wb) * tan_theta if not self.solver_type == 'cvxpy' else x4_tan_theta
        ]

        if self.solver_type == 'cvxpy':
            next_state = current_state + cp.hstack(dx_dt) * self.dt
        elif self.solver_type == 'casadi':
            next_state = ca.vertcat(*[current_state[i] + dx_dt[i] * self.dt for i in range(5)])

        constraints = steering_constraints + acceleration_constraints
        return next_state, constraints

    def get_initial_state(self):
        return self.initial_state.flatten()

    def get_goal_state(self):
        return self.goal_state.flatten()

    def get_position_orientation(self, state):
        if isinstance(state, list):
            state = np.array(state)

        if state.shape == (5, 1):
            return state[:2, 0], state[2, 0]
        else:
            return state[:2], state[2]

    def get_shape(self, orientation):
        front_wheel_front = add_tuple(rotate((0.5, 0), orientation), (1, 0))
        front_wheel_back = add_tuple(rotate((-0.5, 0), orientation), (1, 0))
        return [
            (-1, 0.5), (1, 0.5),
                                (1, 0),
                    front_wheel_back, front_wheel_front,
                                (1, 0),
                       (1, -0.5),
            (-1, -0.5),
        ]

    def get_dim_state(self):
        return self.dim_state

    def get_dim_control_input(self):
        return self.dim_control_input

    def get_a_max(self):
        return self.a_max