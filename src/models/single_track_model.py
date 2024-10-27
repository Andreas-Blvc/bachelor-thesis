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

    c

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
