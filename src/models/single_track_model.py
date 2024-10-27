import numpy as np
import cvxpy as cp
from models.vehicle_model import VehicleModel

BIG_M = 1e6

class SingleTrackModel(VehicleModel):
    def __init__(self, initial_state, goal_state, a_max: float, l_wb: float, v_s: float, steering_velocity_range, steering_angle_range, velocity_range, acceleration_range,  dt: float):
        self.dim_state = 5
        self.dim_control_input = 2
        self.dt = dt

        self.a_max = a_max
        self.l_wb = l_wb
        self.v_s = v_s
        self.steering_velocity_lb = steering_velocity_range[0]
        self.steering_velocity_ub = steering_velocity_range[1]
        self.steering_angle_lb = steering_angle_range[0]
        self.steering_angle_ub = steering_angle_range[1]
        self.velocity_lb = velocity_range[0]
        self.velocity_ub = velocity_range[1]
        self.acceleration_lb = acceleration_range[0]

        self.initial_state = np.reshape(initial_state, (self.dim_state, 1))
        self.goal_state = np.reshape(goal_state, (self.dim_state, 1))

    def acceleration_ub(self, v):
        acc_ub = cp.Variable()
        constraints = [
            acc_ub == self.a_max * (self.v_s / cp.maximum(v, self.v_s))
        ]
        return acc_ub, constraints

    def steer(self, steering_angle, steering_velocity):
        steering_output = cp.Variable()

        z = cp.Variable(boolean=True)
        b1 = cp.Variable(boolean=True)
        b2 = cp.Variable(boolean=True)
        b3 = cp.Variable(boolean=True)
        c1 = cp.Variable(boolean=True)
        c2 = cp.Variable(boolean=True)

        # if (steering_angle <= self.steering_angle_lb and steering_velocity <= 0) or (
        #         steering_angle >= self.steering_angle_ub and steering_velocity >= 0):
        #     return 0
        constraints = [
            self.steering_velocity_lb - steering_angle <= BIG_M * b1,  # if self.steering_velocity_lb > steering_angle, then b1=1
            self.steering_velocity_lb - steering_angle >= -BIG_M * (1 - b1),  # if self.steering_velocity_lb < steering_angle, then b1=0
        ] + [
            steering_angle - self.steering_angle_ub <= BIG_M * b2,  # if steering_angle > self.steering_angle_ub, then b2=1
            steering_angle - self.steering_angle_ub >= -BIG_M * (1 - b2),  # if steering_angle < self.steering_angle_ub, then b2=0
        ] + [
            steering_velocity >= -BIG_M * b3,  # if steering_velocity < 0, then b3=1
            steering_velocity <= BIG_M * (1-b3),  # if steering_velocity > 0 then b3=0
        ] + [
            b1 + b3 <= 1 + c1,  # if b1 and b3, then c1=1
            b1 + b3 >= 2 * c1,  # if not b1 or not b3, then c1=0
        ] + [
            b1 + (1-b3) <= 1 + c2,  # if b1 and not b3, then c2=1
            b1 + (1-b3) >= 2 * c2,  # if not b1 or b3, then c2=0
        ] + [
            c1 + c2 >= z,  # if c1 + c2 = 0, then z=0
            c1 + c2 <= 2 * z,  # if c1 + c2 > 0, then z=1
        ] + [
            steering_output <= BIG_M * (1-z),
            steering_output >= -BIG_M * (1-z),
        ]

        # elif steering_velocity <= self.steering_velocity_lb:
        #     return self.steering_velocity_lb
        # elif steering_velocity >= self.steering_velocity_ub:
        #     return self.steering_velocity_ub
        constraints += [
            steering_output <= self.steering_velocity_ub,
            steering_output >= self.steering_velocity_lb,
        ]

        return steering_output, constraints

    def acc(self, velocity, acceleration):
        acc_output = cp.Variable()

        z = cp.Variable(boolean=True)
        b1 = cp.Variable(boolean=True)
        b2 = cp.Variable(boolean=True)
        b3 = cp.Variable(boolean=True)
        c1 = cp.Variable(boolean=True)
        c2 = cp.Variable(boolean=True)

        # if (velocity <= self.velocity_lb and acceleration <= 0) or (velocity >= self.velocity_ub and acceleration >= 0):
        #   return 0
        constraints = [
          self.velocity_lb - velocity <= BIG_M * b1,  # if self.velocity_lb > velocity, then b1=1
          self.velocity_lb - velocity >= -BIG_M * (1 - b1),  # if self.velocity_lb < velocity, then b1=0
        ] + [
          velocity - self.velocity_ub <= BIG_M * b2,  # if velocity > self.velocity_ub, then b2=1
          velocity - self.velocity_ub >= -BIG_M * (1 - b2),  # if velocity < self.velocity_ub, then b2=0
        ] + [
          acceleration >= -BIG_M * b3,  # if acceleration < 0, then b3=1
          acceleration <= BIG_M * (1 - b3),  # if acceleration > 0 then b3=0
        ] + [
          b1 + b3 <= 1 + c1,  # if b1 and b3, then c1=1
          b1 + b3 >= 2 * c1,  # if not b1 or not b3, then c1=0
        ] + [
          b1 + (1 - b3) <= 1 + c2,  # if b1 and not b3, then c2=1
          b1 + (1 - b3) >= 2 * c2,  # if not b1 or b3, then c2=0
        ] + [
          c1 + c2 >= z,  # if c1 + c2 = 0, then z=0
          c1 + c2 <= 2 * z,  # if c1 + c2 > 0, then z=1
        ] + [
          acc_output <= BIG_M * (1 - z),
          acc_output >= -BIG_M * (1 - z),
        ]

        acc_ub, acc_constraints = self.acceleration_ub(velocity)
        # elif acceleration <= self.acceleration_lb:
        #     return self.acceleration_lb
        # elif acceleration >= self.acceleration_ub(velocity):
        #     return self.acceleration_ub(velocity)

        constraints += [
            acc_output >= self.acceleration_lb,
            acc_output <= acc_ub,
        ]

        return acc_output, constraints + acc_constraints


    def update(self, current_state, control_inputs):
        # Flatten current state and control inputs to ensure compatibility
        x = cp.reshape(current_state, (5,))  # Reshape to 1D vector of shape (5,)
        u = cp.reshape(control_inputs, (2,))  # Reshape to 1D vector of shape (2,)

        # Approximations using CVXPY expressions
        cos_theta = 1 - (x[4] ** 2) / 2  # Approximation for cos(x[4])
        sin_theta = x[4]  # Approximation for sin(x[4]) for small angles
        tan_theta = x[2]  # Approximation for tan(x[2])

        # Get CVXPY expressions from steering and acceleration functions
        steering_output, steering_constraints = self.steer(x[2], u[0])
        acceleration_output, acceleration_constraints = self.acc(x[3], u[1])

        # Define the differential changes using CVXPY variables
        dx_dt = cp.vstack([
            x[3] * cos_theta,
            x[3] * sin_theta,
            steering_output,
            acceleration_output,
            (x[3] / self.l_wb) * tan_theta
        ]).flatten()  # Flatten to ensure shape (5,)

        # Update next state using CVXPY compatible operations
        next_state = x + dx_dt * self.dt  # `x` is now 1D, matching `dx_dt`

        # Define the constraint for acceleration within limits
        constraints = [
            cp.sqrt(u[1]**2 + (x[3]*dx_dt[4])**2) <= self.a_max  # Constrain acceleration inputs
        ] + steering_constraints + acceleration_constraints
        return next_state, constraints

    def get_initial_state(self):
        return self.initial_state

    def get_goal_state(self):
        return self.goal_state

    def get_position_orientation(self, state):
        if isinstance(state, list):
            state = np.array(state)

        if state.shape == (5, 1):
            return state[:2, 0], state[3, 0]
        else:
            return state[:2], state[3]

    def get_shape(self):
        return [
            (-1, 0.5), (1, 0.5),
            (-1, -0.5), (1, -0.5)
        ]

    def get_dim_state(self):
        return self.dim_state

    def get_dim_control_input(self):
        return self.dim_control_input