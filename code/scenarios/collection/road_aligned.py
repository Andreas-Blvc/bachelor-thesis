import numpy as np

from models import RoadAlignedModel
from obstacles import roads
from path_planner import ConvexPathPlanner, Objectives

from .._scenario import Scenario


def create_scenario():
    dt = 1 / 60
    time_horizon = 10
    objective = Objectives.minimize_control_input

    model = RoadAlignedModel(
        initial_state=np.array([0, 0, 0.01, 0]),
        goal_state=np.array([1, 0, 0.01, 0]),
        dt=dt,
        road=roads.s_shaped_road,
        v_x_range=(-5, 40),
        v_y_range=(-1, 1),
        acc_x_range=(-2, 2),
        acc_y_range=(-2, 2),
        yaw_rate_range=(-1, 1),
        yaw_acc_range=(-0.3, 0.3),
        a_max=80,
    )

    planner = ConvexPathPlanner(model, dt, time_horizon, objective)
    car_states, control_inputs = planner.get_optimized_trajectory()

    return Scenario(dt, model, car_states, control_inputs)

