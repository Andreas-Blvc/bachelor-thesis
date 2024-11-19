import numpy as np

from models import PointMassModelAbstract
from path_planner import ConvexPathPlanner, Objectives

from .._scenario import Scenario


def create_scenario():
    dt = 0.1
    time_horizon = 2
    objective = Objectives.minimize_control_input
    model = PointMassModelAbstract(
        initial_state=np.reshape([0, 0, 0, 0], (4,)),
        goal_state=np.reshape([4, 2, 0, 0], (4,)),
        a_max=20,
        dt=dt
    )
    planner = ConvexPathPlanner(model, dt, time_horizon, objective)

    car_states, control_inputs = planner.get_optimized_trajectory()

    return Scenario(dt, model, car_states, control_inputs)


