import numpy as np

from models.point_mass_model import PointMassModel
from path_planner.cvxpy_optimizer import ConvexPathPlanner
from path_planner.objectives import Objectives
from scenarios.scenario import Scenario


def create_scenario():
    dt = 0.1
    time_horizon = 2
    objective = Objectives.minimize_control_input
    model = PointMassModel(
        initial_state=np.reshape([0, 0, 0, 0], (4,)),
        goal_state=np.reshape([4, 2, 0, 0], (4,)),
        a_max=20,
        dt=dt
    )
    planner = ConvexPathPlanner(model, dt, time_horizon, objective)

    car_states, control_inputs = planner.get_optimized_trajectory()

    return Scenario(dt, model, car_states, control_inputs)


