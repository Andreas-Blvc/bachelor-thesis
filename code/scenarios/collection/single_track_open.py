import numpy as np

from models import SingleTrackModelAbstract
from path_planner import NonConvexPathPlanner, Objectives
from scenarios._internal.scenario import Scenario


def create_scenario_simple():
    dt = 1 / 30
    time_horizon = 3
    objective = Objectives.minimize_control_input
    model = SingleTrackModelAbstract(
        initial_state=np.reshape([-6, -2, 0, 0, 0], (5,)),
        goal_state=np.reshape([6, -2, (0 / 180) * np.pi, 0, 0], (5,)),
        l_wb=1.8,
        v_s=30,
        steering_velocity_range=(-1, 1),
        steering_angle_range=((-35 / 180) * np.pi, (35 / 180) * np.pi),
        velocity_range=(-40, 40),
        acceleration_range=(-200, 200),
        dt=dt,
    )
    planner = NonConvexPathPlanner(model, dt, time_horizon, objective)
    car_states, control_inputs = planner.get_optimized_trajectory()
    # actual_car_states = [model.get_initial_state()]
    # for u in control_inputs:
    # 	actual_car_states.append(model.accurate_update(actual_car_states[-1], u))
    return Scenario(dt, model, car_states, control_inputs)


def create_scenario_line_change():
    dt = 0.1
    time_horizon = 10
    objective = Objectives.minimize_control_input
    model = SingleTrackModelAbstract(
        initial_state=np.reshape([-6, -2, 0, 0, 0], (5,)),
        goal_state=np.reshape([2, -1, 0, 0, (0 / 180) * np.pi], (5,)),
        l_wb=1.8,
        v_s=10,
        steering_velocity_range=(-10, 10),
        steering_angle_range=((-30 / 180) * np.pi, (30 / 180) * np.pi),
        velocity_range=(-40, 40),
        acceleration_range=(-5, 5),
        dt=dt,
        solver_type='casadi',
    )
    planner = NonConvexPathPlanner(model, dt, time_horizon, objective)

    car_states, control_inputs = planner.get_optimized_trajectory()

    actual_car_states = [model.get_initial_state()]
    for u in control_inputs:
        actual_car_states.append(model.accurate_update(actual_car_states[-1], u))
    return Scenario(dt, model, actual_car_states, control_inputs)
