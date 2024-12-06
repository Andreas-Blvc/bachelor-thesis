import numpy as np

from models import OrientedRoadFollowingModel
from roads import load_road
from path_planner import NonConvexPathPlanner, ConvexPathPlanner, Objectives

from .._scenario import Scenario


def create_scenario(v_min=None, v_max=None, v_start=None):
    v_min = v_min or 4
    v_max = v_max or 5
    v_start = v_start or 4
    dt = 1 / 30
    time_horizon = 10
    objective = Objectives.minimize_control_input
    road = load_road("./data/right_turn.pkl")
    model = OrientedRoadFollowingModel(
        initial_state=np.array([0, 0, 0, v_start, 0]),
        goal_state=np.array([road.length, 0, 0, v_start, 0]),  # np.array([road.length, 0, 0, 2, 0]),
        dt=dt,
        road=road,
        v_range=(v_min, v_max),
        acc_range=(-2, 2),
        steering_angle_range=((-45 / 180) * np.pi, (46 / 180) * np.pi),
        steering_velocity_range=(-5, 5),
    )

    planner = NonConvexPathPlanner(model, dt, time_horizon, objective)
    car_states, control_inputs = planner.get_optimized_trajectory()
    # plot_control_inputs(control_inputs, model.get_control_input_labels(), dt)
    # plot_control_inputs(car_states, ['s', 'n', 'xi', 'v', 'delta'], dt)
    # plot_control_inputs([(dn.value, dxi.value) for dn, dxi in model.artificial_variables], ['dn_term', 'dxi_term'], dt)

    # model.goal_state = car_states[-1]
    # planner = NonConvexPathPlanner(model, dt, time_horizon, objective)
    # car_states, control_inputs = planner.get_optimized_trajectory(initial_guess=(car_states, control_inputs))
    # plot_control_inputs(control_inputs, model.get_control_input_labels(), dt)
    # plot_control_inputs(car_states, ['s', 'n', 'xi', 'v', 'delta'], dt)

    actual_car_states = [model.initial_state]
    for u in control_inputs:
        next_state, _ = model.update(actual_car_states[-1], u)
        actual_car_states.append(next_state)

    return Scenario(dt, model, car_states, control_inputs, actual_car_states)

