import numpy as np

from models import RoadAlignedModelAbstract
from obstacles import roads
from path_planner import ConvexPathPlanner, Objectives
from scenarios._internal.scenario import Scenario


def create_scenario():
    dt = 1 / 60
    time_horizon = 10
    objective = Objectives.minimize_control_input

    model = RoadAlignedModelAbstract(
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

    # Visualize:
    # model.visualize_constraints(car_states, control_inputs)
    # road.plot_combined_curvature_and_derivative()
    body_fixed_controls = []
    for j in range(len(control_inputs)):
        body_fixed_controls.append(
            (
                model.to_body_fixed(car_states[j], control_inputs[j])[0],
                model.road.get_curvature_at(car_states[j][0]) * car_states[j][2]
            )
        )
    # plot_control_inputs(body_fixed_controls, ['a_x', 'yaw_rate'], dt)

    return Scenario(dt, model, car_states, control_inputs)

