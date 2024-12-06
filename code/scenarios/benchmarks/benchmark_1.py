import numpy as np

from models import OrientedRoadFollowingModel, RoadAlignedModel
from path_planner import ConvexPathPlanner, Objectives
from obstacles import roads
from scenarios._scenario import Scenario
from visualizer import VehiclePathVisualizer


def run():
    objective = Objectives.minimize_remaining_distance
    dt = 1 / 30
    time_horizon = 2
    road = roads.right_curved_road
    path_visualizer = VehiclePathVisualizer(interactive=True)

    def _get_planned_trajectory(model):
        planner = ConvexPathPlanner(model, dt, time_horizon, objective)
        planned_car_states, control_inputs = planner.get_optimized_trajectory()
        actual_car_states = [model.initial_state]
        for u in control_inputs:
            next_state, _ = model.update(actual_car_states[-1], u)
            actual_car_states.append(next_state)
        return planned_car_states, actual_car_states, control_inputs

    model_1 = OrientedRoadFollowingModel(
        initial_state=np.array([0, 0, 0, 0, 0]),
        goal_state=None,  # np.array([road.length, 0, 0, 2, 0]),
        dt=dt,
        road=road,
        v_range=(0, 5),
        acc_range=(-2, 2),
        steering_angle_range=((-30 / 180) * np.pi, (30 / 180) * np.pi),
        steering_velocity_range=(-3, 3),
    )

    model_2 = RoadAlignedModel(
        initial_state=np.array([0, 0, 0, 0]),
        goal_state=None,
        dt=dt,
        road=road,
        v_x_range=(0, 5),
        v_y_range=(-1, 1),
        acc_x_range=(-2, 2),
        acc_y_range=(-2, 2),
        yaw_rate_range=(-1, 1),
        yaw_acc_range=(-0.3, 0.3),
        a_max=80,
    )

    planned_car_states_1, actual_car_states_1, control_inputs_1 = _get_planned_trajectory(model_1)
    planned_car_states_2, actual_car_states_2, control_inputs_2 = _get_planned_trajectory(model_2)

    scenario_1 = Scenario(dt, model_1, planned_car_states_1, control_inputs_1, actual_car_states_1)
    scenario_2 = Scenario(dt, model_2, planned_car_states_2, control_inputs_2, actual_car_states_2)

    return scenario_1, scenario_2


if __name__ == '__main__':
    run()