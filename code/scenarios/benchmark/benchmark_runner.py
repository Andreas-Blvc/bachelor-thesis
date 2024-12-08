import numpy as np

from models import OrientedRoadFollowingModel, RoadAlignedModel
from path_planner import ConvexPathPlanner, NonConvexPathPlanner
from roads import load_road

from .._scenario import Scenario
from .benchmark_configuration import *


def run(config: BenchmarkConfiguration):
    # store for non-convex case
    oriented_road_following_possible_goal_state = None
    oriented_road_following_initial_guess = None
    road_aligned_possible_goal_state = None
    road_aligned_initial_guess = None

    objective = config.objective
    dt = config.time_discretization.value
    time_horizon = config.time_horizon.value
    road = load_road(config.road.value)
    start_velocity = config.start_velocity.value
    start_offset = (
        (1 - config.start_offset.value) * road.width(0)/2 +
        config.start_offset.value * -road.width(0)/2
    )
    velocity_range = (
        start_velocity * config.velocity_range.value[0],
        start_velocity * config.velocity_range.value[1],
    )

    def _get_planned_trajectory(model, non_convex=False, initial_guess=None):
        if non_convex:
            planner = NonConvexPathPlanner(model, dt, time_horizon, objective)
            planned_car_states, control_inputs = planner.get_optimized_trajectory(initial_guess)
        else:
            planner = ConvexPathPlanner(model, dt, time_horizon, objective)
            planned_car_states, control_inputs = planner.get_optimized_trajectory()
        actual_car_states = [model.initial_state]
        if control_inputs is not None:
            for u in control_inputs:
                next_state, _ = model.update(actual_car_states[-1], u)
                actual_car_states.append(next_state)
        return planned_car_states, actual_car_states, control_inputs, planner.solve_time

    scenarios = []

    for model_type, solver_type in config.models:
        match model_type:
            case Model.OrientedRoadFollowingModel:
                vehicle_model = OrientedRoadFollowingModel(
                    initial_state=np.array([0, start_offset, 0, start_velocity, 0]),
                    goal_state=None if solver_type != SolverType.NonConvex else oriented_road_following_possible_goal_state,
                    dt=dt,
                    road=road,
                    v_range=velocity_range,
                    acc_range=(-4, 4),
                    steering_angle_range=((-45 / 180) * np.pi, (45 / 180) * np.pi),
                    steering_velocity_range=(-1, 1),
                    l_wb=2
                )
            case Model.RoadAlignedModel:
                vehicle_model = RoadAlignedModel(
                    initial_state=np.array([0, start_offset, start_velocity, 0]),
                    goal_state=None if solver_type != SolverType.NonConvex else road_aligned_possible_goal_state,
                    dt=dt,
                    road=road,
                    v_x_range=velocity_range,
                    v_y_range=(-1, 1),
                    acc_x_range=(-2, 2),
                    acc_y_range=(-2, 2),
                    yaw_rate_range=(-1, 1),
                    yaw_acc_range=(-0.9, 0.9),
                    a_max=4,
                )
            case _:
                raise ValueError('model type not supported')
        try:
            print(f'Starting planning for {model_type.value} {"(NonConvex)" if solver_type == SolverType.NonConvex else ""}')
            planned_states, actual_states, controls, solve_time = _get_planned_trajectory(
                vehicle_model,
                non_convex=solver_type == SolverType.NonConvex,
                initial_guess= road_aligned_initial_guess if model_type == Model.RoadAlignedModel
                    else oriented_road_following_initial_guess,
            )
        except ValueError as e:
            print(f"{model_type.value} failed")
            print(e)
            continue

        if controls is not None:
            print(
                f"{model_type.value}, solve time {'(NonConvex)' if solver_type == SolverType.NonConvex else ''}: "
                f"{solve_time * 1000:.1f}ms, "
                f"state transitions: {len(controls)}"
            )
            if model_type == Model.RoadAlignedModel:
                road_aligned_possible_goal_state = planned_states[-1]
                road_aligned_initial_guess = np.array(planned_states), np.array(controls)
            elif model_type == Model.OrientedRoadFollowingModel:
                oriented_road_following_possible_goal_state = planned_states[-1]
                oriented_road_following_initial_guess = np.array(planned_states), np.array(controls)

            scenarios.append(
                Scenario(
                    dt,
                    vehicle_model,
                    planned_states,
                    controls,
                    actual_states,
                    title=f"{model_type.value} {'(NonConvex)' if solver_type == SolverType.NonConvex else ''}"
                )
            )
        else:
            print(f"{model_type.value} did not found a solution")

    return scenarios
