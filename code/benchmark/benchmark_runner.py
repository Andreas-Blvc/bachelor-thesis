import numpy as np

from models import OrientedRoadFollowingModel, RoadAlignedModel
from path_planner import ConvexPathPlanner, NonConvexPathPlanner
from roads import load_road
from self_driving_cars import DynamicSingleTrackModel
from visualizer.vehicle_path_visualizer import animate
from .benchmark import Benchmark

from .benchmark_configuration import *


def run(config: BenchmarkConfiguration):
    # store for non-convex case
    oriented_road_following_possible_goal_state = None
    oriented_road_following_initial_guess = None
    road_aligned_possible_goal_state = None
    road_aligned_initial_guess = None

    objective = config.objective
    dt = config.time_discretization
    time_horizon = config.time_horizon
    road = load_road(config.road)
    start_velocity = config.start_velocity
    start_offset = (
        (1 - config.start_offset) * road.width(0)/2 +
        config.start_offset * -road.width(0)/2
    )
    velocity_range = (
        start_velocity * config.velocity_range[0],
        start_velocity * config.velocity_range[1],
    )

    # def _get_planned_trajectory(model, non_convex=False, initial_guess=None):
    #     if non_convex:
    #         planner = NonConvexPathPlanner(model, dt, time_horizon, objective)
    #         planned_car_states, control_inputs = planner.get_optimized_trajectory(initial_guess)
    #     else:
    #         planner = ConvexPathPlanner(model, dt, time_horizon, objective)
    #         planned_car_states, control_inputs = planner.get_optimized_trajectory()
    #     actual_car_states = [model.initial_state]
    #     if control_inputs is not None:
    #         for u in control_inputs:
    #             next_state, _ = model.update(actual_car_states[-1], u)
    #             actual_car_states.append(next_state)
    #     return planned_car_states, actual_car_states, control_inputs, planner.solve_time

    benchmarks = []

    for model_type, solver_type in config.models:
        match model_type:
            case Model.OrientedRoadFollowingModel:
                predictive_model = OrientedRoadFollowingModel(
                    initial_state=np.array([0, start_offset, 0, start_velocity, 0]),
                    goal_state=None if solver_type != SolverType.NonConvex else oriented_road_following_possible_goal_state,
                    road=road,
                    v_range=velocity_range,
                    acc_range=(-4, 4),
                    steering_angle_range=((-30 / 180) * np.pi, (30 / 180) * np.pi),
                    steering_velocity_range=(-1, 1),
                    l_wb=2.5
                )
            case Model.RoadAlignedModel:
                predictive_model = RoadAlignedModel(
                    initial_state=np.array([0, start_offset, start_velocity, 0]),
                    goal_state=None if solver_type != SolverType.NonConvex else road_aligned_possible_goal_state,
                    road=road,
                    v_x_range=velocity_range,
                    v_y_range=(-1, 1),
                    acc_x_range=(-2, 2),
                    acc_y_range=(-2, 2),
                    yaw_rate_range=(-1, 0),
                    yaw_acc_range=(-0.9, 0.9),
                    a_max=4,
                )
            case _:
                raise ValueError('model type not supported')
        match solver_type:
            case SolverType.Convex:
                planner = ConvexPathPlanner(predictive_model, dt, time_horizon, objective)
            case SolverType.NonConvex:
                planner = NonConvexPathPlanner(predictive_model, dt, time_horizon, objective)
            case _:
                raise ValueError('solver type not supported')

        (x, y), _ = predictive_model.convert_vec_to_state(predictive_model.initial_state).get_position_orientation()
        self_driving_car = DynamicSingleTrackModel(
            predictive_model=predictive_model,
            planner=planner,
            initial_state=[x, y, 0, 0, 0, 0, 0],
            steering_range=(0, 0),
            steering_velocity_range=(0, 0),
            velocity_range=(0, 0),
            acceleration_range=(0, 0),
            road=road,
        )

        animation = None
        try:
            animation = animate(self_driving_car, interactive=False)
        except StopIteration as e:
            print('done')

        benchmarks.append(
            Benchmark(
                animation=animation,
            )
        )


    return benchmarks
