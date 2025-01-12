import numpy as np

from models import OrientedRoadFollowingModel, RoadAlignedModel
from path_planner import ConvexPathPlanner, NonConvexPathPlanner
from roads import load_road
from self_driving_cars import DynamicSingleTrackModel
from visualizer.vehicle_path_visualizer import animate

from .benchmark import Benchmark
from .benchmark_configuration import *


def run(config: BenchmarkConfiguration):
    dt = config.time_discretization
    time_horizon = config.time_horizon
    road = load_road(config.road)
    start_velocity = config.start_velocity
    # objective = config.objective(road.length, start_velocity)
    # objective = config.objective(road.length, start_velocity, start_velocity * config.velocity_range[1])
    objective = config.objective
    start_offset = (
        (1 - config.start_offset) * road.n_max(0) +
        config.start_offset * road.n_min(0)
    )
    velocity_range = (
        start_velocity * config.velocity_range[0],
        start_velocity * config.velocity_range[1],
    )

    benchmarks = []
    steering_velocity_range = (-8, 8)
    steering_range = ((-40 / 180) * np.pi, (40 / 180) * np.pi)

    for model_type, solver_type in config.models:
        match model_type:
            case Model.OrientedRoadFollowingModel:
                predictive_model = OrientedRoadFollowingModel(
                    road=road,
                    v_range=velocity_range,
                    acc_range=(-6, 3),
                    steering_angle_range=steering_range,
                    steering_velocity_range=steering_velocity_range,
                    l_wb=0.883+1.508,
                )
            case Model.RoadAlignedModel:
                predictive_model = RoadAlignedModel(
                    road=road,
                    v_x_range=velocity_range,
                    v_y_range=(-4, 4),
                    acc_x_range=(-4, 4),
                    acc_y_range=(-4, 4),
                    yaw_rate_range=(-4, 4),
                    yaw_acc_range=(-4, 4),
                    a_max=8,
                )
            case _:
                raise ValueError('model type not supported')
        match solver_type:
            case SolverType.Convex:
                planner = ConvexPathPlanner(predictive_model, dt, time_horizon, objective, verbose=True)
            case SolverType.NonConvex:
                planner = NonConvexPathPlanner(predictive_model, dt, time_horizon, objective)
            case _:
                raise ValueError('solver type not supported')

        x, y = road.get_global_position(0, start_offset)
        psi = road.get_tangent_angle_at(0)
        self_driving_car = DynamicSingleTrackModel(
            predictive_model=predictive_model,
            initial_state=np.array([
                x, y, 0, start_velocity, psi, 0, 0
            ]),
            planner=planner,
            velocity_range=velocity_range,
            acceleration_range=(-4, 4),
            steering_range=steering_range,
            steering_velocity_range=steering_velocity_range,
            road=road,
        )

        animation = animate(self_driving_car, interactive=False)

        benchmarks.append(
            Benchmark(
                animation=animation,
                car=self_driving_car,
            )
        )

    return benchmarks
