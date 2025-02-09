import numpy as np
from datetime import datetime

from models import OrientedRoadFollowingModel, RoadAlignedModel
from path_planner import ConvexPathPlanner, NonConvexPathPlanner
from roads import load_road
from self_driving_cars import DynamicSingleTrackModel
from utils.range_bounding import calculate_product_range

from .benchmark import Benchmark
from .benchmark_configuration import *


def run(config: BenchmarkConfiguration):
    dt = config.time_discretization
    time_horizon = config.time_horizon
    roads = [(load_road(road_name), road_name) for road_name in config.roads]
    benchmarks = []
    steering_velocity_range = (-8, 8)
    steering_range = ((-40 / 180) * np.pi, (40 / 180) * np.pi)
    l_wb = 0.883+1.508

    for objective in config.objectives:
        for start_velocity in config.start_velocities:
            velocity_range = (
                start_velocity * config.velocity_range[0],
                start_velocity * config.velocity_range[1],
            )
            v_tan_delta = calculate_product_range(velocity_range,
                                                  (np.tan(steering_range[0]), np.tan(steering_range[1])))
            v_delta = calculate_product_range(velocity_range, steering_range)
            yaw_rate_range = (
                1 / l_wb * v_tan_delta[0],
                1 / l_wb * v_tan_delta[1],
            )
            yaw_acc_range = (
                1 / l_wb * (v_tan_delta[0] + v_delta[0] / np.cos(steering_range[0]) ** 2),
                1 / l_wb * (v_tan_delta[1] + v_delta[1] / np.cos(steering_range[0]) ** 2),
            )

            for road, road_name in roads:
                start_offset = (
                        (1 - config.start_offset) * road.n_max(0) +
                        config.start_offset * road.n_min(0)
                )
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
                                acc_x_range=(-6, 3),
                                acc_y_range=(-20, 20),
                                yaw_rate_range=yaw_rate_range,
                                yaw_acc_range=yaw_acc_range,
                                a_max=8,
                            )
                        case _:
                            raise ValueError('model type not supported')
                    match solver_type:
                        case SolverType.Convex:
                            planner = ConvexPathPlanner(predictive_model, dt, time_horizon, objective, verbose=True, use_param=model_type == Model.RoadAlignedModel)
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
                        replanning_steps=config.replanning_steps,
                    )

                    benchmarks.append(
                        Benchmark(
                            car=self_driving_car,
                            folder=f"{road_name.split('/')[-1].split('.')[0]}-{model_type.value}-{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
                            road_name=road_name.split('/')[-1].split('.')[0].replace('_', ' ').title()
                        )
                    )

    return benchmarks
