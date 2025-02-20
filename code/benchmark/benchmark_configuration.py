from enum import Enum
from typing import List, Tuple, Callable, Any

from path_planner import Objectives
from utils import State, ControlInput


# defines a velocity in m/s
class Velocity(Enum):
    VerySlow = 0.5
    Slow = 1
    Medium = 5
    Fast = 10
    VeryFast = 20

# defines the lower and upper bound in percentage of the reference velocity
class VelocityRange(Enum):
    Tight = (0.99, 1.01)
    Small = (0.9, 1.1)
    Medium = (2/3, 4/3)
    Large = (0.4, 1.6)
    ZeroToVelocity = (0, 1)
    VelocityToTripled = (1, 3)

# (1 - value) * left_width + value * right_width
class LateralOffset(Enum):
    FarLeft = 0
    Left = 0.25
    Mid = 0.5
    Right = 0.75
    FarRight = 1

class Road(Enum):
    ElchTest_One = './data/elchtest_1.pkl'
    Right_Turn = './data/right_turn.pkl'
    Left_Turn = './data/left_turn.pkl'
    Left_Turn_Small = './data/left_turn_small.pkl'
    Straight = './data/straight.pkl'
    Lane_Change = './data/lane_change.pkl'
    Slalom = './data/slalom.pkl'
    Feasible_Curve = './data/feasible_curve.pkl'
    Infeasible_Curve = './data/infeasible_curve.pkl'
    Random = './data/random.pkl'
    PmFeasibleCurve = './data/pm_feasible_curve.pkl'


# specifies the duration into the future (in seconds) for which the path planning algorithm optimizes the trajectory.
class TimeHoriozon(Enum):
    Short = 1
    Medium = 3
    Long = 5
    VeryLong = 10

# defines how many time steps per seconds
class TimeDiscretization(Enum):
    Constant = lambda h: lambda _: h
    Linear = lambda a, b: lambda i: a * i + b
    PlateauLinear = lambda a, b, threshold: lambda i: a * (i-threshold) + b if i > threshold else b

    def __call__(self, *args):
        return self.value(*args)

class SolverType(Enum):
    Convex = 1
    NonConvex = 2

class Model(Enum):
    OrientedRoadFollowingModel = "OrientedRoadFollowingModel"
    RoadAlignedModel = "RoadAlignedModel"

class BenchmarkConfiguration:
    def __init__(
            self,
            start_velocities: List[Velocity | float],
            start_offset: LateralOffset | float,
            velocity_range: VelocityRange | Tuple[float, float],
            roads: List[Road] | List[str],
            time_horizon: TimeHoriozon | float,
            time_discretization: Callable[[int], float],
            models: List[Tuple[Model, SolverType]],
            objectives: List[Callable[[List[State], List[ControlInput]], Tuple[Any, Objectives.Type, Any, str]]],
            replanning_steps: int,
    ):
        self.start_velocities = []
        for velocity in start_velocities:
            self.start_velocities.append(
                velocity.value if isinstance(velocity, Velocity) else velocity
            )

        self.start_offset = (
            start_offset.value if isinstance(start_offset, LateralOffset) else start_offset
        )

        self.velocity_range = (
            velocity_range.value if isinstance(velocity_range, VelocityRange) else velocity_range
        )

        self.time_horizon = (
            time_horizon.value if isinstance(time_horizon, TimeHoriozon) else time_horizon
        )

        self.time_discretization = (
           time_discretization
        )

        self.roads = []
        for road in roads:
            self.roads.append(
                road.value if isinstance(road, Road) else road
            )

        self.models = models
        self.objectives = objectives
        self.replanning_steps = replanning_steps


