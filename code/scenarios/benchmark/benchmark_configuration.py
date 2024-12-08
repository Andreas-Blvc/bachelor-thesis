from enum import Enum
from typing import List, Tuple, Callable, Any

from path_planner import Objectives
from utils import State, ControlInput


# defines a velocity in m/s
class Velocity(Enum):
    Slow = 1
    Medium = 10
    Fast = 40
    VeryFast = 100

# defines the lower and upper bound in percentage of the reference velocity
class VelocityRange(Enum):
    Tight = (0.99, 1.01)
    Small = (0.9, 1.1)
    Medium = (2/3, 4/3)
    Large = (0.4, 1.6)
    ZeroToVelocity = (0, 1)

# (1 - value) * left_width + value * right_width
class LateralOffset(Enum):
    FarLeft = 0
    Left = 0.25
    Mid = 0.5
    Right = 0.75
    FarRight = 1

class Road(Enum):
    RightTurn = './data/right_turn_simple.pkl'
    LeftTurn = './data/left_turn_simple.pkl'
    FirstTransition = './data/first_transition.pkl'


# specifies the duration into the future (in seconds) for which the path planning algorithm optimizes the trajectory.
class TimeHoriozon(Enum):
    Short = 1
    Medium = 10
    Long = 30

# defines how many time steps per seconds
class TimeDiscretization(Enum):
    Fine = 1/60
    Medium = 1/30
    Coarse = 1/10

class SolverType(Enum):
    Convex = 1
    NonConvex = 2

class Model(Enum):
    OrientedRoadFollowingModel = "OrientedRoadFollowingModel"
    RoadAlignedModel = "RoadAlignedModel"

class BenchmarkConfiguration:
    def __init__(
            self,
            start_velocity: Velocity,
            start_offset: LateralOffset,
            velocity_range: VelocityRange,
            road: Road,
            time_horizon: TimeHoriozon,
            time_discretization: TimeDiscretization,
            models: List[Tuple[Model, SolverType]],
            objective: Callable[[List[State], List[ControlInput]], Tuple[Any, Objectives.Type]],

    ):
        self.start_velocity = start_velocity
        self.start_offset = start_offset
        self.velocity_range = velocity_range
        self.road = road
        self.time_horizon = time_horizon
        self.time_discretization = time_discretization
        self.models = models
        self.objective = objective


