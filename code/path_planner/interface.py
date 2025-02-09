from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Any
import numpy as np

from models import AbstractVehicleModel
from .objectives import Objectives
from utils import State, ControlInput


class AbstractPathPlanner(ABC):
    solve_time: float
    setup_time: float
    def __init__(self, model: AbstractVehicleModel, dt: Callable[[int], float], time_horizon, get_objective: Callable[[List[State], List[ControlInput]], Tuple[Any, Objectives.Type, Any, str]]):
        self.model = model
        self.dt = dt
        self.time_horizon = time_horizon
        self.get_objective = get_objective

    def get_state_transitions(self, time_horizon, start_index=0):
        sum_time_steps = 0
        i = start_index
        while i < 1e6:
            if sum_time_steps >= time_horizon:
                return i - start_index
            sum_time_steps += self.dt(i)
            i += 1
        raise RuntimeError("Time Horizon too large")


    @abstractmethod
    def get_optimized_trajectory(self, initial_state: np.ndarray):
        pass