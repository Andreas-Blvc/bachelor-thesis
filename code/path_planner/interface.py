from abc import ABC, abstractmethod
import numpy as np

from models import AbstractVehicleModel


class AbstractPathPlanner(ABC):
    solve_time: float
    def __init__(self, model: AbstractVehicleModel, dt, time_horizon, get_objective):
        self.model = model
        self.dt = dt
        self.time_horizon = time_horizon
        self.get_objective = get_objective

    @abstractmethod
    def get_optimized_trajectory(self, initial_state: np.ndarray):
        pass