from abc import ABC, abstractmethod

from models import AbstractVehicleModel


class AbstractPathPlanner(ABC):
    def __init__(self, model: AbstractVehicleModel, dt, time_horizon, get_objective):
        self.model = model
        self.dt = dt
        self.time_horizon = time_horizon
        self.get_objective = get_objective

    @abstractmethod
    def get_optimized_trajectory(self):
        pass