from abc import ABC, abstractmethod
from typing import List, Tuple, Generator
import numpy as np

from models import AbstractVehicleModel
from path_planner import AbstractPathPlanner
from roads import AbstractRoad


class AbstractSelfDrivingCar(ABC):
    current_state: np.ndarray
    controls: List[Tuple[float, np.ndarray]]
    predictive_model: AbstractVehicleModel
    planner: AbstractPathPlanner
    road: AbstractRoad
    state_labels: List[str]
    control_input_labels: List[str]

    @abstractmethod
    def drive(self) -> Generator[Tuple[np.ndarray, List[np.ndarray]], None, None]:
        """
        yields car states
        """
        pass

    # ============================================
    # VISUALIZATION
    # ============================================

    polygon = List[Tuple[float, float]]

    @abstractmethod
    def get_vehicle_polygon(self, state) -> polygon:
        pass
    @abstractmethod
    def get_start(self) -> np.ndarray:
        pass
    @abstractmethod
    def get_goal(self) -> np.ndarray | None:
        pass
    @abstractmethod
    def get_road(self) -> AbstractRoad | None:
        pass
    @abstractmethod
    def get_position(self, state) -> Tuple[float, float]:
        pass
    @abstractmethod
    def get_orientation(self, state) -> float:
        pass
    @abstractmethod
    def get_speed(self, state) -> float:
        pass
    @abstractmethod
    def get_steering_angle(self, state) -> float:
        pass

