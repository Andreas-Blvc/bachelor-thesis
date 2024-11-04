from abc import abstractmethod
from typing import Tuple, List

class Obstacle:
    @abstractmethod
    def get_constraints(self):
        raise NotImplementedError

    @abstractmethod
    def get_polygon_and_color(self) -> Tuple[List[Tuple[float, float]], str]:
        raise NotImplementedError

