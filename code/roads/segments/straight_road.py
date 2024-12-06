from typing import Tuple, List
import math

from ..road_interface import AbstractRoadSegment

class StraightRoad(AbstractRoadSegment):
    """
    s goes from 0 to length
    """
    def __init__(self, width: float, length: float, start_position: Tuple[float, float], direction_angle: float):
        super().__init__(length, lambda _: width)
        self.start_position = start_position
        self.direction_angle = direction_angle  # angle in radians with respect to x-axis

    def get_curvature_at(self, s_param: float) -> float:
        return 0.0  # A straight road has zero curvature.

    def get_curvature_derivative_at(self, s_param: float) -> float:
        return 0.0  # No change in curvature for a straight road.

    def get_global_position(self, s: float, lateral_offset: float) -> Tuple[float, float]:
        if s < -1e-6 or s > self.length+1e-6:
            raise ValueError("s_param is out of bounds. It should be between 0 and the length of the road.")

        x = self.start_position[0] + s * math.cos(self.direction_angle) - lateral_offset * math.sin(self.direction_angle)
        y = self.start_position[1] + s * math.sin(self.direction_angle) + lateral_offset * math.cos(self.direction_angle)
        return x, y

    def get_curvature_min(self, start: float, end: float) -> float:
        return 0.0

    def get_curvature_max(self, start: float, end: float) -> float:
        return 0.0

    def get_polygon_and_color(self) -> Tuple[List[Tuple[float, float]], str]:
        # Approximate the road as a rectangle
        x1, y1 = self.get_global_position(0, self.width(0) / 2)
        x2, y2 = self.get_global_position(self.length, self.width(self.length) / 2)
        x3, y3 = self.get_global_position(self.length, -self.width(self.length) / 2)
        x4, y4 = self.get_global_position(0, -self.width(0) / 2)
        return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], "gray"

    def get_tangent_angle_at(self, s_param: float) -> float:
        return self.direction_angle
