from typing import Tuple, List
import math

from ..interface import AbstractRoad

class StraightRoad(AbstractRoad):
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
        if s < -1e-6 or s > self.length+1e-3:
            raise ValueError(f"s_param={s} is out of bounds. It should be between 0 and the length of the road {self.length}.")

        x = self.start_position[0] + s * math.cos(self.direction_angle) - lateral_offset * math.sin(self.direction_angle)
        y = self.start_position[1] + s * math.sin(self.direction_angle) + lateral_offset * math.cos(self.direction_angle)
        return x, y

    def get_road_position(self, x: float, y: float) -> Tuple[float, float]:
        # Direction vector of the road
        dx = math.cos(self.direction_angle)
        dy = math.sin(self.direction_angle)

        # Vector from the start position to the given point
        vx = x - self.start_position[0]
        vy = y - self.start_position[1]

        # Project the vector onto the road's direction to get the s parameter
        s = vx * dx + vy * dy

        # Check if s is within valid range
        if s < -1e-6 or s > self.length + 1e-3:
            raise ValueError(f"The point ({x}, {y}) is not on the road or out of bounds.")

        # Perpendicular vector to the road direction
        px = -dy
        py = dx

        # Project the vector onto the perpendicular direction to get the lateral offset
        lateral_offset = vx * px + vy * py

        if abs(lateral_offset) > self.width(s)/2:
            raise ValueError("The given point is not on the road curve.")

        return s, lateral_offset

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
        return (self.direction_angle + math.pi) % (2 * math.pi) - math.pi
