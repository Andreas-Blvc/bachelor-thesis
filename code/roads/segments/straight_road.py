from typing import Tuple, List
import math

from ..interface import AbstractRoad, LateralOffsetBound

class StraightRoad(AbstractRoad):
    """
    s goes from 0 to length
    """
    def __init__(
            self,
            n_min: LateralOffsetBound,
            n_max: LateralOffsetBound,
            length: float,
            start_position: Tuple[float, float],
            direction_angle: float
    ):
        super().__init__(length, n_min, n_max)
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

        if not self.n_min(s) <= lateral_offset <= self.n_max(s):
            raise ValueError("The given point is not on the road curve.")

        return s, lateral_offset

    def get_curvature_min(self, start: float, end: float) -> float:
        return 0.0

    def get_curvature_max(self, start: float, end: float) -> float:
        return 0.0

    def get_polygon_and_color(self) -> Tuple[List[Tuple[float, float]], str]:
        segments = 20
        polygon = []
        for i in range(segments + 1):
            s = i / segments * self.length
            polygon.append(self.get_global_position(s, self.n_max(s)))
        for i in range(segments, -1, -1):
            s = i / segments * self.length
            polygon.append(self.get_global_position(s, self.n_min(s)))
        return polygon, "gray"

    def get_tangent_angle_at(self, s_param: float) -> float:
        return (self.direction_angle + math.pi) % (2 * math.pi) - math.pi
