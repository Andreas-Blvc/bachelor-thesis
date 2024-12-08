from typing import Tuple, List
import math

from ..road_interface import AbstractRoad


class CircularCurveRoad(AbstractRoad):
    def __init__(self, width: float, radius: float, center: Tuple[float, float], start_angle: float, angle_sweep: float):
        self.radius = radius
        self.center = center
        self.start_angle = start_angle  # Starting angle of the arc
        self.angle_sweep = angle_sweep  # Sweep of the curve in radians
        super().__init__(self.get_road_length(), lambda _: width)

    def get_road_length(self) -> float:
        return abs(self.radius * self.angle_sweep)

    def get_curvature_at(self, s_param: float) -> float:
        return 1 / self.radius * (-1 if self.angle_sweep < 0 else 1)

    def get_curvature_derivative_at(self, s_param: float) -> float:
        return 0.0  # Constant curvature for a circular arc.

    def get_global_position(self, s: float, lateral_offset: float) -> Tuple[float, float]:
        # Ensure that s parameter is within the valid range
        if s < -1e-6 or s > self.length+1e-3:
            raise ValueError(f"s_param={s} is out of bounds. It should be between 0 and the length of the road {self.length}.")

        # Calculate the angle based on s parameter and the total angle sweep
        angle = self.start_angle + (s / self.length) * self.angle_sweep
        radius_offset = self.radius + lateral_offset * (-1 if self.angle_sweep > 0 else 1)
        x = self.center[0] + radius_offset * math.cos(angle)
        y = self.center[1] + radius_offset * math.sin(angle)
        return x, y

    def get_curvature_min(self, start: float, end: float) -> float:
        return self.get_curvature_at(0)  # curvature is constant

    def get_curvature_max(self, start: float, end: float) -> float:
        return self.get_curvature_at(0)  # curvature is constant

    def get_polygon_and_color(self) -> Tuple[List[Tuple[float, float]], str]:
        segments = 20
        polygon = []
        for i in range(segments + 1):
            s = i / segments * self.length
            polygon.append(self.get_global_position(s, self.width(s) / 2))
        for i in range(segments, -1, -1):
            s = i / segments * self.length
            polygon.append(self.get_global_position(s, -self.width(s) / 2))
        return polygon, "blue"

    def get_tangent_angle_at(self, s: float) -> float:
        return (0.5 if self.angle_sweep > 0 else 1.5) * math.pi + self.start_angle + (s / self.length) * self.angle_sweep
