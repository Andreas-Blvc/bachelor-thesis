from typing import Tuple, List
import math

from ..interface import AbstractRoad, LateralOffsetBound


class CircularCurveRoad(AbstractRoad):
    def __init__(
            self,
            n_min: LateralOffsetBound,
            n_max: LateralOffsetBound,
            radius: float,
            center: Tuple[float, float],
            start_angle: float,
            angle_sweep: float
    ):
        self.radius = radius
        self.center = center
        self.start_angle = start_angle  # Starting angle of the arc
        self.angle_sweep = angle_sweep  # Sweep of the curve in radians
        super().__init__(self.get_road_length(), n_min, n_max)

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

    def get_road_position(self, x: float, y: float) -> Tuple[float, float]:
        # Calculate the vector from the center to the given point
        dx = x - self.center[0]
        dy = y - self.center[1]

        # Calculate the distance from the center to the point
        distance = math.sqrt(dx ** 2 + dy ** 2)

        # Determine the angle of the point relative to the center
        angle = math.atan2(dy, dx)

        # Normalize the angle to the range [0, 2π)
        if self.angle_sweep > 0:
            angle = (angle - self.start_angle) % (2 * math.pi)
        else:
            angle = (self.start_angle - angle) % (2 * math.pi)

        # Ensure the angle lies within the swept range
        if angle > abs(self.angle_sweep):
            raise ValueError(f"The given point is not on the road curve. ({x}, {y})")

        # Calculate the s parameter
        s = angle / abs(self.angle_sweep) * self.length

        # Calculate the lateral offset
        lateral_offset = (distance - self.radius) * (-1 if self.angle_sweep > 0 else 1)

        if not self.n_min(s) <= lateral_offset <= self.n_max(s):
            raise ValueError("The given point is not on the road curve.")

        return s, lateral_offset

    def get_curvature_min(self, start: float, end: float) -> float:
        return self.get_curvature_at(0)  # curvature is constant

    def get_curvature_max(self, start: float, end: float) -> float:
        return self.get_curvature_at(0)  # curvature is constant

    def get_polygon_and_color(self) -> Tuple[List[Tuple[float, float]], str]:
        segments = 100
        polygon = []
        for i in range(segments + 1):
            s = i / segments * self.length
            polygon.append(self.get_global_position(s, self.n_max(s)))
        for i in range(segments, -1, -1):
            s = i / segments * self.length
            polygon.append(self.get_global_position(s, self.n_min(s)))
        return polygon, "blue"

    def get_tangent_angle_at(self, s: float) -> float:
        if self.angle_sweep > 0:
            return ((
                    (math.pi/2 + self.start_angle + self.angle_sweep * (s / self.length))
            ) + math.pi) % (2 * math.pi) - math.pi
        else:
            return ((
                    -math.pi/2 + self.start_angle +  self.angle_sweep * (s / self.length)
            ) + math.pi) % (2 * math.pi) - math.pi
