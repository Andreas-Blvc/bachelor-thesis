from typing import List, Tuple

import numpy as np

from obstacles import AbstractRoad


class ConstantCurvatureRoad(AbstractRoad):
    """
    Represents a road with constant curvature, implemented as a circular arc.
    """

    def __init__(self, midpoint: Tuple[float, float], start_angle: float, width: float, radius: float, length: float):
        super().__init__(width)
        self.radius = radius
        self.length = length  # Length of the road segment
        self.midpoint = np.array(midpoint)
        self.start_angle = start_angle  # Angle at the midpoint of the arc
        self.width = width

    def _arc(self, s_param):
        """
        Generates (x, y) points on a circular arc for parameter s_param.
        Maps s_param (0 to 1) to the length of the road segment.
        """
        arc_length = s_param
        angle = self.start_angle - arc_length / self.radius  # Subtract to move clockwise
        x = self.midpoint[0] + self.radius * np.cos(angle)
        y = self.midpoint[1] + self.radius * np.sin(angle)
        return x, y

    def get_curvature_at(self, s_param) -> float:
        """Returns the constant curvature of the circular arc."""
        return -1 / self.radius

    def get_curvature_derivative_at(self, s_param) -> float:
        """The curvature derivative is zero for a constant curvature road."""
        return 0

    def get_global_position(self, s_param: float, lateral_offset: float) -> Tuple[float, float]:
        """
        Returns the global position on the road for a given parameter s_param (0 to 1) and lateral offset.
        """
        x_center, y_center = self._arc(s_param)

        # Tangent calculation for the circular arc
        arc_length = s_param
        angle = self.start_angle - arc_length / self.radius  # Subtract to move clockwise
        dx, dy = self.radius * np.sin(angle), -self.radius * np.cos(angle)  # Adjusted dx, dy for clockwise direction
        tangent_norm = np.sqrt(dx ** 2 + dy ** 2)

        if tangent_norm == 0:
            raise ValueError("Tangent vector magnitude is zero, cannot compute normal.")

        normal_unit = np.array([-dy, dx]) / tangent_norm
        x_global = x_center + lateral_offset * normal_unit[0]
        y_global = y_center + lateral_offset * normal_unit[1]

        return x_global, y_global

    def get_curvature_min(self, start, end) -> float:
        """Returns the constant minimum curvature (since curvature is constant)."""
        return -1 / self.radius

    def get_curvature_max(self, start, end) -> float:
        """Returns the constant maximum curvature (since curvature is constant)."""
        return -1 / self.radius

    def get_tangent_angle_at(self, s_param: float) -> float:
        """
        Returns the angle of the tangent to the x-axis at a given parameter s_param (0 to 1).
        """
        arc_length = s_param
        angle = self.start_angle - arc_length / self.radius  # Subtract to make the tangent angle clockwise
        return angle - np.pi / 2

    def get_polygon_and_color(self) -> Tuple[List[Tuple[float, float]], str]:
        """
        Returns the polygon representing the road boundaries and its color for visualization.
        """
        s_values = np.linspace(0, self.length, 500)  # evenly spaced s_param values from 0 to 1
        half_width = self.width / 2.0

        upper_points = []
        lower_points = []

        for s in s_values:
            x_center, y_center = self._arc(s)

            # Tangent calculation for normal vector
            arc_length = s
            angle = self.start_angle - arc_length / self.radius  # Subtract for clockwise direction
            dx, dy = self.radius * np.sin(angle), -self.radius * np.cos(angle)  # Adjusted dx, dy for clockwise
            tangent_norm = np.sqrt(dx ** 2 + dy ** 2)

            if tangent_norm == 0:
                raise ValueError("Tangent vector magnitude is zero, cannot compute normal.")

            normal_unit = np.array([-dy, dx]) / tangent_norm

            x_upper = x_center + half_width * normal_unit[0]
            y_upper = y_center + half_width * normal_unit[1]
            x_lower = x_center - half_width * normal_unit[0]
            y_lower = y_center - half_width * normal_unit[1]

            upper_points.append((x_upper, y_upper))
            lower_points.append((x_lower, y_lower))

        polygon = upper_points + lower_points[::-1]
        return polygon, 'grey'
