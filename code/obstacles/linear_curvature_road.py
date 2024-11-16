import numpy as np
from scipy.special import fresnel
from scipy.optimize import minimize
from typing import List, Tuple

from obstacles.road import AbstractRoad

class LinearCurvatureRoad(AbstractRoad):
    """
    Represents a street as a B-spline curve with associated normals.
    """
    def __init__(self, s: List[Tuple[float, float]], width: float):
        super().__init__(width)
        s_array = np.array(s)
        diff = np.diff(s_array, axis=0)
        non_zero = np.any(diff != 0, axis=1)
        s_array = np.vstack([s_array[0], s_array[1:][non_zero]])

        if len(s_array) < 2:
            raise ValueError("At least two distinct points are required to define a street.")

        self.evaluation_points = 500

        self.x = s_array[:, 0]
        self.y = s_array[:, 1]

        # Fit Euler spiral (clothoid) to data points
        self.a_opt = self._fit_clothoid()
        self.t_min, self.t_max = (-5, 5)
        if not np.isfinite(self.a_opt) or self.a_opt <= 0:
            raise ValueError(f"Invalid value for a_opt: {self.a_opt}")


    def _clothoid(self, t, a=None):
        """Generates (x, y) points on a clothoid for parameter t"""
        a = self.a_opt if a is None else a
        S, C = fresnel(t / np.sqrt(np.pi * a))
        x = a * np.sqrt(np.pi) * C
        y = a * np.sqrt(np.pi) * S
        return x, y

    def _fit_clothoid(self):
        """Fit an Euler spiral (clothoid) to the provided data points."""

        def objective(a):
            t_data = np.linspace(-1, 1, len(self.x)) * a
            x_fit, y_fit = self._clothoid(t_data, a)
            return np.sum((x_fit - self.x) ** 2 + (y_fit - self.y) ** 2)

        a0 = np.array(1.0)
        res = minimize(objective, a0, bounds=[(1e-3, 1e3)])  # Set bounds to prevent extreme values
        return res.x[0] if res.success else 1.0  # Use 1.0 as a fallback

    def _transform(self, s_param):
        return (1 - s_param) * self.t_min + s_param * self.t_max

    def get_curvature_at(self, s_param) -> float:
        t = self._transform(s_param)
        return t / (self.a_opt ** 2)

    def get_curvature_derivative_at(self, s_param) -> float:
        return 1 / (self.a_opt ** 2)

    def get_global_position(self, s_param: float, lateral_offset: float) -> Tuple[float, float]:
        t = self._transform(s_param)
        x_center, y_center = self._clothoid(t)

        # Adaptive delta for computing tangent vector
        delta_t = max(1e-4, 1e-2 * abs(t))
        x_next, y_next = self._clothoid(t + delta_t)
        dx, dy = x_next - x_center, y_next - y_center

        tangent_norm = np.sqrt(dx ** 2 + dy ** 2)
        if tangent_norm == 0:
            raise ValueError("Tangent vector magnitude is zero, cannot compute normal.")

        normal_unit = np.array([-dy, dx]) / tangent_norm
        x_global = x_center + lateral_offset * normal_unit[0]
        y_global = y_center + lateral_offset * normal_unit[1]

        return x_global, y_global

    def get_curvature_min(self, start, end) -> float:
        curvatures = [self.get_curvature_at(start), self.get_curvature_at(end)]
        return min(curvatures)

    def get_curvature_max(self, start, end) -> float:
        curvatures = [self.get_curvature_at(start), self.get_curvature_at(end)]
        return max(curvatures)

    # def velocity_limit(self, s_param: float) -> float:
    #     return 40 if s_param < 10 else 42

    def get_tangent_angle_at(self, s_param: float) -> float:
        t = self._transform(s_param)

        # Compute current position on the clothoid
        x, y = self._clothoid(t)

        # Use a small delta to compute the next point for tangent approximation
        delta_t = max(1e-4, 1e-2 * abs(t))
        x_next, y_next = self._clothoid(t + delta_t)

        # Calculate the tangent vector (dx, dy)
        dx = x_next - x
        dy = y_next - y

        # Return the angle of the tangent with respect to the x-axis
        return np.arctan2(dy, dx)

    def get_polygon_and_color(self) -> Tuple[List[Tuple[float, float]], str]:
        s_values = np.linspace(0, 1, 500)
        half_width = self.width / 2.0

        upper_points = []
        lower_points = []

        for s in s_values:
            t = self._transform(s)
            x_center, y_center = self._clothoid(t)

            delta_t = max(1e-4, 1e-2 * abs(t))
            x_next, y_next = self._clothoid(t + delta_t)
            dx, dy = x_next - x_center, y_next - y_center

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