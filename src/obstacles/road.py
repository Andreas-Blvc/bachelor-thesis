import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np
import casadi as ca
from obstacles.obstacle import Obstacle
from scipy.interpolate import splprep, splev

class Road(Obstacle):
    """
    Represents a street as a B-spline curve with associated normals.
    """
    def __init__(self, s: List[Tuple[float, float]], width: float,
                 smooth: float = 0.1, curvature_threshold=1e-1, evaluation_points=1000
                 ):
        self.width = width
        s_array = np.array(s)
        diff = np.diff(s_array, axis=0)
        non_zero = np.any(diff != 0, axis=1)
        s_array = np.vstack([s_array[0], s_array[1:][non_zero]])

        if len(s_array) < 2:
            raise ValueError("At least two distinct points are required to define a street.")

        self.x = s_array[:, 0]
        self.y = s_array[:, 1]
        distances = np.sqrt(np.diff(self.x)**2 + np.diff(self.y)**2)
        self.t = np.concatenate(([0], np.cumsum(distances)))
        self.total_length = self.t[-1]

        if self.total_length == 0:
            raise ValueError("Total length of the street path is zero.")

        # Fit spline to the data points
        self.tck = splprep([self.x, self.y], s=smooth)[0]

        self.curvature_threshold = curvature_threshold
        self.evaluation_points = evaluation_points
        self.split_points, self.curvature_values = self._adaptive_precompute_curvature()

    def _evaluate_splev(self, s, der):
        # Numeric case: s is a float or integer
        if isinstance(s, (float, int)):
            return splev(s, self.tck, der=der)
        else:
            raise ValueError("s should be either a float or a int")

    def _adaptive_precompute_curvature(self):
        # Sample s values over the entire spline range
        s_min, s_max = (0, 1)
        s_values = np.linspace(s_min, s_max, self.evaluation_points)

        split_points = []
        curvature_values = []

        # Compute curvature at each sample point and adaptively choose split points
        last_curvature_derivative = None
        for s in s_values:
            dx, dy = self._evaluate_splev(s, der=1)
            ddx, ddy = self._evaluate_splev(s, der=2)
            dddx, dddy = self._evaluate_splev(s, der=3)

            curvature = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5

            # Corrected curvature derivative calculation
            num1 = (dx ** 2 + dy ** 2) * (dx * dddy - dy * dddx)
            num2 = 3 * (dx * ddy - dy * ddx) * (dx * ddx + dy * ddy)
            numerator = num1 - num2
            denominator = (dx ** 2 + dy ** 2) ** 3
            curvature_derivative = numerator / denominator if denominator != 0 else 0

            if last_curvature_derivative is None or abs(last_curvature_derivative - curvature_derivative) > self.curvature_threshold:
                split_points.append(s)
                curvature_values.append(curvature)
                last_curvature_derivative = curvature_derivative

        split_points.append(s_values[-1])
        curvature_values.append(curvature)
        print(f"Found {len(split_points)} Split Points")
        return split_points, curvature_values

    def _get_piecewise_linear_curvature(self, s_param):
        # Find the segment `s_param` falls into
        if isinstance(s_param, (float, int)):
            # Numeric case
            for i in range(len(self.split_points) - 1):
                if self.split_points[i] <= s_param <= self.split_points[i + 1]:
                    # Linear interpolation within the segment
                    s0, s1 = self.split_points[i], self.split_points[i + 1]
                    k0, k1 = self.curvature_values[i], self.curvature_values[i + 1]
                    # Linear interpolation formula
                    return k0 + (s_param - s0) * (k1 - k0) / (s1 - s0)

        elif isinstance(s_param, ca.MX):
            # Symbolic case: use CasADi conditional expressions for linear interpolation
            curvature = self.curvature_values[-1]  # Default to last value
            for i in range(len(self.split_points) - 1):
                s0, s1 = self.split_points[i], self.split_points[i + 1]
                k0, k1 = self.curvature_values[i], self.curvature_values[i + 1]
                # Linear interpolation formula with conditional selection
                interpolated_curvature = k0 + (s_param - s0) * (k1 - k0) / (s1 - s0)
                curvature = ca.if_else(
                    ca.logic_and(s_param >= s0, s_param <= s1),
                    interpolated_curvature,
                    curvature
                )
            return curvature

        else:
            raise ValueError("s_param should be either a float, int, or CasADi MX variable")


    def _get_piecewise_constant_dc(self, s_param):
        # Find the segment `s_param` falls into
        if isinstance(s_param, (float, int)):
            # Numeric case
            for i in range(len(self.split_points) - 1):
                if self.split_points[i] <= s_param <= self.split_points[i + 1]:
                    s0, s1 = self.split_points[i], self.split_points[i + 1]
                    k0, k1 = self.curvature_values[i], self.curvature_values[i + 1]
                    return (k1 - k0) / (s1 - s0)

        elif isinstance(s_param, ca.MX):
            # Symbolic case: use CasADi conditional expressions for linear interpolation
            curvature = self.curvature_values[-1]  # Default to last value
            for i in range(len(self.split_points) - 1):
                s0, s1 = self.split_points[i], self.split_points[i + 1]
                k0, k1 = self.curvature_values[i], self.curvature_values[i + 1]
                # Linear interpolation formula with conditional selection
                curvature = ca.if_else(
                    ca.logic_and(s_param >= s0, s_param <= s1),
                    (k1 - k0) / (s1 - s0),
                    curvature
                )
            return curvature

        else:
            raise ValueError("s_param should be either a float, int, or CasADi MX variable")


    def get_curvature_at(self, s_param) -> float:
        return self._get_piecewise_linear_curvature(s_param)

    def get_curvature_derivative_at(self, s_param) -> float:
        return self._get_piecewise_constant_dc(s_param)


    def get_global_position(self, s_param: float, lateral_offset: float) -> Tuple[float, float]:
        x_center, y_center = splev(s_param, self.tck)
        dx, dy = splev(s_param, self.tck, der=1)

        tangent_norm = np.sqrt(dx ** 2 + dy ** 2)
        if tangent_norm == 0:
            raise ValueError("Tangent vector magnitude is zero, cannot compute normal.")

        normal_unit = np.array([-dy, dx]) / tangent_norm
        x_global = x_center + lateral_offset * normal_unit[0]
        y_global = y_center + lateral_offset * normal_unit[1]

        return x_global, y_global

    def get_curvature_min(self) -> float:
        sampled_curvatures = [self.get_curvature_at(u) for u in np.linspace(0, 1, self.evaluation_points)]
        return min(sampled_curvatures)

    def get_curvature_max(self) -> float:
        sampled_curvatures = [self.get_curvature_at(u) for u in np.linspace(0, 1, self.evaluation_points)]
        return max(sampled_curvatures)

    def velocity_limit(self, s_param: float) -> float:
        return 40 if s_param < 10 else 42

    def get_constraints(self):
        return []

    def _sample_spline(self) -> np.ndarray:
        """
        Samples points along the B-spline.s

        :return: Array of sampled points on the spline.
        """
        u = np.linspace(0, 1, self.evaluation_points)
        x, y = splev(u, self.tck)
        return np.vstack([x, y]).T

    def get_polygon_and_color(self) -> Tuple[List[Tuple[float, float]], str]:
        """
        Constructs the polygon representing the street boundaries and its color.

        :return: A tuple containing the polygon as a list of (x, y) tuples and the color string.
        """
        sampled_points = self._sample_spline()

        # Compute tangents
        tangents = np.gradient(sampled_points, axis=0)
        tangent_lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
        # Avoid division by zero
        tangent_lengths[tangent_lengths == 0] = 1e-8
        tangents_unit = tangents / tangent_lengths

        # Compute normals
        normals = np.empty_like(tangents_unit)
        normals[:, 0] = -tangents_unit[:, 1]
        normals[:, 1] = tangents_unit[:, 0]

        # Scale normals by half the street width
        half_width = self.width / 2.0
        normals_scaled = normals * half_width

        # Compute upper and lower boundary points
        upper_points = sampled_points + normals_scaled
        lower_points = sampled_points - normals_scaled

        # Convert to list of tuples
        upper_points = [(point[0], point[1]) for point in upper_points]
        lower_points = [(point[0], point[1]) for point in lower_points]

        # Combine upper and lower boundaries to form a closed polygon
        polygon = upper_points + lower_points[::-1]

        return polygon, 'grey'

    def plot_combined_curvature_and_derivative(self):
        """
        Plots both the approximated and exact curvature and its derivative along the road for comparison.
        """
        s_samples = np.linspace(0, 1, self.evaluation_points)

        # Compute approximated curvature and its derivative
        approx_curvatures = [float(self.get_curvature_at(s)) for s in s_samples]
        approx_curvature_derivatives = [float(self.get_curvature_derivative_at(s)) for s in s_samples]

        # Compute exact curvature and its finite difference derivative approximation
        exact_curvatures = []
        exact_curvature_derivatives = []

        # Calculate exact curvature values
        for s in s_samples:
            dx, dy = self._evaluate_splev(s, der=1)
            ddx, ddy = self._evaluate_splev(s, der=2)
            numerator = dx * ddy - dy * ddx
            denominator = (dx ** 2 + dy ** 2) ** 1.5
            exact_curvature = numerator / denominator if denominator != 0 else 0
            exact_curvatures.append(exact_curvature)

        # Use finite differences to approximate the derivative of exact curvature
        for i in range(1, len(exact_curvatures)):
            d_curvature = exact_curvatures[i] - exact_curvatures[i - 1]
            d_s = s_samples[i] - s_samples[i - 1]
            exact_curvature_derivative = d_curvature / d_s if d_s != 0 else 0
            exact_curvature_derivatives.append(exact_curvature_derivative)

        # Pad the derivative array to match lengths for plotting (optional: replicate last derivative)
        exact_curvature_derivatives = [exact_curvature_derivatives[0]] + exact_curvature_derivatives

        # Convert lists to arrays for plotting
        approx_curvatures = np.array(approx_curvatures).flatten()
        approx_curvature_derivatives = np.array(approx_curvature_derivatives).flatten()
        exact_curvatures = np.array(exact_curvatures)
        exact_curvature_derivatives = np.array(exact_curvature_derivatives)

        # Plot both approximated and exact curvature and derivatives
        plt.figure(figsize=(12, 10))

        # Curvature plot
        plt.subplot(2, 1, 1)
        plt.plot(s_samples, approx_curvatures, label='Approximated Curvature')
        plt.plot(s_samples, exact_curvatures, label='Exact Curvature', linestyle='--', color='orange')
        plt.ylabel('Curvature')
        plt.title('Curvature along the Road (Approximated vs Exact)')
        plt.grid()
        plt.legend()

        # Curvature derivative plot
        plt.subplot(2, 1, 2)
        plt.plot(s_samples, approx_curvature_derivatives, label='Approximated Curvature Derivative')
        plt.plot(s_samples, exact_curvature_derivatives, label='Exact Curvature Derivative (Finite Differences)',
                 linestyle='--', color='red')
        plt.ylabel('dκ/ds')
        plt.title('Curvature Derivative along the Road (Approximated vs Exact)')
        plt.grid()
        plt.legend()

        # Show combined plot
        plt.tight_layout()
        plt.show()




