from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Protocol, Any


class LateralOffsetBound(Protocol):
    def __call__(self, s: Any, road_segment_idx: int = None) -> float: ...


class SegmentDependentVariables:
    def __init__(
            self,
            C: Callable[[Any], Any],
            dC: Callable[[Any], Any],
            c_min,
            c_max,
            n_min: LateralOffsetBound,
            n_max: LateralOffsetBound
    ):
        self.C = C
        self.dC = dC
        self.c_min = c_min
        self.c_max = c_max
        self.n_min = n_min
        self.n_max = n_max


class AbstractRoad(ABC):
    """
    Abstract base class for different types of roads.
    Defines a standard interface for road methods and properties.
    """
    def __init__(self, length: float, n_min: LateralOffsetBound, n_max: LateralOffsetBound):
        self.length = length
        self.n_min = n_min
        self.n_max = n_max

    @abstractmethod
    def get_curvature_at(self, s_param: float) -> float:
        """
        Returns the curvature at a given parameter along the road.

        Args:
            s_param (float): Parameter along the road (typically between 0 and road.length).

        Returns:
            float: Curvature at the specified parameter.
        """
        pass

    @abstractmethod
    def get_curvature_derivative_at(self, s_param: float) -> float:
        """
        Returns the derivative of the curvature at a given parameter along the road.

        Args:
            s_param (float): Parameter along the road (typically between 0 and road.length).

        Returns:
            float: Curvature derivative at the specified parameter.
        """
        pass

    @abstractmethod
    def get_global_position(self, s_param: float, lateral_offset: float) -> Tuple[float, float]:
        """
        Returns the global (x, y) position on the road for a given parameter and lateral offset.

        Args:
            s_param (float): Parameter along the road (typically between 0 and road.length).
            lateral_offset (float): Lateral offset from the road center.

        Returns:
            Tuple[float, float]: Global (x, y) coordinates.
        """
        pass

    @abstractmethod
    def get_road_position(self, x: float, y: float) -> Tuple[float, float]:
        """
        Returns the road-relative position (s, lateral_offset) for a given global (x, y) position.

        Args:
            x (float): Global x-coordinate.
            y (float): Global y-coordinate.

        Returns:
            Tuple[float, float]: A tuple containing:
                - s (float): Parameter along the road (typically between 0 and road.length).
                - lateral_offset (float): Lateral offset from the road center.
        """
        pass

    @abstractmethod
    def get_curvature_min(self, start: float, end: float) -> float:
        """
        Returns the minimum curvature between two parameters along the road.

        Args:
            start (float): Starting parameter along the road.
            end (float): Ending parameter along the road.

        Returns:
            float: Minimum curvature between start and end.
        """
        pass

    @abstractmethod
    def get_curvature_max(self, start: float, end: float) -> float:
        """
        Returns the maximum curvature between two parameters along the road.

        Args:
            start (float): Starting parameter along the road.
            end (float): Ending parameter along the road.

        Returns:
            float: Maximum curvature between start and end.
        """
        pass

    @abstractmethod
    def get_polygon_and_color(self) -> Tuple[List[Tuple[float, float]], str]:
        """
        Returns the polygon representing the road boundaries and its color for visualization.

        Returns:
            Tuple[List[Tuple[float, float]], str]: Polygon vertices and color.
        """
        pass

    @abstractmethod
    def get_tangent_angle_at(self, s_param: float) -> float:
        """
        Returns the angle of the tangent to the x-axis at a given parameter along the road.

        Args:
            s_param (float): Parameter along the road (typically between 0 and road.length).

        Returns:
            float: Angle of the tangent in radians with respect to the x-axis.
        """
        pass

    def plot_combined_curvature_and_derivative(self):
        """
        Plots the curvature and its derivative along the road.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        s_samples = np.linspace(0, self.length, 500)

        curvatures = [self.get_curvature_at(s) for s in s_samples]
        curvature_derivatives = [self.get_curvature_derivative_at(s) for s in s_samples]

        plt.figure(figsize=(12, 10))

        plt.subplot(2, 1, 1)
        plt.plot(s_samples, curvatures, label='Curvature')
        plt.ylabel('Curvature κ')
        plt.title('Curvature along the Road')
        plt.grid()
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(s_samples, curvature_derivatives, label='Curvature Derivative')
        plt.ylabel('dκ/ds')
        plt.title('Curvature Derivative along the Road')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()
