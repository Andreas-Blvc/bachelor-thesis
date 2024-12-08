import casadi as ca
import numpy as np
from typing import List, Tuple, Callable, Any

from utils import is_float_in_range
from .road_interface import AbstractRoad

class SegmentDependentVariables:
    def __init__(self, C: Callable[[Any], Any], dC: Callable[[Any], Any], c_min, c_max, n_min, n_max):
        self.C = C
        self.dC = dC
        self.c_min = c_min
        self.c_max = c_max
        self.n_min = n_min
        self.n_max = n_max

class Road(AbstractRoad):
    def __init__(self, segments: List[AbstractRoad]):
        self.segment_dependent_variables = None
        self.prev_road_segment_idx = None

        if len(segments) == 0:
            raise ValueError('No segments provided')
        self.segments = segments
        def width(s):
            current_length = 0
            for segment in self.segments:
                segment_length = segment.length
                if is_float_in_range(s, current_length, current_length + segment_length):
                    local_s = s - current_length
                    return segment.width(local_s)
                current_length += segment_length
            return 0.0
        super().__init__(self.get_total_length(), width)

    def compute_casadi_segment_dependent_variable(self, s, callback: Callable[[AbstractRoad, Any], Any]):
        """
        Computes a segment-dependent variable for a given position `s` on the road
        specifically for the CasADi solver.

        Parameters:
        - s:
            The position on the road for which the segment-dependent variable is computed.
        - callback: function
            A function that takes a road segment and a local position `local_s` as inputs
            and computes the desired variable for that segment.

        Returns:
        - segment_dependent_variable: Depends on the callback's return type
            The computed value for the segment containing the position `s`.
        """
        current_length = 0
        segment_dependent_variable = 0

        for segment in self.segments:
            local_s = s - current_length
            in_segment = ca.logic_and(current_length <= s, s <= current_length + segment.length)
            segment_dependent_variable = ca.if_else(
                in_segment,
                callback(segment, local_s),
                segment_dependent_variable
            )
            current_length += segment.length

        return segment_dependent_variable

    def get_segment_dependent_variables(self, s, use_casadi: bool=False, road_segment_idx: int=None) -> SegmentDependentVariables:
        """
        Retrieves road segment-dependent variables: curvature function, curvature bounds and width bounds.

        Parameters:
        - s: float or other type depending on solver requirements
            The longitudinal position on the road or a reference value.

        Returns:
        - SegmentDependentVariables: An object containing:
            - C: A function to compute the curvature at a given longitudinal offset.
            - dC: A function to compute the curvature derivative at a given longitudinal offset.
            - c_min: The minimum curvature bound of the segment at the given position `s`.
            - c_max: The maximum curvature bound of the segment at the given position `s`.
            - n_min: The minimum width bound of the segment at the given position `s`.
            - n_max: The maximum width bound of the segment at the given position `s`.


        Raises:
        - ValueError: If no road segments are defined.
        - NotImplementedError: If the case cannot be handled
          (e.g., multiple segments convex solver without predefined road segment index).
        """

        # Validate road and segments
        if self.segments is None or len(self.segments) == 0:
            raise ValueError("No road segments available.")

        # Scalar input case
        if np.isscalar(s):
            # do not consider whole segment, just the position at s
            C = self.get_curvature_at
            dC = self.get_curvature_derivative_at
            c_min = c_max = self.get_curvature_at(s)
            width = self.width(s)
            n_min = -width / 2
            n_max = width / 2

        # CasADi solver case
        elif use_casadi:
            C = lambda longitudinal_offset: self.compute_casadi_segment_dependent_variable(
                longitudinal_offset,
                lambda road_segment, local_s: road_segment.get_curvature_at(local_s)
            )
            dC = lambda longitudinal_offset: self.compute_casadi_segment_dependent_variable(
                longitudinal_offset,
                lambda road_segment, local_s: road_segment.get_curvature_derivative_at(local_s)
            )
            c_min = self.compute_casadi_segment_dependent_variable(
                s,
                lambda road_segment, _: road_segment.get_curvature_min(0, road_segment.length)
            )
            c_max = self.compute_casadi_segment_dependent_variable(
                s,
                lambda road_segment, _: road_segment.get_curvature_max(0, road_segment.length)
            )
            # width on whole segment constant:
            n_min = self.compute_casadi_segment_dependent_variable(
                s,
                lambda road_segment, local_s: -road_segment.width(local_s) / 2
            )
            n_max = self.compute_casadi_segment_dependent_variable(
                s,
                lambda road_segment, local_s: road_segment.width(local_s) / 2
            )

        # from here on, segment_dependent_variables only changes if road_segment_idx did change
        elif self.segment_dependent_variables is not None and road_segment_idx == self.prev_road_segment_idx:
            return self.segment_dependent_variables

        # Single road segment case
        elif len(self.segments) == 1:
            C = lambda longitudinal_offset: self.get_curvature_at(longitudinal_offset, road_segment_idx=0)
            dC = lambda longitudinal_offset: self.get_curvature_derivative_at(longitudinal_offset, road_segment_idx=0)
            segment = self.segments[0]
            c_min = segment.get_curvature_min(0, segment.length)
            c_max = segment.get_curvature_max(0, segment.length)
            #  width on whole segment constant:
            width = segment.width(0)
            n_min = -width / 2
            n_max = width / 2

        # Multiple segments with specified road segment index
        elif road_segment_idx is not None:
            C = lambda longitudinal_offset: self.get_curvature_at(
                longitudinal_offset,
                road_segment_idx=road_segment_idx
            )
            dC = lambda longitudinal_offset: self.get_curvature_derivative_at(
                longitudinal_offset,
                road_segment_idx=road_segment_idx
            )
            segment = self.segments[road_segment_idx]
            c_min = segment.get_curvature_min(0, segment.length)
            c_max = segment.get_curvature_max(0, segment.length)
            # width on whole segment constant:
            width = segment.width(0)
            n_min = -width / 2
            n_max = width / 2

        # Unsupported case
        else:
            raise NotImplementedError("Handling multiple road segments convex without a predefined segment not supported.")

        # save result:
        self.segment_dependent_variables = SegmentDependentVariables(C, dC, c_min, c_max, n_min, n_max)
        self.prev_road_segment_idx = road_segment_idx
        return self.segment_dependent_variables

    def get_total_length(self) -> float:
        return sum(segment.length for segment in self.segments)

    def get_all_segments(self) -> List[AbstractRoad]:
        return self.segments

    def _get_prev_length(self, segment_idx):
        return sum(segment.length for segment in self.segments[:segment_idx])

    def get_curvature_at(self, s: float, road_segment_idx=None) -> float:
        # Assuming segments are traversed sequentially, find which segment `s` falls into
        if road_segment_idx is not None:
            return self.segments[road_segment_idx].get_curvature_at(s - self._get_prev_length(road_segment_idx))

        current_length = 0
        for segment in self.segments:
            segment_length = segment.length
            if is_float_in_range(s, current_length, current_length + segment_length):
                local_s = s - current_length
                return segment.get_curvature_at(local_s)
            current_length += segment_length
        return 0.0

    def get_curvature_derivative_at(self, s: float, road_segment_idx=None) -> float:
        if road_segment_idx is not None:
            return (self.segments[road_segment_idx].
                    get_curvature_derivative_at(s - self._get_prev_length(road_segment_idx)))

        current_length = 0
        for segment in self.segments:
            segment_length = segment.length
            if is_float_in_range(s, current_length, current_length + segment_length):
                local_s = s - current_length
                return segment.get_curvature_derivative_at(local_s)
            current_length += segment_length
        return 0.0

    def get_global_position(self, s: float, lateral_offset: float, road_segment_idx=None) -> Tuple[float, float]:
        if road_segment_idx is not None:
            return (self.segments[road_segment_idx].
                    get_global_position(s - self._get_prev_length(road_segment_idx), lateral_offset))

        current_length = 0
        for segment in self.segments:
            segment_length = segment.length
            if is_float_in_range(s, current_length, current_length + segment_length):
                local_s = s - current_length
                return segment.get_global_position(local_s, lateral_offset)
            current_length += segment_length
        return 0.0, 0.0

    def get_curvature_min(self, start: float, end: float, road_segment_idx=None) -> float:
        if road_segment_idx is not None:
            prev_length = self._get_prev_length(road_segment_idx)
            return self.segments[road_segment_idx].get_curvature_min(start - prev_length, end - prev_length)

        get_prev_length = lambda seg_idx: sum(segment.length for segment in self.segments[:seg_idx])
        curvatures = [
            segment.get_curvature_min(
                start-get_prev_length(idx),
                end-get_prev_length(idx)
            ) for idx, segment in enumerate(self.segments)]
        return min(curvatures)

    def get_curvature_max(self, start: float, end: float, road_segment_idx=None) -> float:
        if road_segment_idx is not None:
            prev_length = self._get_prev_length(road_segment_idx)
            return self.segments[road_segment_idx].get_curvature_max(start - prev_length, end - prev_length)

        get_prev_length = lambda seg_idx: sum(segment.length for segment in self.segments[:seg_idx])
        curvatures = [
            segment.get_curvature_max(
                start - get_prev_length(idx),
                end - get_prev_length(idx)
            ) for idx, segment in enumerate(self.segments)]
        return max(curvatures)

    def get_polygon_and_color(self) -> Tuple[List[Tuple[float, float]], str]:
        polygons = []
        for segment in self.segments:
            segment_polygon, _ = segment.get_polygon_and_color()
            mid_index = int(len(polygons)/2)
            polygons = polygons[:mid_index] + segment_polygon + polygons[mid_index:]
        return polygons, "grey"

    def get_tangent_angle_at(self, s: float, road_segment_idx=None) -> float:
        if road_segment_idx is not None:
            return (self.segments[road_segment_idx].
                    get_tangent_angle_at(s - self._get_prev_length(road_segment_idx)))

        current_length = 0
        for segment in self.segments:
            segment_length = segment.length
            if is_float_in_range(s, current_length, current_length + segment_length):
                local_s = s - current_length
                return segment.get_tangent_angle_at(local_s)
            current_length += segment_length
        return 0.0