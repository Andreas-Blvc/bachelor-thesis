from typing import List, Tuple

from utils import is_float_in_range
from .road_interface import AbstractRoadSegment

class Road(AbstractRoadSegment):
    def __init__(self, segments: List[AbstractRoadSegment]):
        if len(segments) == 0:
            raise ValueError('No segments provided')
        self.segments = segments
        super().__init__(self.get_total_length(), lambda _: segments[0].width(0))

    def get_total_length(self) -> float:
        return sum(segment.length for segment in self.segments)

    def get_all_segments(self) -> List[AbstractRoadSegment]:
        return self.segments

    def _get_prev_length(self, segment_idx):
        return sum(segment.length for segment in self.segments[:segment_idx])

    def get_curvature_at(self, s: float, current_road_segment=None) -> float:
        # Assuming segments are traversed sequentially, find which segment `s` falls into
        if current_road_segment is not None:
            return self.segments[current_road_segment].get_curvature_at(s - self._get_prev_length(current_road_segment))

        current_length = 0
        for segment in self.segments:
            segment_length = segment.length
            if is_float_in_range(s, current_length, current_length + segment_length):
                local_s = s - current_length
                return segment.get_curvature_at(local_s)
            current_length += segment_length
        return 0.0

    def get_curvature_derivative_at(self, s: float, current_road_segment=None) -> float:
        if current_road_segment is not None:
            return self.segments[current_road_segment].get_curvature_derivative_at(s - self._get_prev_length(current_road_segment))

        current_length = 0
        for segment in self.segments:
            segment_length = segment.length
            if is_float_in_range(s, current_length, current_length + segment_length):
                local_s = s - current_length
                return segment.get_curvature_derivative_at(local_s)
            current_length += segment_length
        return 0.0

    def get_global_position(self, s: float, lateral_offset: float, current_road_segment=None) -> Tuple[float, float]:
        if current_road_segment is not None:
            return self.segments[current_road_segment].get_global_position(s - self._get_prev_length(current_road_segment), lateral_offset)

        current_length = 0
        for segment in self.segments:
            segment_length = segment.length
            if is_float_in_range(s, current_length, current_length + segment_length):
                local_s = s - current_length
                return segment.get_global_position(local_s, lateral_offset)
            current_length += segment_length
        return 0.0, 0.0

    def get_curvature_min(self, start: float, end: float, current_road_segment=None) -> float:
        if current_road_segment is not None:
            prev_length = self._get_prev_length(current_road_segment)
            return self.segments[current_road_segment].get_curvature_min(start - prev_length, end - prev_length)

        get_prev_length = lambda seg_idx: sum(segment.length for segment in self.segments[:seg_idx])
        curvatures = [
            segment.get_curvature_min(
                start-get_prev_length(idx),
                end-get_prev_length(idx)
            ) for idx, segment in enumerate(self.segments)]
        return min(curvatures)

    def get_curvature_max(self, start: float, end: float, current_road_segment=None) -> float:
        if current_road_segment is not None:
            prev_length = self._get_prev_length(current_road_segment)
            return self.segments[current_road_segment].get_curvature_max(start - prev_length, end - prev_length)

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

    def get_tangent_angle_at(self, s: float, current_road_segment=None) -> float:
        if current_road_segment is not None:
            return self.segments[current_road_segment].get_tangent_angle_at(s - self._get_prev_length(current_road_segment))

        current_length = 0
        for segment in self.segments:
            segment_length = segment.length
            if is_float_in_range(s, current_length, current_length + segment_length):
                local_s = s - current_length
                return segment.get_tangent_angle_at(local_s)
            current_length += segment_length
        return 0.0