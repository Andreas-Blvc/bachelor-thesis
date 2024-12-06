from typing import List, Tuple

from .segments.abstract_road_segment import AbstractRoadSegment

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

    def get_curvature_at(self, s: float) -> float:
        # Assuming segments are traversed sequentially, find which segment `s` falls into
        current_length = 0
        for segment in self.segments:
            segment_length = segment.length
            if current_length <= s < current_length + segment_length:
                local_s = s - current_length
                return segment.get_curvature_at(local_s)
            current_length += segment_length
        return 0.0

    def get_curvature_derivative_at(self, s: float) -> float:
        current_length = 0
        for segment in self.segments:
            segment_length = segment.length
            if current_length <= s < current_length + segment_length:
                local_s = s - current_length
                return segment.get_curvature_derivative_at(local_s)
            current_length += segment_length
        return 0.0

    def get_global_position(self, s: float, lateral_offset: float) -> Tuple[float, float]:
        current_length = 0
        for segment in self.segments:
            segment_length = segment.length
            if current_length <= s < current_length + segment_length:
                local_s = s - current_length
                return segment.get_global_position(local_s, lateral_offset)
            current_length += segment_length
        return 0.0, 0.0

    def get_curvature_min(self, start: float, end: float) -> float:
        curvatures = [segment.get_curvature_min(0, segment.length) for segment in self.segments]
        return min(curvatures)

    def get_curvature_max(self, start: float, end: float) -> float:
        curvatures = [segment.get_curvature_max(0, segment.length) for segment in self.segments]
        return max(curvatures)

    def get_polygon_and_color(self) -> Tuple[List[Tuple[float, float]], str]:
        polygons = []
        for segment in self.segments:
            segment_polygon, color = segment.get_polygon_and_color()
            polygons.extend(segment_polygon)
        return polygons, "black"

    def get_tangent_angle_at(self, s: float) -> float:
        current_length = 0
        for segment in self.segments:
            segment_length = segment.length
            if current_length <= s < current_length + segment_length:
                local_s = s - current_length
                return segment.get_tangent_angle_at(local_s)
            current_length += segment_length
        return 0.0