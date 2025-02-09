class State:
    def __init__(
            self,
            vec,
            get_distance_between,
            get_velocity,
            get_negative_distance_to_closest_border,
            get_traveled_distance,
            get_remaining_distance,
            get_position_orientation,
            to_string,
            get_alignment_error,
            get_lateral_offset,
    ):
        self._vec = vec
        self.get_distance_between = get_distance_between
        self.get_velocity = get_velocity
        self.get_negative_distance_to_closest_border = get_negative_distance_to_closest_border
        self.get_traveled_distance = get_traveled_distance
        self.get_remaining_distance = get_remaining_distance
        self.get_position_orientation = get_position_orientation
        self.get_alignment_error = get_alignment_error
        self.get_lateral_offset = get_lateral_offset
        self.to_string = to_string

    def as_vector(self):
        return self._vec

class ControlInput:
    def __init__(self, vec, to_string):
        self._vec = vec
        self.to_string = to_string

    def as_vector(self):
        return self._vec