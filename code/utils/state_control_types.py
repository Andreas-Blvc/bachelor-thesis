class State:
    def __init__(
            self,
            vec,
            get_distance_between,
            get_velocity,
            get_offset_from_reference_path,
            get_traveled_distance,
            get_remaining_distance,
            get_position_orientation,
            to_string,
    ):
        self._vec = vec
        self.get_distance_between = get_distance_between
        self.get_velocity = get_velocity
        self.get_offset_from_reference_path = get_offset_from_reference_path
        self.get_traveled_distance = get_traveled_distance
        self.get_remaining_distance = get_remaining_distance
        self.get_position_orientation = get_position_orientation
        self.to_string = to_string

    def as_vector(self):
        return self._vec

class ControlInput:
    def __init__(self, vec, to_string):
        self._vec = vec
        self.to_string = to_string

    def as_vector(self):
        return self._vec