class State:
    @staticmethod
    def get_distance_between(state_a, state_b):
        raise NotImplementedError('get_distance_between was not set!')

    @staticmethod
    def get_traveled_distance(state):
        raise NotImplementedError('get_traveled_distance was not set!')

    @staticmethod
    def get_remaining_distance(state):
        raise NotImplementedError('get_remaining_distance was not set!')

    @staticmethod
    def get_offset_from_reference_path(state):
        raise NotImplementedError('get_offset_from_reference_path was not set!')

    @staticmethod
    def get_velocity(state):
        raise NotImplementedError('get_velocity was not set!')

    def __init__(self, vec):
        self._vec = vec
        self.velocity = State.get_velocity(self)
        self.offset_from_reference_path = State.get_offset_from_reference_path(self)
        self.traveled_distance = State.get_traveled_distance(self)
        self.remaining_distance = State.get_remaining_distance(self)

    def as_vector(self):
        return self._vec

    def distance_to(self, other):
        if not isinstance(other, State):
            raise ValueError(f"Cannot calculate distance from {type(self).__name__} to {type(other).__name__}")
        return State.get_distance_between(self, other)


class ControlInput:
    def __init__(self, vec):
        self._vec = vec

    def as_vector(self):
        return self._vec