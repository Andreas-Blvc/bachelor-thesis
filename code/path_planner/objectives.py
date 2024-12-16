from enum import Enum
from typing import List

from pygments.lexers.objective import objective

from utils import ControlInput, State


def _validate_typs(states, control_inputs):
    # Validate states
    if not all(isinstance(state, State) for state in states):
        raise TypeError("All elements in 'states' must be of type 'State'.")

    # Validate control_inputs
    if not all(isinstance(control_input, ControlInput) for control_input in control_inputs):
        raise TypeError("All elements in 'control_inputs' must be of type 'ControlInput'.")


class Objectives:
    norm = lambda x: (_ for _ in ()).throw(NotImplementedError('Norm was not set by Optimizer!'))
    class Type(Enum):
        MAXIMIZE = 0
        MINIMIZE = 1


    @staticmethod
    def minimize_control_input(states: List[State], control_inputs: List[ControlInput]):
        _validate_typs(states, control_inputs)
        objective = None
        for control_input in control_inputs:
            if objective is None:
                objective = Objectives.norm(control_input.as_vector())
            else:
                objective += Objectives.norm(control_input.as_vector())
        return objective, Objectives.Type.MINIMIZE

    @staticmethod
    def maximize_distance(states: List[State], control_inputs: List[ControlInput]):
        _validate_typs(states, control_inputs)
        objective = states[-1].get_traveled_distance() - states[-1].get_offset_from_reference_path()
        return objective, Objectives.Type.MAXIMIZE


    @staticmethod
    def minimize_velocity_deviation(states: List[State], control_inputs: List[ControlInput]):
        _validate_typs(states, control_inputs)
        objective = None
        ref_velocity = states[0].get_velocity()
        for state in states:
            if objective is None:
                objective = (state.get_velocity() - ref_velocity) ** 2
            else:
                objective += (state.get_velocity() - ref_velocity) ** 2
        return objective, Objectives.Type.MINIMIZE

    @staticmethod
    def minimize_remaining_distance(states: List[State], control_inputs: List[ControlInput]):
        _validate_typs(states, control_inputs)
        objective = states[-1].get_remaining_distance() + states[-1].get_offset_from_reference_path()
        return objective, Objectives.Type.MINIMIZE

    @staticmethod
    def minimize_offset_from_reference_path(states: List[State], control_inputs: List[ControlInput]):
        _validate_typs(states, control_inputs)
        objective = 0 * states[-1].get_offset_from_reference_path()
        for state in states:
            objective += state.get_offset_from_reference_path()
        return objective, Objectives.Type.MINIMIZE

    @staticmethod
    def minimize_inputs_and_offset(states: List[State], control_inputs: List[ControlInput]):
        _validate_typs(states, control_inputs)
        objective = None
        for state in states:
            objective = (objective or 0) + state.get_offset_from_reference_path()
        for control in control_inputs:
            objective += Objectives.norm(control.as_vector())
        return objective, Objectives.Type.MINIMIZE

    @staticmethod
    def minimize_alignment_error_and_later_offset(states: List[State], control_inputs: List[ControlInput]):
        _validate_typs(states, control_inputs)
        objective = 0
        for state in states:
            objective += state.get_lateral_offset() ** 2 + state.get_alignment_error() ** 2
        return objective, Objectives.Type.MINIMIZE
