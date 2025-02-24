from enum import Enum
from typing import List

import numpy as np

from utils import ControlInput, State


def _validate_typs(states, control_inputs):
    # Validate states
    if not all(isinstance(state, State) for state in states):
        raise TypeError("All elements in 'states' must be of type 'State'.")

    # Validate control_inputs
    if not all(isinstance(control_input, ControlInput) for control_input in control_inputs):
        raise TypeError("All elements in 'control_inputs' must be of type 'ControlInput'.")


class Objectives:
    sum_squares = lambda x: (_ for _ in ()).throw(NotImplementedError('Norm was not set by Optimizer!'))
    max = lambda x, y: (_ for _ in ()).throw(NotImplementedError('Max was not set by Optimizer!'))
    create_var = lambda: (_ for _ in ()).throw(NotImplementedError('create_var was not set by Optimizer!'))
    dt = None
    Zero = None
    class Type(Enum):
        MAXIMIZE = 0
        MINIMIZE = 1


    @staticmethod
    def minimize_control_input(states: List[State], control_inputs: List[ControlInput]):
        _validate_typs(states, control_inputs)
        objective = Objectives.Zero
        for control_input in control_inputs:
            objective += Objectives.sum_squares(control_input.as_vector())
        return objective, Objectives.Type.MINIMIZE, [], 'minimize_control_input'

    @staticmethod
    def maximize_distance(states: List[State], control_inputs: List[ControlInput]):
        _validate_typs(states, control_inputs)
        if len(states) == 0:
            return Objectives.Zero, Objectives.Type.MAXIMIZE, [], 'maximize_distance'
        objective = states[-1].get_traveled_distance()
        return objective, Objectives.Type.MAXIMIZE, [], 'maximize_distance'


    @staticmethod
    def minimize_velocity_deviation(states: List[State], control_inputs: List[ControlInput]):
        _validate_typs(states, control_inputs)
        objective = Objectives.Zero
        ref_velocity = states[0].get_velocity()
        for state in states:
            objective += (state.get_velocity() - ref_velocity) ** 2
        return objective, Objectives.Type.MINIMIZE, [], 'minimize_velocity_deviation'

    @staticmethod
    def minimize_remaining_distance(states: List[State], control_inputs: List[ControlInput]):
        _validate_typs(states, control_inputs)
        objective = states[-1].get_remaining_distance() + states[-1].get_negative_distance_to_closest_border()
        return objective, Objectives.Type.MINIMIZE, [], 'minimize_remaining_distance'

    @staticmethod
    def minimize_offset_from_reference_path(states: List[State], control_inputs: List[ControlInput]):
        _validate_typs(states, control_inputs)
        objective = Objectives.Zero
        constraints = []
        for i, state in enumerate(states):
            if Objectives.create_var is not None:
                hp_var = Objectives.create_var()
                constraints.append(hp_var >= state.get_negative_distance_to_closest_border())
                constraints.append(0 >= hp_var)
                objective += hp_var ** 2
            else:
                objective += state.get_negative_distance_to_closest_border()
        return objective, Objectives.Type.MINIMIZE, [], 'minimize_offset_from_reference_path'


    @staticmethod
    def minimize_inputs_and_offset(states: List[State], control_inputs: List[ControlInput]):
        _validate_typs(states, control_inputs)
        objective =  Objectives.Zero
        for state in states:
            objective += 10 * state.get_negative_distance_to_closest_border()
        for control in control_inputs:
            objective += Objectives.sum_squares(control.as_vector())
        return objective, Objectives.Type.MINIMIZE, [], 'minimize_inputs_and_offset'

    @staticmethod
    def minimize_alignment_error(states: List[State], control_inputs: List[ControlInput]):
        _validate_typs(states, control_inputs)
        objective = Objectives.Zero
        for idx, state in enumerate(states):
            objective += idx * state.get_alignment_error() ** 2
        return objective, Objectives.Type.MINIMIZE, [], 'minimize_alignment_error'

    @staticmethod
    def minimize_alignment_error_and_later_offset(states: List[State], control_inputs: List[ControlInput]):
        _validate_typs(states, control_inputs)
        objective = Objectives.Zero
        for state in states:
            objective += state.get_lateral_offset() ** 2 + state.get_alignment_error() ** 2
        return objective, Objectives.Type.MINIMIZE, [], 'minimize_alignment_error_and_later_offset'

    @staticmethod
    def slow_down(road_length, start_velocity):
        def ret(states: List[State], control_inputs: List[ControlInput]):
            _validate_typs(states, control_inputs)
            objective = Objectives.Zero
            for state in states:
                ref_velocity = start_velocity * (road_length - state.get_traveled_distance()) / road_length
                objective += (state.get_velocity() - ref_velocity) ** 2
            return objective, Objectives.Type.MINIMIZE, [], 'slow_down'
        return ret

    @staticmethod
    def speed_up(road_length, start_velocity, max_velocity):
        def ret(states: List[State], control_inputs: List[ControlInput]):
            _validate_typs(states, control_inputs)
            objective = Objectives.Zero
            for state in states:
                ref_velocity = start_velocity + (max_velocity - start_velocity) * state.get_traveled_distance() / road_length
                objective += (state.get_velocity() - ref_velocity) ** 2
            return objective, Objectives.Type.MINIMIZE, [], 'speed_up'
        return ret

    @staticmethod
    def minimize_control_derivatives(states: List[State], control_inputs: List[ControlInput]):
        _validate_typs(states, control_inputs)
        objective = Objectives.Zero
        constraints = []
        if len(control_inputs) == 0:
            return Objectives.Zero, Objectives.Type.MINIMIZE, constraints, 'minimize_control_derivatives'
        dim_controls = control_inputs[0].as_vector().size
        N = len(control_inputs)
        for i in range(N-1):
            if Objectives.create_var is not None:
                derivative_vars = [Objectives.create_var() for _ in range(dim_controls)]
                constraints += [
                    control_inputs[i + 1].as_vector()[j]
                    == control_inputs[i].as_vector()[j] + derivative_vars[j] * Objectives.dt(0)
                    for j in range(dim_controls)
                ]
                for idx, var in enumerate(derivative_vars):
                    objective += var**2 * (10 if idx == 1 else 1)  # hacky -> second is steering angle control
            else:
                derivatives: np.ndarray = (control_inputs[i + 1].as_vector() - control_inputs[i].as_vector()) / Objectives.dt(0)
                objective +=  float(derivatives[0]**2 + 10 * derivatives[1]**2)
        return objective, Objectives.Type.MINIMIZE, constraints, 'minimize_control_derivatives'

    @staticmethod
    def minimize_control_derivatives_offset_maximize_distance(states: List[State], control_inputs: List[ControlInput]):
        objective_1, _, constraints_1, _ = Objectives.minimize_control_derivatives(states, control_inputs)
        objective_2, _, constraints_2, _ = Objectives.maximize_distance(states, control_inputs)
        objective_3, _, constraints_3, _ = Objectives.minimize_offset_from_reference_path(states, control_inputs)
        return (
            objective_1 - 1e4 * objective_2 + 1e3 * objective_3,
            Objectives.Type.MINIMIZE,
            constraints_1 + constraints_2 + constraints_3,
            'minimize_control_derivatives_offset_maximize_distance'
        )




