from enum import Enum

class Objectives:
    norm = lambda x: (_ for _ in ()).throw(NotImplementedError('Norm was not set by Optimizer!'))
    class Type(Enum):
        MAXIMIZE = 0
        MINIMIZE = 1

    @staticmethod
    def minimize_control_input(states, control_inputs):
        objective = None
        for control_input in control_inputs:
            if objective is None:
                objective = Objectives.norm(control_input)
            else:
                objective += Objectives.norm(control_input)
        return objective, Objectives.Type.MINIMIZE


