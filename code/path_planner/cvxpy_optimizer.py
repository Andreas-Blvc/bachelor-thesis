import cvxpy as cp
from cvxpy import SolverError
from typing import Tuple

from models import AbstractVehicleModel
from roads import AbstractRoad

from .objectives import Objectives
from .interface import AbstractPathPlanner

class ConvexPathPlanner(AbstractPathPlanner):
    def __init__(self, model: AbstractVehicleModel, dt, time_horizon, get_objective, verbose=False):
        super().__init__(
            model,
            dt,
            time_horizon,
            get_objective
        )

        # Configure others:
        model.solver_type = 'cvxpy'
        Objectives.norm = cp.sum_squares

        self.verbose = verbose
        # for each state_transition, we have to construct a problem with initial state as param.
        self._constructed_problems: dict[Tuple[int, int], Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter, cp.Parameter | None]] = {}


    def _construct_problem(self, state_transitions, include_goal_state=False) -> Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter, cp.Parameter | None]:
        # Define control and state variables for the entire time horizon
        u = cp.Variable((state_transitions, self.model.dim_control_input))  # Control inputs matrix: (N, 2)
        x = cp.Variable((state_transitions + 1, self.model.dim_state))  # State variables matrix: (N + 1, 4)

        initial_state = cp.Parameter((self.model.dim_state,), name="initial_state_param")
        goal_state = cp.Parameter((self.model.dim_state,), name="goal_state_param") if include_goal_state else None

        constraints = [x[0, :] == initial_state]
        additional_min_objective_term = 0

        # Goal state constraint (only position constraints at the final state)
        if goal_state is not None:
            constraints.append(x[state_transitions, :] == goal_state)
        elif (
                self.model.road_segment_idx is not None and
                self.model.road_segment_idx < len(getattr(self.model.road, 'segments', [self.model.road])) - 1
        ):
            next_segment: AbstractRoad = getattr(self.model.road, 'segments')[self.model.road_segment_idx + 1]
            final_state = self.model.convert_vec_to_state(x[state_transitions, :])
            constraints.append(final_state.get_lateral_offset() <= next_segment.n_max(0))
            constraints.append(final_state.get_lateral_offset() >= next_segment.n_min(0))
            # additional_min_objective_term += 1e3 * final_state.get_alignment_error() ** 2 + 5 * final_state.get_lateral_offset() ** 2

        # Dynamics constraints vectorized over a time horizon
        for j in range(state_transitions):
            # State transition: x_{k+1} = x_k + (A * x_k + B * u_k) * dt
            next_state, model_constraints = self.model.update(
                current_state=x[j, :].T,
                control_inputs=u[j, :].T,
                dt=self.dt,
            )
            constraints += model_constraints
            constraints.append(x[j + 1, :].T == next_state)

        # constraint goal_state:
        _, goal_state_constraints = self.model.update(
            current_state=x[state_transitions, :].T,
            control_inputs=cp.Variable((self.model.dim_control_input, 1)),  # dummy control input
            dt=self.dt,
        )
        constraints += goal_state_constraints

        # Define the optimization problem
        states = [self.model.convert_vec_to_state(x[j, :]) for j in range(state_transitions + 1)]
        control_inputs = [self.model.convert_vec_to_control_input(u[j, :]) for j in range(state_transitions)]
        objective, objective_type = self.get_objective(states, control_inputs)
        if objective_type == Objectives.Type.MINIMIZE:
            prob = cp.Problem(cp.Minimize(objective + additional_min_objective_term), constraints)
        else:
            prob = cp.Problem(cp.Maximize(objective - additional_min_objective_term), constraints)

        return prob, x, u, initial_state, goal_state

    def get_optimized_trajectory(self, initial_state):
        max_state_transitions = int(self.time_horizon / self.dt)
        prev_segments_length = 0
        solve_time = 0
        states = [initial_state]
        control_inputs = []
        failed = False
        for i, segment in enumerate(getattr(self.model.road, 'segments', [self.model.road])):
            while True:
                traveled_distance = self.model.convert_vec_to_state(initial_state).get_traveled_distance()
                distance_til_next_segment = prev_segments_length + segment.length - traveled_distance
                min_time_to_reach_next_segment = distance_til_next_segment / self.model.get_v_max()
                state_transitions = min(max_state_transitions - len(control_inputs), int(min_time_to_reach_next_segment / self.dt))
                if state_transitions * self.dt <= .010:
                    break
                else:
                    # print(f'Planning {state_transitions} transitions on {i}')
                    try:
                        self.model.road_segment_idx = i
                        prob, x, u, initial_state_param, _ = self._constructed_problems.setdefault(
                            (i, state_transitions),
                            self._construct_problem(state_transitions)
                        )
                        initial_state_param.value = initial_state
                        prob.solve(solver='MOSEK', warm_start=True)
                        solve_time += prob.solver_stats.solve_time
                    except ValueError as e:
                        print('\nvalue error', e)
                        break
                    except SolverError as e:
                        print('\nsolver error', e)
                        break

                    if x.value is None:
                        failed = True
                        break
                    states += [state for state in x.value][1:]
                    control_inputs += [control_input for control_input in u.value]
                    initial_state = states[-1]
            if failed:
                break
            prev_segments_length += segment.length


        # self.prob.solve(solver='CLARABEL', verbose=self.verbose)
        # print("Solve time:", f"{self.prob.solver_stats.solve_time:.3f}s")
        self.solve_time = solve_time
        # Return optimized state and control trajectories
        if len(control_inputs) == 0:
            print(
                '\ncannot drive further, final state:\n',
                self.model.convert_vec_to_state(states[-1]).to_string()
            )
        return states, control_inputs

