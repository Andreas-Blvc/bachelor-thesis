import cvxpy as cp
from typing import Tuple, Any

from models import AbstractVehicleModel
from utils import lazy_setdefault

from .objectives import Objectives
from .interface import AbstractPathPlanner

class ConvexPathPlanner(AbstractPathPlanner):
    def __init__(self, model: AbstractVehicleModel, dt, time_horizon, get_objective, verbose=False, use_param=False):
        super().__init__(
            model,
            dt,
            time_horizon,
            get_objective
        )

        # Configure others:
        model.solver_type = 'cvxpy'
        Objectives.norm = cp.sum_squares
        Objectives.max = lambda x, y: cp.max(cp.hstack([x, y]))

        self.use_param = use_param
        self.verbose = verbose
        # for each state_transition, we have to construct a problem with initial state as param.
        self._constructed_problems: dict[int, Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter]] = {}


    def _construct_problem(self, state_transitions_on_segment, initial_state) -> Tuple[cp.Problem, cp.Variable, cp.Variable, cp.Parameter | Any]:
        # Define control and state variables for the entire time horizon
        N = max(sum(state_transitions_on_segment), 0)

        u = cp.Variable((N, self.model.dim_control_input))  # Control inputs matrix: (N, 2)
        x = cp.Variable((N + 1, self.model.dim_state))  # State variables matrix: (N + 1, 4)

        if self.use_param:
            initial_state = cp.Parameter((self.model.dim_state,), name="initial_state_param")

        constraints = [x[0, :] == initial_state]
        additional_min_objective_term = 0

        if (
                self.model.road_segment_idx is not None and
                self.model.road_segment_idx == len(getattr(self.model.road, 'segments', [self.model.road])) - 1
        ):
            final_state = self.model.convert_vec_to_state(x[N, :])
            # constraints.append(final_state.get_alignment_error() == 0)
            # additional_min_objective_term += 1e3 * final_state.get_alignment_error() ** 2 + 5 * final_state.get_lateral_offset() ** 2

        # Dynamics constraints vectorized over a time horizon
        model_states_objective = cp.Constant(0)
        states = []
        for j in range(N):
            # State transition: x_{k+1} = x_k + (A * x_k + B * u_k) * dt
            road_segment_idx = next((i for i, x in enumerate(state_transitions_on_segment) if x > 0), None)
            state_transitions_on_segment[road_segment_idx] -= 1
            self.model.road_segment_idx = road_segment_idx
            next_state, model_constraints, model_objective = self.model.forward_euler_step(
                current_state=x[j, :].T,
                control_inputs=u[j, :].T,
                dt=self.dt(j),
                convexify_ref_state=initial_state,
                amount_prev_planning_states=j,
            )
            model_states_objective += model_objective
            constraints += model_constraints
            constraints.append(x[j + 1, :].T == next_state)
            states.append(self.model.convert_vec_to_state(x[j, :], road_segment_idx))

        # constraint goal_state:
        _, goal_state_constraints, model_objective = self.model.forward_euler_step(
            current_state=x[N, :].T,
            control_inputs=cp.Variable((self.model.dim_control_input, 1)),  # dummy control input
            dt=self.dt(N),
            convexify_ref_state=initial_state,
            amount_prev_planning_states=N,
        )
        constraints += goal_state_constraints
        model_states_objective += model_objective

        # Define the optimization problem
        control_inputs = [self.model.convert_vec_to_control_input(u[j, :]) for j in range(N)]
        objective, objective_type = self.get_objective(states, control_inputs)
        # TODO remove later:
        for state in states:
            constraints.append(state.get_velocity() >= 3)
        if objective_type == Objectives.Type.MINIMIZE:
            prob = cp.Problem(cp.Minimize(objective + additional_min_objective_term + model_states_objective), constraints)
        else:
            prob = cp.Problem(cp.Maximize(objective - additional_min_objective_term - model_states_objective), constraints)

        return prob, x, u, initial_state

    def get_optimized_trajectory(self, initial_state):
        max_state_transitions = self.get_state_transitions(self.time_horizon)
        prev_segments_length = 0
        ref_velocity = self.model.convert_vec_to_state(initial_state).get_velocity()
        traveled_distance = self.model.convert_vec_to_state(initial_state).get_traveled_distance()
        segments = getattr(self.model.road, 'segments', [self.model.road])
        state_transitions_on_segment = [0] * len(segments)
        for i, segment in enumerate(segments):
            if prev_segments_length <= traveled_distance <= prev_segments_length + segment.length:
                distance_til_next_segment = prev_segments_length + segment.length - traveled_distance - (0 if i < len(segments) - 1 else 3)
                estimated_time_to_reach_next_segment = distance_til_next_segment / ref_velocity
                state_transitions = min(max_state_transitions,  self.get_state_transitions(estimated_time_to_reach_next_segment, sum(state_transitions_on_segment)))
                state_transitions_on_segment[i] = state_transitions
                max_state_transitions -= state_transitions
                # assuming the car reaches the end of the current segment
                traveled_distance = prev_segments_length + segment.length
            prev_segments_length += segment.length

        # print(str(state_transitions_on_segment).ljust(20), end='')
        if sum(state_transitions_on_segment) <= 0:
            print('\nreached end of road')
            return [], []
        if self.use_param:
            prob, x, u, initial_state_param =  lazy_setdefault(
                self._constructed_problems,
                sum([x * max_state_transitions ** i for i, x in enumerate(state_transitions_on_segment)]),
                lambda: self._construct_problem(state_transitions_on_segment, None)
            )
            initial_state_param.value = initial_state
        else:
            prob, x, u, _ = self._construct_problem(state_transitions_on_segment, initial_state)
        prob.solve(solver='MOSEK', warm_start=True)

        if x.value is not None and u.value is not None:
            states = [state for state in x.value]
            control_inputs = [control_input for control_input in u.value]
        else:
            states = [initial_state]
            control_inputs = []

        # self.prob.solve(solver='CLARABEL', verbose=self.verbose)
        # print("Solve time:", f"{self.prob.solver_stats.solve_time:.3f}s")
        self.solve_time = prob.solver_stats.solve_time
        self.setup_time = prob.solver_stats.setup_time
        # Return optimized state and control trajectories
        if len(control_inputs) == 0:
            print(
                '\ncannot drive further, final state:\n',
                self.model.convert_vec_to_state(states[-1]).to_string()
            )
        return states, control_inputs
