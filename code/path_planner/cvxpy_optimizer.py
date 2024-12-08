import cvxpy as cp
from debugpy.common.timestamp import current

from models import AbstractVehicleModel

from .objectives import Objectives


class ConvexPathPlanner:
    def __init__(self, model: AbstractVehicleModel, dt, time_horizon, get_objective, verbose=False):
        # Configure others:
        model.solver_type = 'cvxpy'
        Objectives.norm = cp.sum_squares

        self.u = None
        self.x = None
        self.solve_time = None
        self.prob = None
        self.verbose = verbose
        self.dt = dt
        self.time_horizon = time_horizon
        self.model = model
        self.get_objective = get_objective


    def construct_problem(self, state_transitions, initial_state, goal_state=None):
        # Define control and state variables for the entire time horizon
        self.u = cp.Variable((state_transitions, self.model.dim_control_input))  # Control inputs matrix: (N, 2)
        self.x = cp.Variable((state_transitions + 1, self.model.dim_state))  # State variables matrix: (N + 1, 4)

        constraints = [self.x[0, :] == initial_state]

        # Goal state constraint (only position constraints at the final state)
        if goal_state is not None:
            constraints.append(self.x[state_transitions, :] == goal_state)

        # Dynamics constraints vectorized over time horizon
        for j in range(state_transitions):
            # State transition: x_{k+1} = x_k + (A * x_k + B * u_k) * dt
            next_state, model_constraints = self.model.update(
                current_state=self.x[j, :].T,
                control_inputs=self.u[j, :].T,
            )
            constraints += model_constraints
            constraints.append(self.x[j + 1, :].T == next_state)

        # Define the optimization problem
        states = [self.model.convert_vec_to_state(self.x[j, :]) for j in range(state_transitions + 1)]
        control_inputs = [self.model.convert_vec_to_control_input(self.u[j, :]) for j in range(state_transitions)]
        objective, objective_type = self.get_objective(states, control_inputs)
        if objective_type == Objectives.Type.MINIMIZE:
            self.prob = cp.Problem(cp.Minimize(objective), constraints)
        else:
            self.prob = cp.Problem(cp.Maximize(objective), constraints)

    def get_optimized_trajectory(self):
        max_state_transitions = int(self.time_horizon / self.dt)
        initial_state = self.model.initial_state
        prev_segments_length = 0
        solve_time = 0
        states = [initial_state]
        control_inputs = []
        for i, segment in enumerate(getattr(self.model.road, 'segments', [self.model.road])):
            self.model.road_segment_idx = i
            while True:
                traveled_distance = self.model.convert_vec_to_state(initial_state).get_traveled_distance()
                distance_til_next_segment = prev_segments_length + segment.length - traveled_distance
                min_time_to_reach_next_segment = distance_til_next_segment / self.model.get_v_max()
                state_transitions = min(max_state_transitions - len(control_inputs), int(min_time_to_reach_next_segment / self.dt))
                if state_transitions == 0:
                    break
                else:
                    self.construct_problem(state_transitions, initial_state)
                    self.prob.solve(solver='CLARABEL', verbose=self.verbose)
                    solve_time += self.prob.solver_stats.solve_time
                    states += [state for state in self.x.value][1:]
                    control_inputs += [control_input for control_input in self.u.value]
                    initial_state = states[-1]
            prev_segments_length += segment.length


        # self.prob.solve(solver='CLARABEL', verbose=self.verbose)
        # print("Solve time:", f"{self.prob.solver_stats.solve_time:.3f}s")
        self.solve_time = solve_time
        # Return optimized state and control trajectories
        return states, control_inputs

