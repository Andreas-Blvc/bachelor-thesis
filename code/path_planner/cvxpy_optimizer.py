import cvxpy as cp

from models import AbstractVehicleModel
from roads import AbstractRoad

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
        additional_min_objective_term = 0

        # Goal state constraint (only position constraints at the final state)
        if goal_state is not None:
            constraints.append(self.x[state_transitions, :] == goal_state)
        elif (
                self.model.road_segment_idx is not None and
                self.model.road_segment_idx < len(getattr(self.model.road, 'segments', [self.model.road])) - 1
        ):
            next_segment: AbstractRoad = getattr(self.model.road, 'segments')[self.model.road_segment_idx + 1]
            final_state = self.model.convert_vec_to_state(self.x[state_transitions, :])
            constraints.append(final_state.get_lateral_offset() <= next_segment.width(0)/2)
            constraints.append(final_state.get_lateral_offset() >= -next_segment.width(0)/2)
            additional_min_objective_term += 1e3 * final_state.get_alignment_error() ** 2 + 5 * final_state.get_lateral_offset() ** 2

        # Dynamics constraints vectorized over time horizon
        for j in range(state_transitions):
            # State transition: x_{k+1} = x_k + (A * x_k + B * u_k) * dt
            next_state, model_constraints = self.model.update(
                current_state=self.x[j, :].T,
                control_inputs=self.u[j, :].T,
            )
            constraints += model_constraints
            constraints.append(self.x[j + 1, :].T == next_state)

        # constraint goal_state:
        _, goal_state_constraints = self.model.update(
            current_state=self.x[state_transitions, :].T,
            control_inputs=cp.Variable((self.model.dim_control_input, 1))  # dummy control input
        )
        constraints += goal_state_constraints

        # Define the optimization problem
        states = [self.model.convert_vec_to_state(self.x[j, :]) for j in range(state_transitions + 1)]
        control_inputs = [self.model.convert_vec_to_control_input(self.u[j, :]) for j in range(state_transitions)]
        objective, objective_type = self.get_objective(states, control_inputs)
        if objective_type == Objectives.Type.MINIMIZE:
            self.prob = cp.Problem(cp.Minimize(objective + additional_min_objective_term), constraints)
        else:
            self.prob = cp.Problem(cp.Maximize(objective - additional_min_objective_term), constraints)

    def get_optimized_trajectory(self):
        max_state_transitions = int(self.time_horizon / self.dt)
        initial_state = self.model.initial_state
        prev_segments_length = 0
        solve_time = 0
        states = [initial_state]
        control_inputs = []
        failed = False
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
                    # print(f'Planning {state_transitions} transitions on {i}')
                    try:
                        self.construct_problem(state_transitions, initial_state)
                        self.prob.solve(solver='MOSEK', verbose=self.verbose)
                        solve_time += self.prob.solver_stats.solve_time
                    except ValueError as e:
                        print('value error', e)

                    if self.x.value is None:
                        print(
                            'cannot drive further, final state:\n',
                            self.model.convert_vec_to_state(states[-1]).to_string()
                        )
                        failed = True
                        break
                    states += [state for state in self.x.value][1:]
                    control_inputs += [control_input for control_input in self.u.value]
                    initial_state = states[-1]
            if failed:
                break
            prev_segments_length += segment.length


        # self.prob.solve(solver='CLARABEL', verbose=self.verbose)
        # print("Solve time:", f"{self.prob.solver_stats.solve_time:.3f}s")
        self.solve_time = solve_time
        # Return optimized state and control trajectories
        return states, control_inputs

