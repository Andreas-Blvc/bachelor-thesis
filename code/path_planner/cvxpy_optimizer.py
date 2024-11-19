import cvxpy as cp

from models import AbstractVehicleModel

from .objectives import Objectives


class ConvexPathPlanner:
    def __init__(self, model: AbstractVehicleModel, dt, time_horizon, get_objective, verbose=False):
        # Configure others:
        model.solver_type = 'cvxpy'
        Objectives.norm = cp.sum_squares

        self.verbose = verbose
        self.dt = dt
        self.model = model
        N = int(time_horizon / dt)

        # Define control and state variables for the entire time horizon
        self.u = cp.Variable((N, model.dim_control_input))  # Control inputs matrix: (N, 2)
        self.x = cp.Variable((N + 1, model.dim_state))  # State variables matrix: (N + 1, 4)

        # Initialize constraints list
        constraints = []

        initial_state = model.initial_state
        goal_state = model.goal_state

        # Initial state constraint
        constraints.append(self.x[0, :] == initial_state)

        # Goal state constraint (only position constraints at the final state)
        if goal_state is not None:
            constraints.append(self.x[N, :] == goal_state)

        # Dynamics constraints vectorized over time horizon
        for j in range(N):
            # State transition: x_{k+1} = x_k + (A * x_k + B * u_k) * dt
            next_state, model_constraints = model.update(
                current_state=self.x[j, :].T,
                control_inputs=self.u[j, :].T,
            )
            constraints += model_constraints
            constraints.append(self.x[j + 1, :].T == next_state)

        # Define the optimization problem
        states = [model.convert_vec_to_state(self.x[j, :]) for j in range(N + 1)]
        control_inputs = [model.convert_vec_to_control_input(self.u[j, :]) for j in range(N)]
        objective, objective_type = get_objective(states, control_inputs)
        if objective_type == Objectives.Type.MINIMIZE:
            self.prob = cp.Problem(cp.Minimize(objective), constraints)
        else:
            self.prob = cp.Problem(cp.Maximize(objective), constraints)

    def get_optimized_trajectory(self):
        self.prob.solve(solver='CLARABEL', verbose=self.verbose)
        print("Solve time:", f"{self.prob.solver_stats.solve_time:.3f}s")
        # Return optimized state and control trajectories
        return self.x.value, self.u.value

