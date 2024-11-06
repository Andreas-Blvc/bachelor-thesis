import cvxpy as cp

from models.vehicle_model import VehicleModel


class ConvexPathPlanner:
    def __init__(self, model: VehicleModel, dt, time_horizon, verbose=False):
        model.solver_type = 'cvxpy'
        self.verbose = verbose
        self.dt = dt
        self.model = model
        N = int(time_horizon / dt)

        # Define control and state variables for the entire time horizon
        self.u = cp.Variable((N, model.get_dim_control_input()))  # Control inputs matrix: (N, 2)
        self.x = cp.Variable((N + 1, model.get_dim_state()))  # State variables matrix: (N + 1, 4)

        # Initialize constraints list
        constraints = []

        initial_state = model.get_initial_state()
        goal_state = model.get_goal_state()

        # Initial state constraint
        constraints.append(self.x[0, :] == initial_state)

        # Goal state constraint (only position constraints at the final state)
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

        # Objective function: minimize the sum of squared control inputs over the horizon
        objective = cp.Minimize(cp.sum_squares(self.u))

        # Define the optimization problem
        self.prob = cp.Problem(objective, constraints)

    def get_optimized_trajectory(self):
        self.prob.solve(solver='CLARABEL', verbose=self.verbose)
        print("Solve time:", f"{self.prob.solver_stats.solve_time:.3f}s")
        # Return optimized state and control trajectories
        return self.x.value, self.u.value

