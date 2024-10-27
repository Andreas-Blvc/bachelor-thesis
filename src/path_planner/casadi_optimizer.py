import casadi as ca
import numpy as np
from models.vehicle_model import VehicleModel

class NonConvexPathPlanner:
    def __init__(self, model: VehicleModel, dt, time_horizon):
        self.dt = dt
        self.model = model
        N = int(time_horizon / dt)
        self.N = N  # Store N for later use

        # Define CasADi control and state variables as vectors
        self.u = ca.MX.sym('u', N * model.get_dim_control_input())
        self.x = ca.MX.sym('x', (N + 1) * model.get_dim_state())

        initial_state = model.get_initial_state().flatten()
        goal_state = model.get_goal_state().flatten()

        # Define constraint expressions and the objective function
        constraints = []
        objective = 0

        # Reshape x and u to matrices for easier handling
        u = ca.reshape(self.u, (N, model.get_dim_control_input()))
        x = ca.reshape(self.x, (N + 1, model.get_dim_state()))

        # Initial state constraint
        constraints.append(x[0, :].T == ca.reshape(initial_state, (self.model.get_dim_state(), 1)))

        # Goal state constraint
        constraints.append(x[0, :].T == ca.reshape(goal_state, (self.model.get_dim_state(), 1)))

        # Dynamics constraints vectorized over time horizon
        for j in range(N):
            current_state = x[j, :]
            control_inputs = u[j, :]
            next_state, model_constraints = model.update(current_state, control_inputs)
            # constraints += model_constraints
            constraints.append(x[0, :].T == ca.reshape(next_state, (self.model.get_dim_state(),1)))

            # Objective: minimize the sum of squared control inputs over the horizon
            objective += ca.sumsqr(u[j, :])

        # Flatten constraints into a single dense vector for CasADi compatibility
        g = ca.vertcat(*[ca.reshape(constr, -1, 1) for constr in constraints])

        # Define the NLP problem
        nlp = {
            'x': ca.vertcat(self.x, self.u),
            'f': objective,
            'g': g,
        }

        # Create the CasADi NLP solver
        self.solver = ca.nlpsol('solver', 'ipopt', nlp)

    def get_optimized_trajectory(self):
        # Define the number of states and controls
        N = self.N
        x_dim = self.model.get_dim_state()
        u_dim = self.model.get_dim_control_input()

        # Provide initial guess for x and u
        total_vars = (N + 1) * x_dim + N * u_dim
        initial_guess = np.zeros((total_vars,))

        # Solve the optimization problem
        solution = self.solver(
            x0=initial_guess,
            lbx=-ca.inf,
            ubx=ca.inf,
            lbg=0,
            ubg=0
        )

        # Extract the solution, which should be a DM object
        solution_xu = solution['x'].full().flatten()

        # Calculate the sizes for slicing
        x_size = (N + 1) * x_dim

        # Extract and reshape the optimized states and controls
        x_opt = solution_xu[:x_size].reshape((N + 1, x_dim))
        u_opt = solution_xu[x_size:].reshape((N, u_dim))

        return x_opt, u_opt
