import casadi as ca
import numpy as np
from models.vehicle_model import VehicleModel

class NonConvexPathPlanner:
    def __init__(self, model: VehicleModel, dt, time_horizon, amax):
        self.dt = dt
        self.model = model
        self.N = int(time_horizon / dt)
        self.amax = amax  # Maximum allowable acceleration

        x_dim = model.get_dim_state()
        u_dim = model.get_dim_control_input()

        # Define CasADi control and state variables as vectors
        self.u = ca.MX.sym('u', self.N * u_dim)
        self.x = ca.MX.sym('x', (self.N + 1) * x_dim)

        # Retrieve initial and goal states and ensure they are CasADi DM with proper shape
        initial_state = ca.DM(model.get_initial_state()).T  # Shape (1, x_dim)
        goal_state = ca.DM(model.get_goal_state()).T       # Shape (1, x_dim)

        # Define separate lists for equality and inequality constraints
        self.constraints_eq = []
        self.constraints_ineq = []
        self.objective = 0

        # Reshape x and u to matrices for easier handling
        u = ca.reshape(self.u, (self.N, u_dim))
        x = ca.reshape(self.x, (self.N + 1, x_dim))

        # Initial state constraint (equality)
        self.constraints_eq.append(x[0, :] - initial_state)

        # Goal state constraint (equality)
        self.constraints_eq.append(x[self.N, :] - goal_state)

        # Dynamics constraints vectorized over time horizon
        for j in range(self.N):
            current_state = x[j, :]
            control_inputs = u[j, :]
            next_state, model_constraints = model.update(current_state, control_inputs)

            # **Transpose next_state to match the shape of x[j + 1, :]**
            # next_state is already in the correct shape as per the updated model

            # Include model constraints (assumed to be equalities)
            self.constraints_eq.extend(model_constraints)

            # Dynamics constraint: x[j + 1, :] - next_state == 0
            self.constraints_eq.append(x[j + 1, :] - next_state)

            # Objective: minimize the sum of squared control inputs over the horizon
            self.objective += ca.sumsqr(u[j, :])

            # Extract necessary variables for the new constraint
            # Assuming state indices:
            # x[j, 3] = v (velocity)
            # x[j, 4] = theta (orientation)
            v = x[j, 3]
            theta_dot = (v / model.l_wb) * ca.tan(current_state[2])  # Assuming update has corrected theta_dot

            # Control input u2 is the second control input
            u2 = u[j, 1]

            # Define the new inequality constraint: u2^2 + (v * theta_dot)^2 <= amax^2
            constraint_expr = u2**2 + (v * theta_dot)**2 - self.amax**2
            self.constraints_ineq.append(constraint_expr)

        # Flatten equality constraints into a single dense vector for CasADi
        g_eq = ca.vertcat(*[ca.reshape(constr, -1, 1) for constr in self.constraints_eq])

        # Flatten inequality constraints into a single dense vector for CasADi
        g_ineq = ca.vertcat(*[ca.reshape(constr, -1, 1) for constr in self.constraints_ineq])

        # Concatenate all constraints
        g = ca.vertcat(g_eq, g_ineq)

        # Define lower and upper bounds for constraints
        # Equality constraints: g_eq = 0
        # Inequality constraints: g_ineq <= 0
        lbg_eq = [0] * g_eq.size1()
        ubg_eq = [0] * g_eq.size1()

        lbg_ineq = [-ca.inf] * g_ineq.size1()
        ubg_ineq = [0] * g_ineq.size1()

        lbg = lbg_eq + lbg_ineq
        ubg = ubg_eq + ubg_ineq

        # Define the NLP problem
        nlp = {
            'x': ca.vertcat(self.x, self.u),
            'f': self.objective,
            'g': g,
        }

        # Create the CasADi NLP solver
        self.solver = ca.nlpsol('solver', 'ipopt', nlp)

        # Store bounds for later use in the solver
        self.lbg = lbg
        self.ubg = ubg

    def get_optimized_trajectory(self, initial_guess=None):
        # Define the number of states and controls
        N = self.N
        x_dim = self.model.get_dim_state()
        u_dim = self.model.get_dim_control_input()

        # Retrieve initial and goal states
        initial_state = self.model.get_initial_state().flatten()
        goal_state = self.model.get_goal_state().flatten()

        if initial_guess is None:
            # Create an initial guess for x by interpolating between initial_state and goal_state
            x_init = np.zeros((N + 1, x_dim))
            for i in range(x_dim):
                x_init[:, i] = np.linspace(initial_state[i], goal_state[i], N + 1)

            # Provide an initial guess for u (e.g., zeros or nominal control inputs)
            u_init = np.zeros((N, u_dim))
        else:
            # Use the provided initial guess
            x_init = initial_guess[: (N + 1) * x_dim].reshape((N + 1, x_dim))
            u_init = initial_guess[(N + 1) * x_dim:].reshape((N, u_dim))

        # Flatten x_init and u_init to create initial_guess
        initial_guess = np.concatenate([x_init.flatten(), u_init.flatten()])

        # Solve the optimization problem
        solution = self.solver(
            x0=initial_guess,
            lbx=-ca.inf,
            ubx=ca.inf,
            lbg=self.lbg,
            ubg=self.ubg
        )

        # Extract the solution, which should be a DM object
        solution_xu = solution['x'].full().flatten()

        # Calculate the sizes for slicing
        x_size = (N + 1) * x_dim

        # Extract and reshape the optimized states and controls
        x_opt = solution_xu[:x_size].reshape((N + 1, x_dim))
        u_opt = solution_xu[x_size:].reshape((N, u_dim))

        return x_opt, u_opt
