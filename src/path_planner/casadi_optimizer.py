import casadi as ca
import numpy as np
from models.vehicle_model import VehicleModel

class NonConvexPathPlanner:
    def __init__(self, model: VehicleModel, dt, time_horizon):
        model.solver_type = 'casadi'
        # Initialize the Opti object
        self.opti = ca.Opti()

        self.dt = dt
        self.model = model
        self.N = int(time_horizon / dt)
        self.a_max = model.get_a_max()  # Maximum allowable acceleration

        x_dim = model.get_dim_state()
        u_dim = model.get_dim_control_input()

        # Define CasADi Opti variables for states and controls
        # States: (N+1) x x_dim
        self.x = self.opti.variable(self.N + 1, x_dim)
        # Controls: N x u_dim
        self.u = self.opti.variable(self.N, u_dim)

        # Retrieve initial and goal states
        initial_state = np.array(model.get_initial_state()).reshape((1, x_dim))
        goal_state = np.array(model.get_goal_state()).reshape((1, x_dim))

        # Initial state constraint (equality)
        self.opti.subject_to(self.x[0, :] == initial_state)

        # Goal state constraint (equality) at final time step
        self.opti.subject_to(self.x[self.N, :] == goal_state)

        # Initialize the objective: prioritize reaching the goal quickly and staying there
        self.objective = 0

        # Define weights: higher weights for earlier time steps
        # This encourages minimizing the distance to the goal early on
        weights = np.linspace(1.0, 0.1, self.N)  # Adjust the range as needed

        for j in range(self.N):
            # Current state and control input
            current_state = self.x[j, :].T
            control_inputs = self.u[j, :].T

            # Update the model to get next state and any additional constraints
            next_state, constraints = model.update(current_state, control_inputs)

            # Add model-specific equality constraints
            if constraints:
                # Assuming constraints is a list of CasADi expressions
                for constr in constraints:
                    self.opti.subject_to(constr)

            # Dynamics constraint: x[j + 1, :] == next_state
            self.opti.subject_to(self.x[j + 1, :] == next_state.T)

            # Weighted penalty for being away from the goal
            distance_to_goal = ca.sumsqr(self.x[j, :] - goal_state)
            self.objective += weights[j] * distance_to_goal

            # Accumulate the control effort
            self.objective += ca.sumsqr(self.u[j, :])

        # Optionally, add a small penalty on the final state deviation to ensure precision
        final_distance = ca.sumsqr(self.x[self.N, :] - goal_state)
        self.objective += 1e3 * final_distance  # Large weight to enforce final state

        # Set the objective in the Opti problem
        self.opti.minimize(self.objective)

        # Optionally, you can set solver options here
        p_opts = {"expand": True}
        s_opts = {"max_iter": 20000}
        self.opti.solver('ipopt', p_opts, s_opts)

    def get_optimized_trajectory(self, initial_guess=None):
        # Optionally set initial guesses for the solver
        if initial_guess is not None:
            x_guess, u_guess = initial_guess
            if x_guess is not None:
                self.opti.set_initial(self.x, x_guess)
            if u_guess is not None:
                self.opti.set_initial(self.u, u_guess)

        try:
            # Solve the optimization problem
            solution = self.opti.solve()
        except RuntimeError as e:
            # Handle solver errors (e.g., infeasibility)
            print("Solver failed:", e)
            return None, None

        # Extract the optimized states and controls
        x_opt = solution.value(self.x)
        u_opt = solution.value(self.u)

        return x_opt, u_opt
