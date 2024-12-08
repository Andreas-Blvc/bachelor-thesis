import casadi as ca
import numpy as np

from models import AbstractVehicleModel

from .objectives import Objectives


class NonConvexPathPlanner:
    def __init__(self, model: AbstractVehicleModel, dt, time_horizon, get_objective):
        # Configure others
        model.solver_type = 'casadi'
        Objectives.norm = ca.sumsqr

        # Initialize the Opti object
        self.opti = ca.Opti()

        self.solve_time = None
        self.dt = dt
        self.model = model
        self.N = int(time_horizon / dt)

        x_dim = model.dim_state
        u_dim = model.dim_control_input

        # Define CasADi Opti variables for states and controls
        # States: (N+1) x x_dim
        self.x = self.opti.variable(self.N + 1, x_dim)
        # Controls: N x u_dim
        self.u = self.opti.variable(self.N, u_dim)

        # Initial state constraint (equality)
        initial_state = np.array(model.initial_state).reshape((1, x_dim))
        self.opti.subject_to(self.x[0, :] == initial_state)

        # Goal state constraint (equality) at final time step
        if model.goal_state is not None:
            goal_state = np.array(model.goal_state).reshape((1, x_dim))
            self.opti.subject_to(self.x[self.N, :] == goal_state)

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


        # Set the objective in the Opti problem
        states = [model.convert_vec_to_state(self.x[j, :].T) for j in range(self.N + 1)]
        control_inputs = [model.convert_vec_to_control_input(self.u[j, :].T) for j in range(self.N)]
        objective, objective_type = get_objective(states, control_inputs)
        if objective_type == Objectives.Type.MINIMIZE:
            self.opti.minimize(objective)
        else:
            self.opti.minimize(-objective)

        # Optionally, you can set solver options here
        p_opts = {"expand": True}
        s_opts = {"max_iter": 500, "print_level": 0}
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

            # Capture the solve time from the solver stats
            self.solve_time = solution.stats()['t_mainloop']
        except RuntimeError as e:
            # Handle solver errors (e.g., infeasibility)
            print("Solver failed:", e)
            return None, None

        # Extract the optimized states and controls
        x_opt = solution.value(self.x)
        u_opt = solution.value(self.u)

        return x_opt, u_opt
