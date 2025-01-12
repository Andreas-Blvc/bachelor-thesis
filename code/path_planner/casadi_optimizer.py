import casadi as ca
import numpy as np
import sys

from models import AbstractVehicleModel

from .objectives import Objectives
from .interface import AbstractPathPlanner

class NonConvexPathPlanner(AbstractPathPlanner):
    def __init__(self, model: AbstractVehicleModel, dt, time_horizon, get_objective):
        super().__init__(
            model,
            dt,
            time_horizon,
            get_objective
        )

        # Configure others
        model.solver_type = 'casadi'
        Objectives.norm = ca.sumsqr
        Objectives.max = lambda x, y: ca.fmax(x, y)

        self.opti = None
        self.solve_time = 0.0
        self.x = None
        self.u = None

    def _construct_problem(self, state_transitions, initial_state, goal_state=None):
        # Initialize the Opti object
        self.opti = ca.Opti()

        x_dim = self.model.dim_state
        u_dim = self.model.dim_control_input

        # Define CasADi Opti variables for states and controls
        # States: (N+1) x x_dim
        self.x = self.opti.variable(state_transitions + 1, x_dim)
        # Controls: N x u_dim
        self.u = self.opti.variable(state_transitions, u_dim)

        # Initial state constraint (equality)
        initial_state = np.array(initial_state).reshape((1, x_dim))
        self.opti.subject_to(self.x[0, :] == initial_state)

        # Goal state constraint (equality) at a final time step
        if goal_state is not None:
            goal_state = np.array(goal_state).reshape((1, x_dim))
            self.opti.subject_to(self.x[state_transitions, :] == goal_state)

        for j in range(state_transitions):
            # Current state and control input
            current_state = self.x[j, :].T
            control_inputs = self.u[j, :].T

            # Update the model to get the next state and any additional constraints
            next_state, constraints = self.model.forward_euler_step(current_state, control_inputs, self.dt)

            # Add model-specific equality constraints
            if constraints:
                # Assuming constraints is a list of CasADi expressions
                for constr in constraints:
                    self.opti.subject_to(constr)

            # Dynamics constraint: x[j + 1, :] == next_state
            self.opti.subject_to(self.x[j + 1, :] == next_state.T)

        # Set the objective in the Opti problem
        states = [self.model.convert_vec_to_state(self.x[j, :].T) for j in range(state_transitions + 1)]
        control_inputs = [self.model.convert_vec_to_control_input(self.u[j, :].T) for j in range(state_transitions)]
        objective, objective_type = self.get_objective(states, control_inputs)
        if objective_type == Objectives.Type.MINIMIZE:
            self.opti.minimize(objective)
        else:
            self.opti.minimize(-objective)

        # Optionally, you can set solver options here
        p_opts = {"expand": True}
        s_opts = {"max_iter": 5000, "print_level": 0}
        self.opti.solver('ipopt', p_opts, s_opts)

    def get_optimized_trajectory(self, initial_state, initial_guess=None):
        self._construct_problem(int(self.time_horizon/self.dt), initial_state)
        # Optionally set initial guesses for the solver
        if initial_guess is not None:
            x_guess, u_guess = initial_guess
            if x_guess is not None:
                self.opti.set_initial(self.x, x_guess)
            if u_guess is not None:
                self.opti.set_initial(self.u, u_guess)

        with open('solver_output.txt', 'w') as output_file:
            stdout_old = sys.stdout
            sys.stdout = output_file

            try:
                solution = self.opti.solve()
                self.solve_time = solution.stats()['t_proc_total']
                sys.stdout = stdout_old
            except RuntimeError:
                # traceback.print_exc(file=stdout_old)
                sys.stdout = stdout_old
                return [], []

        # Extract the optimized states and controls
        x_opt = solution.value(self.x)
        u_opt = solution.value(self.u)

        if x_opt is None or u_opt is None:
            return [], []
        else:
            return x_opt, u_opt
