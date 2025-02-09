from abc import abstractmethod
from typing import Any, List, Tuple, NoReturn
import numpy as np
import casadi as ca
import cvxpy as cp
from jedi.inference.gradual.typing import Callable

from utils import State, ControlInput
from roads import AbstractRoad


class AbstractVehicleModel:
	road: AbstractRoad = None
	road_segment_idx: int = None
	solver_type = None

	def __init__(self, dim_state, dim_control_input, state_labels, control_input_labels):
		self.dim_state = dim_state
		self.dim_control_input = dim_control_input
		self.state_labels = state_labels
		self.control_input_labels = control_input_labels

	@abstractmethod
	def get_name(self):
		raise 'get_name not implemented'
	@abstractmethod
	def forward_euler_step(self, current_state, control_inputs, dt: float, convexify_ref_state=None, amount_prev_planning_states=None) -> Tuple[np.ndarray, List[Any], cp.Expression | ca.MX | int]:
		raise 'update not implemented'
	@abstractmethod
	def convert_vec_to_state(self, vec, road_segment_idx=None) -> State:
		raise "convert_vec_to_state not implemented"
	@abstractmethod
	def plot_additional_information(self, states, controls):
		raise "plot_additional_information not implemented"
	@abstractmethod
	def get_state_vec_from_dsm(self, vec) -> np.ndarray:
		raise "get_state_vec_from_dsm not implemented"
	class CarState:
		def __init__(self, steering_angle, orientation):
			self.steering_angle = steering_angle
			self.orientation = orientation
	@abstractmethod
	def get_dsm_control_from_vec(self, control_vec, state_vec, dynamics, dt=None, remaining_predictive_model_states:List[np.ndarray]=None, car_cur_state: CarState=None) -> np.ndarray:
		raise "get_dsm_control_from_vec not implemented"

	def convert_vec_to_control_input(self, vec) -> ControlInput:
		self._validate__control_dimension(vec)
		return ControlInput(vec, to_string=lambda: self._control_vec_to_string(vec))

	def _validate__state_dimension(self, state):
		if state.shape != (self.dim_state,) and state.shape != (self.dim_state, 1):
			raise ValueError(
				f"current_state must have shape ({self.dim_state},) or ({self.dim_state}, 1), got {state.shape}")

	def _validate__control_dimension(self, control):
		if control.shape != (self.dim_control_input,) and control.shape != (self.dim_control_input, 1):
			raise ValueError(
				f"control_inputs must have shape ({self.dim_control_input},)  "
				f"or ({self.dim_control_input}, 1), got {control.shape}")


	def _state_vec_to_string(self, state_vec):
		self._validate__state_dimension(state_vec)
		if self.state_labels is None:
			raise ValueError("state_labels not defined")
		return "State: " + ", ".join([f"({self.state_labels[i]}: {v:.2f})" for i, v in enumerate(state_vec)])

	def _control_vec_to_string(self, control_vec):
		self._validate__control_dimension(control_vec)
		if self.control_input_labels is None:
			raise ValueError("control_input_labels not defined")
		return ("Control Input: " +
				", ".join([f"({self.control_input_labels[i]}: {v:.2f})" for i, v in enumerate(control_vec)]))

	def _raise_unsupported_solver(self) -> NoReturn:
		raise ValueError(f"solver_type {self.solver_type} not supported")

	def _norm_squared(self, vec):
		if np.isscalar(vec[0]):
			return vec.dot(vec)
		sv_type = self.solver_type
		err =self._raise_unsupported_solver
		return ca.sumsqr(vec) if sv_type == 'casadi' else cp.sum_squares(vec) if sv_type == 'cvxpy' else err()

	def _absolute(self, val):
		if np.isscalar(val):
			return abs(val)
		sv_type = self.solver_type
		err =self._raise_unsupported_solver
		return ca.fabs(val) if sv_type == 'casadi' else cp.abs(val) if sv_type == 'cvxpy' else err()

	def _sqrt(self, val):
		if np.isscalar(val):
			return np.sqrt(val)
		sv_type = self.solver_type
		err =self._raise_unsupported_solver
		return ca.sqrt(val) if sv_type == 'casadi' else cp.sqrt(val) if sv_type == 'cvxpy' else err()