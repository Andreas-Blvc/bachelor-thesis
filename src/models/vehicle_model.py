import numpy as np
from typing import Tuple, Any, List

class VehicleModel:
	def update(self, current_state, control_inputs) -> Tuple[np.ndarray, List[Any], List[Any]]:
		raise 'update not implemented'
	def get_initial_state(self) -> np.ndarray:
		raise "get_initial_state not implemented"
	def get_goal_state(self) -> np.ndarray:
		raise "get_goal_state not implemented"
	def get_position_orientation(self, state) -> Tuple[np.ndarray, float]:
		raise "get_position_orientation not implemented"
	def get_shape(self, state) -> List[Tuple[float, float]]:
		raise "get_shape not implemented"
	def get_dim_state(self) -> int:
		raise "get_dim_state not implemented"
	def get_dim_control_input(self) -> int:
		raise "get_dim_control_input not implemented"
	def get_a_max(self) -> float:
		raise "get_a_max not implemented"

