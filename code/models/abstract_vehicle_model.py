import numpy as np
from typing import Tuple, Any, List
from abc import abstractmethod

from utils.state_space import State


class AbstractVehicleModel:
	road = None
	@abstractmethod
	def update(self, current_state, control_inputs) -> Tuple[np.ndarray, List[Any]]:
		raise 'update not implemented'
	@abstractmethod
	def get_initial_state(self) -> np.ndarray:
		raise "get_initial_state not implemented"
	@abstractmethod
	def get_goal_state(self) -> np.ndarray:
		raise "get_goal_state not implemented"
	@abstractmethod
	def get_position_orientation(self, state) -> Tuple[np.ndarray, float]:
		raise "get_position_orientation not implemented"
	@abstractmethod
	def get_vehicle_polygon(self, state) -> List[Tuple[float, float]]:
		raise "get_vehicle_polygon not implemented"
	@abstractmethod
	def get_dim_state(self) -> int:
		raise "get_dim_state not implemented"
	@abstractmethod
	def get_dim_control_input(self) -> int:
		raise "get_dim_control_input not implemented"
	@abstractmethod
	def get_a_max(self) -> float:
		raise "get_a_max not implemented"
	@abstractmethod
	def to_string(self, state, control):
		raise "print_state_control not implemented"
	@abstractmethod
	def get_control_input_labels(self) -> List[str]:
		raise "get_control_input_labels not implemented"
	def configure_state_class(self):
		State.get_distance_between = self.get_distance_between
		State.get_traveled_distance = self.get_traveled_distance
		State.get_offset_from_reference_path = self.get_offset_from_reference_path
		State.get_velocity = self.get_velocity
		State.get_remaining_distance = self.get_remaining_distance
	@abstractmethod
	def get_distance_between(self, state_a: State, state_b: State):
		raise "get_distance_between not implemented"
	@abstractmethod
	def get_traveled_distance(self, state: State):
		raise "get_traveled_distance not implemented"
	@abstractmethod
	def get_remaining_distance(self, state: State):
		raise "get_remaining_distance not implemented"
	@abstractmethod
	def get_offset_from_reference_path(self, state: State):
		raise "get_offset_from_reference_path not implemented"
	@abstractmethod
	def get_velocity(self, state: State):
		raise "get_velocity not implemented"

