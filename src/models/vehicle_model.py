class VehicleModel:
	def update(self, current_state, acceleration_inputs):
		raise 'update not implemented'
	def get_initial_state(self):
		raise "get_initial_state not implemented"
	def get_goal_state(self):
		raise "get_goal_state not implemented"
	def get_position_orientation(self, state):
		raise "get_position_orientation not implemented"
	def get_shape(self):
		raise "get_shape not implemented"
	def get_dim_state(self):
		raise "get_dim_state not implemented"
	def get_dim_control_input(self):
		raise "get_dim_control_input not implemented"
