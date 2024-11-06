from models.point_mass_model import PointMassModel
from models.single_track_model import SingleTrackModel
from models.road_aligned_model import RoadAlignedModel
from obstacles.road import Road
from path_planner.cvxpy_optimizer import ConvexPathPlanner
from path_planner.casadi_optimizer import NonConvexPathPlanner
from visualizer.vehicle_path_visualizer import VehiclePathVisualizer
from visualizer.control_inputs_plot import plot_control_inputs
import time
import pickle
import os
from math import pi
import numpy as np

class Scenario:
	def __init__(self, dt, model, car_states, control_inputs, obstacles=None):
		if obstacles is None:
			obstacles = []
		start_pos, start_orientation = model.get_position_orientation(model.get_initial_state())
		goal_pos, goal_orientation = model.get_position_orientation(model.get_goal_state())
		start_shape, goal_shape = (model.get_vehicle_polygon(model.get_initial_state()),
								   model.get_vehicle_polygon(model.get_goal_state()))
		self.dt = dt
		self.start_pos = start_pos
		self.start_orientation = start_orientation
		self.start_shape = start_shape
		self.goal_pos = goal_pos
		self.goal_orientation = goal_orientation
		self.goal_shape = goal_shape
		self.car_states = car_states
		self.control_inputs = control_inputs
		self.get_vehicle_polygon = model.get_vehicle_polygon
		self.get_position_orientation= model.get_position_orientation
		self.to_string = model.to_string
		self.control_input_labels = model.get_control_input_labels()
		self.obstacles = obstacles


def scenario_0():
	dt = 0.1
	time_horizon = 2
	model = PointMassModel(
		initial_state=np.reshape([0,0,0,0], (4,)),
		goal_state=np.reshape([4, 2, 0, 0], (4,)),
		a_max=20,
		dt=dt
	)
	planner = ConvexPathPlanner(model, dt, time_horizon)

	car_states, control_inputs = planner.get_optimized_trajectory()

	return Scenario(dt, model, car_states, control_inputs)



def scenario_1():
	dt = 1 / 30
	time_horizon = 30
	model = SingleTrackModel(
		initial_state=np.reshape([-6, -2, 0, 0, 0], (5,)),
		goal_state=np.reshape([6, -2, (0 / 180) * pi, 0, 0], (5,)),
		l_wb=1.8,
		v_s=30,
		steering_velocity_range=(-1, 1),
		steering_angle_range=((-35 / 180) * pi, (35 / 180) * pi),
		velocity_range=(-40, 40),
		acceleration_range=(-200, 200),
		dt=dt,
		solver_type='cvxpy',
	)
	planner = ConvexPathPlanner(model, dt, time_horizon)

	car_states, control_inputs = planner.get_optimized_trajectory()
	# actual_car_states = [model.get_initial_state()]
	# for u in control_inputs:
	# 	actual_car_states.append(model.accurate_update(actual_car_states[-1], u))
	return Scenario(dt, model, car_states, control_inputs)


def scenario_2():
	dt = 0.1
	time_horizon = 100
	model = SingleTrackModel(
		initial_state=np.reshape([-6, -2, 0, 0, 0], (5,)),
		goal_state=np.reshape([-6, -2, 0, 0,  (180 / 180) * pi], (5,)),
		l_wb=1.8,
		v_s=10,
		steering_velocity_range=(-10, 10),
		steering_angle_range=((-30 / 180) * pi, (30 / 180) * pi),
		velocity_range=(-40, 40),
		acceleration_range=(-5, 5),
		dt=dt,
		solver_type='casadi',
	)
	planner = NonConvexPathPlanner(model, dt, time_horizon)

	car_states, control_inputs = planner.get_optimized_trajectory()

	actual_car_states = [model.get_initial_state()]
	for u in control_inputs:
		actual_car_states.append(model.accurate_update(actual_car_states[-1], u))
	return Scenario(dt, model, actual_car_states, control_inputs)


def scenario_3():
	dt = 1 / 30
	time_horizon = 10
	save_file = "data/scenario_3_data.pkl"
	persist_path = False

	road = Road(
		s=[(-8, -8), (1, 1), (5, 5)],
		width=4
	)
	model = RoadAlignedModel(
		initial_state=np.array([0, -1, 0.01, 0]),
		goal_state=np.array([1, .1, 0.01, 0]),
		dt=dt,
		road=road,
		v_x_range=(-5, 40),
		v_y_range=(-1, 1),
		acc_x_range=(-2, 2),
		acc_y_range=(-2, 2),
		yaw_rate_range=(-3, 3),
		yaw_acc_range=(-3, 3),
		a_max=80,
	)

	# Check if data file exists; if it does, load car_states and control_inputs
	if os.path.exists(save_file):
		with open(save_file, "rb") as file:
			data = pickle.load(file)
			car_states = data["car_states"]
			control_inputs = data["control_inputs"]
	else:
		# If the file does not exist, generate car_states and control_inputs
		planner = ConvexPathPlanner(model, dt, time_horizon)
		car_states, control_inputs = planner.get_optimized_trajectory()

		# Save car_states and control_inputs to a file for future use
		if persist_path:
			with open(save_file, "wb") as file:
				pickle.dump({"car_states": car_states, "control_inputs": control_inputs}, file)

	# Visualize:
	# model.visualize_constraints(car_states, control_inputs)
	# road.plot_combined_curvature_and_derivative()
	plot_control_inputs(control_inputs, model.get_control_input_labels(), dt)

	return Scenario(dt, model, car_states, control_inputs, [road])

if __name__ == '__main__':
	visualizer = VehiclePathVisualizer()
	scenario = scenario_3()

	for i, state in enumerate(scenario.car_states):
		pos, orientation = scenario.get_position_orientation(state.T)
		control = scenario.control_inputs[i] if i < len(scenario.control_inputs) else (0, 0)

		# print(
		# 	f"Step {i + 1}: "
		# 	f"Position = ({pos[0]:.2f} m, {pos[1]:.2f} m), "
		# 	+ scenario.to_string(state, control)
		# )

		shape = scenario.get_vehicle_polygon(state)
		visualizer.draw(
			scenario.start_pos,
			scenario.start_orientation,
			scenario.start_shape,
			scenario.goal_pos,
			scenario.goal_orientation,
			scenario.goal_shape,
			pos,
			orientation,
			shape,
			scenario.obstacles
		)
		time.sleep(scenario.dt / 10)  # Add delay to simulate motion
	visualizer.show()