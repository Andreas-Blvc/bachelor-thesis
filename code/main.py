from fontTools.misc.arrayTools import sectRect

from models.point_mass_model import PointMassModel
from models.single_track_model import SingleTrackModel
from models.road_aligned_model import RoadAlignedModel
from models.oriented_road_following import OrientedRoadFollowingModel
from obstacles.road_collection import *
from path_planner.cvxpy_optimizer import ConvexPathPlanner
from path_planner.casadi_optimizer import NonConvexPathPlanner
from path_planner.objectives import Objectives
from visualizer.vehicle_path_visualizer import VehiclePathVisualizer, VehicleObject
from visualizer.control_inputs_plot import plot_control_inputs
import time
import pickle
import os
from math import pi
import numpy as np

class Scenario:
	def __init__(self, dt, model, predicted_car_states, control_inputs, actual_car_states=None, obstacles=None):
		if obstacles is None:
			obstacles = []
		start_pos, start_orientation = model.get_position_orientation(model.get_initial_state())
		start_shape = model.get_vehicle_polygon(model.get_initial_state())
		if model.get_goal_state() is not None:
			goal_pos, goal_orientation = model.get_position_orientation(model.get_goal_state())
			goal_shape = model.get_vehicle_polygon(model.get_goal_state())
		else:
			goal_pos, goal_orientation, goal_shape = None, None, None

		if actual_car_states and len(actual_car_states) != len(predicted_car_states):
			raise ValueError('Number of actual car states does not match number of predicted car states')


		self.dt = dt
		self.start_pos = start_pos
		self.start_orientation = start_orientation
		self.start_shape = start_shape
		self.goal_pos = goal_pos
		self.goal_orientation = goal_orientation
		self.goal_shape = goal_shape
		self.predicted_car_states = predicted_car_states
		self.actual_car_states = actual_car_states
		self.control_inputs = control_inputs
		self.get_vehicle_polygon = model.get_vehicle_polygon
		self.get_position_orientation= model.get_position_orientation
		self.to_string = model.to_string
		self.control_input_labels = model.get_control_input_labels()
		self.obstacles = obstacles
		self.road = model.road if hasattr(model, 'road') else None


def scenario_0():
	dt = 0.1
	time_horizon = 2
	objective = Objectives.minimize_control_input
	model = PointMassModel(
		initial_state=np.reshape([0,0,0,0], (4,)),
		goal_state=np.reshape([4, 2, 0, 0], (4,)),
		a_max=20,
		dt=dt
	)
	planner = ConvexPathPlanner(model, dt, time_horizon, objective)

	car_states, control_inputs = planner.get_optimized_trajectory()

	return Scenario(dt, model, car_states, control_inputs)



def scenario_1():
	dt = 1 / 30
	time_horizon = 30
	objective = Objectives.minimize_control_input
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
	planner = ConvexPathPlanner(model, dt, time_horizon, objective)
	car_states, control_inputs = planner.get_optimized_trajectory()
	# actual_car_states = [model.get_initial_state()]
	# for u in control_inputs:
	# 	actual_car_states.append(model.accurate_update(actual_car_states[-1], u))
	return Scenario(dt, model, car_states, control_inputs)


def scenario_2():
	dt = 0.1
	time_horizon = 10
	objective = Objectives.minimize_control_input
	model = SingleTrackModel(
		initial_state=np.reshape([-6, -2, 0, 0, 0], (5,)),
		goal_state=np.reshape([2, -1, 0, 0,  (0 / 180) * pi], (5,)),
		l_wb=1.8,
		v_s=10,
		steering_velocity_range=(-10, 10),
		steering_angle_range=((-30 / 180) * pi, (30 / 180) * pi),
		velocity_range=(-40, 40),
		acceleration_range=(-5, 5),
		dt=dt,
		solver_type='casadi',
	)
	planner = NonConvexPathPlanner(model, dt, time_horizon, objective)

	car_states, control_inputs = planner.get_optimized_trajectory()

	actual_car_states = [model.get_initial_state()]
	for u in control_inputs:
		actual_car_states.append(model.accurate_update(actual_car_states[-1], u))
	return Scenario(dt, model, actual_car_states, control_inputs)


def scenario_3():
	dt = 1 / 60
	time_horizon = 10
	objective = Objectives.minimize_control_input
	save_file = "data/scenario_3_data.pkl"
	persist_path = False

	model = RoadAlignedModel(
		initial_state=np.array([0, 0, 0.01, 0]),
		goal_state=np.array([1, 0, 0.01, 0]),
		dt=dt,
		road=right_curved_road,
		v_x_range=(-5, 40),
		v_y_range=(-1, 1),
		acc_x_range=(-2, 2),
		acc_y_range=(-2, 2),
		yaw_rate_range=(-1, 1),
		yaw_acc_range=(-0.3, 0.3),
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
		planner = ConvexPathPlanner(model, dt, time_horizon, objective)
		car_states, control_inputs = planner.get_optimized_trajectory()

		# Save car_states and control_inputs to a file for future use
		if persist_path:
			with open(save_file, "wb") as file:
				pickle.dump({"car_states": car_states, "control_inputs": control_inputs}, file)

	# Visualize:
	# model.visualize_constraints(car_states, control_inputs)
	# road.plot_combined_curvature_and_derivative()
	body_fixed_controls = []
	for j in range(len(control_inputs)):
		body_fixed_controls.append(
			(
				model.to_body_fixed(car_states[j], control_inputs[j])[0],
				model.road.get_curvature_at(car_states[j][0]) * car_states[j][2]
			)
		)
	plot_control_inputs(body_fixed_controls, ['a_x', 'yaw_rate'], dt)

	return Scenario(dt, model, car_states, control_inputs)


def scenario_4():
	dt = 1 / 60
	time_horizon = 5
	objective = Objectives.minimize_remaining_distance
	road = right_curved_road
	model = OrientedRoadFollowingModel(
		initial_state=np.array([0, 0, 0, 4, 0]),
		goal_state=None, # np.array([road.length, 0, 0, 2, 0]),
		dt=dt,
		road=road,
		v_range=(0, 4.5),
		acc_range=(-2, 2),
		steering_angle_range=((-30 / 180) * pi, (30 / 180) * pi),
		steering_velocity_range=(-3, 3),
	)

	planner = ConvexPathPlanner(model, dt, time_horizon, objective)
	car_states, control_inputs = planner.get_optimized_trajectory()
	plot_control_inputs(control_inputs, model.get_control_input_labels(), dt)
	plot_control_inputs(car_states, ['s', 'n', 'xi', 'v', 'delta'], dt)
	# plot_control_inputs([(dn.value, dxi.value) for dn, dxi in model.artificial_variables], ['dn_term', 'dxi_term'], dt)

	actual_car_states = [model.get_initial_state()]
	for u in control_inputs:
		next_state, _ = model.update(actual_car_states[-1], u)
		actual_car_states.append(next_state)
	# model.goal_state = car_states[-1]
	# planner = NonConvexPathPlanner(model, dt, time_horizon, objective)
	# car_states, control_inputs = planner.get_optimized_trajectory(initial_guess=(car_states, control_inputs))
	# plot_control_inputs(control_inputs, model.get_control_input_labels(), dt)
	# plot_control_inputs(car_states, ['s', 'n', 'xi', 'v', 'delta'], dt)

	return Scenario(dt, model, actual_car_states, control_inputs, actual_car_states)

if __name__ == '__main__':
	visualizer = VehiclePathVisualizer()
	scenario = scenario_4()

	for i in range(len(scenario.predicted_car_states)):
		predicted_state = scenario.predicted_car_states[i]
		actual_car_state = scenario.actual_car_states[i] if scenario.actual_car_states else None

		predicted_pos, predicted_orientation = scenario.get_position_orientation(predicted_state.T)
		predicted_shape = scenario.get_vehicle_polygon(predicted_state)

		if scenario.actual_car_states:
			actual_pos, actual_orientation = scenario.get_position_orientation(actual_car_state.T)
			actual_shape = scenario.get_vehicle_polygon(actual_car_state)
		else:
			actual_pos, actual_orientation, actual_shape = None, None, None

		control = scenario.control_inputs[i] if i < len(scenario.control_inputs) else (0, 0)

		# print(
		# 	f"Step {i + 1}: "
		# 	f"Position = ({pos[0]:.2f} m, {pos[1]:.2f} m), "
		# 	+ scenario.to_string(state, control)
		# )
		visualizer.draw(
			start=VehicleObject(scenario.start_pos, scenario.start_orientation, scenario.start_shape),
			goal=VehicleObject(scenario.goal_pos, scenario.goal_orientation, scenario.goal_shape) if scenario.goal_pos else None,
			predicted_car=VehicleObject(predicted_pos, predicted_orientation, predicted_shape),
			actual_car=VehicleObject(actual_pos, actual_orientation, actual_shape) if scenario.actual_car_states else None,
			road= scenario.road
		)
		time.sleep(scenario.dt / 10)  # Add delay to simulate motion
	visualizer.show()