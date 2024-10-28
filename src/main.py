from models.point_mass_model import PointMassModel
from models.single_track_model import SingleTrackModel
from path_planner.cvxpy_optimizer import ConvexPathPlanner
from path_planner.casadi_optimizer import NonConvexPathPlanner
from visualizer.matplot_visualization import MatPlotVisualizer
import time
from math import pi
import numpy as np

class Scenario:
	def __init__(self, dt, model, car_states, control_inputs):
		start_pos, start_orientation = model.get_position_orientation(model.get_initial_state())
		goal_pos, goal_orientation = model.get_position_orientation(model.get_goal_state())
		start_shape, goal_shape = (model.get_vehicle_polygon(model.get_initial_state()),
								   model.get_vehicle_polygon(model.get_goal_state()),)
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


def scenario_0():
	dt = 0.1
	time_horizon = 20
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

	actual_car_states = [model.get_initial_state()]
	for u in control_inputs:
		print(u)
		actual_car_states.append(model.accurate_update(actual_car_states[-1], u))
	print(actual_car_states)
	return Scenario(dt, model, actual_car_states, control_inputs)


def scenario_2():
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
		solver_type='casadi',
	)
	planner = NonConvexPathPlanner(model, dt, time_horizon)

	car_states, control_inputs = planner.get_optimized_trajectory()

	actual_car_states = [model.get_initial_state()]
	for u in control_inputs:
		print(u)
		actual_car_states.append(model.accurate_update(actual_car_states[-1], u))
	print(actual_car_states)
	return Scenario(dt, model, car_states, control_inputs)


if __name__ == '__main__':
	visualizer = MatPlotVisualizer()
	scenario = scenario_2()

	for i, state in enumerate(scenario.car_states):
		pos, orientation = scenario.get_position_orientation(state.T)
		shape = scenario.get_vehicle_polygon(state)
		control_input = scenario.control_inputs[i] if i < len(scenario.control_inputs) else [0, 0]
		print(f"Step {i + 1}: Position, Orientation = ({pos[0]:.2f}, {pos[1]:.2f}), {(orientation*180/pi):.2f}), Control Inputs = [{control_input[0]:.5f}, {control_input[1]:.5f}]")
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
			[]
		)
		time.sleep(scenario.dt)  # Add delay to simulate motion
	input('END')