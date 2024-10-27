from models.double_integrator_model import DoubleIntegrator
from models.single_track_model import SingleTrackModel
from path_planner.optimization import PathPlanner
from visualizer.motion_visualization import MotionVisualizer
import time
from math import pi

if __name__ == '__main__':
	dt = 1/30
	time_horizon = 3
	# Create Path Planning Instance
	# model = DoubleIntegrator( [0, 0, 0, 0], [7, 5, 0, 0], 10, dt)
	model = SingleTrackModel(
		initial_state=[0, 0, 0, 0, 0],
		goal_state=[3, 8, 0, 0, 0],
		a_max=20,
		l_wb=1.8,
		v_s=30,
		steering_velocity_range=[-5, 5],
		steering_angle_range=[(-15/180)*pi, (15/180)*pi],
		velocity_range=[0, 40],
		acceleration_range=[-20, 20],
		dt=dt
	)
	planner = PathPlanner(model, dt, time_horizon)
	visualizer = MotionVisualizer()

	# Get Start and Goal, Car Shape
	start_pos, start_orientation = model.get_position_orientation(model.get_initial_state())
	goal_pos, goal_orientation = model.get_position_orientation(model.get_goal_state())
	car_shape = model.get_shape()

	# Get Car Path
	car_states, control_inputs = planner.get_optimized_trajectory()
	car_path = []

	print(model.get_initial_state())
	for i, state in enumerate(car_states):
		pos, orientation = model.get_position_orientation(state)
		car_path.append(pos)
		control_input = control_inputs[i] if i < len(control_inputs) else [0, 0]
		print(f"Step {i + 1}: Position = ({pos[0]:.2f}, {pos[1]:.2f}), Control Inputs = [{control_input[0]:.5f}, {control_input[1]:.5f}]")
		visualizer.draw(
			start_pos,
			start_orientation,
			goal_pos,
			goal_orientation,
			pos,
			orientation,
			car_shape,
			car_path,
			[]
		)
		time.sleep(dt)  # Add delay to simulate motion
	input('END')