from models.point_mass_model import PointMassModel
from models.single_track_model import SingleTrackModel
from path_planner.cvxpy_optimizer import ConvexPathPlanner
from path_planner.casadi_optimizer import NonConvexPathPlanner
from visualizer.matplot_visualization import MatPlotVisualizer
import time
from math import pi
import numpy as np



if __name__ == '__main__':
	dt = 1/30
	time_horizon = 100
	visualizer = MatPlotVisualizer()
	# Create Path Planning Instance
	model = SingleTrackModel(
		initial_state=np.reshape([0, 0, 0, -1, 0], (5,)),
		goal_state=np.reshape([-4, -6, (60/180)*pi, 0, 0], (5,)),
		a_max=20,
		l_wb=1.8,
		v_s=30,
		steering_velocity_range=(-1, 1),
		steering_angle_range=((-35/180)*pi, (35/180)*pi),
		velocity_range=(-40, 40),
		acceleration_range=(-20, 20),
		dt=dt,
		solver_type='cvxpy',
	)
	# model = PointMassModel(np.reshape([0, 0, 0, 0], (4, 1)), np.reshape([7, 5, 0, 0], (4, 1)), 10, dt)
	planner = ConvexPathPlanner(model, dt, time_horizon)
	# nlp_planner = NonConvexPathPlanner(model, dt, time_horizon)

	# Get Start and Goal, Car Shape
	start_pos, start_orientation = model.get_position_orientation(model.get_initial_state())
	goal_pos, goal_orientation = model.get_position_orientation(model.get_goal_state())

	# Get Car Path
	car_states, control_inputs = planner.get_optimized_trajectory()
	car_path = []

	for i, state in enumerate(car_states):
		pos, orientation = model.get_position_orientation(state.T)
		car_path.append(pos)
		control_input = control_inputs[i] if i < len(control_inputs) else [0, 0]
		print(f"Step {i + 1}: Position, Orientation = ({pos[0]:.2f}, {pos[1]:.2f}), {(orientation*180/pi):.2f}), Control Inputs = [{control_input[0]:.5f}, {control_input[1]:.5f}]")
		visualizer.draw(
			start_pos,
			start_orientation,
			goal_pos,
			goal_orientation,
			pos,
			orientation,
			model.get_shape(state),
			car_path,
			[]
		)
		time.sleep(dt)  # Add delay to simulate motion
	input('END')