from models.double_integrator_model import DoubleIntegrator
from path_planner.optimization import PathPlanner
from visualizer.motion_visualization import MotionVisualizer
import time

if __name__ == '__main__':
	start_state = [0, 0, 0, 0]
	goal_state = [7, 5, 0, 0]
	dt = 1/30
	time_horizon = 3
	# Create Path Planning Instance
	model = DoubleIntegrator(start_state, goal_state, 10, dt)
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