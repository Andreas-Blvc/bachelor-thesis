from models.double_integrator_model import DoubleIntegrator
from path_planner.optimization import PathPlanner
from visualizer.motion_visualization import MotionVisualizer
import time

if __name__ == '__main__':
	start_state = [0, 0, 0, 0]
	goal_state = [10, 10, 0, 0]
	dt = 0.1
	model = DoubleIntegrator(start_state, 10, dt)
	planner = PathPlanner(goal_state, model, dt, 30)
	visualizer = MotionVisualizer()

	car_path, control_inputs = planner.get_optimized_trajectory()
	car_path = [(car_path[i][0], car_path[i][1]) for i in range(len(car_path))]
	for i, position in enumerate(car_path):
		control_input = control_inputs[i] if i < len(control_inputs) else [0, 0]
		print(f"Step {i + 1}: Position = ({position[0]:.2f}, {position[1]:.2f}), Control Inputs = [{control_input[0]:.5f}, {control_input[1]:.5f}]")
		visualizer.draw(start_state[:2], goal_state[:2], position, car_path, [])
		time.sleep(dt)  # Add delay to simulate motion
	input('END')