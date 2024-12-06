import scenarios
from visualizer import VehiclePathVisualizer, plot_states_or_inputs
from roads import launch_editor, load_road

if __name__ == '__main__':
	visualizer = VehiclePathVisualizer(interactive=True)
	scenario = scenarios.collection.oriented_road_following.create_scenario(1,4,1)
	# plot_states_or_inputs(scenario.control_inputs, scenario.control_input_labels, scenario.dt)
	# plot_states_or_inputs(scenario.predicted_car_states, scenario.state_labels, scenario.dt)
	scenario.visualize(visualizer)
	# launch_editor()

