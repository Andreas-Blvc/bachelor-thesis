import scenarios
from visualizer import VehiclePathVisualizer, visualize_mccormick, visualize_mccormick_2d_interactive
from roads import launch_editor, load_road

if __name__ == '__main__':
	# visualizer = VehiclePathVisualizer(interactive=True)
	# scenario = scenarios.collection.oriented_road_following.create_scenario(1,4,1)
	# scenario.visualize(visualizer)

	road = load_road("./data/right_turn.pkl")
	road.plot_combined_curvature_and_derivative()
	# launch_editor()

