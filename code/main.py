import scenarios
from visualizer import VehiclePathVisualizer, visualize_mccormick, visualize_mccormick_2d_interactive


if __name__ == '__main__':
	visualizer = VehiclePathVisualizer()
	scenario = scenarios.collection.oriented_road_following.create_scenario()
	# scenario.visualize(visualizer)
	# visualize_mccormick((-5, 5), (-1, 1), resolution=100)
	visualize_mccormick_2d_interactive(
		x_bounds=(-2, 200),
		y_bounds=(4, 6),
	)

