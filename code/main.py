import scenarios
from visualizer import VehiclePathVisualizer, visualize_mccormick, visualize_mccormick_2d_interactive


if __name__ == '__main__':
	visualizer = VehiclePathVisualizer(interactive=True)
	scenario = scenarios.collection.oriented_road_following.create_scenario()
	scenario.visualize(visualizer)

