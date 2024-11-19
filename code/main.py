import scenarios
from visualizer import VehiclePathVisualizer

if __name__ == '__main__':
	visualizer = VehiclePathVisualizer()
	scenario = scenarios._internal.oriented_road_following.create_scenario()
	scenario.visualize(visualizer)

