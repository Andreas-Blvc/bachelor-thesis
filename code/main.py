from visualizer.vehicle_path_visualizer import VehiclePathVisualizer
import scenarios

if __name__ == '__main__':
	visualizer = VehiclePathVisualizer()
	scenario = scenarios.oriented_road_following.create_scenario()
	scenario.visualize(visualizer)

