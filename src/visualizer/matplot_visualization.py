import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import use
import numpy as np

use("TkAgg")


class MatPlotVisualizer:
    def __init__(self):
        """
        Initializes the visualizer with a figure and axis.
        """
        # Enable interactive mode for live updates
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.setup_plot()

    def setup_plot(self):
        """
        Sets up the initial plot configuration such as axis labels and limits.
        """
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_title('Car Motion Planning Visualization')
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.grid(True)
        self.ax.set_aspect('equal', adjustable='box')

    def clear(self):
        """
        Clears the current plot to allow for redrawing.
        """
        self.ax.clear()
        self.setup_plot()

    def transform_points(self, points, position, orientation):
        """
        Transforms a list of points based on the car's position and orientation.

        Parameters:
        - points (list of tuples): List of (x, y) points defining the shape of the car.
        - position (tuple): (x, y) position of the car.
        - orientation (float): Orientation of the car in radians.

        Returns:
        - List of transformed (x, y) points.
        """
        transformed_points = []
        cos_theta = np.cos(orientation)
        sin_theta = np.sin(orientation)

        for point in points:
            # Rotate and translate the points
            x_new = position[0] + point[0] * cos_theta - point[1] * sin_theta
            y_new = position[1] + point[0] * sin_theta + point[1] * cos_theta
            transformed_points.append((x_new, y_new))

        return transformed_points

    def draw(self, start_pos, start_orientation, goal_pos, goal_orientation, car_position, car_orientation, car_shape,
             car_path, obstacles):
        """
        Draws the car's path, start and goal positions as car shapes, and obstacles.

        Parameters:
        - start_pos (tuple): Starting position of the car as (x, y)
        - goal_pos (tuple): Goal position of the car as (x, y)
        - car_position (tuple): Current position of the car as (x, y)
        - car_orientation (float): Orientation of the car in radians
        - car_shape (list of tuples): List of (x, y) points defining the car's shape
        - car_path (list of tuples): A list of (x, y) points representing the car's planned path
        - obstacles (list of dict): Each obstacle is a dictionary with:
            * 'position' (tuple): Center of the obstacle (x, y)
            * 'size' (tuple): Width and height of the rectangular obstacle (w, h)
        """
        # Clear the previous drawing
        self.clear()

        # Only draw start and goal shapes if car_shape is not empty
        if car_shape:
            # Draw the car shape at the start position
            start_shape = self.transform_points(car_shape, start_pos, start_orientation)
            start_polygon = patches.Polygon(start_shape, closed=True, facecolor='lightgreen', edgecolor='green',
                                            label='Start')
            self.ax.add_patch(start_polygon)

            # Draw the car shape at the goal position
            goal_shape = self.transform_points(car_shape, goal_pos, goal_orientation)
            goal_polygon = patches.Polygon(goal_shape, closed=True, facecolor='lightcoral', edgecolor='red',
                                           label='Goal')
            self.ax.add_patch(goal_polygon)

            # Transform current car shape points
            transformed_car_shape = self.transform_points(car_shape, car_position, car_orientation)
            # Draw the current car position as a filled polygon
            car_polygon = patches.Polygon(transformed_car_shape, closed=True, facecolor='cyan', edgecolor='blue',
                                          label='Car')
            self.ax.add_patch(car_polygon)

        # Plot the car's path as a trail of markers
        if car_path:
            car_path = np.array(car_path)
            self.ax.plot(car_path[:, 0], car_path[:, 1], 'b-', label='Planned Path', linewidth=2)

        # Highlight start/end position
        self.ax.plot(start_pos[0], start_pos[1], 'go', label='Start', markersize=4)
        self.ax.plot(goal_pos[0], goal_pos[1], 'ro', label='Goal', markersize=4)

        # Highlight car's current position
        self.ax.plot(car_position[0], car_position[1], 'bo', markersize=4)

        # Plot obstacles
        for obstacle in obstacles:
            obs_pos = obstacle['position']
            obs_size = obstacle['size']
            rect = patches.Rectangle(
                (obs_pos[0] - obs_size[0] / 2, obs_pos[1] - obs_size[1] / 2),
                obs_size[0], obs_size[1],
                linewidth=1, edgecolor='r', facecolor='gray', label='Obstacle'
            )
            self.ax.add_patch(rect)

        # Redraw the updated plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# Example usage of the dynamic visualizer
if __name__ == "__main__":
    # Create a MotionVisualizer instance
    visualizer = MatPlotVisualizer()

    # Define start, goal, path, car position, and obstacles

    # Start and goal positions
    start_pos = (0, 0)
    goal_pos = (8, 8)

    # A simple path as a list of (x, y) tuples
    car_path = [(0, 0), (2, 2), (4, 3), (6, 5), (8, 8)]

    # Initial car position (this will change dynamically)
    car_position = (0, 0)
    car_orientation = 0.0  # Initial orientation in radians

    # Define the car shape as a list of points (e.g., triangle or arbitrary polygon)
    car_shape = [(-0.5, -0.25), (0.5, -0.25), (0, 0.5)]  # Simple triangle shape

    # Define obstacles as a list of dictionaries
    obstacles = [
        {'position': (4, 4), 'size': (2, 2)},
        {'position': (6, 6), 'size': (1.5, 1.5)}
    ]

    # Draw the initial scene with the car's position and path
    visualizer.draw(start_pos, goal_pos, car_position, car_orientation, car_shape, car_path, obstacles)

    # Simulate some updates to the car's position and orientation
    import time

    for t in range(11):
        # Update the car's position and orientation dynamically
        car_position = (t * 0.8, t * 0.8)
        car_orientation = t * 0.1  # Rotate a bit each time
        visualizer.draw(start_pos, goal_pos, car_position, car_orientation, car_shape, car_path, obstacles)
        time.sleep(0.5)  # Pause for half a second before redrawing

    input()
