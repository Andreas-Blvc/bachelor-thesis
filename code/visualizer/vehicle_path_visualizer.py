import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import use
import numpy as np
from obstacles.road import AbstractRoad
from typing import List

use("TkAgg")


def transform_points(points, position, orientation):
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


class VehiclePathVisualizer:
    def __init__(self):
        """
        Initializes the visualizer with a figure and axis.
        """
        # Enable interactive mode for live updates
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.setup_plot()
        self.car_path = []

    def setup_plot(self):
        """
        Sets up the initial plot configuration such as axis labels and limits.
        """
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_title('Car Motion Planning Visualization')
        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-12, 12)
        self.ax.grid(True)
        self.ax.set_aspect('equal', adjustable='box')

    def clear(self):
        """
        Clears the current plot to allow for redrawing.
        """
        self.ax.clear()
        self.setup_plot()

    def draw(self, start_pos, start_orientation, start_shape, goal_pos, goal_orientation, goal_shape, car_position,
             car_orientation, car_shape, road: AbstractRoad):
        """
        Draws the car's path, start and goal positions as car shapes, and obstacles.

        Parameters:
        - start_pos (tuple): Starting position of the car as (x, y)
        - goal_pos (tuple): Goal position of the car as (x, y)
        - car_position (tuple): Current position of the car as (x, y)
        - car_orientation (float): Orientation of the car in radians
        - car_shape (list of tuples): List of (x, y) points defining the car's shape
        - car_path (list of tuples): A list of (x, y) points representing the car's planned path
        - obstacles (list of obstacle)
        """
        # Clear the previous drawing
        self.clear()
        self.car_path.append(car_position)

        # Plot Road
        if road is not None:
            shape, color = road.get_polygon_and_color()
            polygon = patches.Polygon(shape, closed=True, facecolor=color, edgecolor='black')
            self.ax.add_patch(polygon)

        # Only draw start and goal shapes if car_shape is not empty
        if car_shape:
            # Draw the car shape at the start position
            start_shape = transform_points(start_shape, start_pos, start_orientation)
            start_polygon = patches.Polygon(start_shape, closed=True, facecolor='lightgreen', edgecolor='green',
                                            label='Start')
            self.ax.add_patch(start_polygon)

            # Draw the car shape at the goal position
            goal_shape = transform_points(goal_shape, goal_pos, goal_orientation)
            goal_polygon = patches.Polygon(goal_shape, closed=True, facecolor='lightcoral', edgecolor='red',
                                           label='Goal')
            self.ax.add_patch(goal_polygon)

            # Transform current car shape points
            transformed_car_shape = transform_points(car_shape, car_position, car_orientation)
            # Draw the current car position as a filled polygon
            car_polygon = patches.Polygon(transformed_car_shape, closed=True, facecolor='cyan', edgecolor='blue',
                                          label='Car')
            self.ax.add_patch(car_polygon)

        # Plot the car's path as a trail of markers
        path = np.array(self.car_path)
        self.ax.plot(path[:, 0], path[:, 1], 'b-', label='Planned Path', linewidth=2)

        # Highlight start/end position
        self.ax.plot(start_pos[0], start_pos[1], 'go', label='Start', markersize=4)
        self.ax.plot(goal_pos[0], goal_pos[1], 'ro', label='Goal', markersize=4)

        # Highlight car's current position
        self.ax.plot(car_position[0], car_position[1], 'bo', markersize=4)

        # Redraw the updated plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    @staticmethod
    def show():
        plt.show(block=True)