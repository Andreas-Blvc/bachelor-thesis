import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import use
import numpy as np
from obstacles.road import AbstractRoad
from typing import List

use("TkAgg")


class VehicleObject:
    def __init__(self, position, orientation, shape):
        self.position = position
        self.orientation = orientation
        self.shape = shape

    def polygon_coordinates(self):
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
        cos_theta = np.cos(self.orientation)
        sin_theta = np.sin(self.orientation)

        for point in self.shape:
            # Rotate and translate the points
            x_new = self.position[0] + point[0] * cos_theta - point[1] * sin_theta
            y_new = self.position[1] + point[0] * sin_theta + point[1] * cos_theta
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
        self.actual_path = []
        self.predicted_path = []

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


    def plot_car(self, car, label, face_col, edge_col, marker):
        start_polygon = patches.Polygon(car.polygon_coordinates(), closed=True, facecolor=face_col,
                                        edgecolor=edge_col,
                                        label=label)
        self.ax.add_patch(start_polygon)
        self.ax.plot(car.position[0], car.position[1], marker, label=label, markersize=4)

    def draw(self, start: VehicleObject, goal: VehicleObject, predicted_car: VehicleObject, actual_car: VehicleObject, road: AbstractRoad):

        # Clear the previous drawing
        self.clear()
        if predicted_car:
            self.actual_path.append(actual_car.position)
        self.predicted_path.append(predicted_car.position)

        # Plot Road
        if road is not None:
            shape, color = road.get_polygon_and_color()
            polygon = patches.Polygon(shape, closed=True, facecolor=color, edgecolor='black')
            self.ax.add_patch(polygon)


        # Draw the car shape at the start position
        self.plot_car(
            car=start,
            label='Start',
            face_col='lightgreen',
            edge_col='green',
            marker='go',
        )

        # Draw the car shape at the goal position
        if goal:
            self.plot_car(
                car=goal,
                label='Goal',
                face_col='lightcoral',
                edge_col='red',
                marker='ro',
            )

        # Draw the predicted car
        self.plot_car(
            car=predicted_car,
            label='Predicted',
            face_col='cyan',
            edge_col='blue',
            marker='bo',
        )

        # Draw the actual car
        if actual_car:
            self.plot_car(
                car=actual_car,
                label='Actual',
                face_col='grey',
                edge_col='black',
                marker='ko',
            )

        # Plot the car's path as a trail of markers
        predicted_path = np.array(self.predicted_path)
        self.ax.plot(predicted_path[:, 0], predicted_path[:, 1], 'b-', label='Planned Path', linewidth=2)
        if actual_car:
            actual_path = np.array(self.actual_path)
            self.ax.plot(actual_path[:, 0], actual_path[:, 1], 'k-', label='Actual Path', linewidth=2)

        # Redraw the updated plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    @staticmethod
    def show():
        plt.show(block=True)