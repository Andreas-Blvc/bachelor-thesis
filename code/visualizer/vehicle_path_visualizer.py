import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

WIDTH=200
HEIGHT=200

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
    def __init__(self, interactive: bool, title: str=''):
        self.anim = None
        self.interactive = interactive
        self.fig, self.ax = plt.subplots()
        if not interactive:
            plt.close(self.fig)  # Suppress display right after creation
        self.setup_plot(title)

        # Storage for paths
        self.actual_path = []
        self.predicted_path = []

        # Patches for dynamic updates
        self.start_patch = None
        self.goal_patch = None
        self.predicted_patch = None
        self.actual_patch = None
        self.predicted_path_line, = self.ax.plot([], [], 'b-')
        self.actual_path_line, = self.ax.plot([], [], 'k-')

    def setup_plot(self, title):
        self.ax.set_xlim(0, WIDTH)
        self.ax.set_ylim(0, HEIGHT)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True)
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.set_title(f"Vehicle Path Visualization - {title}")
        # self.ax.legend()

    def init_patches(self, start, goal, predicted_car, actual_car):
        """
        Initialize the car patches for the animation.
        """
        # START
        self.start_patch = patches.Polygon(
            start.polygon_coordinates(), closed=True, facecolor='lightgreen', edgecolor='green', label='Start'
        )
        self.ax.add_patch(self.start_patch)

        # PREDICTED CAR-STATE
        self.predicted_patch = patches.Polygon(
            predicted_car.polygon_coordinates(), closed=True, facecolor='cyan', edgecolor='blue', label='Predicted'
        )
        self.ax.add_patch(self.predicted_patch)

        if goal is not None:
            self.goal_patch = patches.Polygon(
                goal.polygon_coordinates(), closed=True, facecolor='lightcoral', edgecolor='red', label='Goal'
            )
            self.ax.add_patch(self.goal_patch)

        if actual_car is not None:
            self.actual_patch = patches.Polygon(
                actual_car.polygon_coordinates(), closed=True, facecolor='grey', edgecolor='black', label='Actual'
            )
            self.ax.add_patch(self.actual_patch)

        self.ax.legend()

    def update_frame(self, step, predicted_car, actual_car):
        """
        Update the animation frame with new car positions.
        """
        zoom_width = 30  # Width of the zoomed-in view
        zoom_height = 30  # Height of the zoomed-in view

        # Update paths
        if actual_car is not None:
            self.actual_path.append(actual_car.position)
            self.actual_patch.set_xy(actual_car.polygon_coordinates())
            self.actual_path_line.set_data(
                [p[0] for p in self.actual_path], [p[1] for p in self.actual_path]
            )

        # Recenter the view around the predicted car
        center_x, center_y = predicted_car.position

        self.predicted_path.append(predicted_car.position)
        self.predicted_patch.set_xy(predicted_car.polygon_coordinates())
        self.predicted_path_line.set_data(
            [p[0] for p in self.predicted_path], [p[1] for p in self.predicted_path]
        )

        # Set the zoomed-in view around the car
        self.ax.set_xlim(center_x - zoom_width / 2, center_x + zoom_width / 2)
        self.ax.set_ylim(center_y - zoom_height / 2, center_y + zoom_height / 2)

        # Redraw
        return self.predicted_patch, self.actual_patch, self.predicted_path_line, self.actual_path_line

    def animate(self, start, goal, predicted_car, actual_car, road, num_frames, update_func, dt):
        """
        Run the animation.
        """
        if road:  # Draw road only once
            shape, color = road.get_polygon_and_color()
            road_patch = patches.Polygon(shape, closed=True, facecolor=color, edgecolor='black')
            self.ax.add_patch(road_patch)

        self.init_patches(start, goal, predicted_car, actual_car)

        self.anim = FuncAnimation(
            self.fig,
            update_func,
            frames=num_frames,
            interval=int(dt * 1000),  # Interval in ms
            blit=False,
            repeat=False
        )

        if not self.interactive:
            centered_html = f"""
            <div style="display: flex; justify-content: center; align-items: center;">
                {self.anim.to_jshtml()}
            </div>
            """
            return HTML(centered_html)
        else:
            self.show()

    @staticmethod
    def show():
        plt.show()