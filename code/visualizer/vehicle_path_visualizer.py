import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from self_driving_cars import AbstractSelfDrivingCar
from utils.constants import PATH_PLANNER_HEIGHT as HEIGHT, PATH_PLANNER_WIDTH as WIDTH

def _polygon_coordinates(position, orientation, shape):
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

    for point in shape:
        # Rotate and translate the points
        x_new = position[0] + point[0] * cos_theta - point[1] * sin_theta
        y_new = position[1] + point[0] * sin_theta + point[1] * cos_theta
        transformed_points.append((x_new, y_new))

    return transformed_points


def animate(car: AbstractSelfDrivingCar, interactive: bool, title: str=''):
    car_states_gen = car.drive()
    fig, ax = plt.subplots()
    if not interactive:
        plt.close(fig)  # Suppress display right after creation

    # setup plot
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(f"Vehicle Path Visualization - {title}")

    # initialize car dependent variables
    start: np.ndarray = car.get_start()
    start_pos, start_orientation = car.get_position(start), car.get_orientation(start)
    start_shape = car.get_vehicle_polygon(start)

    # Goal attributes
    goal: np.ndarray | None = car.get_goal()
    if goal is not None:
        goal_pos, goal_orientation = car.get_position(goal), car.get_orientation(goal)
        goal_shape = car.get_vehicle_polygon(goal)
    else:
        goal_pos, goal_orientation, goal_shape = None, None, None

    car_path = []
    car_path_line, = ax.plot([], [], 'b-')

    """
        INIT PATCHES
    """
    if getattr(car, "road", None):  # Draw road only once
        shape, color = car.road.get_polygon_and_color()
        road_patch = patches.Polygon(np.array(shape), closed=True, facecolor=color, edgecolor='black')
        ax.add_patch(road_patch)

    # Start
    start_polygon_coordinates = _polygon_coordinates(start_pos, start_orientation, start_shape)
    start_patch = patches.Polygon(
        np.array(start_polygon_coordinates), closed=True, facecolor='lightgreen', edgecolor='green', label='Start'
    )
    ax.add_patch(start_patch)

    # Car Patch, which will get updated
    initial_car_state = next(car_states_gen)
    car_polygon_coordinates = _polygon_coordinates(
        car.get_position(initial_car_state),
        car.get_orientation(initial_car_state),
        car.get_vehicle_polygon(initial_car_state),
    )
    car_patch = patches.Polygon(
        np.array(car_polygon_coordinates), closed=True, facecolor='cyan', edgecolor='blue', label='Predicted'
    )
    ax.add_patch(car_patch)

    # Goal
    if goal is not None:
        goal_polygon_coordinates = _polygon_coordinates(goal_pos, goal_orientation, goal_shape)
        goal_patch = patches.Polygon(
            np.array(goal_polygon_coordinates), closed=True, facecolor='lightcoral', edgecolor='red', label='Goal'
        )
        ax.add_patch(goal_patch)

    ax.legend()

    """
        RUN THE ANIMATION:
    """
    # TODO
    num_steps = 8
    dt = car.planner.dt

    def update_frame():
        """
        Update the animation frame with new car positions.
        """
        zoom_width = 30  # Width of the zoomed-in view
        zoom_height = 30  # Height of the zoomed-in view

        car_state = next(car_states_gen)
        car_position = car.get_position(car_state)

        _car_polygon_coordinates = _polygon_coordinates(
            car.get_position(car_state),
            car.get_orientation(car_state),
            car.get_vehicle_polygon(car_state),
        )

        # Recenter the view around the predicted car
        center_x, center_y = car_position

        car_path.append(car_position)
        car_patch.set_xy(_car_polygon_coordinates)
        car_path_line.set_data(
            [p[0] for p in car_path], [p[1] for p in car_path]
        )

        # Set the zoomed-in view around the car
        ax.set_xlim(center_x - zoom_width / 2, center_x + zoom_width / 2)
        ax.set_ylim(center_y - zoom_height / 2, center_y + zoom_height / 2)

        # Redraw
        return car_patch, car_path_line

    anim = FuncAnimation(
        fig,
        lambda step: update_frame(),
        frames=num_steps,
        interval=int(dt * 1000),  # Interval in ms
        blit=False,
        repeat=False
    )

    if not interactive:
        centered_html = f"""
                <div style="display: flex; justify-content: center; align-items: center;">
                    {anim.to_jshtml()}
                </div>
                """
        return HTML(centered_html)
    else:
        plt.show()


# class VehiclePathVisualizer:
#     def __init__(self, car: AbstractSelfDrivingCar, interactive: bool, title: str=''):
#         self.anim = None
#         self.car = car
#         self.car_states_gen = self.car.drive()
#         self.interactive = interactive
#         self.fig, self.ax = plt.subplots()
#         if not interactive:
#             plt.close(self.fig)  # Suppress display right after creation
#         self._setup_plot(title)
#         self._initialize_car_dependent_attributes(car)
#
#         # Storage for paths
#         self.actual_path = []
#         self.predicted_path = []
#
#         # Patches for dynamic updates
#         self.start_patch = None
#         self.goal_patch = None
#         self.predicted_patch = None
#         self.actual_patch = None
#         self.predicted_path_line, = self.ax.plot([], [], 'b-')
#         self.actual_path_line, = self.ax.plot([], [], 'k-')
#
#     def _setup_plot(self, title):
#         self.ax.set_xlim(0, WIDTH)
#         self.ax.set_ylim(0, HEIGHT)
#         self.ax.set_aspect('equal', adjustable='box')
#         self.ax.grid(True)
#         self.ax.set_xlabel("X Position")
#         self.ax.set_ylabel("Y Position")
#         self.ax.set_title(f"Vehicle Path Visualization - {title}")
#         # self.ax.legend()
#
#     def _init_patches(self, start, goal, predicted_car, actual_car):
#         """
#         Initialize the car patches for the animation.
#         """
#         # START
#         self.start_patch = patches.Polygon(
#             start.polygon_coordinates(), closed=True, facecolor='lightgreen', edgecolor='green', label='Start'
#         )
#         self.ax.add_patch(self.start_patch)
#
#         # PREDICTED CAR-STATE
#         self.predicted_patch = patches.Polygon(
#             predicted_car.polygon_coordinates(), closed=True, facecolor='cyan', edgecolor='blue', label='Predicted'
#         )
#         self.ax.add_patch(self.predicted_patch)
#
#         if goal is not None:
#             self.goal_patch = patches.Polygon(
#                 goal.polygon_coordinates(), closed=True, facecolor='lightcoral', edgecolor='red', label='Goal'
#             )
#             self.ax.add_patch(self.goal_patch)
#
#         if actual_car is not None:
#             self.actual_patch = patches.Polygon(
#                 actual_car.polygon_coordinates(), closed=True, facecolor='grey', edgecolor='black', label='Actual'
#             )
#             self.ax.add_patch(self.actual_patch)
#
#         self.ax.legend()
#
#     def _initialize_car_dependent_attributes(self, car: AbstractSelfDrivingCar):
#         """
#         Initialize attributes derived from the car, such as start and goal positions.
#         """
#         # Start attributes
#         start: np.ndarray = car.get_start()
#         self.start_pos, self.start_orientation = car.get_position(start), car.get_orientation(start)
#         self.start_shape = car.get_vehicle_polygon(start)
#
#         # Goal attributes
#         goal: np.ndarray | None = car.get_goal()
#         if goal is not None:
#             self.goal_pos, self.goal_orientation = car.get_position(goal), car.get_orientation(goal)
#             self.goal_shape = car.get_vehicle_polygon(goal)
#         else:
#             self.goal_pos, self.goal_orientation, self.goal_shape = None, None, None
#
#         # Model-dependent methods and attributes
#         self.to_string = lambda state_vec, control_vec: (
#                 ", ".join([f'{label}: {state}' for label, state in zip(car.state_labels, state_vec)]) + "\n" +
#                 ", ".join([f'{label}: {control}' for label, control in zip(car.control_input_labels, control_vec)])
#         )
#
#     def update_frame(self):
#         """
#         Update the animation frame with new car positions.
#         """
#         zoom_width = 30  # Width of the zoomed-in view
#         zoom_height = 30  # Height of the zoomed-in view
#
#         car_state = next(self.car_states_gen)
#
#         vehicle_object = VehicleObject(
#             self.car.get_position(car_state),
#             self.car.get_orientation(car_state),
#             self.car.get_vehicle_polygon(car_state),
#         )
#
#         # Recenter the view around the predicted car
#         center_x, center_y = vehicle_object.position
#
#         self.predicted_path.append(vehicle_object.position)
#         self.predicted_patch.set_xy(vehicle_object.polygon_coordinates())
#         self.predicted_path_line.set_data(
#             [p[0] for p in self.predicted_path], [p[1] for p in self.predicted_path]
#         )
#
#         # Set the zoomed-in view around the car
#         self.ax.set_xlim(center_x - zoom_width / 2, center_x + zoom_width / 2)
#         self.ax.set_ylim(center_y - zoom_height / 2, center_y + zoom_height / 2)
#
#         # Redraw
#         return self.predicted_patch, self.actual_patch, self.predicted_path_line, self.actual_path_line
#
#     def animate(self):
#         """
#         Run the animation.
#         """
#         # TODO
#         num_steps = 1000
#
#         # Assign the animation to a variable to prevent garbage collection
#         initial_car_state = next(self.car_states_gen)
#
#         if getattr(self.car, "road", None):  # Draw road only once
#             shape, color = self.car.road.get_polygon_and_color()
#             road_patch = patches.Polygon(np.array(shape), closed=True, facecolor=color, edgecolor='black')
#             self.ax.add_patch(road_patch)
#
#         self._init_patches(
#             start= VehicleObject(self.start_pos, self.start_orientation, self.start_shape),
#             goal= (
#                 VehicleObject(self.goal_pos, self.goal_orientation, self.goal_shape)
#                 if self.goal_orientation is not None
#                 else None
#             ),
#             predicted_car=VehicleObject(
#                 self.car.get_position(initial_car_state),
#                 self.car.get_orientation(initial_car_state),
#                 self.car.get_vehicle_polygon(initial_car_state),
#             ),
#             actual_car=None,
#         )
#
#         self.anim = FuncAnimation(
#             self.fig,
#             lambda step: self.update_frame(),
#             frames=num_steps,
#             interval=int(self.dt * 1000),  # Interval in ms
#             blit=False,
#             repeat=False
#         )
#
#         if not self.interactive:
#             centered_html = f"""
#             <div style="display: flex; justify-content: center; align-items: center;">
#                 {self.anim.to_jshtml()}
#             </div>
#             """
#             return HTML(centered_html)
#         else:
#             self.show()
#
#     @staticmethod
#     def show():
#         plt.show()