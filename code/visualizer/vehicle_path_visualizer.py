import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import HTML

from self_driving_cars import AbstractSelfDrivingCar
from utils.constants import PATH_PLANNER_HEIGHT as HEIGHT, PATH_PLANNER_WIDTH as WIDTH

import matplotlib as mpl
mpl.rcParams['animation.embed_limit'] = 500

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


def animate(car: AbstractSelfDrivingCar, interactive: bool, title: str='', save_only: bool=False, path:str=''):
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
    car_polygon_coordinates = _polygon_coordinates(
        start_pos,
        start_orientation,
        start_polygon_coordinates,
    )
    car_patch = patches.Polygon(
        np.array(car_polygon_coordinates), closed=True, facecolor='cyan', edgecolor='blue', label='Vehicle'
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

    # Info box
    info_box = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, fontsize=10, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
    )

    dt = car.planner.dt

    def update_frame(car_state):
        zoom_width = 30  # Width of the zoomed-in view
        zoom_height = 30  # Height of the zoomed-in view

        car_position = car.get_position(car_state)
        car_orientation = car.get_orientation(car_state)
        car_speed = car.get_speed(car_state)
        car_steering_angle = car.get_steering_angle(car_state)

        _car_polygon_coordinates = _polygon_coordinates(
            car_position,
            car_orientation,
            car.get_vehicle_polygon(car_state),
        )

        center_x, center_y = car_position

        car_path.append(car_position)
        car_patch.set_xy(_car_polygon_coordinates)
        car_path_line.set_data(
            [p[0] for p in car_path], [p[1] for p in car_path]
        )

        # Update info box text
        info_box.set_text(
            f"Position: ({car_position[0]:.2f}, {car_position[1]:.2f})\n"
            f"Orientation: {np.degrees(car_orientation):.2f}°\n"
            f"Steering Angle: {np.degrees(car_steering_angle):.2f}°\n"
            f"Speed: {car_speed:.2f} m/s"
        )

        ax.set_xlim(center_x - zoom_width / 2, center_x + zoom_width / 2)
        ax.set_ylim(center_y - zoom_height / 2, center_y + zoom_height / 2)
        return car_patch, car_path_line, info_box

    frames = list(car.drive())  # Convert to list to prevent exhaustion

    anim: FuncAnimation = FuncAnimation(
        fig,
        update_frame,
        frames=frames,
        interval=int(dt(0) * 1000),  # Interval in ms
        blit=True,
        repeat=False
    )


    if save_only:
        # SAVING FEATURE
        save_path = f"{path}animation.mp4"  # Default save path
        writer = FFMpegWriter(fps=1 / dt(0), metadata={'artist': 'Me'}, bitrate=1800)
        anim.save(save_path, writer=writer)
        return

    if not interactive:
        centered_html = f"""
        <div style="display: flex; justify-content: center; align-items: center;">
            {anim.to_jshtml()}
        </div>
        """
        return HTML(centered_html)
    else:
        plt.show()
