import matplotlib.pyplot as plt
import matplotlib.patches as patches
from  matplotlib import use
import numpy as np

use("TkAgg")

class MotionVisualizer:
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

    def draw(self, start_pos, goal_pos, car_position, car_path, obstacles):
        """
        Draws the car's path, start and goal positions, and obstacles.

        Parameters:
        - start_pos (tuple): Starting position of the car as (x, y)
        - goal_pos (tuple): Goal position of the car as (x, y)
        - car_position (tuple): Current position of the car as (x, y)
        - car_path (list of tuples): A list of (x, y) points representing the car's planned path
        - obstacles (list of dict): Each obstacle is a dictionary with:
            * 'position' (tuple): Center of the obstacle (x, y)
            * 'size' (tuple): Width and height of the rectangular obstacle (w, h)
        """
        # Clear the previous drawing
        self.clear()

        # Plot start and goal positions
        self.ax.plot(start_pos[0], start_pos[1], 'go', label='Start', markersize=10)  # Start as green circle
        self.ax.plot(goal_pos[0], goal_pos[1], 'ro', label='Goal', markersize=10)    # Goal as red circle

        # Plot the car's path
        if car_path:
            car_path = np.array(car_path)
            self.ax.plot(car_path[:, 0], car_path[:, 1], 'b-', label='Planned Path', linewidth=2)

        # Plot the car's current position as a blue point
        self.ax.plot(car_position[0], car_position[1], 'bo', label='Car', markersize=10)

        # Plot obstacles
        for obstacle in obstacles:
            obs_pos = obstacle['position']
            obs_size = obstacle['size']
            rect = patches.Rectangle(
                (obs_pos[0] - obs_size[0] / 2, obs_pos[1] - obs_size[1] / 2),  # Bottom left corner
                obs_size[0],  # Width
                obs_size[1],  # Height
                linewidth=1, edgecolor='r', facecolor='gray', label='Obstacle'
            )
            self.ax.add_patch(rect)

        # Redraw the updated plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# Example usage of the dynamic visualizer
if __name__ == "__main__":
    # Create a MotionVisualizer instance
    visualizer = MotionVisualizer()

    # Define start, goal, path, car position, and obstacles

    # Start and goal positions
    start_pos = (0, 0)
    goal_pos = (8, 8)

    # A simple path as a list of (x, y) tuples
    car_path = [(0, 0), (2, 2), (4, 3), (6, 5), (8, 8)]

    # Initial car position (this will change dynamically)
    car_position = (0, 0)

    # Define obstacles as a list of dictionaries
    obstacles = [
        {'position': (4, 4), 'size': (2, 2)},
        {'position': (6, 6), 'size': (1.5, 1.5)}
    ]

    # Draw the initial scene with the car's position and path
    visualizer.draw(start_pos, goal_pos, car_position, car_path, obstacles)

    # Simulate some updates to the car's position and redraw the scene
    import time
    for t in range(10):
        # Update the car's position dynamically (e.g., simulating a moving car)
        car_position = (t * 0.8, t * 0.8)  # Update car position independently
        visualizer.draw(start_pos, goal_pos, car_position, car_path, obstacles)
        time.sleep(0.5)  # Pause for half a second before redrawing
    input()