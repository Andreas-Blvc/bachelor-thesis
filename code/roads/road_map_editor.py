import tkinter as tk
from tkinter import Canvas, Frame, Button, TOP, LEFT, simpledialog, Label, filedialog
import math
import os
import dill as pickle
from typing import List, Tuple

from .road import Road
from .road_interface import AbstractRoadSegment
from .segments import StraightRoad, CircularCurveRoad

# CONSTANTS
TK_WIDTH = 800
TK_HEIGHT = 800
PATH_PLANNER_HEIGHT = 20
PATH_PLANNER_WIDTH = 20
GRID_SPACING = 20
DATA_FOLDER = "data"

def tk_to_path_coord(x: float, y: float) -> Tuple[float, float]:
    return (
        x * (PATH_PLANNER_WIDTH / TK_WIDTH),
        -y * (PATH_PLANNER_HEIGHT / TK_HEIGHT)
    )

def path_to_tk_coord(x: float, y: float) -> Tuple[float, float]:
    return (
        x * (TK_WIDTH / PATH_PLANNER_WIDTH),
        -y * (TK_HEIGHT / PATH_PLANNER_HEIGHT)
    )

class RoadMapEditor:
    def __init__(self, tk_root):
        self.tk_root = tk_root
        self.tk_root.title("Road Drawing Tool")

        # Create the toolbar frame
        self.toolbar = Frame(tk_root)
        self.toolbar.pack(side=TOP, fill=tk.X)

        # Create road selection buttons in the toolbar
        self.selected_road_type = tk.StringVar(value="None")  # Keeps track of which road is selected

        self.straight_road_btn = Button(self.toolbar, text="Straight Road", command=self.select_straight_road)
        self.straight_road_btn.pack(side=LEFT, padx=2, pady=2)

        self.circular_curve_btn = Button(self.toolbar, text="Circular Curve", command=self.select_circular_curve)
        self.circular_curve_btn.pack(side=LEFT, padx=2, pady=2)

        # Add clear button to the toolbar
        self.clear_btn = Button(self.toolbar, text="Clear", command=self.clear_canvas)
        self.clear_btn.pack(side=LEFT, padx=2, pady=2)

        # Add delete last segment button to the toolbar
        self.delete_last_btn = Button(self.toolbar, text="Delete Last Segment", command=self.delete_last_segment)
        self.delete_last_btn.pack(side=LEFT, padx=2, pady=2)

        # Add save and load buttons to the toolbar
        self.save_btn = Button(self.toolbar, text="Save", command=self.save_road)
        self.save_btn.pack(side=LEFT, padx=2, pady=2)

        self.load_btn = Button(self.toolbar, text="Load", command=self.load_road)
        self.load_btn.pack(side=LEFT, padx=2, pady=2)

        # Add cell width and height information
        cell_width = PATH_PLANNER_WIDTH / (TK_WIDTH / GRID_SPACING)
        cell_height = PATH_PLANNER_HEIGHT / (TK_HEIGHT / GRID_SPACING)
        self.grid_info_label = Label(self.toolbar, text=f"Cell Size: {cell_width:.2f} x {cell_height:.2f}")
        self.grid_info_label.pack(side=LEFT, padx=10, pady=2)

        # Canvas for drawing
        self.canvas = Canvas(tk_root, width=TK_WIDTH, height=TK_HEIGHT, bg='white')
        self.canvas.pack()

        # Draw underlying grid
        self.draw_grid()

        # Event bindings for road drawing
        self.canvas.bind("<Button-1>", self.on_click)

        self.roads: List[AbstractRoadSegment] = []  # Store created road segments
        self.last_endpoint = None  # Track the last endpoint of the most recent road
        self.full_road = None  # Full road composed of segments

        # Ensure data directory exists
        if not os.path.exists(DATA_FOLDER):
            os.makedirs(DATA_FOLDER)

    def highlight_selected_button(self):
        """
        Update button styles to indicate which road type is currently selected.
        """

        # Reset all button styles
        self.straight_road_btn.config(default="normal")
        self.circular_curve_btn.config(default="normal")

        # Highlight the selected button
        if self.selected_road_type.get() == "StraightRoad":
            self.straight_road_btn.config(default="active")
        elif self.selected_road_type.get() == "CircularCurve":
            self.circular_curve_btn.config(default="active")

    def select_straight_road(self):
        """Selects the straight road as the road type to draw."""
        self.selected_road_type.set("StraightRoad")
        self.highlight_selected_button()
        print("Straight Road Selected")

    def select_circular_curve(self):
        """Selects the circular curve as the road type to draw."""
        self.selected_road_type.set("CircularCurve")
        self.highlight_selected_button()
        print("Circular Curve Selected")

    def on_click(self, event):
        # Handle canvas click and road creation
        x, y = tk_to_path_coord(event.x, event.y)

        # If there's a last endpoint, use it as the starting position for the next road
        if self.last_endpoint:
            x, y = self.last_endpoint

        # Create different roads depending on which road type is selected
        if self.selected_road_type.get() == "StraightRoad":
            # Get parameters for the StraightRoad from the user
            length = simpledialog.askfloat("Input", "Enter the length of the Straight Road:", minvalue=.01)
            width = simpledialog.askfloat("Input", "Enter the width of the Straight Road:", minvalue=.01)
            if len(self.roads) == 0:
                direction_angle = simpledialog.askfloat("Input",
                                                        "Enter the direction angle (in degrees) of the Straight Road:")
                direction_angle = math.radians(direction_angle) if direction_angle else 0.0
            else:
                direction_angle = self.roads[-1].get_tangent_angle_at(1)

            # Example: create a StraightRoad starting at last endpoint or click point
            road = StraightRoad(width=width, length=length, start_position=(x, y), direction_angle=direction_angle)
            self.roads.append(road)
            self.draw_road(road)

            # Update the last endpoint based on the new road
            self.last_endpoint = self.calculate_endpoint(road)

        elif self.selected_road_type.get() == "CircularCurve":
            # Get parameters for the CircularCurveRoad from the user
            radius = simpledialog.askfloat("Input", "Enter the radius of the Circular Curve Road:", minvalue=.01)
            width = simpledialog.askfloat("Input", "Enter the width of the Circular Curve Road:", minvalue=.01)
            start_angle = None
            if len(self.roads) == 0:
                start_angle = simpledialog.askfloat("Input", "Enter the start angle (in degrees) of the Circular Curve:")
                start_angle = math.radians(start_angle) if start_angle else 0.0

            angle_sweep = simpledialog.askfloat("Input", "Enter the angle sweep (in degrees) of the Circular Curve:")
            angle_sweep = math.radians(angle_sweep) if angle_sweep else math.pi / 2

            if start_angle is None:
                start_angle = (1.5 if angle_sweep > 0 else 0.5) * math.pi + self.roads[-1].get_tangent_angle_at(1)

            previous_angle = self.roads[-1].get_tangent_angle_at(1) if len(self.roads) > 0 else 0
            road = CircularCurveRoad(
                width=width,
                radius=radius,
                center=(
                    x + math.cos(math.pi/2 + previous_angle) * radius * (1 if angle_sweep > 0 else -1),
                    y + math.sin(math.pi/2 + previous_angle) * radius * (1 if angle_sweep > 0 else -1)
                ),
                start_angle=start_angle,
                angle_sweep=angle_sweep
            )
            self.roads.append(road)
            self.draw_road(road)

            # Update the last endpoint based on the new road
            self.last_endpoint = self.calculate_endpoint(road)

        else:
            print(f"No road type selected, clicked at: {x}, {y}")

        # Create a full road from segments
        self.full_road = Road(self.roads)

    def draw_road(self, road):
        # Draw road polygons on the canvas
        polygon, color = road.get_polygon_and_color()
        points = [coord for x, y in polygon for coord in path_to_tk_coord(x, y)]
        self.canvas.create_polygon(points, outline=color, fill='', width=2)

    def draw_grid(self):
        """Draws an underlying grid on the canvas."""
        for i in range(0, TK_WIDTH, GRID_SPACING):
            self.canvas.create_line([(i, 0), (i, TK_HEIGHT)], fill='lightgray', width=1)
        for i in range(0, TK_HEIGHT, GRID_SPACING):
            self.canvas.create_line([(0, i), (TK_WIDTH, i)], fill='lightgray', width=1)

    def clear_canvas(self):
        """Clears the canvas and resets the road list and endpoint."""
        self.canvas.delete("all")
        self.draw_grid()  # Redraw the grid after clearing
        self.roads.clear()
        self.last_endpoint = None
        self.full_road = None

    def delete_last_segment(self):
        """Deletes the last added road segment."""
        if self.roads:
            # Remove the last road segment
            last_segment = self.roads.pop()
            self.last_endpoint = self.calculate_endpoint(self.roads[-1]) if self.roads else None
            self.canvas.delete("all")
            self.draw_grid()
            for segment in self.roads:
                self.draw_road(segment)
            print("Last segment deleted.")
        else:
            print("No segments to delete.")

    def save_road(self):
        """Saves the current road to a file in the data folder."""
        if self.full_road is None:
            print("No road to save.")
            return

        filename = simpledialog.askstring("Input", "Enter a name for the road file:")
        if not filename:
            print("Save cancelled.")
            return

        filepath = os.path.join(DATA_FOLDER, f"{filename}.pkl")
        with open(filepath, "wb") as file:
            pickle.dump(self.full_road, file)
        print(f"Road saved as {filepath}.")

    def load_road(self):
        """Loads a road from a file in the data folder."""
        filepath = filedialog.askopenfilename(initialdir=DATA_FOLDER, title="Select Road File",
                                              filetypes=(("Pickle files", "*.pkl"), ("All files", "*.*")))
        if not filepath:
            print("Load cancelled.")
            return

        with open(filepath, "rb") as file:
            loaded_road = pickle.load(file)

        if isinstance(loaded_road, Road):
            self.clear_canvas()
            self.full_road = loaded_road
            self.roads = loaded_road.get_all_segments()
            for segment in self.roads:
                self.draw_road(segment)
            print(f"Road loaded from {filepath}.")
        else:
            print("Invalid road file.")

    @staticmethod
    def calculate_endpoint(road):
        """
        Calculate the endpoint of a road-based on its type and geometry.
        This is a placeholder implementation. Adjust according to actual road geometry.
        """
        if isinstance(road, StraightRoad):
            x_start, y_start = road.start_position
            x_end = x_start + road.length * math.cos(road.direction_angle)
            y_end = y_start + road.length * math.sin(road.direction_angle)
            return x_end, y_end

        elif isinstance(road, CircularCurveRoad):
            # Assuming we use the end point at the end of the sweep
            angle_end = road.start_angle + road.angle_sweep
            x_end = road.center[0] + road.radius * math.cos(angle_end)
            y_end = road.center[1] + road.radius * math.sin(angle_end)
            return x_end, y_end

        return None

def launch_editor():
    root = tk.Tk()
    app = RoadMapEditor(root)
    root.mainloop()


def load_road(filepath: str =None):
    """Loads a road from a file in the data folder."""
    if filepath is None:
        filepath = filedialog.askopenfilename(initialdir=DATA_FOLDER, title="Select Road File",
                                          filetypes=(("Pickle files", "*.pkl"), ("All files", "*.*")))
    if not filepath:
        print("Load cancelled.")
        return

    with open(filepath, "rb") as file:
        print(filepath)
        loaded_road = pickle.load(file)

    if isinstance(loaded_road, Road):
        print(f"Road loaded from {filepath}.")
        return loaded_road
    else:
        print("Invalid road file.")