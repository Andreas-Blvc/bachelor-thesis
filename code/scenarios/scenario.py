import time

from models.vehicle_model import VehicleModel
from visualizer.vehicle_path_visualizer import VehicleObject


class Scenario:
    """
    Represents a driving scenario including vehicle states, control inputs, and obstacles.

    Attributes:
        dt (float): Time step duration.
        start_pos (tuple): Starting position of the vehicle.
        start_orientation (float): Starting orientation of the vehicle.
        start_shape (Polygon): Polygon representing the vehicle's initial shape.
        goal_pos (tuple or None): Goal position of the vehicle, if defined.
        goal_orientation (float or None): Goal orientation of the vehicle, if defined.
        goal_shape (Polygon or None): Polygon representing the vehicle's goal shape, if defined.
        predicted_car_states (list): Predicted states of the vehicle.
        actual_car_states (list or None): Actual states of the vehicle.
        control_inputs (list): Control inputs applied to the vehicle.
        obstacles (list): List of obstacles in the scenario.
        road (object or None): Road model, if available.
    """

    def __init__(self, dt, model: VehicleModel, predicted_car_states, control_inputs, actual_car_states=None, obstacles=None):
        """
        Initialize a Scenario object.

        Args:
            dt (float): Time step duration.
            model (VehicleModel): Vehicle model providing state and shape functions.
            predicted_car_states (list): Predicted vehicle states over time.
            control_inputs (list): Control inputs applied to the vehicle.
            actual_car_states (list, optional): Actual vehicle states, if available.
            obstacles (list, optional): List of obstacles in the scenario.
        """
        self.dt = dt
        self.predicted_car_states = predicted_car_states
        self.actual_car_states = actual_car_states or []
        self.control_inputs = control_inputs
        self.obstacles = obstacles or []

        self._validate_states()
        self._initialize_model_dependent_attributes(model)

    def _validate_states(self):
        """
        Validate that the actual car states match the number of predicted car states if provided.
        """
        if self.actual_car_states and len(self.actual_car_states) != len(self.predicted_car_states):
            raise ValueError(
                f"Number of actual car states ({len(self.actual_car_states)}) does not match "
                f"the number of predicted car states ({len(self.predicted_car_states)})."
            )

    def _initialize_model_dependent_attributes(self, model):
        """
        Initialize attributes derived from the model, such as start and goal positions.
        """
        # Start attributes
        initial_state = model.get_initial_state()
        self.start_pos, self.start_orientation = model.get_position_orientation(initial_state)
        self.start_shape = model.get_vehicle_polygon(initial_state)

        # Goal attributes
        goal_state = model.get_goal_state()
        if goal_state is not None:
            self.goal_pos, self.goal_orientation = model.get_position_orientation(goal_state)
            self.goal_shape = model.get_vehicle_polygon(goal_state)
        else:
            self.goal_pos, self.goal_orientation, self.goal_shape = None, None, None

        # Model-dependent methods and attributes
        self.to_string = model.to_string
        self.get_vehicle_polygon = model.get_vehicle_polygon
        self.get_position_orientation = model.get_position_orientation
        self.control_input_labels = model.get_control_input_labels()
        self.road = getattr(model, "road", None)

    def visualize(self, visualizer):
        for i in range(len(self.predicted_car_states)):
            predicted_state = self.predicted_car_states[i]
            actual_car_state = self.actual_car_states[i] if self.actual_car_states else None

            predicted_pos, predicted_orientation = self.get_position_orientation(predicted_state.T)
            predicted_shape = self.get_vehicle_polygon(predicted_state)

            if self.actual_car_states:
                actual_pos, actual_orientation = self.get_position_orientation(actual_car_state.T)
                actual_shape = self.get_vehicle_polygon(actual_car_state)
            else:
                actual_pos, actual_orientation, actual_shape = None, None, None

            control = self.control_inputs[i] if i < len(self.control_inputs) else (0, 0)

            # print(
            # 	f"Step {i + 1}: "
            # 	f"Position = ({pos[0]:.2f} m, {pos[1]:.2f} m), "
            # 	+ scenario.to_string(state, control)
            # )
            visualizer.draw(
                start=VehicleObject(self.start_pos, self.start_orientation, self.start_shape),
                goal=VehicleObject(self.goal_pos, self.goal_orientation,
                                   self.goal_shape) if self.goal_orientation is not None else None,
                predicted_car=VehicleObject(predicted_pos, predicted_orientation, predicted_shape),
                actual_car=VehicleObject(actual_pos, actual_orientation,
                                         actual_shape) if self.actual_car_states else None,
                road=self.road
            )
            time.sleep(self.dt / 10)  # Add delay to simulate motion
        visualizer.show()

    def __repr__(self):
        """
        Provide a string representation of the Scenario object for debugging purposes.
        """
        return (
            f"Scenario(dt={self.dt}, start_pos={self.start_pos}, goal_pos={self.goal_pos}, "
            f"num_predicted_states={len(self.predicted_car_states)}, num_actual_states={len(self.actual_car_states)}, "
            f"num_obstacles={len(self.obstacles)})"
        )
