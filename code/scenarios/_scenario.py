from models import AbstractVehicleModel
from visualizer import VehicleObject


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

    def __init__(self,
                 dt,
                 model: AbstractVehicleModel,
                 predicted_car_states,
                 control_inputs,
                 actual_car_states=None,
                 obstacles=None):
        """
        Initialize a Scenario object.

        Args:
            dt (float): Time step duration.
            model (AbstractVehicleModel): Vehicle model providing state and shape functions.
            predicted_car_states (list): Predicted vehicle states over time.
            control_inputs (list): Control inputs applied to the vehicle.
            actual_car_states (list, optional): Actual vehicle states, if available.
            obstacles (list, optional): List of obstacles in the scenario.
        """
        self.anim = None
        self.dt = dt
        self.predicted_car_states = predicted_car_states
        self.actual_car_states = actual_car_states or []
        self.control_inputs = control_inputs
        self.obstacles = obstacles or []
        self.model = model

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

    def _initialize_model_dependent_attributes(self, model: AbstractVehicleModel):
        """
        Initialize attributes derived from the model, such as start and goal positions.
        """
        # Start attributes
        initial_state = model.initial_state
        self.start_pos, self.start_orientation = model.convert_vec_to_state(initial_state).get_position_orientation()
        self.start_shape = model.get_vehicle_polygon(initial_state)

        # Goal attributes
        goal_state = model.goal_state
        if goal_state is not None:
            self.goal_pos, self.goal_orientation = model.convert_vec_to_state(goal_state).get_position_orientation()
            self.goal_shape = model.get_vehicle_polygon(goal_state)
        else:
            self.goal_pos, self.goal_orientation, self.goal_shape = None, None, None

        # Model-dependent methods and attributes
        self.to_string = lambda state_vec, control_vec: (model.state_vec_to_string(state_vec) + "\n" +
                                                         model.control_vec_to_string(control_vec))
        self.get_vehicle_polygon = model.get_vehicle_polygon
        self.get_position_orientation = lambda state_vec: (model.convert_vec_to_state(state_vec).
                                                           get_position_orientation())
        self.control_input_labels = model.control_input_labels
        self.state_labels = model.state_labels
        self.road = getattr(model, "road", None)

    def get_predicted_actual_car(self, step):
        predicted_state = self.predicted_car_states[step]
        actual_car_state = self.actual_car_states[step] if self.actual_car_states else None

        predicted_pos, predicted_orientation = self.get_position_orientation(predicted_state.T)
        predicted_shape = self.get_vehicle_polygon(predicted_state)

        if self.actual_car_states:
            actual_pos, actual_orientation = self.get_position_orientation(actual_car_state.T)
            actual_shape = self.get_vehicle_polygon(actual_car_state)
        else:
            actual_pos, actual_orientation, actual_shape = None, None, None

        # Create VehicleObject instances
        predicted_car = VehicleObject(predicted_pos, predicted_orientation, predicted_shape)
        actual_car = VehicleObject(actual_pos, actual_orientation,
                                   actual_shape) if actual_car_state is not None else None
        return predicted_car, actual_car

    def visualize(self, visualizer):
        """
        Visualizes the simulation using the updated animation-based visualizer.
        """
        num_steps = len(self.predicted_car_states)

        def update_func(step):
            """
            Update function for animation.
            """
            predicted_car_update, actual_car_update = self.get_predicted_actual_car(step)

            # Update the visualizer for the current step
            visualizer.update_frame(
                step,
                predicted_car=predicted_car_update,
                actual_car=actual_car_update,
            )

        # Create start and goal vehicle objects
        start_vehicle = VehicleObject(self.start_pos, self.start_orientation, self.start_shape)
        goal_vehicle = (
            VehicleObject(self.goal_pos, self.goal_orientation, self.goal_shape)
            if self.goal_orientation is not None
            else None
        )

        # Assign the animation to a variable to prevent garbage collection
        predicted_car, actual_car = self.get_predicted_actual_car(0)
        self.anim = visualizer.animate(
            start=start_vehicle,
            goal=goal_vehicle,
            predicted_car=predicted_car,  # Placeholder, updated in `update_func`
            actual_car=actual_car,  # Placeholder, updated in `update_func`
            road=self.road,
            num_frames=num_steps,
            update_func=update_func,
            dt=self.dt,
        )
        return self.anim

    def __repr__(self):
        """
        Provide a string representation of the Scenario object for debugging purposes.
        """
        return (
            f"Scenario(dt={self.dt}, start_pos={self.start_pos}, goal_pos={self.goal_pos}, "
            f"num_predicted_states={len(self.predicted_car_states)}, num_actual_states={len(self.actual_car_states)}, "
            f"num_obstacles={len(self.obstacles)})"
        )
