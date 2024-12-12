# import numpy as np
#
# from self_driving_cars import AbstractSelfDrivingCar
# from visualizer import VehicleObject, plot_states_or_inputs, VehiclePathVisualizer
#
#
# class Scenario:
#     """
#     Represents a driving scenario including vehicle states, control inputs, and obstacles.
#
#     Attributes:
#         dt (float): Time step duration.
#         start_pos (tuple): Starting position of the vehicle.
#         start_orientation (float): Starting orientation of the vehicle.
#         start_shape (Polygon): Polygon representing the vehicle's initial shape.
#         goal_pos (tuple or None): Goal position of the vehicle, if defined.
#         goal_orientation (float or None): Goal orientation of the vehicle, if defined.
#         goal_shape (Polygon or None): Polygon representing the vehicle's goal shape, if defined.
#         road (object or None): Road car, if available.
#     """
#
#     def __init__(self,
#                  dt,
#                  car: AbstractSelfDrivingCar,
#                  interactive=False,
#                  title: str = ''
#                  ):
#         """
#         Initialize a Scenario object.
#
#         Args:
#             dt (float): Time step duration.
#             car (AbstractVehicleModel): Vehicle car providing state and shape functions.
#         """
#         self.anim = None
#         self.dt = dt
#         self.car = car
#         self.car_states_gen = car.drive()
#
#         self._initialize_car_dependent_attributes(car)
#         self.visualizer = VehiclePathVisualizer(interactive, title)
#
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
#         self.get_vehicle_polygon = car.get_vehicle_polygon
#         self.get_position_orientation = lambda state_vec: (
#             car.get_position(state_vec),
#             car.get_orientation(state_vec),
#         )
#         self.control_input_labels = car.control_input_labels
#         self.state_labels = car.state_labels
#         self.road = getattr(car, "road", None)
#
#     def visualize(self):
#         """
#         Visualizes the simulation using the updated animation-based visualizer.
#         """
#         # TODO
#         num_steps = 1e3
#
#         def update_func(step):
#             """
#             Update function for animation.
#             """
#             car_state = next(self.car_states_gen)
#
#             # Update the visualizer for the current step
#             self.visualizer.update_frame(
#                 step,
#                 predicted_car=VehicleObject(
#                     self.car.get_position(car_state),
#                     self.car.get_orientation(car_state),
#                     self.car.get_vehicle_polygon(car_state),
#                 ),  # todo: is now actual car!
#                 # actual_car=actual_car_update,
#             )
#
#         # Create start and goal vehicle objects
#         start_vehicle = VehicleObject(self.start_pos, self.start_orientation, self.start_shape)
#         goal_vehicle = (
#             VehicleObject(self.goal_pos, self.goal_orientation, self.goal_shape)
#             if self.goal_orientation is not None
#             else None
#         )
#
#         # Assign the animation to a variable to prevent garbage collection
#         initial_car_state = next(self.car_states_gen)
#         self.anim = self.visualizer.animate(
#             start=start_vehicle,
#             goal=goal_vehicle,
#             predicted_car=VehicleObject(
#                 self.car.get_position(initial_car_state),
#                 self.car.get_orientation(initial_car_state),
#                 self.car.get_vehicle_polygon(initial_car_state),
#             ),  # Placeholder, updated in `update_func`
#             # actual_car=actual_car,  # Placeholder, updated in `update_func` TODO
#             road=self.road,
#             num_frames=num_steps,
#             update_func=update_func,
#             dt=self.dt,
#         )
#         return self.anim
#
#     # def plot_states(self):
#     #     plot_states_or_inputs(self.predicted_car_states, self.state_labels, self.dt)
#     #
#     # def plot_controls(self):
#     #     plot_states_or_inputs(self.control_inputs, self.control_input_labels, self.dt)
#
#     # def plot_additional_information(self):
#     #     self.car.plot_additional_information(self.predicted_car_states, self.control_inputs)
#
#     def plot_combined_curvature_and_derivative(self):
#         self.car.road.plot_combined_curvature_and_derivative()
#
#     def __repr__(self):
#         """
#         Provide a string representation of the Scenario object for debugging purposes.
#         """
#         return (
#             f"Scenario(dt={self.dt}, start_pos={self.start_pos}, goal_pos={self.goal_pos}"
#         )
