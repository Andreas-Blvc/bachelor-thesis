from self_driving_cars import AbstractSelfDrivingCar
from visualizer import plot_states_or_inputs


class Benchmark:
    def __init__(self, animation, car: AbstractSelfDrivingCar):
        self.animation = animation
        self.car = car

    def plot_car_states(self):
        plot_states_or_inputs(self.car.car_states, self.car.state_labels, self.car.planner.dt, "Car State")

    def plot_controls(self):
        plot_states_or_inputs(self.car.executed_controls, self.car.control_input_labels, self.car.planner.dt, "Car Control Inputs")


    def plot_predictive_car_states(self):
        plot_states_or_inputs(self.car.predictive_model_states, self.car.predictive_model.state_labels, self.car.planner.dt, "Car State")

    def plot_predictive_controls(self):
        plot_states_or_inputs(self.car.predictive_model_controls, self.car.predictive_model.control_input_labels, self.car.planner.dt, "Car Control Inputs")