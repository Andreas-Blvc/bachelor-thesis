from typing import List, Tuple, Callable
from scipy.integrate import solve_ivp
import time
import math
import matplotlib.pyplot as plt
import numpy as np

from roads import AbstractRoad
from utils import add_coordinates, rotate_coordinates
from path_planner import AbstractPathPlanner
from models import AbstractVehicleModel

from .interface import AbstractSelfDrivingCar

class DynamicSingleTrackModel(AbstractSelfDrivingCar):
    def __init__(
            self,
            predictive_model: AbstractVehicleModel,
            planner: AbstractPathPlanner,
            steering_range: Tuple[float, float],
            steering_velocity_range: Tuple[float, float],
            velocity_range: Tuple[float, float],
            acceleration_range: Tuple[float, float],
            road: AbstractRoad = None,
            initial_state=None,
            goal_state=None,
            vehicle_length = 4.298,
            vehicle_width = 1.674,
            total_vehicle_mass = 1.225,
            moment_of_inertia = 1.538,
            distance_from_center_of_gravity_to_front_axle = 0.883,
            distance_from_center_of_gravity_to_rear_axle = 1.508,
            center_of_gravity_height_of_total_mass = 0.557,
            cornering_stiffness_coefficient_front = 20.89,
            cornering_stiffness_coefficient_rear = 20.89,
            friction_coefficient = 1.048,
    ):
        """
        State: Global Position x, Global Postion y, Steering Angle, Velocity, Orientation, Yaw Rate, Slip Angle
        """
        # Metrics for plotting
        self.calculation_times = []  # Store time for planning
        self.solve_times = []
        self.setup_times = []

        self.dim_state = 7
        self.dim_control_input = 2
        self.state_labels = ['Global Position x', 'Global Postion y', 'Steering Angle', 'Velocity', 'Orientation', 'Yaw Rate', 'Slip Angle']
        self.control_input_labels = ['Steering Angle Rate', 'Longitudinal Acceleration']
        # Planning:
        self.initial_state = initial_state
        if self.initial_state is None:
            x, y = road.get_global_position(0, 0)
            psi = road.get_tangent_angle_at(0)
            self.initial_state = np.array([
                x, y, 0, 0, psi, 0 , 0
            ])
        self.goal_state = goal_state
        self.road = road
        self.predictive_model = predictive_model
        self.planner = planner

        self.dt = planner.dt
        self.steering_range = steering_range
        self.steering_velocity_range = steering_velocity_range
        self.velocity_range = velocity_range
        self.acceleration_range = acceleration_range

        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width
        self.total_vehicle_mass = total_vehicle_mass
        self.moment_of_inertia = moment_of_inertia
        self.distance_from_center_of_gravity_to_front_axle = distance_from_center_of_gravity_to_front_axle
        self.distance_from_center_of_gravity_to_rear_axle = distance_from_center_of_gravity_to_rear_axle
        self.center_of_gravity_height_of_total_mass = center_of_gravity_height_of_total_mass
        self.cornering_stiffness_coefficient_front = cornering_stiffness_coefficient_front
        self.cornering_stiffness_coefficient_rear = cornering_stiffness_coefficient_rear
        self.friction_coefficient = friction_coefficient

        # store for plotting
        self.executed_controls = []
        self.car_states = []
        self.predictive_model_states = []
        self.predictive_model_controls = []

    def _on_road(self):
        try:
            self.road.get_road_position(float(self.current_state[0]), float(self.current_state[1]))
            return True
        except ValueError:
            return False

    def drive(self):
        yield self.initial_state
        self.controls: List[Tuple[float, Callable[[], np.ndarray]]] = []
        self.current_state = self.initial_state
        current_time = 0
        N = 5
        while self._on_road():
            if len(self.controls) == 0:
                start_time = time.time()

                predictive_model_states, predictive_model_controls = self.planner.get_optimized_trajectory(
                    self.predictive_model.get_state_vec_from_dsm(self.current_state)
                )

                calculation_time = time.time() - start_time
                self.calculation_times.append(calculation_time)

                self.predictive_model_states += list(predictive_model_states[:N])
                self.predictive_model_controls += list(predictive_model_controls[:N])

                self.controls = [
                    (
                        current_time + self.dt * i,
                        lambda: self.predictive_model.get_dsm_control_from_vec(
                            control,
                            state,
                            dt=self.dt,
                            dynamics=lambda u1, u2: self._dynamics(self.current_state, np.array([float(u1), u2])),
                            remaining_predictive_model_states=predictive_model_controls[i+1:],
                            car_cur_state=AbstractVehicleModel.CarState(self.current_state[2], self.current_state[4])
                        )
                    ) for i, (control, state) in enumerate(list(zip(predictive_model_controls[:-2], predictive_model_states[:-2]))[:N])
                ]

                if len(self.controls) == 0:
                    break

            planned_control_time, lazy_control = self.controls.pop(0)
            control = lazy_control()
            self.current_state = self._update(self.current_state, control)
            current_time = planned_control_time + self.dt
            self.solve_times.append(self.planner.solve_time)
            self.setup_times.append(self.planner.setup_time)
            try:
                s = self.road.get_road_position(float(self.current_state[0]), float(self.current_state[1]))[0]
                print(f"\rProgress: [{'█' * math.floor(s) + '-' * math.ceil(self.road.length-s)}] {(s/self.road.length)*100:.2f}%  "
                      f"of the Road Complete, current state: {', '.join(f'{label}: {val:.2f}' for label, val in zip(self.state_labels, self.current_state))}, "
                      f"planned next {N * self.dt * 1000:.2f}ms in {self.planner.solve_time * 1000:.2f}ms{' ' * 10}", end='')
            except ValueError:
                break
            yield self.current_state

            self.car_states.append(self.current_state)
            self.executed_controls.append(control)
        print()

    # Method to plot metrics
    def plot_metrics(self, store_as_pgf=False, pgf_name="metrics_plot.pgf"):
        """
        Plots the calculation times, solver times, and setup times, and optionally saves the plot as a .pgf file.

        Parameters:
        - store_as_pgf: Whether to save the plot as a .pgf file (default: False).
        - pgf_name: Name of the .pgf file (default: "metrics_plot.pgf").
        """
        # Update matplotlib settings for .pgf output if needed
        if store_as_pgf:
            plt.rcParams.update({
                "text.usetex": True,  # Use LaTeX for rendering text
                "font.family": "serif",  # Use a serif font to match LaTeX
                "font.serif": ["Palatino"],  # Use Palatino to match your LaTeX document
                "pgf.texsystem": "pdflatex",  # Use pdflatex for .pgf output
                "pgf.rcfonts": False,  # Prevent matplotlib from overriding LaTeX fonts
                "font.size": 11,  # Set the font size to match your LaTeX document
                "axes.titlesize": 11,  # Title font size to match the document
                "axes.labelsize": 11,  # Axis label font size
                "xtick.labelsize": 9,  # X-tick label size
                "ytick.labelsize": 9,  # Y-tick label size
                "legend.fontsize": 10,  # Legend font size
            })

        # Create three subplots: calculation time, solver time, and setup time
        plt.figure(figsize=(10, 9))

        # Plot calculation times (first subplot)
        plt.subplot(3, 1, 1)
        plt.plot(self.calculation_times, label="Calculation Time (s)", marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Time (s)")
        plt.title("Calculation Time per Iteration")
        plt.grid(True)
        plt.legend()

        # Plot solver times (second subplot)
        plt.subplot(3, 1, 2)
        plt.plot(self.solve_times, label="Solver Time (s)", marker='o')
        plt.axhline(y=self.dt, color='red', linestyle='--', label="Distance between two time discretizations")
        plt.xlabel("Solver Iteration")
        plt.ylabel("Time (s)")
        plt.title("Solver Time per Iteration")
        plt.grid(True)
        plt.legend()

        # Plot error values (third subplot)
        plt.subplot(3, 1, 3)
        plt.plot(self.setup_times, label="Setup Time (S)", marker='o', color='green')
        plt.xlabel("Iteration")
        plt.ylabel("Time (s)")
        plt.title("Setup Time per Iteration")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        # Save as .pgf if specified
        if store_as_pgf:
            plt.savefig(pgf_name, bbox_inches='tight')
            plt.close()  # Close the figure to prevent display
        else:
            plt.show()

    def _update(self, current_state, control_inputs) -> np.ndarray:
        t_span = (0, self.dt)

        # Wrap dynamics to pass control input as argument
        def dyn(t, x): return self._dynamics(x, control_inputs)

        # Solve it over one time step
        result = solve_ivp(dyn, t_span, current_state, method='BDF')

        return np.array(result.y[:, -1])

    def _dynamics(self, current_state, control_inputs) -> np.ndarray:
        l = self.vehicle_length
        # w = self.vehicle_width
        m = self.total_vehicle_mass
        I_z = self.moment_of_inertia
        l_f = self.distance_from_center_of_gravity_to_front_axle
        l_r = self.distance_from_center_of_gravity_to_rear_axle
        h = self.center_of_gravity_height_of_total_mass
        C_f = self.cornering_stiffness_coefficient_front
        C_r = self.cornering_stiffness_coefficient_rear
        mu = self.friction_coefficient
        g = 9.81

        # x, y, delta, v, psi, dpsi, beta
        x1, x2, x3, x4, x5, x6, x7 = current_state.flatten()
        # v_delta, a_x
        u1, u2 = control_inputs.flatten()
        # print('updating with inputs', u1, u2)

        delta_lb, delta_ub = self.steering_range
        v_delta_lb, v_delta_ub = self.steering_velocity_range
        v_lb, v_ub = self.velocity_range
        a_lb, a_ub = self.acceleration_range


        def f_steer(delta, v_delta):
            C1 = (delta <= delta_lb and v_delta <= 0) or (delta >= delta_ub and v_delta >= 0)
            if C1:
                return 0
            if not C1 and v_delta <= v_delta_lb:
                return v_delta_lb
            if not C1 and v_delta >= v_delta_ub:
                return v_delta_ub
            return v_delta

        def f_acc(v, a):
            C2 = (v <= v_lb and a <= 0) or (v >= v_ub and a >= 0)
            if C2:
                return 0
            if not C2 and a <= a_lb:
                return a_lb
            if not C2 and a >= a_ub:
                return a_ub
            return a


        if np.abs(x4) >= 0.1:
            dx_dt = np.array([
                x4 * np.cos(x5 + x7),
                x4 * np.sin(x5 + x7),
                f_steer(x3, u1),
                f_acc(x4, u2),
                x6,
                mu * m / (I_z * (l_r + l_f)) * (l_f * C_f * (g * l_r - u2 * h) * x3 + (l_r * C_r * (g * l_f + u2 * h) - l_f * C_f * (g * l_r - u2 * h)) * x7 - (l_f ** 2 * C_f * (g * l_r - u2 * h) + l_r ** 2 * C_r * (g * l_f + u2 * h)) * x6 / x4),
                mu / (x4 * (l_r + l_f)) * (C_f * (g * l_r - u2 * h) * x3 - (C_r * (g * l_f + u2 * h) + C_f * (g * l_r - u2 * h)) * x7 + (C_r * (g * l_f + u2 * h) * l - C_f * (g * l_r - u2 * h) * l_f) * x6 / x4) - x6,
            ])
        else:
            l_wb = l_r + l_f
            dx7_dt = 1/(1 + (np.tan(x3) * l_r /l_wb) ** 2) * l_r / (l_wb * np.cos(x3) ** 2) * f_steer(x3, u1)
            dx_dt = np.array([
                x4 * np.cos(x5 + x7),
                x4 * np.sin(x5 + x7),
                f_steer(x3, u1),
                f_acc(x4, u2),
                x4 * np.cos(x7) / l_wb * np.tan(x3),
                1/l_wb * (f_acc(x4, u2) * np.cos(x7) * np.tan(x3) - x4 * np.sin(x7) * np.tan(x3) * dx7_dt * self.dt + x4 * np.cos(x7) / (np.cos(x3) ** 2) * f_steer(x3, u1)),
                dx7_dt,
            ])

        return dx_dt

    # ============================================
    # VISUALIZATION
    # ============================================

    def get_vehicle_polygon(self, state) -> List[Tuple[float, float]]:
        # l_f = self.distance_from_center_of_gravity_to_front_axle
        length = self.vehicle_length
        width = self.vehicle_width
        front_wheel_front = add_coordinates(rotate_coordinates((0.5, 0), float(state[2])), (length/2, 0))
        front_wheel_back = add_coordinates(rotate_coordinates((-0.5, 0), float(state[2])), (length/2, 0))
        return [
            (-length/2, width/2), (length/2, width/2),
            (length/2, 0),
            front_wheel_back, front_wheel_front,
            (length/2, 0),
            (length/2, -width/2),
            (-length/2, -width/2),
        ]


    def get_start(self) -> np.ndarray:
        return self.initial_state

    def get_goal(self) -> np.ndarray | None:
        return self.goal_state

    def get_road(self) -> AbstractRoad | None:
        return self.road

    def get_position(self, state) -> Tuple[float, float]:
        return (
            state[0],
            state[1]
        )

    def get_orientation(self, state) -> float:
        return state[4]

    def get_speed(self, state) -> float:
        return state[3]

    def get_steering_angle(self, state) -> float:
        return state[2]


