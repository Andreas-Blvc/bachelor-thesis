import os
import subprocess
import csv
import math

from path_planner import Objectives
from self_driving_cars import DynamicSingleTrackModel
from visualizer import plot_states_or_inputs
from visualizer.vehicle_path_visualizer import animate


class Benchmark:
    def __init__(self, car: DynamicSingleTrackModel, folder: str, road_name: str):
        self.path = f'benchmark-results/{folder}/'
        self.car = car
        self.folder = folder
        self.road_name = road_name
        self.animation_called = False

    def plot_car_states(self):
        plot_states_or_inputs(
            [state for state, _ in self.car.car_states],
            self.car.state_labels,
            [t for _, t in self.car.car_states],
            "Car State"
        )

    def plot_controls(self):
        plot_states_or_inputs(
            [control for control, _ in self.car.executed_controls],
            self.car.control_input_labels,
            [t for _, t in self.car.executed_controls],
            "Car Control Inputs"
        )

    def plot_predictive_car_states(self):
        plot_states_or_inputs(
            [state for state, _ in self.car.predictive_model_states],
            self.car.predictive_model.state_labels,
            [t for _, t in self.car.predictive_model_states],
            "Car State"
        )

    def plot_predictive_controls(self):
        plot_states_or_inputs(
            [control for control, _ in self.car.predictive_model_controls],
            self.car.predictive_model.control_input_labels,
            [t for _, t in self.car.predictive_model_controls],
            "Car Control Inputs"
        )

    def run_simulation(self):
        os.makedirs(self.path, exist_ok=True)
        simulation = list(self.car.drive())
        print(f"Done, amount car states: {len(simulation)}")

    def save_animation(self):
        os.makedirs(self.path, exist_ok=True)
        animate(self.car, interactive=False, save_only=True, path=self.path)
        self.animation_called = True

    def get_animation(self):
        return animate(self.car, interactive=False)

    def save_stats(self):
        if len(self.car.car_states) == 0:
            return
        plot_states_or_inputs(
            [state for state, _ in self.car.car_states],
            self.car.state_labels,
            [t for _, t in self.car.car_states],
            "Car State",
            store_as_pgf=True,
            pgf_name=self.path + "car_states.pgf"
        )
        plot_states_or_inputs(
            [control for control, _ in self.car.executed_controls],
            self.car.control_input_labels,
            [t for _, t in self.car.executed_controls],
          "Car Control Inputs",
            store_as_pgf=True,
            pgf_name=self.path + "car_controls.pgf"
        )
        plot_states_or_inputs(
            [state for state, _ in self.car.predictive_model_states],
            self.car.predictive_model.state_labels,
            [t for _, t in self.car.predictive_model_states],
            "Car State",
            store_as_pgf=True,
            pgf_name=self.path + "predictive_model_states.pgf"
        )
        plot_states_or_inputs(
            [control for control, _ in self.car.predictive_model_controls],
            self.car.predictive_model.control_input_labels,
            [t for _, t in self.car.predictive_model_controls],
            "Car Control Inputs",
            store_as_pgf=True,
            pgf_name=self.path + "predictive_model_controls.pgf"
        )
        self.car.plot_metrics(store_as_pgf=True, pgf_name=self.path + "solver_metrics.pgf")

        # todo solver-metrics -> solver_metrics
        latex_content = r"""
\documentclass[a4paper]{article}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{pgf}

% Custom command with adjustable scaling
\newcommand{\includePGF}[1]{%
    \resizebox{0.9\textwidth}{!}{\input{#1}}% Adjust scaling
}

\begin{document}

\includePGF{car_states.pgf}
\clearpage

\includePGF{car_controls.pgf}
\clearpage

\includePGF{predictive_model_states.pgf}
\clearpage

\includePGF{predictive_model_controls.pgf}
\vspace{1cm}

\includePGF{solver_metrics.pgf}

\end{document}
        """

        file_name = self.path + "stats.tex"
        with open(file_name, "w") as file:
            file.write(latex_content)

        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "stats.tex"],
            cwd=self.path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )

        aux_extensions = [".aux", ".log", ".out", ".toc", ".tex"]
        for ext in aux_extensions:
            aux_file = os.path.join(self.path, f"stats{ext}")
            if os.path.exists(aux_file):
                os.remove(aux_file)


    def save_to_csv(self):
        old_max = Objectives.max
        old_sum_squares = Objectives.sum_squares
        old_create_var = Objectives.create_var
        old_zero = Objectives.Zero
        Objectives.max = max
        Objectives.sum_squares = lambda vec: sum(x ** 2 for x in vec)
        Objectives.create_var = None
        Objectives.Zero = 0

        objective_val, objective_type, _, objective_name  = self.car.planner.get_objective(
            [self.car.convert_vec_to_state(vec) for vec, _ in self.car.car_states],
            [self.car.convert_vec_to_control_input(vec) for vec, _ in self.car.executed_controls],
        )


        # File paths
        animation_file = os.path.join(self.folder, "animation.mp4")
        stats_file = os.path.join(self.folder, "stats.pdf")

        # Ensure the file paths are properly encoded for URLs
        url_animation = f"http://wg-server.net:8000/{animation_file}"
        url_stats = f"http://wg-server.net:8000/{stats_file}"
        print(objective_val)
        new_data = [
            # model:
            self.car.predictive_model.get_name(),
            # dt:
            f"{self.car.dt(0)*1000:.2f}ms",
            # re-plan after
            self.car.N,
            # planning horizon:
            self.car.planner.time_horizon,
            # v_min [m/s]:
            self.car.velocity_range[0],
            # v_max [m/s]:
            self.car.velocity_range[1],
            # a_min [m/s^2]:
            self.car.acceleration_range[0],
            # a_max [m/s^2]:
            self.car.acceleration_range[1],
            # delta_min [radians]:
            self.car.steering_range[0],
            # delta_max [radians]:
            self.car.steering_range[1],
            # v_delta_min [radians/s]:
            self.car.steering_velocity_range[0],
            # v_delta_max [radians/s]:
            self.car.steering_velocity_range[1],
            # road name:
            self.road_name,
            # road completion:
            f"{100 * self.car.convert_vec_to_state(self.car.car_states[-1][0]).get_traveled_distance() / self.car.road.length:.2f}%"
            if len(self.car.car_states) > 0 else "0%",
            # objective name:
            f"{objective_name}",
            # objective value:
            f"{objective_val:.2f}",
            # avg solve time:
            f"{sum(self.car.solve_times) * 1000 / len(self.car.solve_times):.2f}ms"
            if len(self.car.car_states) > 0 else "-",
            # Standard deviation solve time:
            f"{math.sqrt(sum((x - sum(self.car.solve_times) / len(self.car.solve_times)) ** 2 for x in self.car.solve_times) / len(self.car.solve_times)) * 1000:.2f}ms"
            if len(self.car.car_states) > 0 else "-",
            # max solve time:
            f"{max(self.car.solve_times) * 1000:.2f}ms"
            if len(self.car.car_states) > 0 else "-",
            # min solve time:
            f"{min(self.car.solve_times) * 1000:.2f}ms"
            if len(self.car.car_states) > 0 else "-",
            # url animation:
            f'=HYPERLINK("{url_animation}"; "Animation")' if self.animation_called else "-",
            # url stats:
            f'=HYPERLINK("{url_stats}"; "Stats")',
        ]

        # Save to a CSV file
        filename = "output.csv"
        with open(filename, mode="a", newline="\n") as file:  # 'a' for append mode
            writer = csv.writer(file, delimiter=';')
            writer.writerow(new_data)

        Objectives.max = old_max
        Objectives.sum_squares = old_sum_squares
        Objectives.create_var = old_create_var
        Objectives.Zero = old_zero