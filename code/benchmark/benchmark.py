import os
import subprocess

from self_driving_cars import AbstractSelfDrivingCar
from visualizer import plot_states_or_inputs
from visualizer.vehicle_path_visualizer import animate


class Benchmark:
    def __init__(self, car: AbstractSelfDrivingCar, folder: str):
        self.path = f'benchmark-results/{folder}/'
        os.makedirs(self.path, exist_ok=True)
        self.car = car
        self.folder = folder
        self.animation_called = False

    def plot_car_states(self):
        plot_states_or_inputs(self.car.car_states, self.car.state_labels, self.car.planner.dt, "Car State")

    def plot_controls(self):
        plot_states_or_inputs(self.car.executed_controls, self.car.control_input_labels, self.car.planner.dt, "Car Control Inputs")

    def plot_predictive_car_states(self):
        plot_states_or_inputs(self.car.predictive_model_states, self.car.predictive_model.state_labels, self.car.planner.dt, "Car State")

    def plot_predictive_controls(self):
        plot_states_or_inputs(self.car.predictive_model_controls, self.car.predictive_model.control_input_labels, self.car.planner.dt, "Car Control Inputs")

    def save_animation(self):
        animate(self.car, interactive=False, save_only=True, path=self.path)
        self.animation_called = True

    def get_animation(self):
        return animate(self.car, interactive=False)

    def save_stats(self):
        if not self.animation_called:
            print("Stats are not available, because animation hat not been called.")
        plot_states_or_inputs(
            self.car.car_states,
            self.car.state_labels,
            self.car.planner.dt,
            "Car State",
            store_as_pgf=True,
            pgf_name=self.path + "car_states.pgf"
        )
        plot_states_or_inputs(
            self.car.executed_controls,
            self.car.control_input_labels,
            self.car.planner.dt,
          "Car Control Inputs",
            store_as_pgf=True,
            pgf_name=self.path + "car_controls.pgf"
        )
        plot_states_or_inputs(
            self.car.predictive_model_states,
            self.car.predictive_model.state_labels,
            self.car.planner.dt,
            "Car State",
            store_as_pgf=True,
            pgf_name=self.path + "predictive_model_states.pgf"
        )
        plot_states_or_inputs(
            self.car.predictive_model_controls,
            self.car.predictive_model.control_input_labels,
            self.car.planner.dt,
            "Car Control Inputs",
            store_as_pgf=True,
            pgf_name=self.path + "predictive_model_controls.pgf"
        )
        self.car.plot_metrics(store_as_pgf=True, pgf_name=self.path + "solver_metrics.pgf")

        # todo: solver-metrics -> solver_metrics
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

\includePGF{solver-metrics.pgf}

\end{document}
        """

        file_name = self.path + "stats.tex"
        with open(file_name, "w") as file:
            file.write(latex_content)

        subprocess.run(["pdflatex", " -interaction=nonstopmode", "helper.tex"], cwd=self.path, check=True)

        aux_extensions = [".aux", ".log", ".out", ".toc", ".tex"]
        for ext in aux_extensions:
            aux_file = os.path.join(self.path, f"stats{ext}")
            if os.path.exists(aux_file):
                os.remove(aux_file)

