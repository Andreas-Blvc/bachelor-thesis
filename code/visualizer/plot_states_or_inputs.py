import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_states_or_inputs(
        data, labels, time_points: List[float],
        title="States or Control Inputs over Time",
        delta=1e-5,
        store_as_pgf=False,
        pgf_name='output'
):
    """
    Plots states or control inputs with separation of positive, negative, and near-zero values.

    Parameters:
    - data: list or numpy array of shape (timesteps, num_dimensions).
    - labels: List of labels corresponding to each dimension.
    - time_points: Timepoints of the samples.
    - title: Title of the plot.
    - delta: Threshold for separating near-zero values.
    """
    if store_as_pgf:
        plt.rcParams.update({
            "text.usetex": True,  # Use LaTeX for rendering text
            "font.family": "serif",  # Use a serif font to match LaTeX
            "font.serif": ["Palatino"],  # Use Palatino to match your LaTeX document
            "pgf.texsystem": "pdflatex",  # Use pdflatex for .pgf output
            "pgf.rcfonts": False,  # Prevent matplotlib from overriding LaTeX fonts
            "font.size": 11,  # Set the font size to match your LaTeX document (11pt)
            "axes.titlesize": 11,  # Title font size to match the document
            "axes.labelsize": 11,  # Axis label font size
            "xtick.labelsize": 9,  # X-tick label size (slightly smaller than labels)
            "ytick.labelsize": 9,  # Y-tick label size (slightly smaller than labels)
            "legend.fontsize": 10,  # Legend font size
        })

    data = np.array(data)
    num_dimensions = data.shape[1] if len(data.shape) > 1 else 1
    time_steps = np.array(time_points)

    if num_dimensions != len(labels):
        raise ValueError("Number of labels must match the number of data dimensions.")

    # Limit subplots to a maximum of 5
    num_subplots = min(num_dimensions, 5)

    plt.figure(figsize=(10, 2 * num_subplots))
    if not store_as_pgf:
        plt.suptitle(title)

    for i in range(num_subplots):
        ax = plt.subplot(num_subplots, 1, i + 1)

        if num_dimensions > 1:
            positive_values = np.where(data[:, i] > delta, data[:, i], np.nan)
            negative_values = np.where(data[:, i] < -delta, data[:, i], np.nan)
            zero_values = np.where((np.abs(data[:, i]) <= delta), data[:, i], np.nan)
        else:
            positive_values = np.where(data[:] > delta, data[:], np.nan)
            negative_values = np.where(data[:] < -delta, data[:], np.nan)
            zero_values = np.where((np.abs(data[:]) <= delta), data[:], np.nan)

        ax.plot(time_steps, positive_values, color='blue', label="pos")
        ax.plot(time_steps, negative_values, color='red', label="neg")
        ax.plot(time_steps, zero_values, color='green', label="= 0")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(labels[i])
        ax.legend()

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    if store_as_pgf:
        plt.savefig(pgf_name, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
