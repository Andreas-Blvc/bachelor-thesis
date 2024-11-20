import matplotlib.pyplot as plt
import numpy as np
from matplotlib import use


def plot_states_or_inputs(data, labels, dt, title="States or Control Inputs over Time", delta=1e-5):
    """
    Plots states or control inputs with separation of positive, negative, and near-zero values.

    Parameters:
    - data: 2D list or numpy array of shape (timesteps, num_dimensions).
    - labels: List of labels corresponding to each dimension.
    - dt: Time step between samples.
    - title: Title of the plot.
    - delta: Threshold for separating near-zero values.
    """
    data = np.array(data)
    num_dimensions = data.shape[1]
    time_steps = np.arange(data.shape[0]) * dt

    if num_dimensions != len(labels):
        raise ValueError("Number of labels must match the number of data dimensions.")

    # Limit subplots to a maximum of 5
    num_subplots = min(num_dimensions, 5)

    plt.figure(figsize=(10, 2 * num_subplots))
    plt.suptitle(title)

    for i in range(num_subplots):
        ax = plt.subplot(num_subplots, 1, i + 1)

        positive_values = np.where(data[:, i] > delta, data[:, i], np.nan)
        negative_values = np.where(data[:, i] < -delta, data[:, i], np.nan)
        zero_values = np.where((np.abs(data[:, i]) <= delta), data[:, i], np.nan)

        ax.plot(time_steps, positive_values, color='blue', label=f"{labels[i]} (Positive)")
        ax.plot(time_steps, negative_values, color='red', label=f"{labels[i]} (Negative)")
        ax.plot(time_steps, zero_values, color='green', label=f"{labels[i]} (Zero)")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(labels[i])
        ax.legend()

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.show()
