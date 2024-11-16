import matplotlib.pyplot as plt
import numpy as np


def plot_control_inputs(control_inputs, control_input_labels, dt, delta=1e-5):
    # Convert control_inputs to a numpy array for easy indexing
    control_inputs = np.array(control_inputs)
    time_steps = np.arange(len(control_inputs)) * dt  # Create time steps based on dt

    # Check if the number of labels matches the number of control input dimensions
    if control_inputs.shape[1] != len(control_input_labels):
        raise ValueError("Number of labels must match the number of control input dimensions.")

    # Set up the figure for plotting
    plt.figure(figsize=(10, 5))
    plt.suptitle("Control Inputs over Time")

    # Plot each control input
    for i in range(control_inputs.shape[1]):
        plt.subplot(control_inputs.shape[1], 1, i + 1)  # Dynamically set number of rows based on control inputs

        # Separate positive, negative, and near-zero values using the delta tolerance
        positive_values = np.where(control_inputs[:, i] > delta, control_inputs[:, i], np.nan)
        negative_values = np.where(control_inputs[:, i] < -delta, control_inputs[:, i], np.nan)
        zero_values = np.where((np.abs(control_inputs[:, i]) <= delta), control_inputs[:, i], np.nan)

        # Plot positive values in blue, negative values in red, and near-zero values in green
        plt.plot(time_steps, positive_values, color='blue', label=f"{control_input_labels[i]} (Positive)")
        plt.plot(time_steps, negative_values, color='red', label=f"{control_input_labels[i]} (Negative)")
        plt.plot(time_steps, zero_values, color='green', label=f"{control_input_labels[i]} (Zero)")

        plt.xlabel("Time [s]")
        plt.ylabel(control_input_labels[i])
        plt.legend()

    # plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.show()
