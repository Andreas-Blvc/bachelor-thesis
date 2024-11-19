import matplotlib.pyplot as plt


def plot_with_bounds(y_label, lower_bound, upper_bound, y_values, x_values=None):
    """
    Plots values with shaded regions for upper and lower bounds, with dynamic y-axis scaling.

    Parameters:
    - y_label (str): The label for the y-axis.
    - lower_bound (float): The lower bound value.
    - upper_bound (float): The upper bound value.
    - y_values (list or array): The data values to plot.
    - x_values (list or array, optional): The x-axis values. If None, the index will be used.
    """
    # Create a new figure to separate this plot from others
    plt.figure()

    # Create an index for x-axis
    if x_values is None:
        x_values = range(len(y_values))

    # Plot the values
    plt.plot(x_values, y_values, label=y_label, color="blue")

    # Adjust y-axis limits based on y_values with a small margin for better visibility
    y_min, y_max = min(y_values), max(y_values)
    margin = 0.1 * (y_max - y_min) if y_max != y_min else 1  # Add a 10% margin, fallback if range is 0
    plt.ylim(y_min - margin, y_max + margin)

    # Fill the area between lower and upper bounds
    plt.fill_between(x_values, lower_bound, upper_bound, color="green", alpha=0.2, label="Bounds")

    # Add horizontal lines for bounds
    plt.axhline(lower_bound, color="red", linestyle=":", label="Lower Bound")
    plt.axhline(upper_bound, color="red", linestyle="--", label="Upper Bound")

    # Add labels, legend, and grid
    plt.xlabel("Index")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, linestyle=":", color="gray", alpha=0.7)
    plt.show()
