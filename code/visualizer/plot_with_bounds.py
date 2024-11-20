import matplotlib.pyplot as plt


def plot_with_bounds(
    y_label,
    bounds,
    y_values_list,
    x_values=None,
    y_labels=None
):
    """
    Plots multiple functions with shared bounds for each x_value.

    Parameters:
    - y_label (str): The label for the y-axis.
    - bounds (list or array of tuples): The lower-,upper-bound values for each x_value.
    - y_values_list (list of lists): Each entry contains multiple y-values (one list per x_value).
    - x_values (list or array, optional): The x-axis values. If None, the index will be used.
    - y_labels (list of str, optional): The labels for each function. Defaults to "Function i".
    """
    # Create a new figure
    plt.figure()

    # Determine x_values if not provided
    if x_values is None:
        x_values = range(len(bounds))

    # Validate dimensions
    if len(bounds) != len(y_values_list):
        raise ValueError("Length of bounds and y_values_list must match x_values.")

    # Default function labels
    if y_labels is None:
        num_functions = len(y_values_list[0])  # Number of functions per x_value
        y_labels = [f"Function {i + 1}" for i in range(num_functions)]

    # Extract number of functions
    num_functions = len(y_values_list[0])

    # Prepare arrays for each function across all x_values
    function_values = [[] for _ in range(num_functions)]
    for y_values in y_values_list:
        for i, value in enumerate(y_values):
            function_values[i].append(value)

    # Plot each function
    for i, func_values in enumerate(function_values):
        plt.plot(x_values, func_values, label=y_labels[i])

    # Fill the area between lower and upper bounds
    lower_bounds, upper_bounds = tuple([list(group) for group in zip(*bounds)])
    plt.fill_between(
        x_values,
        lower_bounds,
        upper_bounds,
        color="green",
        alpha=0.2,
        label="Bounds"
    )

    # Add horizontal lines for bounds (optional for better clarity)
    plt.plot(x_values, lower_bounds, linestyle=":", color="red", label="Lower Bound")
    plt.plot(x_values, upper_bounds, linestyle="--", color="red", label="Upper Bound")

    # Add labels, legend, and grid
    plt.xlabel("X-axis")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, linestyle=":", color="gray", alpha=0.7)
    plt.show()
