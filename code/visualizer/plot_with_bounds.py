import matplotlib.pyplot as plt


def plot_with_bounds(
    y_label,
    y_values_list,
    bounds=None,
    x_values=None,
    y_labels=None,
    dt=1,
    store_as_pgf=False,
    pgf_name='output',
    no_bounds=False,
):
    """
    Plots multiple functions with shared bounds for each x_value.

    Parameters:
    - y_label (str): The label for the y-axis.
    - bounds (list or array of tuples): The lower-,upper-bound values for each x_value.
    - y_values_list (list of lists): Each entry contains multiple y-values (one list per x_value).
    - x_values (list or array, optional): The x-axis values. If None, the index*dt will be used.
    - y_labels (list of str, optional): The labels for each function. Defaults to "Function i".
    """
    if bounds is None:
        bounds = []
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

    # Create a new figure
    plt.figure()

    # Determine x_values if not provided
    if x_values is None:
        x_values = [i * dt for i in range(len(y_values_list))]

    # Validate dimensions
    if not no_bounds and len(bounds) != len(y_values_list):
        raise ValueError("Length of bounds and y_values_list must match x_values.")

    if not no_bounds:
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

    # Add labels, legend, and grid
    plt.xlabel("X-axis")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, linestyle=":", color="gray", alpha=0.7)
    if store_as_pgf:
        plt.savefig(pgf_name, bbox_inches='tight')
    plt.show()
