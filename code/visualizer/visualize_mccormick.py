import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, fixed


def visualize_mccormick(x_bounds, y_bounds, resolution=50):
    """
    Visualize the McCormick envelopes for a bilinear term z = x * y.

    Parameters:
    - x_bounds: tuple (x_L, x_U), bounds for x.
    - y_bounds: tuple (y_L, y_U), bounds for y.
    - resolution: int, number of points for x and y grids.
    """
    x_L, x_U = x_bounds
    y_L, y_U = y_bounds

    # Generate grid for x and y
    x = np.linspace(x_L, x_U, resolution)
    y = np.linspace(y_L, y_U, resolution)
    X, Y = np.meshgrid(x, y)
    Z = X * Y  # Actual bilinear function

    # Compute McCormick bounds
    # All possible bounds
    Z_bound_1 = x_U * Y + X * y_L - x_U * y_L
    Z_bound_2 = x_L * Y + X * y_U - x_L * y_U
    Z_bound_3 = x_U * Y + X * y_U - x_U * y_U
    Z_bound_4 = x_L * Y + X * y_L - x_L * y_L

    Z_upper = np.minimum.reduce([Z_bound_1, Z_bound_2])
    Z_lower = np.maximum.reduce([Z_bound_3, Z_bound_4])

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 12))  # Set a square figure size, 12x12

    # First heatmap: Difference to upper bound
    Z_diff_upper = Z_upper - Z
    im1 = axes[0].imshow(Z_diff_upper, extent=[x.min(), x.max(), y.min(), y.max()],
                         origin='lower', cmap='cividis', interpolation='bilinear')
    axes[0].set_title("Difference to upper bound")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect(abs((x_U - x_L) / (y_U - y_L)))  # Adjust aspect ratio to make the figure look square
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)  # Add colorbar for the first heatmap with adjusted height

    # Second heatmap: Difference to lower bound
    Z_diff_lower = Z - Z_lower
    im2 = axes[1].imshow(Z_diff_lower, extent=[x.min(), x.max(), y.min(), y.max()],
                         origin='lower', cmap='cividis', interpolation='bilinear')
    axes[1].set_title("Difference to lower bound")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect(abs((x_U - x_L) / (y_U - y_L)))  # Adjust aspect ratio to make the figure look square
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)  # Add colorbar for the second heatmap with adjusted height

    # Adjust layout
    plt.tight_layout()
    plt.show()


def _visualize_mccormick_export_latex(x_bounds, y_bounds, resolution=50):
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

    x_L, x_U = x_bounds
    y_L, y_U = y_bounds
    aspect_ratio = abs((x_U - x_L) / (y_U - y_L))  # Calculate aspect ratio based on bounds

    # Generate grid for x and y
    x = np.linspace(x_L, x_U, resolution)
    y = np.linspace(y_L, y_U, resolution)
    X, Y = np.meshgrid(x, y)
    Z = X * Y  # Actual bilinear function

    # Compute McCormick upper and lower bounds
    Z_bound_1 = x_U * Y + X * y_L - x_U * y_L
    Z_bound_2 = x_L * Y + X * y_U - x_L * y_U
    Z_upper = np.minimum(Z_bound_1, Z_bound_2)
    Z_diff_upper = Z_upper - Z

    # Plot Difference to Upper Bound
    plt.figure(figsize=(5, 4))
    plt.imshow(Z_diff_upper, extent=(x.min(), x.max(), y.min(), y.max()),
               origin='lower', cmap='cividis', interpolation='bilinear', aspect=aspect_ratio)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("mccormick-bounds-1-upper.pgf", bbox_inches='tight')
    plt.close()

    # Compute McCormick lower bounds
    Z_bound_3 = x_U * Y + X * y_U - x_U * y_U
    Z_bound_4 = x_L * Y + X * y_L - x_L * y_L
    Z_lower = np.maximum(Z_bound_3, Z_bound_4)
    Z_diff_lower = Z - Z_lower

    # Plot Difference to Lower Bound
    plt.figure(figsize=(5, 4))
    plt.imshow(Z_diff_lower, extent=(x.min(), x.max(), y.min(), y.max()),
               origin='lower', cmap='cividis', interpolation='bilinear', aspect=aspect_ratio)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("mccormick-bounds-1-lower.pgf", bbox_inches='tight')
    plt.close()


def visualize_mccormick_2d_interactive(x_bounds, y_bounds, resolution=100):
    """
    Create an interactive 2D visualization of the McCormick envelopes for z = x * y,
    where y_fixed can be adjusted using a slider.

    Parameters:
    - x_bounds: tuple (x_L, x_U), bounds for x.
    - y_bounds: tuple (y_L, y_U), bounds for y.
    - resolution: int, number of points for x.
    """
    x_L, x_U = x_bounds
    y_L, y_U = y_bounds

    def plot_for_y_fixed(y_fixed):
        x = np.linspace(x_L, x_U, resolution)
        y = y_fixed

        # Compute actual bilinear function
        z = x * y

        # Compute McCormick bounds for fixed y
        z_upper = np.minimum(
            x_U * y + x * y_L - x_U * y_L,
            x_L * y + x * y_U - x_L * y_U
        )
        z_lower = np.maximum(
            x_L * y + x * y_L - x_L * y_L,
            x_U * y + x * y_U - x_U * y_U
        )

        # Clear the current figure
        plt.figure(figsize=(10, 6))
        plt.clf()

        # Plot actual bilinear function
        plt.plot(x, z, label=r'$z = x \cdot y$', color='green')

        # Plot McCormick upper and lower bounds
        plt.plot(x, z_upper, label='McCormick Upper Bound', color='red', linestyle='--')
        plt.plot(x, z_lower, label='McCormick Lower Bound', color='blue', linestyle='--')

        # Shade the feasible region
        plt.fill_between(x, z_lower, z_upper, color='gray', alpha=0.3, label='Feasible Region')

        # Labels and title
        plt.title(f"McCormick Envelopes for $z = x \\cdot y$ with $y = {y_fixed:.2f}$", fontsize=14)
        plt.xlabel("$x$", fontsize=12)
        plt.ylabel("$z$", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.show()

    # Create an interactive slider for y_fixed
    y_slider = FloatSlider(
        value=(y_L + y_U) / 2,
        min=y_L,
        max=y_U,
        step=(y_U - y_L) / 100,
        description='y_fixed',
        continuous_update=False
    )

    interact(plot_for_y_fixed, y_fixed=y_slider)
