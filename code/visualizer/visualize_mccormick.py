import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, fixed
from IPython.display import display

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
    Z_upper = np.minimum(
        np.minimum(x_U * Y + X * y_L - x_U * y_L, x_L * Y + X * y_U - x_L * y_U),
        np.minimum(x_U * Y + X * y_U - x_U * y_U, x_L * Y + X * y_L - x_L * y_L)
    )

    Z_lower = np.maximum(
        np.maximum(x_L * Y + X * y_L - x_L * y_L, x_U * Y + X * y_U - x_U * y_U),
        np.maximum(x_L * Y + X * y_U - x_L * y_U, x_U * Y + X * y_L - x_U * y_L)
    )

    # Start plotting
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the actual bilinear function
    surf = ax.plot_surface(X, Y, Z, alpha=0.8, cmap='viridis', edgecolor='none', label='Bilinear Function')

    # Plot McCormick envelopes (upper and lower bounds)
    # Upper bound plane
    ax.plot_surface(X, Y, Z_upper, alpha=0.3, color='red', edgecolor='none')
    # Lower bound plane
    ax.plot_surface(X, Y, Z_lower, alpha=0.3, color='blue', edgecolor='none')

    # Plot the feasible region as a volume between the upper and lower bounds
    # For better visualization, we can plot the edges of the feasible region
    ax.plot_wireframe(X, Y, Z_upper, color='red', alpha=0.1)
    ax.plot_wireframe(X, Y, Z_lower, color='blue', alpha=0.1)

    # Add contour plots on the bottom (x-y plane)
    cset = ax.contour(X, Y, Z, zdir='z', offset=np.min(Z_lower)-0.1*(np.max(Z_upper)-np.min(Z_lower)), cmap='viridis')

    # Labels and titles with mathematical expressions
    ax.set_title("Visualization of McCormick Envelopes for $z = x \cdot y$", fontsize=16)
    ax.set_xlabel("$x$", fontsize=14, labelpad=10)
    ax.set_ylabel("$y$", fontsize=14, labelpad=10)
    ax.set_zlabel("$z$", fontsize=14, labelpad=10)

    # Set the limits of the axes
    ax.set_xlim(x_L, x_U)
    ax.set_ylim(y_L, y_U)
    z_L = np.min(Z_lower)-0.1*(np.max(Z_upper)-np.min(Z_lower))
    z_U = np.max(Z_upper)+0.1*(np.max(Z_upper)-np.min(Z_lower))
    ax.set_zlim(z_L, z_U)

    # Adjust viewing angle for better visualization
    ax.view_init(elev=30, azim=-60)

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='purple', lw=4, label='Bilinear Function $z = x \cdot y$'),
        Line2D([0], [0], color='red', lw=4, label='McCormick Upper Bound'),
        Line2D([0], [0], color='blue', lw=4, label='McCormick Lower Bound')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12)

    # Show the plot
    plt.tight_layout()
    plt.legend()
    plt.show()


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
        continuous_update=True
    )

    interact(plot_for_y_fixed, y_fixed=y_slider)


