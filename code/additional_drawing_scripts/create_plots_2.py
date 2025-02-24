import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, And, Or

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

# Define variables
u2, x3 = symbols("u2 x3")

# Define the constraints from the given logical formula
constraints = Or(
    And(-u2 <= 0, x3 == 0, -1600 + 400*u2 + x3**2 <= 0),
    And(x3 == 0, -1600 + 400*u2 + x3**2 <= 0, -320000 - 80000*u2 - 199*x3**2 <= 0),
    And(x3 == 0, -320000 + 80000*u2 + 199*x3**2 <= 0, -320000 - 80000*u2 - 199*x3**2 <= 0),
    And(-10 + x3 <= 0, -x3 < 0, -4000 + x3**2 == 0, -1600 + 400*u2 + x3**2 <= 0, -320000 - 80000*u2 - 199*x3**2 <= 0),
    And(-10 + x3 <= 0, -x3 <= 0, -1600 + 400*u2 + x3**2 <= 0, -79400 - x3 + 20*x3**2 == 0,
        -79400 + x3 + 20*x3**2 == 0, -320000 - 80000*u2 - 199*x3**2 <= 0, 790000 + 10*x3 - 199*x3**2 < 0),
    And(-u2 <= 0, -10 + x3 <= 0, -x3 <= 0, -1600 + 400*u2 + x3**2 <= 0),
    And(-u2 <= 0, -10 + x3 <= 0, -x3 < 0, -1600 + 400*u2 + x3**2 <= 0),
    And(-10 + x3 <= 0, -x3 <= 0, -4000 + x3**2 <= 0, -1600 + 400*u2 + x3**2 <= 0, -320000 - 80000*u2 - 199*x3**2 <= 0),
    And(-10 + x3 <= 0, -x3 <= 0, -1600 + 400*u2 + x3**2 <= 0, -79400 - x3 + 20*x3**2 <= 0, -320000 - 80000*u2 - 199*x3**2 <= 0),
    And(-10 + x3 <= 0, -x3 <= 0, -1600 + 400*u2 + x3**2 <= 0, -80300 + x3 + 20*x3**2 <= 0, -320000 - 80000*u2 - 199*x3**2 <= 0),
    And(-10 + x3 <= 0, -x3 < 0, -4000 + x3**2 <= 0, -1600 + 400*u2 + x3**2 <= 0, -320000 - 80000*u2 - 199*x3**2 <= 0),
    And(-10 + x3 <= 0, -x3 < 0, -1600 + 400*u2 + x3**2 <= 0, -79400 - x3 + 20*x3**2 <= 0, -320000 - 80000*u2 - 199*x3**2 <= 0),
    And(-10 + x3 <= 0, -x3 < 0, -1600 + 400*u2 + x3**2 <= 0, -80300 + x3 + 20*x3**2 <= 0, -320000 - 80000*u2 - 199*x3**2 <= 0)
)

# Convert symbolic constraints into a function for evaluation
def check_constraints(u2_val, x3_val):
    """Evaluates whether a point (u2, x3) satisfies the constraints."""
    return constraints.subs({u2: u2_val, x3: x3_val})

# Generate grid for visualization
u2_vals = np.linspace(-5, 5, 400)
x3_vals = np.linspace(-1, 11, 400)
U2, X3 = np.meshgrid(u2_vals, x3_vals)

# Evaluate constraints on grid
Z = np.array([[1 if check_constraints(u, x) else 0 for u in u2_vals] for x in x3_vals])

# Plot the valid regions
plt.figure(figsize=(8, 6))
plt.contourf(U2, X3, Z, levels=[0.5, 1], colors=["red"], alpha=0.5)
plt.xlabel("$u_2$")
plt.ylabel("$x_3$")
plt.title("Valid Region of Constraints")
plt.grid(True)

# Save as TikZ for LaTeX use
plt.savefig("region_plot_u2_x3.pgf",  bbox_inches='tight')

# Show the plot
plt.show()
