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
u1, x3 = symbols("u1 x3")

# Define the constraints from the given logical formula
constraints = Or(
    And(u1 == 0, -10 + x3 <= 0, -x3 < 0),
    And(-6 + u1 == 0, -600 + 100*u1 - x3 == 0, -10 + x3 <= 0, -x3 <= 0,
        -600 + 100*u1 + x3 == 0, -1200 + 199*u1 + 2*x3 < 0, -4000 + 5*u1 + x3**2 <= 0),
    And(-6 + u1 < 0, -3 - u1 <= 0, x3 == 0),
    And(-6 + u1 <= 0, -3 - u1 <= 0, x3 == 0),
    And(-6 + u1 < 0, -u1 <= 0, x3 == 0),
    And(-6 + u1 <= 0, -u1 <= 0, x3 == 0),
    And(-6 + u1 < 0, -3 - u1 < 0, x3 == 0),
    And(-6 + u1 <= 0, -3 - u1 < 0, x3 == 0),
    And(-6 + u1 < 0, -3 - u1 <= 0, -600 - 199*u1 + 2*x3 <= 0, -300 - 100*u1 + x3 <= 0,
        -10 + x3 <= 0, -x3 <= 0, -600 + 100*u1 + x3 < 0, -1200 + 199*u1 + 2*x3 < 0,
        -4000 + 5*u1 + x3**2 <= 0),
    And(u1 <= 0, -3 - u1 <= 0, -300 - 100*u1 + x3 <= 0, -10 + x3 <= 0, -x3 < 0),
    And(-6 + u1 < 0, -u1 <= 0, -10 + x3 <= 0, -x3 <= 0, -600 + 100*u1 + x3 < 0,
        -4000 + 5*u1 + x3**2 <= 0),
    And(-6 + u1 <= 0, -u1 <= 0, -10 + x3 <= 0, -x3 <= 0, -600 + 100*u1 + x3 <= 0,
        -1200 + 199*u1 + 2*x3 < 0, -4000 + 5*u1 + x3**2 <= 0),
    And(-6 + u1 <= 0, -u1 <= 0, -10 + x3 <= 0, -x3 < 0, -600 + 100*u1 + x3 <= 0,
        -1200 + 199*u1 + 2*x3 < 0, -4000 + 5*u1 + x3**2 <= 0)
)

# Convert symbolic constraints into a function for evaluation
def check_constraints(u1_val, x3_val):
    """Evaluates whether a point (u1, x3) satisfies the constraints."""
    return constraints.subs({u1: u1_val, x3: x3_val})

# Generate grid for visualization
u1_vals = np.linspace(-4, 7, 400)
x3_vals = np.linspace(-1, 11, 400)
U1, X3 = np.meshgrid(u1_vals, x3_vals)

# Evaluate constraints on grid
Z = np.array([[1 if check_constraints(u, x) else 0 for u in u1_vals] for x in x3_vals])

# Plot the valid regions
plt.figure(figsize=(8, 6))
plt.contourf(U1, X3, Z, levels=[0.5, 1], colors=["blue"], alpha=0.5)
plt.xlabel("$u_1$")
plt.ylabel("$x_3$")
plt.title("Valid Region of Constraints")
plt.grid(True)

# Save as TikZ for LaTeX use
plt.savefig('region_plot_x1_u1.pgf', bbox_inches='tight')

# Show the plot
plt.show()
