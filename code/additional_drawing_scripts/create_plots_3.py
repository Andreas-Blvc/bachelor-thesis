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
u1, u2 = symbols("u1 u2")

# Define the constraints from the given logical formula
constraints = Or(
    And(u1 == 0, u2 < -4, u2 >= -(3399/800)),
    And(u1 == 0, u2 < 4, u2 >= -4),
    And(u1 == 0, u2 < 4, u2 >= 0),
    And(u1 == 0, u2 < 4, u2 >= 15/4),
    And(u1 == 0, u2 <= 15/4, u2 >= -(3399/800)),
    And(u1 < 6, u1 >= -3, u2 == 4),
    And(u1 < 6, u1 >= 0, u2 == 4),
    And(u1 <= 6, u1 >= 0, u2 == 4),
    And(u1 == 6, -23999*u1 + 2000*u1**2 <= -71200, u2 >= 0, -300*u1 + 25*u1**2 + u2 <= -896),
    And(u1 == 6, -23999*u1 + 2000*u1**2 <= -71200, -300*u1 + 25*u1**2 + u2 <= -896, -2388*u1 + 199*u1**2 + 8*u2 >= -7196),
    And(u1 < 0, u1 > -3, u2 < 4, u2 >= -4),
    And(u1 < 0, u1 > -3, u2 < 4, u2 >= 0),
    And(u1 < 0, u1 >= -(29/10), u2 <= 15/4, u2 >= -(3399/800)),
    And(u1 < 0, u1 >= -(29/10), u2 <= 15/4, u2 >= 0),
    And(u1 < 59/10, u1 >= -(29/10), u2 <= 15/4, u2 >= -(3399/800)),
    And(u1 < 59/10, u1 >= 0, u2 <= 15/4, u2 >= -(3399/800)),
    And(u1 < 6, u1 > -3, u2 <= 4, u2 >= -4),
    And(u1 < 6, u1 > -3, u2 <= 4, u2 >= 0),
    And(u1 < 6, u1 > 0, u2 == 4, -300*u1 + 25*u1**2 + u2 < -896),
    And(u1 < 6, u1 > 0, u2 == 4, -477600*u1 + 39601*u1**2 + 1600*u2 < -1433600),
    And(u1 < 6, u1 >= -3, 199*u1 - 16000*u2 == 223200, u2 == -4),
    And(u1 < 6, u1 >= -3, 199*u1 - 16000*u2 <= 223200, u2 == -4),
    And(u1 < 6, u1 >= -3, u2 <= 4, u2 >= -4),
    And(u1 < 6, u1 >= -3, u2 <= 4, u2 >= 0),
    And(u1 < 6, u1 >= 0, 199*u1 - 16000*u2 == 223200, u2 == -4),
    And(u1 < 6, u1 >= 0, u2 < 4, u2 >= -4),
    And(u1 < 6, u1 >= 0, u2 < 4, u2 >= 0),
    And(u1 < 6, u1 >= 0, 199*u1 - 16000*u2 <= 223200, u2 == -4),
    And(u1 < 6, u1 >= 0, u2 <= 4, u2 >= -4),
    And(u1 < 6, u1 >= 0, u2 <= 4, u2 >= 0)
)

# Convert symbolic constraints into a function for evaluation
def check_constraints(u1_val, u2_val):
    """Evaluates whether a point (u1, u2) satisfies the constraints."""
    return constraints.subs({u1: u1_val, u2: u2_val})

# Generate grid for visualization
u1_vals = np.linspace(-4, 7, 400)
u2_vals = np.linspace(-5, 5, 400)
U1, U2 = np.meshgrid(u1_vals, u2_vals)

# Evaluate constraints on grid
Z = np.array([[1 if check_constraints(u, v) else 0 for u in u1_vals] for v in u2_vals])

# Plot the valid regions
plt.figure(figsize=(8, 6))
plt.contourf(U1, U2, Z, levels=[0.5, 1], colors=["green"], alpha=0.5)
plt.xlabel("$u_1$")
plt.ylabel("$u_2$")
plt.title("Valid Region of Constraints in $(u_1, u_2)$ Plane")
plt.grid(True)

# Save as TikZ for LaTeX use
plt.savefig("region_plot_u1_u2.pgf",  bbox_inches='tight')

# Show the plot
plt.show()
