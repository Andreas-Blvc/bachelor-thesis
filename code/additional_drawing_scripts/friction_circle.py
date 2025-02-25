import numpy as np
import matplotlib.pyplot as plt

# Define constants
v_star = 10  # max velocity, adjust as needed
delta_star = 0.698   # max steering angle, adjust as needed=
l_wb = 2.4  # wheelbase length
a_max = 11.5  # max acceleration

# Create meshgrid
samples = 1000
v_values = np.linspace(-18, 18, samples)
delta_values = np.linspace(-delta_star-0.5, delta_star+0.5, samples)
V, Delta = np.meshgrid(v_values, delta_values)

# Compute constraint values
a_values = np.linspace(0, 2, 2)  # Different values of a to visualize effect
fig, ax = plt.subplots()

for a_i in [-5]:
    # Make a_i broadcastable
    A = np.full_like(V, a_i)

    # Second Constraint (Original Comparison)
    constraint_friction_circle = np.sqrt(A**2 + ((V**2 / l_wb) * np.tan(Delta))**2)
    constraint_stricter = np.sqrt(A**2 + ((V**2 / l_wb) * (np.tan(delta_star)/delta_star * Delta))**2)

    # Diamond Constraint with w = a_max^2 - a^2
    C = (1/l_wb * np.tan(delta_star) / delta_star) ** 2
    C_ = np.sqrt((a_max ** 2 - a_i ** 2)/C)
    delta_ = lambda v: C_ * (1/v**2)
    d_v = (1 / v_star)
    d_delta = (1 / delta_star)
    v_ = (2*C_*d_delta/d_v)**(1/3)

    w = d_v * v_ + d_delta * delta_(v_)
    print(a_max + w ** (-1/4))
    print(w, d_v, d_delta)


    diamond_constraint_1 = d_v * V + d_delta * Delta - w
    diamond_constraint_2 = d_v * V - d_delta * Delta - w
    diamond_constraint_3 = -d_v * V + d_delta * Delta - w
    diamond_constraint_4 = -d_v * V - d_delta * Delta - w

    # Contour plot for the second constraint
    ax.contour(V, Delta, constraint_friction_circle, levels=[a_max], alpha=0.7, colors='r')
    ax.contour(V, Delta, constraint_stricter, levels=[a_max], alpha=0.7, colors='b')

    # Diamond set visualization (solid black lines)
    ax.contour(V, Delta, diamond_constraint_1, levels=[0], colors='black', linestyles='solid')
    ax.contour(V, Delta, diamond_constraint_2, levels=[0], colors='black', linestyles='solid')
    ax.contour(V, Delta, diamond_constraint_3, levels=[0], colors='black', linestyles='solid')
    ax.contour(V, Delta, diamond_constraint_4, levels=[0], colors='black', linestyles='solid')


ax.set_xlabel('Velocity (v)')
ax.set_ylabel('Steering Angle (δ)')
ax.set_title('Constraint Comparison with Diamond Set')
plt.show()


# test it

# import cvxpy as cp
# import numpy as np
# # Define constants
# c_1 = 0.9040595325562741  # Example value
# c_2 = 12.525535603937156  # Example value
# n = 2
#
# # Define variables
# a = cp.Variable()  # The variable in the denominator term
# w = cp.Variable()  # The main variable we want to optimize
#
# # Constraints
# constraints = [
#     0 <= w,
#     w <= c_1 - cp.power((-a + c_2), (-2*n)),
#     w <= c_1 - cp.power((a + c_2), (-2*n)),
#     # a <= 12.5,
#     # a**2 <= 11.5**2,
# ]
#
# # Define an example objective function (can be changed)
# objective = cp.Minimize(w + a)  # Example: Minimize w
#
# # Solve the problem
# problem = cp.Problem(objective, constraints)
# problem.solve()
#
# # Print results
# print("Optimal a:", a.value)
# print("Optimal w:", w.value)


