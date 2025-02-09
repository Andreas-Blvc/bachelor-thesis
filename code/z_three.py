from z3 import *

# Variables
s = Real('s')
n = Real('n')
ds = Real('ds')
dn = Real('dn')
ut = Real('ut')
un = Real('un')

# Constants (replace these with your actual values)
C = 0.0003
s_min, s_max = 0, 20
n_min, n_max = -2, 2
v_x_min, v_x_max = 0, 10
v_y_min, v_y_max = -2, 2
a_x_min, a_x_max = -3, 6
a_y_min, a_y_max = -4, 4
dpsi_min, dpsi_max = -5, 5
apsi_min, apsi_max = -8, 8

# Define the main constraints (Z)
Z = And(
    v_x_min <= ds * (1 - n * C),
    ds * (1 + n * C) <= v_x_max,
    dpsi_min <= C * ds,
    C * ds <= dpsi_max,
    apsi_min <= C * ut,
    C * ut <= apsi_max,
    a_x_min <= (1 - n * C) * ut - 2 * dn * C * ds,
    (1 - n * C) * ut - 2 * dn * C * ds <= a_x_max,
    a_y_min <= un + C * ds * ds * (1 - n * C),
    un + C * ds * ds * (1 - n * C) <= a_y_max
)

# Define quantifiers
forall_constraint = ForAll([s, n, dn], Implies(
    And(
        s_min <= s, s <= s_max,
        n_min <= n, n <= n_max,
        v_y_min <= dn, dn <= v_y_max
    ),
    Z
))

# Solver setup
solver = Solver()

# Add the quantified constraint to the solver
solver.add(forall_constraint)

# Check satisfiability (whether there exists a solution that meets all conditions)
if solver.check() == sat:
    print("SAT (The conditions hold)")
    print("Example model:", solver.model())
else:
    print("UNSAT (The conditions do not hold for all variables)")
