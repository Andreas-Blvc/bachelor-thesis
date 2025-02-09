from sage.all import *

var('s', 'n', 'ds', 'dn', 'ut', 'un')

qf = qepcad_formula

s_min = 0
s_max = 20
n_min = -2
n_max = 2
v_x_min = 0
v_x_max = 10
v_y_min = -2
v_y_max = 2
a_x_min = -3
a_x_max = 6
a_y_min = -4
a_y_max = 4
C = 0.0003
dpsi_min = -5
dpsi_max = 5
apsi_min = -8
apsi_max = 8

term1 = (1 - n * C)
Z = qf.and_(
    v_x_min <= ds * term1,
    ds * term1 <= v_x_max,
    dpsi_min <= C * ds,
    C * ds <= dpsi_max,
    apsi_min <= C * ut,
    C * ut <= apsi_max,
    a_x_min <= term1 * ut - 2 * dn * C * ds,
    term1 * ut - 2 * dn * C * ds <= a_x_max,
    a_y_min <= un + C * ds * ds * term1,
    un + C * ds * ds * term1 <= a_y_max
)



F = qf.forall(
    s,
    qf.implies(
        qf.and_(
            s_min <= s,
            s <= s_max
        ),
        Z
    )
)
print("Formula", F)
# Use options to reduce QEPCAD overhead
result = qepcad(F)
print("Result:", result)
