from sympy import symbols, Le

def get_z(
    Curvature=1 / 400,
    sMin=0,
    sMax=10,
    nMin=-0,
    nMax=2,
    vxMin=0,
    vxMax=10,
    vyMin=-2,
    vyMax=2,
    axMin=-3,
    axMax=6,
    ayMin=-4,
    ayMax=4,
    dpsiMin=-5,
    dpsiMax=5,
    apsiMin=-2,
    apsiMax=2,
):
    u1, u2, x1, x2, x3, x4 = symbols('u1 u2 x1 x2 x3 x4')
    return [
        Le(sMin, x1),
        Le(x1, sMax),
        Le(nMin, x2),
        Le(x2, nMax),
        Le(vyMin, x4),
        Le(x4, vyMax),

        Le(vxMin, x3 * (1 - x2 * Curvature)),
        Le(x3 * (1 - x2 * Curvature), vxMax),
        Le(dpsiMin, Curvature * x3),
        Le(Curvature * x3, dpsiMax),
        Le(apsiMin, Curvature * u1),
        Le(Curvature * u1, apsiMax),
        Le(axMin, (1 - x2 * Curvature) * u1 - 2 * x3 * Curvature * x4),
        Le((1 - x2 * Curvature) * u1 - 2 * x3 * Curvature * x4, axMax),
        Le(ayMin, u2 + Curvature * x3 ** 2 * (1 - x2 * Curvature)),
        Le(u2 + Curvature * x3 ** 2 * (1 - x2 * Curvature), ayMax)
    ], [x1, x2, x3, x4, u1, u2]
