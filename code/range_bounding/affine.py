from sympy import symbols, Le

from utils import StateRanges, MainPaperConstraintsReduction, optimal_range_bound


def affine_ranges(
    Curvature = 1/200,
    sMin = 0,
    sMax = 10,
    nMin = -4,
    nMax = 4,
    vxMin = 0,
    vxMax = 30,
    vyMin = -2,
    vyMax = 2,
    axMin = -3,
    axMax = 6,
    ayMin = -4,
    ayMax = 4,
    dpsiMin = -3,
    dpsiMax = 3,
    apsiMin = -2,
    apsiMax = 2,
):
    u1, u2, x1, x2, x3, x4 = symbols('u1 u2 x1 x2 x3 x4')
    ranges = StateRanges(
        n=(nMin, nMax),
        c=(Curvature, Curvature),
        ds=None,
        dn=(vyMin, vyMax),
        u_n=None,
        u_t=None,
    )
    # updates ranges in place
    MainPaperConstraintsReduction.apply_all(
        state_ranges=ranges,
        v_x_range=(vxMin, vxMax),
        acc_x_range=(axMin, axMax),
        acc_y_range=(ayMin, ayMax),
        yaw_rate_range=(dpsiMin, dpsiMax),
        yaw_acc_range=(apsiMin, apsiMax),
        curvature_derivative=0  # constant for all s
    )

    return [
        Le(sMin, x1), Le(x1, sMax),
        Le(ranges.n[0], x2), Le(x2, ranges.n[1]),
        Le(ranges.ds[0], x3), Le(x3, ranges.ds[1]),
        Le(ranges.dn[0], x4), Le(x4, ranges.dn[1]),
        Le(ranges.u_t[0], u1), Le(u1, ranges.u_t[1]),
        Le(ranges.u_n[0], u2), Le(u2, ranges.u_n[1]),
    ], [x1, x2, x3, x4, u1, u2]

if __name__ == "__main__":
    res1, symbols1 = affine_ranges()
    print(res1)
