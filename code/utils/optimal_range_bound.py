import sys
from functools import reduce
from itertools import product
from typing import Tuple
import casadi as ca

from utils import StateRanges


def _calculate_product_range(*ranges: Tuple[float, float]) -> Tuple[float, float]:
    """
    Calculate the minimum and maximum product of multiple ranges.

    Args:
        *ranges (Tuple[float, float]): Variable number of tuples, each containing the min and max of a range.

    Returns:
        Tuple[float, float]: The minimum and maximum product.
    """
    if not ranges:
        raise ValueError("At least one range must be provided.")

    combinations = product(*[(range_[0], range_[1]) for range_ in ranges])

    # Calculate all possible products
    products = []
    for combo in combinations:
        product_val = 1
        for num in combo:
            product_val *= num
        products.append(product_val)

    # Reduce the list to find symbolic min and max
    min_val = reduce(lambda acc, x: ca.fmin(acc, x), products)
    max_val = reduce(lambda acc, x: ca.fmax(acc, x), products)

    return min_val, max_val


def _affine_range_bounding(slope_range, intercept_range, lower_bound, upper_bound):
    """
        Calculates the largest set X such that, for all slopes a and intercepts b in the given ranges:
        lower_bound <= ax+b <= upper_bound holds for all x in X
        """
    slope_min, slope_max = slope_range
    intercept_min, intercept_max = intercept_range

    condition_0 = ca.logic_and(ca.logic_and(slope_min <= 0, 0 <= slope_max), ca.logic_or(intercept_min < lower_bound, intercept_max > upper_bound))
    range_0_lower, range_0_upper = (0, 0)

    # Case Distinction:
    #  slope_min = slope_max:
    #  1. slope_min = slope_max = 0
    #  2. slope_min = slope_max < 0
    #  3. slope_min = slope_max > 0
    #  slope_min < slope_max:
    #  4. slope_max < 0
    #  5. 0 < slope_min
    #  6. slope_max = 0
    #  7. 0 = slope_min
    #  8. slope_min < 0 < slope_max

    # 1.
    condition_1 = ca.logic_and(slope_min == slope_max, slope_max == 0)
    range_1_lower = ca.if_else(
        # lower_bound <= b <= upper_bound -> either for all x (no range) or for no x ([0, 0])
        ca.logic_or(intercept_min < lower_bound, intercept_max > upper_bound),
        ca.MX(0),
        ca.MX.nan()
    )
    range_1_upper = ca.if_else(
        # lower_bound <= b <= upper_bound -> either for all x (no range) or for no x ([0, 0])
        ca.logic_or(intercept_min < lower_bound, intercept_max > upper_bound),
        ca.MX(0),  # True branch: symbolic zero
        ca.MX.nan()
    )

    # 2.
    condition_2 = ca.logic_and(slope_min == slope_max, slope_max < 0)
    x_lb = (upper_bound - intercept_max) / slope_max
    x_ub = (lower_bound - intercept_min) / slope_max
    range_2_lower = ca.if_else(
        x_lb <= x_ub,
        x_lb,
        ca.MX(0),
    )
    range_2_upper = ca.if_else(
        x_lb <= x_ub,
        x_ub,
        ca.MX(0),
    )

    # 3.
    condition_3 = ca.logic_and(slope_min == slope_max, slope_max > 0)
    x_lb = (lower_bound - intercept_min) / slope_max
    x_ub = (upper_bound - intercept_max) / slope_max
    range_3_lower = ca.if_else(
        x_lb <= x_ub,
        x_lb,
        ca.MX(0)
    )
    range_3_upper = ca.if_else(
        x_lb <= x_ub,
        x_ub,
        ca.MX(0)
    )

    # 4.
    condition_4 = slope_max < 0
    x_lb = ca.if_else(
        upper_bound - intercept_max < 0,
        (upper_bound - intercept_max) / slope_max,
        (upper_bound - intercept_max) / slope_min
    )
    x_ub = ca.if_else(
        lower_bound - intercept_min < 0,
        (lower_bound - intercept_min) / slope_max,
        (lower_bound - intercept_min) / slope_min
    )
    range_4_lower = ca.if_else(
        x_lb <= x_ub,
        x_lb,
        ca.MX(0)
    )
    range_4_upper = ca.if_else(
        x_lb <= x_ub,
        x_ub,
        ca.MX(0)
    )

    # 5.
    condition_5 = 0 < slope_min
    x_lb = ca.if_else(
        lower_bound - intercept_min > 0,
        (lower_bound - intercept_min) / slope_min,
        (lower_bound - intercept_min) / slope_max
    )
    x_ub = ca.if_else(
        upper_bound - intercept_max > 0,
        (upper_bound - intercept_max) / slope_min,
        (upper_bound - intercept_max) / slope_max
    )
    range_5_lower = ca.if_else(
        x_lb <= x_ub,
        x_lb,
        ca.MX(0)
    )
    range_5_upper = ca.if_else(
        x_lb <= x_ub,
        x_ub,
        ca.MX(0)
    )

    #  6. 7. 8.
    # lower_bound < intercept_min and intercept_max < upper_bound:
    condition_6 = ca.logic_or(intercept_min < lower_bound, intercept_max > upper_bound)
    range_6_lower, range_6_upper = (float('nan'), float('nan'))


    condition_7 = ca.logic_and(slope_min < 0, 0 < slope_max)
    range_7_lower = ca.fmax(
                ca.fmin(0, (lower_bound - intercept_min) / slope_max), ca.fmin(0, (upper_bound - intercept_max) / slope_min)
            )
    range_7_upper = ca.fmin(
                ca.fmax(0, (upper_bound - intercept_max) / slope_max), ca.fmax(0, (lower_bound - intercept_min) / slope_min)
            )

    condition_8 = slope_min < 0
    range_8_lower, range_8_upper = (
            ca.fmin(0, (upper_bound - intercept_max) / slope_min),
            ca.fmax(0, (lower_bound - intercept_min) / slope_min)
        )

    condition_9 = slope_max > 0
    range_9_lower, range_9_upper = (
            ca.fmin(0, (lower_bound - intercept_min) / slope_max),
            ca.fmax(0, (upper_bound - intercept_max) / slope_max)
        )


    # prettiest code ever:
    conditions = [condition_0, condition_1, condition_2, condition_3, condition_4, condition_5, condition_6, condition_7, condition_8, condition_9]
    lower_bounds = [range_0_lower, range_1_lower, range_2_lower, range_3_lower, range_4_lower, range_5_lower, range_6_lower, range_7_lower, range_8_lower, range_9_lower]
    upper_bounds = [range_0_upper, range_1_upper, range_2_upper, range_3_upper, range_4_upper, range_5_upper, range_6_upper, range_7_upper, range_8_upper, range_9_upper]
    return (
        reduce(lambda acc, t: ca.if_else(t[0], t[1], acc), zip(conditions[::-1], lower_bounds[::-1]), float('nan')),
        reduce(lambda acc, t: ca.if_else(t[0], t[1], acc), zip(conditions[::-1], upper_bounds[::-1]), float('nan'))
    )



def optimal_range_bound(road_width_range, v_x_range, v_y_range, yaw_rate_range, yaw_acc_range, a_x_range, a_y_range, curvature) -> StateRanges:
    opti = ca.Opti()
    new_range = lambda: (opti.variable(), opti.variable())
    n = new_range()
    ds = new_range()
    dn = new_range()
    u_t = new_range()
    u_n = new_range()
    contained_in = lambda inner, outer: [outer[0] <= inner[0], inner[1] <= outer[1]]

    a_1 = (
        1 - n[1] * curvature,
        1 - n[0] * curvature
    ) if curvature > 0 else (
        1 - n[0] * curvature,
        1 - n[1] * curvature
    )
    b_1 = (0, 0)

    a_3 = a_1
    dn_ds = _calculate_product_range(dn, ds)
    b_3 = (
        -2 * curvature * dn_ds[1],
        -2 * curvature * dn_ds[0]
    ) if curvature > 0 else (
        -2 * curvature * dn_ds[0],
        -2 * curvature * dn_ds[1],
    )

    a_4 = (1, 1)
    ds_ds = _calculate_product_range(ds, ds)
    dn_ds_ds = _calculate_product_range(dn, ds, ds)
    b_4 = (
        curvature * ds_ds[0] - curvature**2 * dn_ds_ds[1],
        curvature * ds_ds[1] - curvature**2 * dn_ds_ds[0],
    ) if curvature > 0 else (
        curvature * ds_ds[1] - curvature**2 * dn_ds_ds[1],
        curvature * ds_ds[0] - curvature**2 * dn_ds_ds[0],
    )

    constraints = [
        *contained_in(n, road_width_range),
        *contained_in(dn, v_y_range),
        *([
            *contained_in((curvature * ds[0], curvature * ds[1]), yaw_rate_range),
            *contained_in((curvature * u_t[0], curvature * u_t[1]), yaw_acc_range),
        ] if curvature != 0 else []),
        *contained_in(ds, _affine_range_bounding(a_1, b_1, v_x_range[0], v_x_range[1])),
        *contained_in(u_t, _affine_range_bounding(a_3, b_3, a_x_range[0], a_x_range[1])),
        *contained_in(u_n, _affine_range_bounding(a_4, b_4, a_y_range[0], a_y_range[1])),
        # thresholds:
        n[1] - n[0] > 1,
        dn[1] - dn[0] > .1,
        ds[1] - ds[0] > 2,
        u_t[1] - u_t[0] > .1,
        u_n[1] - u_n[0] > .1,
    ]

    for c in constraints:
        opti.subject_to(c)

    opti.minimize(
        -1 * (n[1] - n[0]) ** 2 +
        -10 * (ds[1] - ds[0]) ** 2 +
        -1 * (dn[1] - dn[0]) ** 2 +
        -1 * (u_t[1] - u_t[0]) ** 2 +
        -1 * (u_n[1] - u_n[0]) ** 2
    )

    p_opts = {"expand": True}
    s_opts = {"max_iter": 5000, "print_level": 0}
    opti.solver('ipopt', p_opts, s_opts)

    with open('/dev/null', 'w') as output_file:
        stdout_old = sys.stdout
        sys.stdout = output_file
        try:
            solution = opti.solve()
            ranges_optimal = StateRanges(
                c=(curvature, curvature),
                n=(solution.value(n[0]), solution.value(n[1])),
                ds=(solution.value(ds[0]), solution.value(ds[1])),
                dn=(solution.value(dn[0]), solution.value(dn[1])),
                u_t=(solution.value(u_t[0]), solution.value(u_t[1])),
                u_n=(solution.value(u_n[0]), solution.value(u_n[1])),
            )
        except RuntimeError:
            ranges_optimal = None
        finally:
            sys.stdout = stdout_old

    return ranges_optimal



if __name__ == '__main__':
    v_x_range = (0, 15)
    v_y_range = (-4, 4)
    acc_x_range = (-6, 3)
    acc_y_range = (-16, 16)
    yaw_rate_range = (-4, 4)
    yaw_acc_range = (-8, 8)
    optimal_range_bound(
        (-2 , 2),
        v_x_range,
        v_y_range,
        yaw_rate_range,
        yaw_acc_range,
        acc_x_range,
        acc_y_range,
        0
    )
