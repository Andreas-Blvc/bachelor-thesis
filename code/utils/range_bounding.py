# Purpose: Given Constraints, make them align the DCP rules, by shaping them into rectangles
# Example:
# Given: f: R^n -> R and some constants k,l with k < l
# Choose integer i, with 1<=i<=n
# For j=1,...,n with j!=i FIND k_j, l_j, with k_j < l_j:
# and define M = {x_i | k <= f(x_1, ..., x_i, ..., x_n) <= l, for all j for all x_j in [k_j, l_j] }
# Such that either:
# 1. |M| is maximized or (l_1 - k_1) + ... + (l_(i-1) - k_(i-1)) + (l_(i+1) - k_(i+1)) + ... + (l_n - k_n) > threshold
# 2. (l_1 - k_1) + ... + (l_(i-1) - k_(i-1)) + (l_(i+1) - k_(i+1)) + ... + (l_n - k_n) is maximized and |M| > threshold

# M should result in a Set equal to an 1-D interval
# -> Z := M x [k_1, l_1] x ... x [k_(i-1), l_(i-1)] x [k_(i+1), l_(i+1)] x ... x [k_n, l_n]

from utils.ranges import Ranges
from itertools import product
from typing import Tuple



def affine_range_bounding(slope_range, intercept_range, lower_bound, upper_bound):
    """
    Calculates the largest set X such that, for all slopes a and intercepts b in the given ranges:
    lower_bound <= ax+b <= upper_bound holds for all x in X
    """
    slope_min, slope_max = slope_range
    intercept_min, intercept_max = intercept_range
    if slope_min > slope_max or intercept_min > intercept_max:
        raise ValueError(f"invalid range: slope_min, slope_max = {slope_min, slope_max}; intercept_min, intercept_max = {intercept_min, intercept_max} ")
    if slope_min == 0 or slope_max == 0:
        raise NotImplementedError('unhandled case')

    if slope_min < 0 < slope_max:
        X = (
            max(
                min(0, (lower_bound - intercept_min) / slope_max), min(0, (upper_bound - intercept_max) / slope_min)
            ),
            min(
                max(0, (upper_bound - intercept_max) / slope_max), max(0, (lower_bound - intercept_min) / slope_min)
            )

        )
    elif slope_max < 0:
        X = (
            min(0, (upper_bound - intercept_max) / slope_min),
            max(0, (lower_bound - intercept_min) /slope_min)
        )
    else: # slope_min > 0
        X = (
            min(0, (lower_bound - intercept_min) / slope_max),
            max(0, (upper_bound - intercept_max) / slope_max)
        )

    print(
        "------------------\n"
        f"Slope: {slope_range[0]:.3f}, {slope_range[1]:.3f}\n"
        f"Intercept: {intercept_range[0]:.3f}, {intercept_range[1]:.3f}\n"
        f"Lower: {lower_bound}\n"
        f"Upper: {upper_bound}\n"
        f"X: {X[0]:.3f}, {X[1]:.3f}\n"
        "------------------"
    )
    return X

def calculate_product_range(*ranges: Tuple[float, float]) -> Tuple[float, float]:
    """
    Calculate the minimum and maximum product of multiple ranges.

    Args:
        *ranges (Tuple[float, float]): Variable number of tuples, each containing the min and max of a range.

    Returns:
        Tuple[float, float]: The minimum and maximum product.
    """
    if not ranges:
        raise ValueError("At least one range must be provided.")

    # Generate all combinations of min and max for each range
    combinations = product(*[(range_[0], range_[1]) for range_ in ranges])

    # Calculate all possible products
    products = []
    for combo in combinations:
        product_val = 1
        for num in combo:
            product_val *= num
        products.append(product_val)
    return min(products), max(products)



def v_x_constraint_reduction(v_x_range, c_range, n_range):
    v_x_min, v_x_max = v_x_range
    nc_min, nc_max = calculate_product_range(c_range, n_range)

    slope_range = (
        1 - nc_max,
        1 - nc_min,
    )
    return Ranges(
        ds=affine_range_bounding(
            slope_range=slope_range,
            intercept_range=(0, 0),
            lower_bound=v_x_min,
            upper_bound=v_x_max,
        )
    )

def yaw_rate_constraint_reduction(yaw_rate_range, c_range):
    yaw_rate_min, yaw_rate_max = yaw_rate_range
    slope_range = c_range
    return Ranges(
        ds=affine_range_bounding(
            slope_range=slope_range,
            intercept_range=(0, 0),
            lower_bound=yaw_rate_min,
            upper_bound=yaw_rate_max,
        )
    )

def yaw_acceleration_constraint_reduction(yaw_acceleration_range, c_range, ds_range, dc_ds):
    yaw_acceleration_min, yaw_acceleration_max = yaw_acceleration_range

    ds2_min, ds2_max = calculate_product_range(ds_range, ds_range)
    c_ds2_min = dc_ds * (ds2_min if dc_ds > 0 else ds2_max)
    c_ds2_max = dc_ds * (ds2_max if dc_ds > 0 else ds2_min)

    slope_range = c_range
    intercept_range = (c_ds2_min, c_ds2_max)

    return Ranges(
        u_t=affine_range_bounding(
            slope_range=slope_range,
            intercept_range=intercept_range,
            lower_bound=yaw_acceleration_min,
            upper_bound=yaw_acceleration_max,
        )
    )


def x_acceleration_constraint_reduction(x_acceleration_range, c_range, n_range, dn_range, ds_range, dc_ds):
    x_acceleration_min, x_acceleration_max = x_acceleration_range
    nc_min, nc_max = calculate_product_range(c_range, n_range)

    dncds_min, dncds_max = calculate_product_range(dn_range, c_range, ds_range)
    nds2_min, nds2_max = calculate_product_range(n_range, ds_range, ds_range)
    c_nds2_min = dc_ds * (nds2_min if dc_ds > 0 else nds2_max)
    c_nds2_max = dc_ds * (nds2_max if dc_ds > 0 else nds2_min)
    slope_range = (
        1 - nc_max,
        1 - nc_min,
    )
    intercept_range = (
        -2 * dncds_max - c_nds2_max,
        -2 * dncds_min - c_nds2_min,
    )

    return Ranges(
        u_t=affine_range_bounding(
            slope_range=slope_range,
            intercept_range=intercept_range,
            lower_bound=x_acceleration_min,
            upper_bound=x_acceleration_max,
        )
    )

def y_acceleration_constraint_reduction(y_acceleration_range, c_range, n_range, ds_range):
    y_acceleration_min, y_acceleration_max = y_acceleration_range
    cds2_min, cds2_max = calculate_product_range(c_range, ds_range, ds_range)
    c2ds2n_min, c2ds2n_max = calculate_product_range(c_range, c_range, ds_range, ds_range, n_range)

    slope_range = (1, 1)
    intercept_range = (
        cds2_min - c2ds2n_max,
        cds2_max + c2ds2n_min
    )

    return Ranges(
        u_n=affine_range_bounding(
            slope_range=slope_range,
            intercept_range=intercept_range,
            lower_bound=y_acceleration_min,
            upper_bound=y_acceleration_max,
        )
    )