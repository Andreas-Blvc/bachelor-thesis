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

from itertools import product
from typing import Tuple

from .state_ranges import StateRanges


def _affine_range_bounding(slope_range, intercept_range, lower_bound, upper_bound):
    """
    Calculates the largest set X such that, for all slopes a and intercepts b in the given ranges:
    lower_bound <= ax+b <= upper_bound holds for all x in X
    """
    slope_min, slope_max = slope_range
    intercept_min, intercept_max = intercept_range
    if slope_min > slope_max or intercept_min > intercept_max:
        raise ValueError(f"invalid range: slope_min, slope_max = {slope_min, slope_max}; "
                         f"intercept_min, intercept_max = {intercept_min, intercept_max} ")

    if slope_min <= 0 <= slope_max:
        if intercept_min < lower_bound or intercept_max > upper_bound:
            return 0, 0

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
    if slope_min == slope_max == 0:
        # lower_bound <= b <= upper_bound -> either for all x (no range) or for no x ([0, 0])
        if intercept_min < lower_bound or intercept_max > upper_bound:
            return 0, 0
        else:
            return None

    # 2.
    if slope_min == slope_max < 0:
        x_lb = (upper_bound - intercept_max) / slope_max
        x_ub = (lower_bound - intercept_min) / slope_max
        return (x_lb, x_ub) if x_lb <= x_ub else (0, 0)

    # 3.
    if slope_min == slope_max > 0:
        x_lb = (lower_bound - intercept_min) / slope_max
        x_ub = (upper_bound - intercept_max) / slope_max
        return (x_lb, x_ub) if x_lb <= x_ub else (0, 0)

    # 4.
    if slope_max < 0:
        if upper_bound - intercept_max < 0:
            x_lb = (upper_bound - intercept_max) / slope_max
        else:
            x_lb = (upper_bound - intercept_max) / slope_min
        if lower_bound - intercept_min < 0:
            x_ub = (lower_bound - intercept_min) / slope_max
        else:
            x_ub = (lower_bound - intercept_min) / slope_min
        return (x_lb, x_ub) if x_lb <= x_ub else (0, 0)

    # 5.
    if 0 < slope_min:
        if lower_bound - intercept_min > 0:
            x_lb = (lower_bound - intercept_min) / slope_min
        else:
            x_lb = (lower_bound - intercept_min) / slope_max
        if upper_bound - intercept_max > 0:
            x_ub = (upper_bound - intercept_max) / slope_min
        else:
            x_ub = (upper_bound - intercept_max) / slope_max
        return (x_lb, x_ub) if x_lb <= x_ub else (0, 0)

    #  6. 7. 8.
    # lower_bound < intercept_min and intercept_max < upper_bound:
    if intercept_min < lower_bound or intercept_max > upper_bound:
        raise AssertionError("lower_bound < intercept_min and intercept_max < upper_bound does not hold")
    if slope_min < 0 < slope_max:
        X = (
            max(
                min(0, (lower_bound - intercept_min) / slope_max), min(0, (upper_bound - intercept_max) / slope_min)
            ),
            min(
                max(0, (upper_bound - intercept_max) / slope_max), max(0, (lower_bound - intercept_min) / slope_min)
            )

        )
    elif slope_min < 0:  # slope_max = 0
        X = (
            min(0, (upper_bound - intercept_max) / slope_min),
            max(0, (lower_bound - intercept_min) /slope_min)
        )
    elif slope_max > 0:  # slope_min = 0
        X = (
            min(0, (lower_bound - intercept_min) / slope_max),
            max(0, (upper_bound - intercept_max) / slope_max)
        )
    else:
        raise NotImplementedError(
            'Unhandled case occurred\n'
            f"slope_min, slope_max = {slope_min, slope_max}; "
            f"intercept_min, intercept_max = {intercept_min, intercept_max}"
        )

    # print(
    #     "------------------\n"
    #     f"Slope: {slope_range[0]:.3f}, {slope_range[1]:.3f}\n"
    #     f"Intercept: {intercept_range[0]:.3f}, {intercept_range[1]:.3f}\n"
    #     f"Lower: {lower_bound}\n"
    #     f"Upper: {upper_bound}\n"
    #     f"X: {X[0]:.3f}, {X[1]:.3f}\n"
    #     "------------------"
    # )
    return X

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


class MainPaperConstraintsReduction:
    @staticmethod
    def apply_all(
            state_ranges: StateRanges,
            v_x_range,
            acc_x_range,
            acc_y_range,
            yaw_rate_range,
            yaw_acc_range,
            curvature_derivative
    ):
        # self.v_x_min / (1 + self.nc_max) <= ds <=  self.v_x_max / (1 + self.nc_min)
        state_ranges.update(
            MainPaperConstraintsReduction.v_x_constraint_reduction(
                v_x_range=v_x_range,
                c_range=state_ranges.c,
                n_range=state_ranges.n,
            )
        )
        # self.yaw_rate_min <= C(s) * ds <= self.yaw_rate_max,
        state_ranges.update(
            MainPaperConstraintsReduction.yaw_rate_constraint_reduction(
                yaw_rate_range=yaw_rate_range,
                c_range=state_ranges.c,
            )
        )

        # self.yaw_acc_min <= C'(s) * ds**2 + C(s) * u_t <= self.yaw_acc_max,
        state_ranges.update(
            MainPaperConstraintsReduction.yaw_acceleration_constraint_reduction(
                yaw_acceleration_range=yaw_acc_range,
                c_range=state_ranges.c,
                ds_range=state_ranges.ds,
                dc_ds=curvature_derivative  # constant for all s
            )
        )

        # self.acc_x_min <= g[0] <= self.acc_x_max,
        state_ranges.update(
            MainPaperConstraintsReduction.x_acceleration_constraint_reduction(
                x_acceleration_range=acc_x_range,
                c_range=state_ranges.c,
                n_range=state_ranges.n,
                dn_range=state_ranges.dn,
                ds_range=state_ranges.ds,
                dc_ds=curvature_derivative  # constant for all s
            )
        )

        # self.acc_y_min <= g[1] <= self.acc_y_max,
        state_ranges.update(
            MainPaperConstraintsReduction.y_acceleration_constraint_reduction(
                y_acceleration_range=acc_y_range,
                c_range=state_ranges.c,
                n_range=state_ranges.n,
                ds_range=state_ranges.ds,
            )
        )


    @staticmethod
    def v_x_constraint_reduction(v_x_range, c_range, n_range):
        v_x_min, v_x_max = v_x_range
        nc_min, nc_max = _calculate_product_range(c_range, n_range)

        slope_range = (
            1 - nc_max,
            1 - nc_min,
        )
        return StateRanges(
            ds=_affine_range_bounding(
                slope_range=slope_range,
                intercept_range=(0, 0),
                lower_bound=v_x_min,
                upper_bound=v_x_max,
            )
        )

    @staticmethod
    def yaw_rate_constraint_reduction(yaw_rate_range, c_range):
        yaw_rate_min, yaw_rate_max = yaw_rate_range
        slope_range = c_range
        return StateRanges(
            ds=_affine_range_bounding(
                slope_range=slope_range,
                intercept_range=(0, 0),
                lower_bound=yaw_rate_min,
                upper_bound=yaw_rate_max,
            )
        )

    @staticmethod
    def yaw_acceleration_constraint_reduction(yaw_acceleration_range, c_range, ds_range, dc_ds):
        yaw_acceleration_min, yaw_acceleration_max = yaw_acceleration_range

        ds2_min, ds2_max = _calculate_product_range(ds_range, ds_range)
        c_ds2_min = dc_ds * (ds2_min if dc_ds > 0 else ds2_max)
        c_ds2_max = dc_ds * (ds2_max if dc_ds > 0 else ds2_min)

        slope_range = c_range
        intercept_range = (c_ds2_min, c_ds2_max)

        return StateRanges(
            u_t=_affine_range_bounding(
                slope_range=slope_range,
                intercept_range=intercept_range,
                lower_bound=yaw_acceleration_min,
                upper_bound=yaw_acceleration_max,
            )
        )

    @staticmethod
    def x_acceleration_constraint_reduction(x_acceleration_range, c_range, n_range, dn_range, ds_range, dc_ds):
        x_acceleration_min, x_acceleration_max = x_acceleration_range
        nc_min, nc_max = _calculate_product_range(c_range, n_range)

        dncds_min, dncds_max = _calculate_product_range(dn_range, c_range, ds_range)
        nds2_min, nds2_max = _calculate_product_range(n_range, ds_range, ds_range)
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

        return StateRanges(
            u_t=_affine_range_bounding(
                slope_range=slope_range,
                intercept_range=intercept_range,
                lower_bound=x_acceleration_min,
                upper_bound=x_acceleration_max,
            )
        )

    @staticmethod
    def y_acceleration_constraint_reduction(y_acceleration_range, c_range, n_range, ds_range):
        y_acceleration_min, y_acceleration_max = y_acceleration_range
        cds2_min, cds2_max = _calculate_product_range(c_range, ds_range, ds_range)
        c2ds2n_min, c2ds2n_max = _calculate_product_range(c_range, c_range, ds_range, ds_range, n_range)

        slope_range = (1, 1)
        intercept_range = (
            cds2_min - c2ds2n_max,
            cds2_max + c2ds2n_min
        )

        return StateRanges(
            u_n=_affine_range_bounding(
                slope_range=slope_range,
                intercept_range=intercept_range,
                lower_bound=y_acceleration_min,
                upper_bound=y_acceleration_max,
            )
        )