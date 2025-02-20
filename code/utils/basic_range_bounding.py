from .state_ranges import StateRanges
from .range_bounding import calculate_product_range, affine_range_bounding


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
        # self.v_x_min / (1 - self.nc_max) <= ds <=  self.v_x_max / (1 - self.nc_min)
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
        nc_min, nc_max = calculate_product_range(c_range, n_range)

        slope_range = (
            1 - nc_max,
            1 - nc_min,
        )
        print(slope_range)
        return StateRanges(
            ds=affine_range_bounding(
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
            ds=affine_range_bounding(
                slope_range=slope_range,
                intercept_range=(0, 0),
                lower_bound=yaw_rate_min,
                upper_bound=yaw_rate_max,
            )
        )

    @staticmethod
    def yaw_acceleration_constraint_reduction(yaw_acceleration_range, c_range, ds_range, dc_ds):
        yaw_acceleration_min, yaw_acceleration_max = yaw_acceleration_range

        ds2_min, ds2_max = calculate_product_range(ds_range, ds_range)
        c_ds2_min = dc_ds * (ds2_min if dc_ds > 0 else ds2_max)
        c_ds2_max = dc_ds * (ds2_max if dc_ds > 0 else ds2_min)

        slope_range = c_range
        intercept_range = (c_ds2_min, c_ds2_max)

        return StateRanges(
            u_t=affine_range_bounding(
                slope_range=slope_range,
                intercept_range=intercept_range,
                lower_bound=yaw_acceleration_min,
                upper_bound=yaw_acceleration_max,
            )
        )

    @staticmethod
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

        return StateRanges(
            u_t=affine_range_bounding(
                slope_range=slope_range,
                intercept_range=intercept_range,
                lower_bound=x_acceleration_min,
                upper_bound=x_acceleration_max,
            )
        )

    @staticmethod
    def y_acceleration_constraint_reduction(y_acceleration_range, c_range, n_range, ds_range):
        y_acceleration_min, y_acceleration_max = y_acceleration_range
        cds2_min, cds2_max = calculate_product_range(c_range, ds_range, ds_range)
        c2ds2n_min, c2ds2n_max = calculate_product_range(c_range, c_range, ds_range, ds_range, n_range)

        slope_range = (1, 1)
        intercept_range = (
            cds2_min - c2ds2n_max,
            cds2_max - c2ds2n_min
        )
        print(intercept_range)

        return StateRanges(
            u_n=affine_range_bounding(
                slope_range=slope_range,
                intercept_range=intercept_range,
                lower_bound=y_acceleration_min,
                upper_bound=y_acceleration_max,
            )
        )