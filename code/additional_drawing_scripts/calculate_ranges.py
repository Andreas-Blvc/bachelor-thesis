from range_bounding import eliminate_quantifier
from utils import StateRanges, MainPaperConstraintsReduction
import numpy as np

from utils.range_bounding import calculate_product_range

nMin, nMax = 0, 2
Curvature = -0.033
vyMin, vyMax = -10, 10
vxMin, vxMax = 14, 14.5
axMin, axMax = -6, 3
ayMin, ayMax = -5, 5
steering_range = ((-40 / 180) * np.pi, (40 / 180) * np.pi)
l_wb = 0.883 + 1.508

v_tan_delta = calculate_product_range((vxMin, vxMax), (np.tan(steering_range[0]), np.tan(steering_range[1])))
v_delta = calculate_product_range((vxMin, vxMax), steering_range)
yaw_rate_range = (
    1 / l_wb * v_tan_delta[0],
    1 / l_wb * v_tan_delta[1],
)
yaw_acc_range = (
    1 / l_wb * (v_tan_delta[0] + v_delta[0] / np.cos(steering_range[0]) ** 2),
    1 / l_wb * (v_tan_delta[1] + v_delta[1] / np.cos(steering_range[0]) ** 2),
)
dpsiMin, dpsiMax = yaw_rate_range
apsiMin, apsiMax = yaw_acc_range


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
    yaw_rate_range=yaw_rate_range,
    yaw_acc_range=yaw_acc_range,
    curvature_derivative=0  # constant for all s
)

print(ranges)

# res2, symbols2 = eliminate_quantifier(
#     Curvature,
#     20,
#     20 + 94.2,
#     nMin,
#     nMax,
#     vxMin,
#     vxMax,
#     vyMin,
#     vyMax,
#     axMin,
#     axMax,
#     ayMin,
#     ayMax,
#     dpsiMin,
#     dpsiMax,
#     apsiMin,
#     apsiMax,
# )
# print(res2)