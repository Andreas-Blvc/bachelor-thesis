from utils import StateRanges, MainPaperConstraintsReduction, optimal_range_bound
import sys

curvature = .0

road_width_range = (-2, 2)
v_x_range = (0, 4)
v_y_range = (-2, 2)
acc_x_range = (-6, 3)
acc_y_range = (-2, 2)
yaw_rate_range = (0, 0)
yaw_acc_range = (0, 0)

ranges_basic = StateRanges(
	n=road_width_range,
	c=(curvature, curvature),
	ds=v_x_range,
	dn=v_y_range,
	u_n=None,
	u_t=None,
)

# updates ranges in place
MainPaperConstraintsReduction.apply_all(
	state_ranges=ranges_basic,
	v_x_range=v_x_range,
	acc_x_range=acc_x_range,
	acc_y_range=acc_y_range,
	yaw_rate_range=yaw_rate_range,
	yaw_acc_range=yaw_acc_range,
	curvature_derivative=curvature  # constant for all s
)
stdout_old = sys.stdout
sys.stdout = None
try:
	ranges_optimal = optimal_range_bound(
		road_width_range,
		v_x_range,
		v_y_range,
		yaw_rate_range,
		yaw_acc_range,
		acc_x_range,
		acc_y_range,
		0
	)
except RuntimeError:
	ranges_optimal = None
sys.stdout = stdout_old

print(ranges_basic)
print(ranges_optimal)


# if __name__ == '__main__':
# 	launch_editor()

