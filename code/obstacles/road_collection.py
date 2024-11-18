from obstacles.constant_curvature_road import ConstantCurvatureRoad
from obstacles.linear_curvature_road import LinearCurvatureRoad
from math import pi

class _Roads:
	def __init__(self):
		self.s_shaped_road = LinearCurvatureRoad(
			s=[(-8, -15), (1, 1), (5, 15)],
			width=5
		)
		self.right_curved_road = ConstantCurvatureRoad(
			midpoint=(-8,-8),
			start_angle=pi/2,
			width=4,
			radius=16,
			length=25,
		)

roads = _Roads()