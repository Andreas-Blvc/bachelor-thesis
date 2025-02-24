import cvxpy as cp
from typing import Tuple

class McCormickConvexRelaxation:
    def __init__(self, x, y, x_L, x_U, y_L, y_U):
        self._z = cp.Variable()
        self._x = x
        self._y = y
        self._x_L = x_L
        self._x_U = x_U
        self._y_L = y_L
        self._y_U = y_U

    def get_envelopes(self):
        constraints = [
            self._z >= self._x_L * self._y + self._x * self._y_L - self._x_L * self._y_L,
            self._z >= self._x_U * self._y + self._x * self._y_U - self._x_U * self._y_U,
            self._z <= self._x_U * self._y + self._x * self._y_L - self._x_U * self._y_L,
            self._z <= self._x_L * self._y + self._x * self._y_U - self._x_L * self._y_U,
        ]
        return constraints

    def get_relaxation_variable(self):
        return self._z

    def get_lower_upper_bound(self) -> Tuple[float, float]:
        # Only Callable if prob.solve() was called
        self._check()
        l = max(
            self._x_L * self._y.value + self._x.value * self._y_L - self._x_L * self._y_L,
            self._x_U * self._y.value + self._x.value * self._y_U - self._x_U * self._y_U,
        )
        u = min(
            self._x_U * self._y.value + self._x.value * self._y_L - self._x_U * self._y_L,
            self._x_L * self._y.value + self._x.value * self._y_U - self._x_L * self._y_U,
        )
        return l, u

    def get_bilinear_value(self):
        # Only Callable if prob.solve() was called
        self._check()
        return self._x.value * self._y.value

    def _check(self):
        # Check if prob.solve() was called
        if self._x.value is None or self._y.value is None:
            raise RuntimeError("Solver must be called before accessing bilinear value.")