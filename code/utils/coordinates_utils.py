from typing import Tuple
from math import cos, sin

def add_coordinates(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (
        a[0] + b[0],
        a[1] + b[1],
    )

def rotate_coordinates(point: Tuple[float, float], theta: float) -> Tuple[float, float]:
    x, y = point
    return (
        x * cos(theta) - y * sin(theta),
        y * cos(theta) + x * sin(theta),
    )
