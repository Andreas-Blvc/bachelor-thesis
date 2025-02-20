import numpy as np


def solve_system(v1, delta1):
    # Define the coefficient matrix A
    delta2 = -delta1
    v2 = 0
    A = np.array([
        [v1 ** 2, delta1 ** 2, v1, delta1, 1],
        [v2 ** 2, delta1 ** 2, v2, delta1, 1],
        [v1 ** 2, delta2 ** 2, v1, delta2, 1],
        [v2 ** 2, delta2 ** 2, v2, delta2, 1],
    ])

    # Define the right-hand side vector B
    B = np.array([
        [v1 ** 4 * delta1 ** 2],
        [v2 ** 4 * delta1 ** 2],
        [v1 ** 4 * delta2 ** 2],
        [v2 ** 4 * delta2 ** 2],
        [0]
    ])

    # Solve the linear system Ax = B
    X = np.linalg.lstsq(A, B)

    return X


# Example usage
v = 2  # Example value for v
delta = 3  # Example value for delta
solution = solve_system(v, delta)
print("Solution:")
print(solution)
