from sympy import symbols
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from range_bounding import *
from roads import load_road

road = load_road('./data/straight.pkl')
print(road.get_road_position(7.7142857142857135, 115.57142857142857))
exit()

# Define bounds for the region
sMin, sMax = -1, 11
nMin, nMax = -3, 3
vyMin, vyMax = -4, 4
bounds = [
    (sMin, sMax),  # x1
    (nMin, nMax),  # x2
    (-25, 25),     # x3
    (vyMin, vyMax),  # x4
    (-20, 20),     # u1
    (-20, 20),     # u2
]


# Helper function to sample points
def sample_points(n):
    """Generate random samples concentrated around the center of the bounds."""
    lower_bounds, upper_bounds = zip(*bounds)

    # Calculate the midpoints and the span of each bound
    midpoints = np.array([(low + high) / 2 for low, high in bounds])
    span = np.array([(high - low) / 2 for low, high in bounds])

    # Sample using a normal distribution centered on midpoints, scaled by span
    points = np.random.normal(loc=midpoints, scale=span / 3, size=(n, len(bounds)))

    # Clip points to stay within bounds
    points = np.clip(points, lower_bounds, upper_bounds)

    return points


# Helper function to check if points satisfy constraints
def point_satisfies_constraints(point, constraints, symbols):
    """Check if a point satisfies the given SymPy constraints."""
    x1, x2, x3, x4, u1, u2 = symbols
    subs = {x1: point[0], x2: point[1], x3: point[2], x4: point[3], u1: point[4], u2: point[5]}
    return all(constraint.subs(subs) for constraint in constraints)

# Load constraints and symbols
res0, symbols0 = get_z()
res1, symbols1 = affine_ranges()
res2, symbols2 = eliminate_quantifier()
print(res0)
print(res1)
print(res2)

# Monte Carlo simulation with fewer points for generating regions
num_samples = 100000
points = sample_points(num_samples)
region0_points, region1_points, region2_points = [], [], []

# Classify points based on regions and constraints
for point in points:
    if point_satisfies_constraints(point, res0, symbols0):
        region0_points.append(point)
    if point_satisfies_constraints(point, res1, symbols1):
        region1_points.append(point)
    if point_satisfies_constraints(point, res2, symbols2):
        region2_points.append(point)

# Convert points to NumPy arrays for visualization
region0_points = np.array(region0_points)[:, [2, 4, 5]]  # X3, U1, U2 coordinates
region1_points = np.array(region1_points)[:, [2, 4, 5]]
region2_points = np.array(region2_points)[:, [2, 4, 5]]

# --- Plotting with Plotly ---
fig = go.Figure()

# Helper function to add a convex hull surface with a toggleable legend entry
def add_convex_hull(points, color, name, legend_group):
    """Add a convex hull surface and a legend entry to the Plotly figure."""
    if len(points) > 3:  # Convex hull requires at least 4 points
        hull = ConvexHull(points)

        # Add Mesh3d surface
        fig.add_trace(
            go.Mesh3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                i=hull.simplices[:, 0],
                j=hull.simplices[:, 1],
                k=hull.simplices[:, 2],
                color=color,
                opacity=0.5,
                name=name,
                legendgroup=legend_group,  # Group the Mesh and legend marker together
                showlegend=False  # Hide Mesh3d from the legend
            )
        )

        # Add an invisible Scatter3d trace to serve as the legend entry
        fig.add_trace(
            go.Scatter3d(
                x=[points[0, 0]], y=[points[0, 1]], z=[points[0, 2]],
                mode='markers',
                marker=dict(size=5, color=color),
                name=name,
                legendgroup=legend_group,  # Same group as Mesh3d
                hoverinfo='none'
            )
        )

# Add convex hulls for each region
add_convex_hull(region0_points, 'blue', 'Region 0 (X3, U1, U2)', 'region0')
add_convex_hull(region1_points, 'green', 'Region 1 (X3, U1, U2)', 'region1')
add_convex_hull(region2_points, 'red', 'Region 2 (X3, U1, U2)', 'region2')

print(len(region0_points))
print(len(region1_points))
print(len(region2_points))

# Customize plot layout
fig.update_layout(
    scene=dict(
        xaxis_title='X3',
        yaxis_title='U1',
        zaxis_title='U2',
    ),
    title='Interactive 3D Visualization of Regions with Toggleable Legend',
    showlegend=True
)

# Show interactive plot
fig.show()