import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from sympy import Le
from range_bounding import *


# Define bounds for the region
sMin = 0
sMax = 10
nMin = -0
nMax = 2
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

# Helper function to add convex hull projection on a 2D plane
def add_2d_convex_hull(points, axis_indices, color, name, subplot, fig):
    """Add a convex hull projection to a specific subplot."""
    if len(points) > 3:
        projected_points = points[:, axis_indices]  # Select axes for the projection
        hull = ConvexHull(projected_points)
        vertices = hull.vertices

        # Add 2D scatter plot with hull
        fig.add_trace(
            go.Scatter(
                x=projected_points[vertices, 0],
                y=projected_points[vertices, 1],
                fill='toself',
                mode='lines+markers',
                marker=dict(color=color),
                line=dict(color=color),
                name=name,
                legendgroup=name,
            ),
            row=subplot[0], col=subplot[1]
        )

# Load constraints and symbols
res0, symbols0 = get_z()
print(res0)
res1, symbols1 = affine_ranges()
print(res1)
res2, symbols2 = eliminate_quantifier()
res2 += [
    Le(sMin, symbols2[0]),
    Le(symbols2[0], sMax),
    Le(nMin, symbols2[1]),
    Le(symbols2[1], nMax),
]
print(res2)

# Monte Carlo simulation with fewer points for generating regions
# num_samples = 100,000
# points = sample_points(num_samples)
region0_points, region1_points, region2_points = [], [], []

def parse_bounds_from_constraints(constraints):
    """Extract lower and upper bounds for each variable from constraints."""
    lower = {}
    upper = {}

    for constraint in constraints:
        if constraint.lhs.is_symbol:
            variable = constraint.lhs
            if constraint.rel_op == '<=':
                upper[variable] = float(constraint.rhs)
            elif constraint.rel_op == '>=':
                lower[variable] = float(constraint.rhs)
        elif constraint.rhs.is_symbol:
            variable = constraint.rhs
            if constraint.rel_op == '<=':
                lower[variable] = float(constraint.lhs)
            elif constraint.rel_op == '>=':
                upper[variable] = float(constraint.lhs)

    return lower, upper


def sample_points_near_bounds(n, lower_bounds, upper_bounds, symbols):
    """Generate random samples concentrated around the lower and upper bounds."""
    num_vars = len(symbols)

    # Initialize the points array
    points = []

    # Sample near each bound (lower and upper)
    for _ in range(n):
        point = []
        for symbol in symbols:
            low = lower_bounds[symbol]
            high = upper_bounds[symbol]

            # Randomly choose to sample near either the lower or upper bound
            if np.random.rand() > 0.5:
                # Sample near the lower bound
                sampled_value = np.random.normal(loc=low, scale=(high - low) / 20)
            else:
                # Sample near the upper bound
                sampled_value = np.random.normal(loc=high, scale=(high - low) / 20)

            point.append(sampled_value)

        points.append(point)

    return np.array(points)


# Parse rectangle bounds from `res1`
lower_bounds, upper_bounds = parse_bounds_from_constraints(res1)
# Sample points based on the parsed bounds
symbols = symbols1  # Assuming `symbols1` is the correct set of symbols
num_samples = 100000
points = sample_points_near_bounds(num_samples, lower_bounds, upper_bounds, symbols)

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

# --- Create subplots for projections ---
from plotly.subplots import make_subplots

# Create a 1x3 subplot structure
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=('Projection on (X3, U1)', 'Projection on (X3, U2)', 'Projection on (U1, U2)'),
    horizontal_spacing=0.1
)

# Define projection planes and add convex hulls
projections = [
    ([0, 1], 'Projection on (X3, U1)', (1, 1)),
    ([0, 2], 'Projection on (X3, U2)', (1, 2)),
    ([1, 2], 'Projection on (U1, U2)', (1, 3))
]

colors = ['blue', 'green', 'red']
regions = [region0_points, region1_points, region2_points]
region_names = ['Region 0', 'Region 1', 'Region 2']

# Add projections for each region on each plane
for region_points, color, name in zip(regions, colors, region_names):
    for axis_indices, projection_title, subplot in projections:
        add_2d_convex_hull(region_points, axis_indices, color, name, subplot, fig)

# Customize layout
fig.update_layout(
    title='2D Projections of Regions onto Different Planes',
    showlegend=True,
    height=600, width=1200
)

fig.show()

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