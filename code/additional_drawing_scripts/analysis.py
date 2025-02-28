import pandas as pd
from typing import Dict, List, Tuple

# Load the file
file_path = "../final_output.csv"
df = pd.read_csv(file_path, delimiter=';')

# Convert relevant columns to numeric types for analysis
df["objective value"] = pd.to_numeric(df["objective value"], errors='coerce')

# Solver Metrices
df["avg solve time"] = pd.to_numeric(df["avg solve time"].replace("-", pd.NA).str.replace("ms", "", regex=True), errors="coerce")
df["Standard deviation"] = pd.to_numeric(df["Standard deviation"].replace("-", pd.NA).str.replace("ms", "", regex=True), errors="coerce")

# Group by dt and model, then compute the mean, ignoring NaN values
solver_metrices = df.groupby(["dt", "model"], dropna=True)[["avg solve time", "Standard deviation"]].mean()

# print(solver_metrices)

# Group by dt, road_name, objective_name, v_max and compare objective values across models
obj_val = df.groupby(["dt", "road name", "objective name", "v_max [m/s]"])[["model", "road completion","objective value"]]

# Model Name -> (dt, road_name, objective_name, v_max, completion
failed: Dict[str, List[Tuple[str, str, str, int, str]]] = {"Point Mass": [], "Bicycle": []}
# Iterate over groups and print
for name, group in obj_val:
    dt, road_name, objective_name, v_max = name
    for _, row in group.iterrows():  # Iterate over rows in the group
        model_name = row["model"]
        road_completion = row["road completion"]

        # Convert road_completion to float safely
        try:
            completion_value = float(road_completion.replace("%", ""))
        except ValueError:
            print(f"Warning: Invalid road completion value '{road_completion}'")
            continue

        # Check if completion is below 90%
        if completion_value < 90:
            failed[model_name].append((dt, road_name, objective_name, v_max, road_completion))

    # Print failed results (for debugging)

from collections import Counter

# Dictionary to store failed road names per model
failed_road_count = {"Point Mass": Counter(), "Bicycle": Counter()}

# Iterate over grouped data
for name, group in obj_val:
    dt, road_name, objective_name, v_max = name  # Unpacking group keys

    for _, row in group.iterrows():  # Iterate over rows in the group
        model_name = row["model"]
        road_completion = row["road completion"]

        # Convert road_completion to float safely
        try:
            completion_value = float(road_completion.replace("%", ""))
        except ValueError:
            print(f"Warning: Invalid road completion value '{road_completion}'")
            continue

        # Check if completion is below 90%
        if completion_value < 90 and road_name not in ["Random", "Infeasible Curve"]:
            failed[model_name].append((dt, road_name, objective_name, v_max, road_completion))
            failed_road_count[model_name][(road_name, v_max, objective_name)] += 1  # Count failed road names

# Print counts of failed road names per model+
print(failed)
print(failed_road_count)
for model, counts in failed_road_count.items():
    print(f"Model: {model}")
    for road_name, count in counts.items():  # Iterate over road name counts
        print(f"  {road_name}: {count} failures")





