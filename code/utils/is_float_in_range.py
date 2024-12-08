def is_float_in_range(value, lower_bound, upper_bound, margin=1e-3):
    """
    Check if a float value is within a range [lower_bound, upper_bound]
    with a specified margin of error.

    Parameters:
        value (float): The float value to check.
        lower_bound (float): The lower bound of the range.
        upper_bound (float): The upper bound of the range.
        margin (float): The margin of error (default: 1e-6).

    Returns:
        bool: True if the value is within the range with margin, False otherwise.
    """
    return (lower_bound - margin) <= value <= (upper_bound + margin)