def _update_range(current, new):
    if new is None:
        return current  # No update needed

    if current is None:
        return new  # Initialize with the new range

    current_lower, current_upper = current
    new_lower, new_upper = new

    # Update lower bound if the new lower is higher
    if new_lower is not None:
        updated_lower = max(current_lower, new_lower)
    else:
        updated_lower = current_lower

    # Update upper bound if the new upper is smaller
    if new_upper is not None:
        updated_upper = min(current_upper, new_upper)
    else:
        updated_upper = current_upper

    if updated_lower > updated_upper:
        raise ValueError(
            f"Invalid range update: lower bound {updated_lower} exceeds upper bound {updated_upper}."
        )

    return updated_lower, updated_upper


class Ranges:
    def __init__(self, c=None, n=None, ds=None, dn=None, u_t=None, u_n=None):
        """
        Initialize the Ranges object with optional range tuples.
        Each range should be a tuple of (lower_bound, upper_bound) or None.
        """
        self.c = c  # C(s)
        self.n = n
        self.ds = ds
        self.dn = dn
        self.u_t = u_t
        self.u_n = u_n

    def update(self, new_ranges):
        self.c = _update_range(self.c, new_ranges.c)
        self.n = _update_range(self.n, new_ranges.n)
        self.ds = _update_range(self.ds, new_ranges.ds)
        self.dn = _update_range(self.dn, new_ranges.dn)
        self.u_t = _update_range(self.u_t, new_ranges.u_t)
        self.u_n = _update_range(self.u_n, new_ranges.u_n)

    def __str__(self):
        ranges_str = [
            f"C(s): {self.c[0]:.8f}, {self.c[1]:.8f}" if self.c is not None else "c: -",
            f"n: {self.n[0]:.3f}, {self.n[1]:.3f}" if self.n is not None else "n: -",
            f"ds: {self.ds[0]:.3f}, {self.ds[1]:.3f}" if self.ds is not None else "ds: -",
            f"dn: {self.dn[0]:.3f}, {self.dn[1]:.3f}" if self.dn is not None else "dn: -",
            f"u_t: {self.u_t[0]:.3f}, {self.u_t[1]:.3f}" if self.u_t is not None else "u_t: -",
            f"u_n: {self.u_n[0]:.3f}, {self.u_n[1]:.3f}" if self.u_n is not None else "u_n: -",
        ]
        return "\n".join(ranges_str)