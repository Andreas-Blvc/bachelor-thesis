from sympy import Le
from sympy.core.relational import Relational
from sympy.core.operations import AssocOp


def sympy_to_cvxpy_constraints(sympy_constraints, cvxpy_variables):
    """
    Convert a list of SymPy constraints to CVXPY constraints.

    Parameters:
    - sympy_constraints: List of SymPy constraints (e.g., inequalities).
    - cvxpy_variables: Dictionary mapping SymPy variables to CVXPY variables.

    Returns:
    - List of CVXPY constraints.
    """

    def _sympy_to_cvxpy_expr(sympy_expr):
        """Helper function to recursively convert a SymPy expression to CVXPY."""
        if sympy_expr in cvxpy_variables:
            return cvxpy_variables[sympy_expr]

        if sympy_expr.is_Number:
            return float(sympy_expr)

        if isinstance(sympy_expr, AssocOp):
            # Handle addition and multiplication operations
            args = [_sympy_to_cvxpy_expr(arg) for arg in sympy_expr.args]
            if sympy_expr.is_Add:
                return sum(args)
            elif sympy_expr.is_Mul:
                result = args[0]
                for arg in args[1:]:
                    result *= arg
                return result

        # Handle negation
        if sympy_expr.is_Mul and len(sympy_expr.args) == 2 and -1 in sympy_expr.args:
            return -_sympy_to_cvxpy_expr(sympy_expr.args[1])

        # Handle nested expressions
        return sympy_expr.func(*(_sympy_to_cvxpy_expr(arg) for arg in sympy_expr.args))

    cvxpy_constraints = []
    for constraint in sympy_constraints:
        if isinstance(constraint, Relational) and isinstance(constraint, Le):
            lhs = _sympy_to_cvxpy_expr(constraint.lhs)
            rhs = _sympy_to_cvxpy_expr(constraint.rhs)
            cvxpy_constraints.append(lhs <= rhs)
    return cvxpy_constraints