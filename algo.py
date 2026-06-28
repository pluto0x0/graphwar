from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

Number = int | float | str | sp.Expr
Point = tuple[Number, Number]


def _number(value: Number) -> sp.Expr:
    """Convert UI text/numbers without introducing unnecessary float noise."""
    result = value if isinstance(value, sp.Expr) else sp.sympify(str(value).strip())
    if not result.is_number or result.is_real is False:
        raise ValueError(f"不是有效实数: {value}")
    return result


def generate_expression(
    points: Sequence[Point],
    *,
    limit: Number | None = 10,
    factor: Number = 20,
    freq: Number = 10,
    width: Number = "0.5",
) -> sp.Expr:
    """Generate an expression using the original trajectory algorithm."""
    print(points)
    if len(points) < 2:
        raise ValueError("至少需要选择两个点")

    ordered = sorted(
        [(_number(point[0]), _number(point[1])) for point in points],
        key=lambda point: float(point[0]),
    )
    if any(ordered[i][0] == ordered[i + 1][0] for i in range(len(ordered) - 1)):
        raise ValueError("相邻点的 x 坐标不能相同")

    x = sp.symbols("x")
    factor_value = _number(factor)

    def sigmoid(value: sp.Expr) -> sp.Expr:
        # Preserve exp(-factor * (x - p)) as written. If SymPy distributes the
        # product first, a decimal p makes it evaluate exp(factor*p) into an
        # enormous floating-point coefficient.
        exponent = sp.Mul(-factor_value, value, evaluate=False)
        return 1 / (1 + sp.exp(exponent, evaluate=False))

    y: sp.Expr | None = None
    k: sp.Expr | None = None
    for i in range(len(ordered) - 1):
        p, _q = ordered[i]
        dx = ordered[i + 1][0] - p
        dy = ordered[i + 1][1] - ordered[i][1]
        if i == 0:
            y = x * dy / dx
        else:
            assert y is not None and k is not None
            y += sigmoid(x - p) * (dy / dx - k) * (x - p)
        k = dy / dx

    assert y is not None
    if limit is not None:
        limit_value = _number(limit)
        y += (
            sigmoid(x - limit_value)
            * sp.sin((x - limit_value) * _number(freq))
            * _number(width)
        )
    return y


def expression_text(*args: object, **kwargs: object) -> str:
    return sp.sstr(generate_expression(*args, **kwargs))


if __name__ == "__main__":
    demo_points = [(-10.1, 0.1), (0.1, 10.123123), (10, 0), (20, 10)]
    print(generate_expression(demo_points))
