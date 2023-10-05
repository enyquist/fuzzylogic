# third party libraries
import numpy as np
import pytest

# fuzzy logic libraries
from fuzzylogic.mf.linear import Linear


def test_linear_evaluation():
    m = 1
    b = 0
    linear_fn = Linear(m=m, b=b)

    # Evaluate at a point
    x_val = np.array([0.5])
    y = linear_fn(x_val)
    assert y == pytest.approx(0.5, 0.01)

    # Evaluate at another point
    x_val = np.array([-0.5])
    y = linear_fn(x_val)
    assert y == pytest.approx(0.0, 0.01)  # Clipped to 0

    x_val = np.array([1.5])
    y = linear_fn(x_val)
    assert y == pytest.approx(1.0, 0.01)  # Clipped to 1


def test_linear_array_evaluation():
    linear_fn = Linear(m=2, b=-1)  # y = 2x - 1
    x_vals = np.array([-1, 0, 0.5, 1, 2])
    y = linear_fn(x_vals)

    expected_values = np.array(
        [
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
        ]
    )

    assert np.allclose(y, expected_values, atol=0.01)
