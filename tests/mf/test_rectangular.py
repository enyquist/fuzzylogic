# third party libraries
import numpy as np
import pytest

# fuzzy logic libraries
from fuzzylogic.mf.rectangular import Rectangular


def test_rectangular_initialization():
    with pytest.raises(ValueError):
        Rectangular(low=1, high=0)


def test_rectangular_evaluation():
    rectangular_fn = Rectangular(low=0, high=1)

    # Evaluate at a point within the range
    x_val = np.array([0.5])
    y = rectangular_fn(x_val)
    assert y == pytest.approx(1.0, 0.01)

    # Evaluate at a point outside the range
    x_val = np.array([-0.5])
    y = rectangular_fn(x_val)
    assert y == pytest.approx(0.0, 0.01)

    x_val = np.array([1.5])
    y = rectangular_fn(x_val)
    assert y == pytest.approx(0.0, 0.01)


def test_rectangular_array_evaluation():
    rectangular_fn = Rectangular(low=-0.5, high=0.5)
    x_vals = np.array([-1, -0.5, 0, 0.5, 1])
    y = rectangular_fn(x_vals)

    # outside range  # on boundary  # inside range  # on boundary  # outside range
    expected_values = np.array([0.0, 1.0, 1.0, 1.0, 0.0])

    assert np.allclose(y, expected_values, atol=0.01)
