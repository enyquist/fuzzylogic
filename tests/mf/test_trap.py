# third party libraries
import numpy as np
import pytest

# fuzzy logic libraries
from fuzzylogic.mf.trap import Trapezoid


def test_trapezoid_initialization():
    with pytest.raises(ValueError, match="a must be less than or equal to b"):
        Trapezoid(1, 0, 2, 3)

    with pytest.raises(ValueError, match="b must be less than or equal to c"):
        Trapezoid(0, 1, 0, 3)

    with pytest.raises(ValueError, match="c must be less than or equal to d"):
        Trapezoid(0, 1, 2, 1)

    # Valid initialization
    Trapezoid(0, 1, 2, 3)


def test_trapezoid_evaluation():
    trap_fn = Trapezoid(0, 1, 2, 3)

    # Test for values outside of the trapezoid
    x_vals = np.array([-1, 4])
    y = trap_fn(x_vals)
    expected_values = np.array([0, 0])
    assert np.array_equal(y, expected_values)

    # Test for values within the trapezoid
    x_vals = np.array([0.5, 1.5, 2.5])
    y = trap_fn(x_vals)
    expected_values = np.array([0.5, 1, 0.5])
    assert np.array_equal(y, expected_values)

    # Test for boundary values
    x_vals = np.array([0, 1, 2, 3])
    y = trap_fn(x_vals)
    expected_values = np.array([0, 1, 1, 0])
    assert np.array_equal(y, expected_values)


def test_trapezoid_edges_evaluation():
    # Case where Trapezoid is just a triangle
    trap_fn = Trapezoid(0, 1, 1, 2)

    x_vals = np.array([0, 0.5, 1, 1.5, 2])
    y = trap_fn(x_vals)
    expected_values = np.array([0, 0.5, 1, 0.5, 0])
    assert np.array_equal(y, expected_values)
