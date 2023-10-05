# third party libraries
import numpy as np
import pytest

# fuzzy logic libraries
from fuzzylogic.mf.triangle import Triangle


def test_triangle_initialization():
    # Correct initialization
    triangle_fn = Triangle(a=0, b=1, c=2)
    assert triangle_fn.a == 0
    assert triangle_fn.b == 1
    assert triangle_fn.c == 2

    # Incorrect initialization
    with pytest.raises(ValueError):
        Triangle(a=2, b=1, c=0)


def test_triangle_evaluation():
    triangle_fn = Triangle(a=0, b=1, c=2)

    x_vals = np.array([-1, 0, 0.5, 1, 1.5, 2, 3])
    y = triangle_fn(x_vals)
    expected_values = np.array([0, 0, 0.5, 1, 0.5, 0, 0])

    assert np.array_equal(y, expected_values)


def test_triangle_boundaries():
    triangle_fn = Triangle(a=0, b=1, c=2)

    assert triangle_fn(0) == 0
    assert triangle_fn(1) == 1
    assert triangle_fn(2) == 0
    assert triangle_fn(-1) == 0
    assert triangle_fn(3) == 0


def test_triangle_peak():
    triangle_fn = Triangle(a=0, b=1, c=2)
    assert triangle_fn(1) == 1


def test_triangle_outside_range():
    triangle_fn = Triangle(a=0, b=1, c=2)
    assert triangle_fn(-10) == 0
    assert triangle_fn(10) == 0
