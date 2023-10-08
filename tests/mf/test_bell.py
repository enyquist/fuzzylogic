# third party libraries
import numpy as np
import pytest

# fuzzy logic libraries
from fuzzylogic.mf.bell import Bell


def test_basic_functionality():
    bell_fn = Bell(center=0, width=1, intensity=2)
    x = np.array([0, 1, 2])
    y = bell_fn(x)
    assert np.isclose(y, np.array([1, 1 / (1 + 1**4), 1 / (1 + 2**4)])).all()


def test_mean_shift():
    bell_fn = Bell(center=1, width=1, intensity=2)
    x = np.array([0, 1, 2])
    y = bell_fn(x)
    assert np.isclose(y, np.array([1 / (1 + 1**4), 1, 1 / (1 + 1**4)])).all()


def test_slope_variation():
    bell_fn = Bell(center=0, width=1, intensity=3)
    x = np.array([0, 1, 2])
    y = bell_fn(x)
    assert np.isclose(y, np.array([1, 1 / (1 + 1**6), 1 / (1 + 2**6)])).all()


def test_error_handling():
    with pytest.raises(ValueError, match="width must be non-zero"):
        bell_fn = Bell(center=0, width=0, intensity=2)
        _ = bell_fn(np.array([1]))


def test_asymptotic_behavior():
    """
    This test checks if the function converges to 0 for very large inputs
    """
    bell_fn = Bell(center=0, width=1, intensity=2)
    x = np.array([1e6])
    y = bell_fn(x)
    assert np.isclose(y, [0], atol=1e-5).all()
