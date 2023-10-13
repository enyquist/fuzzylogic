# third party libraries
import numpy as np
import pytest

# fuzzy logic libraries
from fuzzylogic.mf.constant import ConstantMF


def test_basic_functionality():
    """
    This test checks if the function returns the correct value for a given input
    """

    constant_fn = ConstantMF(value=0.5)
    x = np.array([0, 1, 2])
    y = constant_fn(x)
    assert np.isclose(y, [0.5, 0.5, 0.5]).all()


def test_error_handling():
    """
    This test checks if the function raises an error when value is not between 0 and 1
    """

    with pytest.raises(ValueError, match="value must be between 0 and 1"):
        constant_fn = ConstantMF(value=1.5)
        _ = constant_fn(np.array([1]))


def test_asymptotic_behavior():
    """
    This test checks if the function converges to value for very large inputs
    """

    constant_fn = ConstantMF(value=0.5)
    x = np.array([1e6])
    y = constant_fn(x)
    assert np.isclose(y, [0.5], atol=1e-5).all()
