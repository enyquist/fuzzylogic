# third party libraries
import numpy as np
import pytest

# fuzzy logic libraries
from fuzzylogic.mf.user_defined import UserDefinedMF


def test_user_defined_mf_basic_functionality(simple_mf):
    """
    Test the basic functionality of the user defined membership function.
    """

    user_defined_mf = UserDefinedMF(simple_mf)

    # Test inputs
    x = np.array([0, 1, 2])
    y = np.array([2, 1, 0])

    # Expected output
    expected = np.array([2, 2, 2])

    # Calculate the output
    membership_values = user_defined_mf(x, y)

    # Check if the output is correct
    assert np.allclose(membership_values, expected), "The membership values do not match the expected values."


def test_user_defined_mf_invalid_input_type(simple_mf):
    """
    Test error handling for invalid inputs types.
    """

    user_defined_mf = UserDefinedMF(simple_mf)

    # Test invalid inputs
    x = [0, 1, 2]  # not a numpy array
    y = np.array([2, 1, 0])

    with pytest.raises(TypeError):
        user_defined_mf(x, y)


def test_user_defined_mf_mismatched_input_shapes(simple_mf):
    """
    Test error handling for mismatched input shapes.
    """

    user_defined_mf = UserDefinedMF(simple_mf)

    # Test invalid inputs
    x = np.array([0, 1, 2])
    y = np.array([2, 1])  # mismatched shape

    with pytest.raises(ValueError):
        user_defined_mf(x, y)
