# third party libraries
import numpy as np
import pytest

# fuzzy logic libraries
from fuzzylogic.mf.gaussian import Gaussian


def test_gaussian_evaluation():
    mean = 0
    std = 1
    gaussian_fn = Gaussian(mean=mean, std=std)

    # Expected value at mean
    x_val = np.array([mean])
    y = gaussian_fn(x_val)
    assert y == pytest.approx(1.0, 0.01)

    # Expected value at one standard deviation from the mean
    x_val = np.array([mean + std])
    y = gaussian_fn(x_val)
    assert y == pytest.approx(0.6065, 0.01)

    # Expected value at two standard deviations from the mean
    x_val = np.array([mean + 2 * std])
    y = gaussian_fn(x_val)
    assert y == pytest.approx(0.1353, 0.01)


def test_gaussian_array_evaluation():
    gaussian_fn = Gaussian(mean=0, std=1)
    x_vals = np.array([-2, -1, 0, 1, 2])
    y = gaussian_fn(x_vals)

    expected_values = np.array([0.1353, 0.6065, 1.0, 0.6065, 0.1353])

    assert np.allclose(y, expected_values, atol=0.01)
