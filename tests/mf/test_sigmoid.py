# third party libraries
import numpy as np
import pytest

# fuzzy logic libraries
from fuzzylogic.mf.sigmoid import Sigmoid


def test_sigmoid_evaluation():
    sigmoid_fn = Sigmoid(a=1, b=0)

    # Evaluate around the inflection point (b value)
    x_val = np.array([0])
    y = sigmoid_fn(x_val)
    assert y == pytest.approx(0.5, 0.01)

    # Evaluate far to the left and right of inflection point
    x_val_left = np.array([-10])
    y_left = sigmoid_fn(x_val_left)
    assert y_left < 0.1

    x_val_right = np.array([10])
    y_right = sigmoid_fn(x_val_right)
    assert y_right > 0.9


def test_sigmoid_array_evaluation():
    sigmoid_fn = Sigmoid(a=1, b=0)
    x_vals = np.array([-2, -1, 0, 1, 2])
    y = sigmoid_fn(x_vals)

    expected_values = np.array(
        [
            1 / (1 + np.exp(2)),  # -2
            1 / (1 + np.exp(1)),  # -1
            0.5,  # 0
            1 / (1 + np.exp(-1)),  # 1
            1 / (1 + np.exp(-2)),  # 2
        ]
    )

    assert np.allclose(y, expected_values, atol=0.01)


def test_sigmoid_slope():
    # Steeper slope
    sigmoid_steep = Sigmoid(a=5, b=0)
    x_val = np.array([0])
    y_steep = sigmoid_steep(x_val)
    assert y_steep == pytest.approx(0.5, 0.01)

    x_val_left = np.array([-1])
    y_left = sigmoid_steep(x_val_left)
    assert y_left < 0.1

    x_val_right = np.array([1])
    y_right = sigmoid_steep(x_val_right)
    assert y_right > 0.9

    # Shallower slope
    sigmoid_shallow = Sigmoid(a=0.5, b=0)
    x_val = np.array([0])
    y_shallow = sigmoid_shallow(x_val)
    assert y_shallow == pytest.approx(0.5, 0.01)

    x_val_left = np.array([-1])
    y_left = sigmoid_shallow(x_val_left)
    assert y_left > 0.2 and y_left < 0.4

    x_val_right = np.array([1])
    y_right = sigmoid_shallow(x_val_right)
    assert y_right < 0.8 and y_right > 0.6
