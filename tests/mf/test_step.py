# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.mf.step import Step


def test_step_evaluation():
    step_fn = Step(limit=0.5)

    # Evaluate around the limit value
    assert step_fn(np.array([0.5])) == 1
    assert step_fn(np.array([0.49])) == 0

    # Evaluate far from the limit value
    assert step_fn(np.array([-100])) == 0
    assert step_fn(np.array([100])) == 1


def test_step_array_evaluation():
    step_fn = Step(limit=0)

    x_vals = np.array([-2, -1, 0, 1, 2])
    y = step_fn(x_vals)

    expected_values = np.array([0, 0, 1, 1, 1])
    assert np.array_equal(y, expected_values)


def test_step_array_mixed_evaluation():
    step_fn = Step(limit=0.5)

    x_vals = np.array([-1, -0.5, 0, 0.5, 1])
    y = step_fn(x_vals)

    expected_values = np.array([0, 0, 0, 1, 1])
    assert np.array_equal(y, expected_values)
