# third party libraries
import numpy as np
import pytest

# fuzzy logic libraries
from fuzzylogic.defuzz.defuzz import Bisector, Centroid, LargestOfMaximum, MeanOfMaximum, SmallestOfMaximum


def test_centroid_defuzz(dummy_mf_1):
    """
    Test the Centroid defuzzification method.
    """

    x = np.linspace(0, 2 * np.pi, 1000)
    centroid_value = Centroid.defuzz(x, dummy_mf_1)
    assert centroid_value == pytest.approx(np.sum(x * dummy_mf_1(x)) / np.sum(dummy_mf_1(x)))


def test_bisector_defuzz(dummy_mf_1):
    """
    Test the Bisector of Area defuzzification method.
    """

    x = np.linspace(0, 2 * np.pi, 1000)
    bisector_value = Bisector.defuzz(x, dummy_mf_1)
    cumulative_mf = np.cumsum(dummy_mf_1(x))
    total_area = cumulative_mf[-1]
    bisector_index = np.argmin(np.abs(cumulative_mf - total_area / 2))
    assert bisector_value == x[bisector_index]


def test_mean_of_maximum_defuzz(dummy_mf_1):
    """
    Test the Mean of Maximum defuzzification method.
    """

    x = np.linspace(0, 2 * np.pi, 1000)
    mean_max_value = MeanOfMaximum.defuzz(x, dummy_mf_1)
    max_mf = np.max(dummy_mf_1(x))
    max_indices = np.where(dummy_mf_1(x) == max_mf)[0]
    assert mean_max_value == pytest.approx(np.mean(x[max_indices]))


def test_largest_of_maximum_defuzz(dummy_mf_1):
    """
    Test the Largest of Maximum defuzzification method.
    """

    x = np.linspace(0, 2 * np.pi, 1000)
    largest_max_value = LargestOfMaximum.defuzz(x, dummy_mf_1)
    max_mf = np.max(dummy_mf_1(x))
    max_indices = np.where(dummy_mf_1(x) == max_mf)[0]
    assert largest_max_value == np.max(x[max_indices])


def test_smallest_of_maximum_defuzz(dummy_mf_1):
    """
    Test the Smallest of Maximum defuzzification method.
    """

    x = np.linspace(0, 2 * np.pi, 1000)
    smallest_max_value = SmallestOfMaximum.defuzz(x, dummy_mf_1)
    max_mf = np.max(dummy_mf_1(x))
    max_indices = np.where(dummy_mf_1(x) == max_mf)[0]
    assert smallest_max_value == np.min(x[max_indices])
