# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.tconorms import AlgebraicSumTCoNorm, BoundedSumTCoNorm, DrasticSumTCoNorm, MaximumTCoNorm


def test_maximum_tconorm(dummy_mf_1, dummy_mf_2):
    """
    Test the Maximum T-CoNorm.
    """

    combined_mf = MaximumTCoNorm.combine(dummy_mf_1, dummy_mf_2)
    x = np.array([0, np.pi / 4, np.pi / 2])
    assert np.array_equal(combined_mf(x), np.maximum(dummy_mf_1(x), dummy_mf_2(x)))


def test_algebraic_sum_tconorm(dummy_mf_1, dummy_mf_2):
    """
    Test the Algebraic Sum T-CoNorm.
    """

    combined_mf = AlgebraicSumTCoNorm.combine(dummy_mf_1, dummy_mf_2)
    x = np.array([0, np.pi / 4, np.pi / 2])
    assert np.array_equal(combined_mf(x), dummy_mf_1(x) + dummy_mf_2(x) - (dummy_mf_1(x) * dummy_mf_2(x)))


def test_bounded_sum_tconorm(dummy_mf_1, dummy_mf_2):
    """
    Test the Bounded Sum T-CoNorm.
    """

    combined_mf = BoundedSumTCoNorm.combine(dummy_mf_1, dummy_mf_2)
    x = np.array([0, np.pi / 4, np.pi / 2])
    assert np.array_equal(combined_mf(x), np.minimum(1, dummy_mf_1(x) + dummy_mf_2(x)))


def test_drastic_sum_tconorm(dummy_mf_1, dummy_mf_2):
    """
    Test the Drastic Sum T-CoNorm.
    """

    combined_mf = DrasticSumTCoNorm.combine(dummy_mf_1, dummy_mf_2)
    x = np.array([0, np.pi / 4, np.pi / 2])
    assert np.array_equal(
        combined_mf(x), np.where(dummy_mf_1(x) == 0, dummy_mf_2(x), np.where(dummy_mf_2(x) == 0, dummy_mf_1(x), 1))
    )
