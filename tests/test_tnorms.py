# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.tnorms import AlgebraicProductTNorm, BoundedProductTNorm, DrasticProductTNorm, MinimumTNorm


def test_minimum_tnorm(dummy_mf_1, dummy_mf_2):
    """
    Test the Minimum T-Norm.
    """

    combined_mf = MinimumTNorm.combine(dummy_mf_1, dummy_mf_2)
    x = np.array([0, np.pi / 4, np.pi / 2])
    assert np.array_equal(combined_mf(x), np.minimum(dummy_mf_1(x), dummy_mf_2(x)))


def test_algebraic_product_tnorm(dummy_mf_1, dummy_mf_2):
    """
    Test the Algebraic Product T-Norm.
    """

    combined_mf = AlgebraicProductTNorm.combine(dummy_mf_1, dummy_mf_2)
    x = np.array([0, np.pi / 4, np.pi / 2])
    assert np.array_equal(combined_mf(x), dummy_mf_1(x) * dummy_mf_2(x))


def test_bounded_product_tnorm(dummy_mf_1, dummy_mf_2):
    """
    Test the Bounded Product T-Norm.
    """

    combined_mf = BoundedProductTNorm.combine(dummy_mf_1, dummy_mf_2)
    x = np.array([0, np.pi / 4, np.pi / 2])
    assert np.array_equal(combined_mf(x), np.maximum(0, dummy_mf_1(x) + dummy_mf_2(x) - 1))


def test_drastic_product_tnorm(dummy_mf_1, dummy_mf_2):
    """
    Test the Drastic Product T-Norm.
    """

    combined_mf = DrasticProductTNorm.combine(dummy_mf_1, dummy_mf_2)
    x = np.array([0, np.pi / 4, np.pi / 2])
    assert np.array_equal(
        combined_mf(x), np.where(dummy_mf_1(x) == 1, dummy_mf_2(x), np.where(dummy_mf_2(x) == 1, dummy_mf_1(x), 0))
    )
