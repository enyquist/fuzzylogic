# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.tconorms.tconorms import AlgebraicSumTCoNorm, BoundedSumTCoNorm, DrasticSumTCoNorm, MaximumTCoNorm


def test_maximum_tconorm(dummy_mf_1, dummy_mf_2):
    """
    Test the Maximum T-CoNorm.
    """

    combined_mf = MaximumTCoNorm.combine(dummy_mf_1, dummy_mf_2)
    x = np.array([0, np.pi / 4, np.pi / 2])

    # Create a meshgrid from the two arrays
    X1, X2 = np.meshgrid(x, x, indexing="ij")

    # Assert that the 2D combined membership function is equal to the expected output
    assert np.array_equal(
        combined_mf(x, x), np.maximum(dummy_mf_1(X1).ravel(), dummy_mf_2(X2).ravel()).reshape(X1.shape)
    )


def test_algebraic_sum_tconorm(dummy_mf_1, dummy_mf_2):
    """
    Test the Algebraic Sum T-CoNorm.
    """

    combined_mf = AlgebraicSumTCoNorm.combine(dummy_mf_1, dummy_mf_2)
    x = np.array([0, np.pi / 4, np.pi / 2])

    X1, X2 = np.meshgrid(x, x, indexing="ij")
    expected_result = (
        dummy_mf_1(X1.ravel()) + dummy_mf_2(X2.ravel()) - (dummy_mf_1(X1.ravel()) * dummy_mf_2(X2.ravel()))
    ).reshape(X1.shape)
    result = combined_mf(x, x)

    assert np.array_equal(result, expected_result)


def test_bounded_sum_tconorm(dummy_mf_1, dummy_mf_2):
    """
    Test the Bounded Sum T-CoNorm.
    """

    combined_mf = BoundedSumTCoNorm.combine(dummy_mf_1, dummy_mf_2)
    x = np.array([0, np.pi / 4, np.pi / 2])

    # Evaluate the membership functions over the grid using meshgrid
    X1, X2 = np.meshgrid(x, x, indexing="ij")

    expected_result = np.minimum(1, dummy_mf_1(X1.ravel()) + dummy_mf_2(X2.ravel())).reshape(X1.shape)

    assert np.array_equal(combined_mf(x, x), expected_result)


def test_drastic_sum_tconorm(dummy_mf_1, dummy_mf_2):
    """
    Test the Drastic Sum T-CoNorm.
    """

    combined_mf = DrasticSumTCoNorm.combine(dummy_mf_1, dummy_mf_2)
    x = np.array([0, np.pi / 4, np.pi / 2])

    expected_result = np.where(dummy_mf_1(x) == 0, dummy_mf_2(x), np.where(dummy_mf_2(x) == 0, dummy_mf_1(x), 1))

    # Extracting the diagonal of the 2D result
    result_diagonal = np.diagonal(combined_mf(x, x))

    assert np.array_equal(result_diagonal, expected_result)
