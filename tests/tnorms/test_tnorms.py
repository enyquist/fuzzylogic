# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.tnorms.tnorms import AlgebraicProductTNorm, BoundedProductTNorm, DrasticProductTNorm, MinimumTNorm


def test_minimum_tnorm(dummy_mf_1, dummy_mf_2):
    """
    Test the Minimum T-Norm.
    """

    combined_mf = MinimumTNorm.combine(dummy_mf_1, dummy_mf_2)
    x = np.array([0, np.pi / 4, np.pi / 2])

    # Create a meshgrid to evaluate the combined function in 2D
    X1, X2 = np.meshgrid(x, x, indexing="ij")
    combined_values = combined_mf(x, x)

    # Evaluate dummy membership functions and get the minimum
    expected_values = np.minimum(dummy_mf_1(X1.ravel()), dummy_mf_2(X2.ravel())).reshape(X1.shape)

    assert np.array_equal(combined_values, expected_values)


def test_algebraic_product_tnorm(dummy_mf_1, dummy_mf_2):
    """
    Test the Algebraic Product T-Norm.
    """

    combined_mf = AlgebraicProductTNorm.combine(dummy_mf_1, dummy_mf_2)
    x = np.array([0, np.pi / 4, np.pi / 2])

    # Create a meshgrid from the input array
    X1, X2 = np.meshgrid(x, x, indexing="ij")

    # Expected output
    expected_output = dummy_mf_1(X1) * dummy_mf_2(X2)

    assert np.allclose(combined_mf(x, x), expected_output)


def test_bounded_product_tnorm(dummy_mf_1, dummy_mf_2):
    """
    Test the Bounded Product T-Norm.
    """

    combined_mf = BoundedProductTNorm.combine(dummy_mf_1, dummy_mf_2)
    x = np.array([0, np.pi / 4, np.pi / 2])

    expected_result = np.maximum(0, dummy_mf_1(x[:, None]) + dummy_mf_2(x[None, :]) - 1)

    assert np.array_equal(combined_mf(x, x), expected_result)


def test_drastic_product_tnorm(dummy_mf_1, dummy_mf_2):
    """
    Test the Drastic Product T-Norm.
    """

    combined_mf = DrasticProductTNorm.combine(dummy_mf_1, dummy_mf_2)
    x = np.array([0, np.pi / 4, np.pi / 2])

    # Create meshgrid for evaluating the combined membership function
    X1, X2 = np.meshgrid(x, x, indexing="ij")

    # Evaluate the dummy membership functions over the grid
    z1 = dummy_mf_1(X1.ravel())
    z2 = dummy_mf_2(X2.ravel())

    # Compute the expected result
    expected_result = np.where(z1 == 1, z2, np.where(z2 == 1, z1, 0)).reshape(X1.shape)

    assert np.array_equal(combined_mf(x, x), expected_result)
