# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.connectives import And, Or


class TestAndConnective:
    def test_intersection_effect(self, dummy_mf_1, dummy_mf_2):
        """
        Test the intersection effect of the And connective.
        """

        intersected_mf = And.combine(dummy_mf_1, dummy_mf_2)
        x_vals = np.linspace(-2 * np.pi, 2 * np.pi, 500)
        assert np.array_equal(intersected_mf(x_vals), np.minimum(dummy_mf_1(x_vals), dummy_mf_2(x_vals)))

    def test_output_range(self, dummy_mf_1, dummy_mf_2):
        """
        Test the output range of the And connective.
        """

        intersected_mf = And.combine(dummy_mf_1, dummy_mf_2)
        x_vals = np.linspace(-100, 100, 500)
        y_vals = intersected_mf(x_vals)
        assert np.all(y_vals >= 0)
        assert np.all(y_vals <= 1)


class TestOrConnective:
    def test_union_effect(self, dummy_mf_1, dummy_mf_2):
        """
        Test the union effect of the Or connective.
        """

        unioned_mf = Or.combine(dummy_mf_1, dummy_mf_2)
        x_vals = np.linspace(-2 * np.pi, 2 * np.pi, 500)
        assert np.array_equal(unioned_mf(x_vals), np.maximum(dummy_mf_1(x_vals), dummy_mf_2(x_vals)))

    def test_output_range(self, dummy_mf_1, dummy_mf_2):
        """
        Test the output range of the Or connective.
        """

        unioned_mf = Or.combine(dummy_mf_1, dummy_mf_2)
        x_vals = np.linspace(-100, 100, 500)
        y_vals = unioned_mf(x_vals)
        assert np.all(y_vals >= 0)
        assert np.all(y_vals <= 1)
