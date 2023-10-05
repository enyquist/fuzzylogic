# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.hedges.hedges import Con, Dil, Int, Not


class TestNotHedge:
    def test_not_transformation(self, dummy_mf_1):
        """
        Test the Not hedge transformation.
        """

        # Apply Not hedge
        complimented_mf = Not.transform(dummy_mf_1)

        # Define test data
        x_vals = np.array([0, 1, 2])
        expected_values = np.array([0.5, 1 - (np.sin(1) + 1) / 2, 1 - (np.sin(2) + 1) / 2])

        # Check transformation
        assert np.array_equal(complimented_mf(x_vals), expected_values)

    def test_output_range(self, dummy_mf_1):
        """
        Test the output range of the Not hedge.
        """

        complimented_mf = Not.transform(dummy_mf_1)

        x_vals = np.linspace(-100, 100, 500)
        y_vals = complimented_mf(x_vals)

        assert np.all(y_vals >= 0)
        assert np.all(y_vals <= 1)


class TestConHedge:
    def test_con_transformation(self, dummy_mf_1):
        """
        Test the Con hedge transformation.
        """

        # Apply Con hedge
        concentrated_mf = Con.transform(dummy_mf_1)

        # Define test data
        x_vals = np.array([0, 1, 2])
        expected_values = ((np.sin(x_vals) + 1) / 2) ** 2

        # Check transformation
        assert np.array_equal(concentrated_mf(x_vals), expected_values)

    def test_output_range(self, dummy_mf_1):
        """
        Test the output range of the Con hedge.
        """

        concentrated_mf = Con.transform(dummy_mf_1)

        x_vals = np.linspace(-100, 100, 500)
        y_vals = concentrated_mf(x_vals)

        assert np.all(y_vals >= 0)
        assert np.all(y_vals <= 1)

    def test_concentration_effect(self, dummy_mf_1):
        """
        Test the concentration effect of the Con hedge.
        """

        concentrated_mf = Con.transform(dummy_mf_1)

        x_vals = np.linspace(-2 * np.pi, 2 * np.pi, 500)
        original_y_vals = dummy_mf_1(x_vals)
        concentrated_y_vals = concentrated_mf(x_vals)

        # All concentrated values should be less than or equal to the original values.
        assert np.all(concentrated_y_vals <= original_y_vals)


class TestDilHedge:
    def test_dil_transformation(self, dummy_mf_1):
        """
        Test the Dil hedge transformation.
        """

        # Apply Dil hedge
        dilated_mf = Dil.transform(dummy_mf_1)

        # Define test data
        x_vals = np.array([0, 1, 2])
        expected_values = np.sqrt((np.sin(x_vals) + 1) / 2)

        # Check transformation
        assert np.array_equal(dilated_mf(x_vals), expected_values)

    def test_dilation_effect(self, dummy_mf_1):
        """
        Test the dilation effect of the Dil hedge.
        """

        # Ensure that the dilation is having the desired effect
        dilated_mf = Dil.transform(dummy_mf_1)

        x_vals = np.linspace(-2 * np.pi, 2 * np.pi, 500)
        original_y_vals = dummy_mf_1(x_vals)
        dilated_y_vals = dilated_mf(x_vals)

        # All dilated values should be greater than or equal to the original values
        assert np.all(dilated_y_vals >= original_y_vals)

    def test_output_range(self, dummy_mf_1):
        """
        Test the output range of the Dil hedge.
        """

        # Ensure the output is always between 0 and 1
        dilated_mf = Dil.transform(dummy_mf_1)

        x_vals = np.linspace(-100, 100, 500)
        y_vals = dilated_mf(x_vals)

        assert np.all(y_vals >= 0)
        assert np.all(y_vals <= 1)


class TestIntHedge:
    def test_int_transformation(self, dummy_mf_1):
        """
        Test the Int hedge transformation.
        """

        # Apply Int hedge
        intensified_mf = Int.transform(dummy_mf_1)

        # Define test data
        x_vals = np.array([0, 1, 2])
        u_values = (np.sin(x_vals) + 1) / 2
        expected_values = np.where(u_values < 0.5, 2 * u_values**2, 1 - 2 * (1 - u_values) ** 2)

        # Check transformation
        assert np.array_equal(intensified_mf(x_vals), expected_values)

    def test_output_range(self, dummy_mf_1):
        """
        Test the output range of the Int hedge.
        """

        intensified_mf = Int.transform(dummy_mf_1)

        x_vals = np.linspace(-100, 100, 500)
        y_vals = intensified_mf(x_vals)

        assert np.all(y_vals >= 0)
        assert np.all(y_vals <= 1)

    def test_intensification_effect(self, dummy_mf_1):
        """
        Test the intensification effect of the Int hedge.
        """

        intensified_mf = Int.transform(dummy_mf_1)

        x_vals = np.linspace(-2 * np.pi, 2 * np.pi, 500)
        original_y_vals = dummy_mf_1(x_vals)
        intensified_y_vals = intensified_mf(x_vals)

        # For original values below 0.5, intensified values should be less than original.
        # For original values above 0.5, intensified values should be more than original.
        assert np.all(intensified_y_vals[original_y_vals < 0.5] <= original_y_vals[original_y_vals < 0.5])
        assert np.all(intensified_y_vals[original_y_vals > 0.5] >= original_y_vals[original_y_vals > 0.5])
