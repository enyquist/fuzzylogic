# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.core.defuzz import Defuzzification
from fuzzylogic.core.mf import MembershipFunction1D


class Centroid(Defuzzification):
    """
    Centroid defuzzification method.
    """

    @classmethod
    def defuzz(cls, x: np.ndarray, mf: MembershipFunction1D) -> float:
        """
        Defuzzifies the membership function.
        """

        # Compute the summation of x[i] * mf(x[i]) and the summation of mf(x[i])
        numerator = np.sum(x * mf(x))

        denominator = np.sum(mf(x))

        # Avoid division by zero
        if denominator == 0:
            raise ValueError("The membership function has an area of zero, cannot compute COA.")

        return numerator / denominator


class Bisector(Defuzzification):
    """
    Bisector of Area defuzzification method.
    """

    @classmethod
    def defuzz(cls, x: np.ndarray, mf: MembershipFunction1D) -> float:
        """
        Defuzzifies the membership function.
        """

        # Compute the cumulative membership values
        cumulative_mf = np.cumsum(mf(x))

        # Total area
        total_area = cumulative_mf[-1]

        # If total area is zero, raise an error
        if total_area == 0:
            raise ValueError("The membership function has an area of zero, cannot compute BOA.")

        # Find the index where the cumulative area is half of the total area
        bisector_index = np.argmin(np.abs(cumulative_mf - total_area / 2))

        return x[bisector_index]


class MeanOfMaximum(Defuzzification):
    """
    Mean of Maximum defuzzification method.
    """

    @classmethod
    def defuzz(cls, x: np.ndarray, mf: MembershipFunction1D) -> float:
        """
        Defuzzifies the membership function.
        """

        # Find the maximum membership values
        maximum_mf = np.max(mf(x))

        # Find the indices where the maximum membership values are
        maximum_indices = np.where(mf(x) == maximum_mf)[0]

        # Return the mean of the indices
        return np.mean(x[maximum_indices])


class LargestOfMaximum(Defuzzification):
    """
    Largest of Maximum defuzzification method.
    """

    @classmethod
    def defuzz(cls, x: np.ndarray, mf: MembershipFunction1D) -> float:
        """
        Defuzzifies the membership function.
        """

        # Find the maximum membership values
        maximum_mf = np.max(mf(x))

        # Find the indices where the maximum membership values are
        maximum_indices = np.where(mf(x) == maximum_mf)[0]

        # Return the largest of the indices
        return np.max(x[maximum_indices])


class SmallestOfMaximum(Defuzzification):
    """
    Smallest of Maximum defuzzification method.
    """

    @classmethod
    def defuzz(cls, x: np.ndarray, mf: MembershipFunction1D) -> float:
        """
        Defuzzifies the membership function.
        """

        # Find the maximum membership values
        maximum_mf = np.max(mf(x))

        # Find the indices where the maximum membership values are
        maximum_indices = np.where(mf(x) == maximum_mf)[0]

        # Return the smallest of the indices
        return np.min(x[maximum_indices])


DEFUZZ = {
    "centroid": Centroid,
    "bisector": Bisector,
    "mom": MeanOfMaximum,
    "lom": LargestOfMaximum,
    "som": SmallestOfMaximum,
}
