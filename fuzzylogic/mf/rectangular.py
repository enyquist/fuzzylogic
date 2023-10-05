# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction


class Rectangular(MembershipFunction):
    """
    Rectangular membership function
    """

    def __init__(self, low: float, high: float):
        """
        Initializes the rectangular membership function.
        """
        if low > high:
            raise ValueError("low must be less than or equal to high")

        self.low = low
        self.high = high

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the membership function at x.

        Args:
            x (np.ndarray): observations

        Returns:
            np.ndarray: membership values
        """
        return np.logical_and(x >= self.low, x <= self.high).astype(float)
