# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction


class Gaussian(MembershipFunction):
    """
    Gaussian membership function
    """

    def __init__(self, mean: float, std: float):
        """
        Initializes the Gaussian membership function.
        """

        self.mean = mean
        self.std = std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the membership function at x.

        Args:
            x (np.ndarray): observations

        Returns:
            np.ndarray: membership values
        """
        return np.exp(-0.5 * ((x - self.mean) / self.std) ** 2)
