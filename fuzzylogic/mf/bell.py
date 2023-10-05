# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction


class Bell(MembershipFunction):
    """
    Bell membership function
    """

    def __init__(self, mean: float, std: float, slope: float):
        """
        Initializes the Bell membership function.
        """

        self.mean = mean
        self.std = std
        self.slope = slope

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the membership function at x.

        Args:
            x (np.ndarray): observations

        Returns:
            np.ndarray: membership values
        """
        return 1 / (1 + np.abs((x - self.mean) / self.std) ** (2 * self.slope))
