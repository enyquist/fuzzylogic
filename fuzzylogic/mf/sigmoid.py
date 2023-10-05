# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction


class Sigmoid(MembershipFunction):
    """
    Sigmoid membership function
    """

    def __init__(self, a: float, b: float):
        """
        Initializes the sigmoid membership function.
        """

        self.a = a
        self.b = b

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the membership function at x.

        Args:
            x (np.ndarray): observations

        Returns:
            np.ndarray: membership values
        """
        return 1 / (1 + np.exp(-self.a * (x - self.b)))
