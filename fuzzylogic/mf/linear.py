# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction


class Linear(MembershipFunction):
    """
    Linear membership function
    """

    def __init__(self, a: float, b: float):
        """
        Initializes the linear membership function.
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
        return np.clip(self.a * x + self.b, 0, 1)
