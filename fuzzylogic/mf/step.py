# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction


class Step(MembershipFunction):
    """
    Step membership function
    """

    def __init__(self, limit: float):
        """
        Initializes the step membership function.
        """

        self.limit = limit

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the membership function at x.

        Args:
            x (np.ndarray): observations

        Returns:
            np.ndarray: membership values
        """
        return np.where(x < self.limit, 0, 1)
