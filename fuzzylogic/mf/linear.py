# standard libraries
from dataclasses import dataclass

# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction1D


@dataclass
class Linear(MembershipFunction1D):
    """
    Linear membership function
    """

    m: float
    b: float

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the membership function at x.

        Args:
            x (np.ndarray): observations

        Returns:
            np.ndarray: membership values
        """
        return np.clip(self.m * x + self.b, 0, 1)
