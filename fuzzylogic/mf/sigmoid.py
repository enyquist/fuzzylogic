# standard libraries
from dataclasses import dataclass

# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.core.mf import MembershipFunction1D


@dataclass
class Sigmoid(MembershipFunction1D):
    """
    Sigmoid membership function
    """

    center_slope: float
    center: float

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the membership function at x.

        Args:
            x (np.ndarray): observations

        Returns:
            np.ndarray: membership values
        """
        return 1 / (1 + np.exp(-self.center_slope * (x - self.center)))
