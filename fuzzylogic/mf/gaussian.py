# standard libraries
from dataclasses import dataclass

# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.core.mf import MembershipFunction1D


@dataclass
class Gaussian(MembershipFunction1D):
    """
    Gaussian membership function
    """

    mean: float
    std: float

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the membership function at x.

        Args:
            x (np.ndarray): observations

        Returns:
            np.ndarray: membership values
        """
        return np.exp(-0.5 * ((x - self.mean) / self.std) ** 2)
