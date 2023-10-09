# standard libraries
from dataclasses import dataclass

# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction1D


@dataclass
class Rectangular(MembershipFunction1D):
    """
    Rectangular membership function
    """

    low: float
    high: float

    def __post_init__(self):
        """
        Checks that low is less than or equal to high.
        """
        if self.low > self.high:
            raise ValueError("low must be less than or equal to high")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the membership function at x.

        Args:
            x (np.ndarray): observations

        Returns:
            np.ndarray: membership values
        """
        return np.logical_and(x >= self.low, x <= self.high).astype(float)
