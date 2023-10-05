# standard libraries
from dataclasses import dataclass

# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction


@dataclass
class Bell(MembershipFunction):
    """
    Bell membership function
    """

    mean: float
    std: float
    slope: float

    def __post_init__(self):
        """
        Checks that std is not 0.
        """
        if self.std == 0:
            raise ValueError("std must be non-zero")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the membership function at x.

        Args:
            x (np.ndarray): observations

        Returns:
            np.ndarray: membership values
        """
        return 1 / (1 + np.abs((x - self.mean) / self.std) ** (2 * self.slope))
