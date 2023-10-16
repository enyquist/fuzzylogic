# standard libraries
from dataclasses import dataclass

# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.core.mf import MembershipFunction1D


@dataclass
class Bell(MembershipFunction1D):
    """
    Bell membership function
    """

    center: float
    width: float
    intensity: float

    def __post_init__(self):
        """
        Checks that width is not 0.
        """
        if self.width == 0:
            raise ValueError("width must be non-zero")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the membership function at x.

        Args:
            x (np.ndarray): observations

        Returns:
            np.ndarray: membership values
        """
        return 1 / (1 + np.abs((x - self.center) / self.width) ** (2 * self.intensity))
