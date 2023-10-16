# standard libraries
from dataclasses import dataclass

# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.core.mf import MembershipFunction1D


@dataclass
class ConstantMF(MembershipFunction1D):
    """
    Constant membership function.
    """

    value: float

    def __post_init__(self):
        """
        Checks that value is between 0 and 1.
        """
        if self.value < 0 or self.value > 1:
            raise ValueError("value must be between 0 and 1")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the membership function at x.

        Args:
            x (np.ndarray): observations

        Returns:
            np.ndarray: membership values
        """
        return np.full_like(x, fill_value=self.value, dtype=np.float64)
