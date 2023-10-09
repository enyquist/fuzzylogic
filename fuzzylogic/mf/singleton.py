# standard libraries
from dataclasses import dataclass

# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction


@dataclass
class FuzzySingleton(MembershipFunction):
    """
    Fuzzy singleton membership function.
    """

    value: float  # the crisp value of the fuzzy singleton

    def __post_init__(self):
        """
        Validates the value of the fuzzy singleton.
        """

        if not isinstance(self.value, (int, float)):
            raise ValueError(f"Invalid value type. Received {type(self.value)}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the membership function at x.

        Args:
            x (np.ndarray): inputs

        Returns:
            np.ndarray: Membership function evaluated at x.
        """

        return np.where(x == self.value, 1, 0)
