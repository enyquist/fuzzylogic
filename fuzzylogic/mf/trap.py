# standard libraries
from dataclasses import dataclass

# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.core.mf import MembershipFunction1D


@dataclass
class Trapezoid(MembershipFunction1D):
    """
    Trapezoid membership function
    """

    a: float
    b: float
    c: float
    d: float

    def __post_init__(self):
        """
        Checks that a <= b <= c <= d.
        """
        if self.a > self.b:
            raise ValueError("a must be less than or equal to b")
        if self.b > self.c:
            raise ValueError("b must be less than or equal to c")
        if self.c > self.d:
            raise ValueError("c must be less than or equal to d")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the membership function at x.

        Args:
            x (np.ndarray): observations

        Returns:
            np.ndarray: membership values
        """
        return np.maximum(
            np.minimum(np.minimum((x - self.a) / (self.b - self.a), (self.d - x) / (self.d - self.c)), 1), 0
        )
