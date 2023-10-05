# standard libraries
from dataclasses import dataclass

# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction


@dataclass
class Triangle(MembershipFunction):
    """
    Triangle membership function
    """

    a: float
    b: float
    c: float

    def __post_init__(self):
        """
        Checks that a <= b <= c.
        """
        if self.a > self.b:
            raise ValueError("a must be less than or equal to b")
        if self.b > self.c:
            raise ValueError("b must be less than or equal to c")
        if self.a > self.c:
            raise ValueError("a must be less than or equal to c")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the membership function at x.

        Args:
            x (np.ndarray): observations

        Returns:
            np.ndarray: membership values
        """
        return np.maximum(np.minimum((x - self.a) / (self.b - self.a), (self.c - x) / (self.c - self.b)), 0)
