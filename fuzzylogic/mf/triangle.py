# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction


class Triangle(MembershipFunction):
    """
    Triangle membership function
    """

    def __init__(self, a: float, b: float, c: float):
        """
        Initializes the triangle membership function.
        """
        if a > b:
            raise ValueError("a must be less than or equal to b")
        if b > c:
            raise ValueError("b must be less than or equal to c")
        if a > c:
            raise ValueError("a must be less than or equal to c")

        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the membership function at x.

        Args:
            x (np.ndarray): observations

        Returns:
            np.ndarray: membership values
        """
        return np.maximum(np.minimum((x - self.a) / (self.b - self.a), (self.c - x) / (self.c - self.b)), 0)
