# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction


class Trapezoid(MembershipFunction):
    """
    Trapezoid membership function
    """

    def __init__(self, a: float, b: float, c: float, d: float):
        """
        Initializes the trapezoid membership function.
        """
        if a > b:
            raise ValueError("a must be less than or equal to b")
        if b > c:
            raise ValueError("b must be less than or equal to c")
        if c > d:
            raise ValueError("c must be less than or equal to d")

        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the membership function at x.

        Args:
            x (np.ndarray): observations

        Returns:
            np.ndarray: membership values
        """
        return np.maximum(
            np.minimum(np.minimum(x - self.a) / (self.b - self.a), 1, (self.d - x) / (self.d - self.c)), 0
        )
