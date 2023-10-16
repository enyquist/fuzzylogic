# standard libraries
from dataclasses import dataclass
from typing import Callable

# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.core.mf import MembershipFunction2D


@dataclass
class UserDefinedMF(MembershipFunction2D):
    """
    User defined membership function.

    Used for TSK Engines
    """

    func: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Evaluates the membership function.

        Args:
            x (np.ndarray): Inputs in universe of discourse x
            y (np.ndarray): Inputs in universe of discourse y

        Returns:
            np.ndarray: membership values
        """

        # Check if the inputs are valid
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be a numpy array. Got {type(x)}.")

        if not isinstance(y, np.ndarray):
            raise TypeError(f"y must be a numpy array. Got {type(y)}.")

        if x.shape != y.shape:
            raise ValueError(f"x and y must have the same shape. Got {x.shape} and {y.shape}.")

        return self.func(x, y)
