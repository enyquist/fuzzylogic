# standard libraries
from abc import ABC, abstractmethod

# third party libraries
import numpy as np


class MembershipFunction(ABC):
    """
    Abstract class for membership functions.
    """

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the membership function at x.
        """
        pass

    def inverse(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the complement of the membership function (1 - membership function).

        Args:
            x (np.ndarray): observations

        Returns:
            np.ndarray: complement of the membership function
        """
        return 1 - self(x)
