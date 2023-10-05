# standard libraries
from abc import ABC, abstractmethod
from dataclasses import dataclass

# third party libraries
import numpy as np


@dataclass
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
