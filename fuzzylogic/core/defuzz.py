# standard libraries
from abc import ABC, abstractclassmethod

# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.core.mf import MembershipFunction1D


class Defuzzification(ABC):
    """
    Fuzzy Inference Engine base class.
    """

    @abstractclassmethod
    def defuzz(cls, x: np.ndarray, mf: MembershipFunction1D) -> float:
        """
        Defuzzifies the membership function.
        """
        pass
