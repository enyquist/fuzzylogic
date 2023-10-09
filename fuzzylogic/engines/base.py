# standard libraries
from abc import ABC, abstractclassmethod

# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction1D


class FuzzyEngine(ABC):
    """
    Fuzzy Inference Engine base class.
    """

    @abstractclassmethod
    def defuzz(cls, x: np.ndarray, mf: MembershipFunction1D) -> float:
        """
        Defuzzifies the membership function.
        """
        pass
