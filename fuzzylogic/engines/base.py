# standard libraries
from abc import ABC, abstractclassmethod

# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction


class FuzzyEngine(ABC):
    """
    Fuzzy Inference Engine base class.
    """

    @abstractclassmethod
    def defuzz(cls, x: np.ndarray, mf: MembershipFunction) -> float:
        """
        Defuzzifies the membership function.
        """
        pass
