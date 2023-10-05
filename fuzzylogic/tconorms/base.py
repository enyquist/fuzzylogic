# standard libraries
from abc import ABC, abstractclassmethod

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction


class TCoNorm(ABC):
    """
    Base class for T-CoNorms a.k.a. S-Norms.
    """

    @abstractclassmethod
    def combine(cls, mf1: MembershipFunction, mf2: MembershipFunction) -> MembershipFunction:
        """
        Transforms the membership function.
        """
        pass
