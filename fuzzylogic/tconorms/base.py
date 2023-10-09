# standard libraries
from abc import ABC, abstractclassmethod

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction1D


class TCoNorm(ABC):
    """
    Base class for T-CoNorms a.k.a. S-Norms.
    """

    @abstractclassmethod
    def combine(cls, mf1: MembershipFunction1D, mf2: MembershipFunction1D) -> MembershipFunction1D:
        """
        Transforms the membership function.
        """
        pass
