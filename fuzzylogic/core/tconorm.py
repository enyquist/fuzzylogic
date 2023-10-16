# standard libraries
from abc import ABC, abstractclassmethod

# fuzzy logic libraries
from fuzzylogic.core.mf import MembershipFunction1D


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

    def __repr__(self) -> str:
        """
        Returns a string representation of the T-CoNorm.
        """
        return f"{self.__class__.__name__}"
