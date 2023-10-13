# standard libraries
from abc import ABC, abstractclassmethod

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction1D


class TNorm(ABC):
    """
    Base class for T-Norms.
    """

    @abstractclassmethod
    def combine(cls, mf1: MembershipFunction1D, mf2: MembershipFunction1D) -> MembershipFunction1D:
        """
        Transforms the membership function.
        """
        pass

    def __repr__(self) -> str:
        """
        Returns a string representation of the T-Norm.
        """
        return f"{self.__class__.__name__}"
