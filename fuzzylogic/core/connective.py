# standard libraries
from abc import ABC, abstractclassmethod

# fuzzy logic libraries
from fuzzylogic.core.mf import MembershipFunction1D


class Connective(ABC):
    """
    Linguistic Connective base class i.e. AND, OR, etc.
    """

    @abstractclassmethod
    def combine(cls, mf1: MembershipFunction1D, mf2: MembershipFunction1D) -> MembershipFunction1D:
        """
        Transforms the membership function.
        """
        pass
