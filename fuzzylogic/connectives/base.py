# standard libraries
from abc import ABC, abstractclassmethod

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction


class Connective(ABC):
    """
    Linguistic Connective base class i.e. AND, OR, etc.
    """

    @abstractclassmethod
    def combine(cls, mf1: MembershipFunction, mf2: MembershipFunction) -> MembershipFunction:
        """
        Transforms the membership function.
        """
        pass
