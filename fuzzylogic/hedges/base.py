# standard libraries
from abc import ABC, abstractclassmethod

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction1D


class Hedge(ABC):
    """
    Linguistic Hedge base class.
    """

    @abstractclassmethod
    def transform(cls, mf: MembershipFunction1D) -> MembershipFunction1D:
        """
        Transforms the membership function.
        """
        pass
