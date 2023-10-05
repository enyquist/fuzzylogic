# standard libraries
from abc import ABC, abstractclassmethod

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction


class Hedge(ABC):
    """
    Linguistic Hedge base class.
    """

    @abstractclassmethod
    def transform(cls, mf: MembershipFunction) -> MembershipFunction:
        """
        Transforms the membership function.
        """
        pass
