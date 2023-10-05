# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.connectives.base import Connective
from fuzzylogic.mf.base import MembershipFunction


class And(Connective):
    """
    And connective. Creates the intersection of the membership functions.
    """

    @classmethod
    def combine(cls, mf1: MembershipFunction, mf2: MembershipFunction) -> MembershipFunction:
        """
        Transforms the membership function.
        """

        class IntersectedMF(MembershipFunction):
            """
            Intersected membership function.
            """

            def __call__(self, x: np.ndarray) -> np.ndarray:
                """
                Evaluates the membership function at x.
                """
                return np.minimum(mf1(x), mf2(x))

        return IntersectedMF()


class Or(Connective):
    """
    Or connective. Creates the union of the membership functions.
    """

    @classmethod
    def combine(cls, mf1: MembershipFunction, mf2: MembershipFunction) -> MembershipFunction:
        """
        Transforms the membership function.
        """

        class UnionedMF(MembershipFunction):
            """
            Unioned membership function.
            """

            def __call__(self, x: np.ndarray) -> np.ndarray:
                """
                Evaluates the membership function at x.
                """

                return np.maximum(mf1(x), mf2(x))

        return UnionedMF()
