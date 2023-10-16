# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.core.mf import MembershipFunction1D
from fuzzylogic.core.tnorm import TNorm


class MinimumTNorm(TNorm):
    """
    Minimum T-Norm.
    """

    @classmethod
    def combine(cls, mf1: MembershipFunction1D, mf2: MembershipFunction1D) -> MembershipFunction1D:
        """
        Combines two membership functions using the minimum operator.
        """

        class CombinedMF(MembershipFunction1D):
            """
            Combined membership function using the minimum operator.
            """

            def __call__(self, x: np.ndarray) -> np.ndarray:
                """
                Evaluates the membership function.
                """

                return np.minimum(mf1(x), mf2(x))

        return CombinedMF()


class AlgebraicProductTNorm(TNorm):
    """
    Algebraic Product T-Norm.
    """

    @classmethod
    def combine(cls, mf1: MembershipFunction1D, mf2: MembershipFunction1D) -> MembershipFunction1D:
        """
        Combines two membership functions using the algebraic product operator.
        """

        class CombinedMF(MembershipFunction1D):
            """
            Combined membership function using the algebraic product operator.
            """

            def __call__(self, x: np.ndarray) -> np.ndarray:
                """
                Evaluates the membership function.
                """

                return mf1(x) * mf2(x)

        return CombinedMF()


class BoundedProductTNorm(TNorm):
    """
    Bounded Product T-Norm.
    """

    @classmethod
    def combine(cls, mf1: MembershipFunction1D, mf2: MembershipFunction1D) -> MembershipFunction1D:
        """
        Combines two membership functions using the bounded product operator.
        """

        class CombinedMF(MembershipFunction1D):
            """
            Combined membership function using the bounded product operator.
            """

            def __call__(self, x: np.ndarray) -> np.ndarray:
                """
                Evaluates the membership function.
                """

                return np.maximum(0, mf1(x) + mf2(x) - 1)

        return CombinedMF()


class DrasticProductTNorm(TNorm):
    """
    Drastic Product T-Norm.
    """

    @classmethod
    def combine(cls, mf1: MembershipFunction1D, mf2: MembershipFunction1D) -> MembershipFunction1D:
        """
        Combines two membership functions using the drastic product operator.
        """

        class CombinedMF(MembershipFunction1D):
            """
            Combined membership function using the drastic product operator.
            """

            def __call__(self, x: np.ndarray) -> np.ndarray:
                """
                Evaluates the membership function.
                """

                return np.where(mf1(x) == 1, mf2(x), np.where(mf2(x) == 1, mf1(x), 0))

        return CombinedMF()
