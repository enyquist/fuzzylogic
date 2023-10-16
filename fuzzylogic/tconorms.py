# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.core.mf import MembershipFunction1D
from fuzzylogic.core.tconorm import TCoNorm


class MaximumTCoNorm(TCoNorm):
    """
    Maximum T-CoNorm a.k.a. OR operator.
    """

    @classmethod
    def combine(cls, mf1: MembershipFunction1D, mf2: MembershipFunction1D) -> MembershipFunction1D:
        """
        Transforms the membership function.
        """

        class CombinedMF(MembershipFunction1D):
            """
            Combined membership function.
            """

            def __call__(self, x: np.ndarray) -> np.ndarray:
                """
                Evaluates the membership function.
                """

                return np.maximum(mf1(x), mf2(x))

        return CombinedMF()


class AlgebraicSumTCoNorm(TCoNorm):
    """
    Algebraic sum T-CoNorm.
    """

    @classmethod
    def combine(cls, mf1: MembershipFunction1D, mf2: MembershipFunction1D) -> MembershipFunction1D:
        """
        Transforms the membership function.
        """

        class CombinedMF(MembershipFunction1D):
            """
            Combined membership function.
            """

            def __call__(self, x: np.ndarray) -> np.ndarray:
                """
                Evaluates the membership function.
                """

                return mf1(x) + mf2(x) - mf1(x) * mf2(x)

        return CombinedMF()


class BoundedSumTCoNorm(TCoNorm):
    """
    Bounded sum T-CoNorm.
    """

    @classmethod
    def combine(cls, mf1: MembershipFunction1D, mf2: MembershipFunction1D) -> MembershipFunction1D:
        """
        Transforms the membership function.
        """

        class CombinedMF(MembershipFunction1D):
            """
            Combined membership function.
            """

            def __call__(self, x: np.ndarray) -> np.ndarray:
                """
                Evaluates the membership function.
                """

                return np.minimum(1, mf1(x) + mf2(x))

        return CombinedMF()


class DrasticSumTCoNorm(TCoNorm):
    """
    Drastic Sum T-CoNorm.
    """

    @classmethod
    def combine(cls, mf1: MembershipFunction1D, mf2: MembershipFunction1D) -> MembershipFunction1D:
        """
        Transforms the membership function.
        """

        class CombinedMF(MembershipFunction1D):
            """
            Combined membership function.
            """

            def __call__(self, x: np.ndarray) -> np.ndarray:
                """
                Evaluates the membership function.
                """

                return np.where(mf1(x) == 0, mf2(x), np.where(mf2(x) == 0, mf1(x), 1))

        return CombinedMF()
