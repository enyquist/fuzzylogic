# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction
from fuzzylogic.tconorms.base import TCoNorm


class MaximumTCoNorm(TCoNorm):
    """
    Maximum T-CoNorm a.k.a. OR operator.
    """

    @classmethod
    def combine(cls, mf1: MembershipFunction, mf2: MembershipFunction) -> MembershipFunction:
        """
        Transforms the membership function.
        """

        class CombinedMF(MembershipFunction):
            """
            Combined membership function.
            """

            def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
                """
                Evaluates the membership function.
                """

                # If x1 and x2 are 1D, turn them into a meshgrid
                if len(x1.shape) == 1 and len(x2.shape) == 1:
                    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
                else:
                    X1, X2 = x1, x2

                # Evaluate the membership functions
                z1 = mf1(X1)
                z2 = mf2(X2)

                # Return the combined membership function using the maximum operator
                return np.maximum(z1, z2)

        return CombinedMF()


class AlgebraicSumTCoNorm(TCoNorm):
    """
    Algebraic sum T-CoNorm.
    """

    @classmethod
    def combine(cls, mf1: MembershipFunction, mf2: MembershipFunction) -> MembershipFunction:
        """
        Transforms the membership function.
        """

        class CombinedMF(MembershipFunction):
            """
            Combined membership function.
            """

            def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
                """
                Evaluates the membership function.
                """

                # If x1 and x2 are 1D, turn them into a meshgrid
                if len(x1.shape) == 1 and len(x2.shape) == 1:
                    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
                else:
                    X1, X2 = x1, x2

                # Evaluate the membership functions
                z1 = mf1(X1)
                z2 = mf2(X2)

                # Return the combined membership function using the algebraic sum operator
                return z1 + z2 - z1 * z2

        return CombinedMF()


class BoundedSumTCoNorm(TCoNorm):
    """
    Bounded sum T-CoNorm.
    """

    @classmethod
    def combine(cls, mf1: MembershipFunction, mf2: MembershipFunction) -> MembershipFunction:
        """
        Transforms the membership function.
        """

        class CombinedMF(MembershipFunction):
            """
            Combined membership function.
            """

            def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
                """
                Evaluates the membership function.
                """

                # If x1 and x2 are 1D, turn them into a meshgrid
                if len(x1.shape) == 1 and len(x2.shape) == 1:
                    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
                else:
                    X1, X2 = x1, x2

                # Evaluate the membership functions
                z1 = mf1(X1)
                z2 = mf2(X2)

                # Return the combined membership function using the bounded sum operator
                return np.minimum(1, z1 + z2)

        return CombinedMF()


class DrasticSumTCoNorm(TCoNorm):
    """
    Drastic Sum T-CoNorm.
    """

    @classmethod
    def combine(cls, mf1: MembershipFunction, mf2: MembershipFunction) -> MembershipFunction:
        """
        Transforms the membership function.
        """

        class CombinedMF(MembershipFunction):
            """
            Combined membership function.
            """

            def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
                """
                Evaluates the membership function.
                """

                # If x1 and x2 are 1D, turn them into a meshgrid
                if len(x1.shape) == 1 and len(x2.shape) == 1:
                    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
                else:
                    X1, X2 = x1, x2

                # Evaluate the membership functions
                z1 = mf1(X1)
                z2 = mf2(X2)

                # Return the combined membership function
                return np.where(z1 == 0, z2, np.where(z2 == 0, z1, 1)).reshape(X1.shape)

        return CombinedMF()
