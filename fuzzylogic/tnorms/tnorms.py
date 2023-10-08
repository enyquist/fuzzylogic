# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction
from fuzzylogic.tnorms.base import TNorm


class MinimumTNorm(TNorm):
    """
    Minimum T-Norm.
    """

    @classmethod
    def combine(cls, mf1: MembershipFunction, mf2: MembershipFunction) -> MembershipFunction:
        """
        Combines two membership functions using the minimum operator.
        """

        class CombinedMF(MembershipFunction):
            """
            Combined membership function using the minimum operator.
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

                # Return the combined membership function using the minimum operator
                return np.minimum(z1, z2)

        return CombinedMF()


class AlgebraicProductTNorm(TNorm):
    """
    Algebraic Product T-Norm.
    """

    @classmethod
    def combine(cls, mf1: MembershipFunction, mf2: MembershipFunction) -> MembershipFunction:
        """
        Combines two membership functions using the algebraic product operator.
        """

        class CombinedMF(MembershipFunction):
            """
            Combined membership function using the algebraic product operator.
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

                # Return the combined membership function using the algebraic product operator
                return z1 * z2

        return CombinedMF()


class BoundedProductTNorm(TNorm):
    """
    Bounded Product T-Norm.
    """

    @classmethod
    def combine(cls, mf1: MembershipFunction, mf2: MembershipFunction) -> MembershipFunction:
        """
        Combines two membership functions using the bounded product operator.
        """

        class CombinedMF(MembershipFunction):
            """
            Combined membership function using the bounded product operator.
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

                # Return the combined membership function using the bounded product operator
                return np.maximum(0, z1 + z2 - 1)

        return CombinedMF()


class DrasticProductTNorm(TNorm):
    """
    Drastic Product T-Norm.
    """

    @classmethod
    def combine(cls, mf1: MembershipFunction, mf2: MembershipFunction) -> MembershipFunction:
        """
        Combines two membership functions using the drastic product operator.
        """

        class CombinedMF(MembershipFunction):
            """
            Combined membership function using the drastic product operator.
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

                # Combine using the drastic product
                result = np.where(z1 == 1, z2, np.where(z2 == 1, z1, 0))

                # Reshape the result to match the meshgrid shape
                return result.reshape(X1.shape)

        return CombinedMF()
