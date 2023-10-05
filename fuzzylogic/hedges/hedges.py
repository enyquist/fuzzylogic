# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.hedges.base import Hedge
from fuzzylogic.mf.base import MembershipFunction


class Not(Hedge):
    """
    Not hedge. Creates the complement of the membership function.
    """

    @classmethod
    def transform(cls, mf: MembershipFunction) -> MembershipFunction:
        """
        Transforms the membership function.
        """

        class ComplimentedMF(MembershipFunction):
            """
            Complimented membership function.
            """

            def __call__(self, x: np.ndarray) -> np.ndarray:
                """
                Evaluates the membership function at x.
                """
                return 1 - mf(x)

        return ComplimentedMF()


class Con(Hedge):
    """
    Con hedge. Creates the concentration of the membership function.
    """

    @classmethod
    def transform(cls, mf: MembershipFunction) -> MembershipFunction:
        """
        Transforms the membership function.
        """

        class ConcentratedMF(MembershipFunction):
            """
            Concentrated membership function.
            """

            def __call__(self, x: np.ndarray) -> np.ndarray:
                """
                Evaluates the membership function at x.
                """
                return mf(x) ** 2

        return ConcentratedMF()


class Dil(Hedge):
    """
    Dil hedge. Creates the dilation of the membership function.
    """

    @classmethod
    def transform(cls, mf: MembershipFunction) -> MembershipFunction:
        """
        Transforms the membership function.
        """

        class DilatedMF(MembershipFunction):
            """
            Dilated membership function.
            """

            def __call__(self, x: np.ndarray) -> np.ndarray:
                """
                Evaluates the membership function at x.
                """
                return np.sqrt(mf(x))

        return DilatedMF()


class Int(Hedge):
    """
    Contrast Intensifier hedge. Creates the intensification of the membership function.
    """

    @classmethod
    def transform(cls, mf: MembershipFunction) -> MembershipFunction:
        """
        Transforms the membership function.
        """

        class IntensifiedMF(MembershipFunction):
            """
            Intensified membership function.
            """

            def __call__(self, x: np.ndarray) -> np.ndarray:
                """
                Evaluates the membership function at x.
                """
                u = mf(x)
                result = np.where(u < 0.5, 2 * u**2, 1 - 2 * (1 - u) ** 2)
                return result

        return IntensifiedMF()
