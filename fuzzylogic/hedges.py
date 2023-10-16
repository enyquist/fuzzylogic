# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.core.hedge import Hedge
from fuzzylogic.core.mf import MembershipFunction1D


class Not(Hedge):
    """
    Not hedge. Creates the complement of the membership function.
    """

    @classmethod
    def transform(cls, mf: MembershipFunction1D) -> MembershipFunction1D:
        """
        Transforms the membership function.
        """

        class ComplimentedMF(MembershipFunction1D):
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
    def transform(cls, mf: MembershipFunction1D) -> MembershipFunction1D:
        """
        Transforms the membership function.
        """

        class ConcentratedMF(MembershipFunction1D):
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
    def transform(cls, mf: MembershipFunction1D) -> MembershipFunction1D:
        """
        Transforms the membership function.
        """

        class DilatedMF(MembershipFunction1D):
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
    def transform(cls, mf: MembershipFunction1D) -> MembershipFunction1D:
        """
        Transforms the membership function.
        """

        class IntensifiedMF(MembershipFunction1D):
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


class Dim(Hedge):
    """
    Contrast Diminisher hedge. Creates the diminishment of the membership function.
    """

    @classmethod
    def transform(cls, mf: MembershipFunction1D) -> MembershipFunction1D:
        """
        Transforms the membership function.
        """

        class DiminishedMF(MembershipFunction1D):
            """
            Diminished membership function.
            """

            def __call__(self, x: np.ndarray) -> np.ndarray:
                """
                Evaluates the membership function at x.
                """
                u = mf(x)
                result = np.where(u < 0.5, 0.5 * u**0.5, 1 - 0.5 * (1 - u) ** 0.5)
                return result

        return DiminishedMF()
