# third party libraries
import numpy as np
import pytest

# fuzzy logic libraries
from fuzzylogic.core.mf import MembershipFunction1D


@pytest.fixture(scope="session")
def dummy_mf_1():
    """
    Dummy membership function for testing connectives.
    """

    class DummyMembershipFunction1(MembershipFunction1D):
        """
        Sin function scaled to be between 0 and 1.
        """

        def __call__(self, x: np.ndarray) -> np.ndarray:
            return (np.sin(x) + 1) / 2

    return DummyMembershipFunction1()


@pytest.fixture(scope="session")
def dummy_mf_2():
    """
    Dummy membership function for testing connectives.
    """

    class DummyMembershipFunction2(MembershipFunction1D):
        """
        Cos function scaled to be between 0 and 1.
        """

        def __call__(self, x: np.ndarray) -> np.ndarray:
            return (np.cos(x) + 1) / 2

    return DummyMembershipFunction2()


@pytest.fixture(scope="session")
def simple_mf():
    """
    Simple membership function for testing.
    """

    class SimpleMF(MembershipFunction1D):
        """
        Simple membership function.
        """

        def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return x + y

    return SimpleMF()
