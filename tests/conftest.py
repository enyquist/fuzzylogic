# third party libraries
import numpy as np
import pytest

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction


@pytest.fixture(scope="session")
def dummy_mf_1():
    """
    Dummy membership function for testing connectives.
    """

    class DummyMembershipFunction1(MembershipFunction):
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

    class DummyMembershipFunction2(MembershipFunction):
        """
        Cos function scaled to be between 0 and 1.
        """

        def __call__(self, x: np.ndarray) -> np.ndarray:
            return (np.cos(x) + 1) / 2

    return DummyMembershipFunction2()
