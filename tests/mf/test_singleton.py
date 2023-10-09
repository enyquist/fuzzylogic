# third party libraries
import numpy as np
import pytest

# fuzzy logic libraries
from fuzzylogic.mf.singleton import FuzzySingleton


def test_basic_functionality():
    singleton_mf = FuzzySingleton(value=5)
    x = np.array([3, 4, 5, 6, 7])
    y = singleton_mf(x)
    assert np.isclose(y, np.array([0, 0, 1, 0, 0])).all()


def test_multiple_singletons():
    singleton_mf = FuzzySingleton(value=5)
    x = np.array([5, 5, 5])
    y = singleton_mf(x)
    assert np.isclose(y, np.array([1, 1, 1])).all()


def test_no_singleton():
    singleton_mf = FuzzySingleton(value=8)
    x = np.array([3, 4, 5, 6, 7])
    y = singleton_mf(x)
    assert np.isclose(y, np.array([0, 0, 0, 0, 0])).all()


def test_float_singleton():
    singleton_mf = FuzzySingleton(value=5.5)
    x = np.array([5, 5.5, 6])
    y = singleton_mf(x)
    assert np.isclose(y, np.array([0, 1, 0])).all()


def test_error_handling():
    with pytest.raises(ValueError, match="Invalid value type. Received <class 'str'>"):
        singleton_mf = FuzzySingleton(value="invalid")
        _ = singleton_mf(np.array([5]))


def test_large_array():
    singleton_mf = FuzzySingleton(value=1e6)
    x = np.array([1e6 - 1, 1e6, 1e6 + 1])
    y = singleton_mf(x)
    assert np.isclose(y, np.array([0, 1, 0])).all()
