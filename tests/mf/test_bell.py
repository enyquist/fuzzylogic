# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.mf.bell import Bell


def test_basic_functionality():
    bell_fn = Bell(mean=0, std=1, slope=2)
    x = np.array([0, 1, 2])
    y = bell_fn(x)
    assert np.isclose(y, np.array([1, 1 / (1 + 1**4), 1 / (1 + 2**4)])).all()
