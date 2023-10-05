# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction
from fuzzylogic.tnorms.base import TNorm


class Composition:
    """
    Composition class for composing membership functions with a t-norm operator.
    """

    def __init__(self, operator: TNorm):
        """
        Initializes the composition method.

        Args:
            operator (TNorm): t-norm operator
        """

        if not isinstance(operator, TNorm):
            raise ValueError(f"Provided operator is not an instance of TNorm. Received {type(operator)}")

        self.operator = operator

    def compose(self, mf1: MembershipFunction, x1: np.ndarray, mf2: MembershipFunction, x2: np.ndarray) -> np.ndarray:
        """
        Evaluates the composition of two membership functions.

        Args:
            mf1 (MembershipFunction): Membership function 1.
            x1 (np.ndarray): Universe of discourse for membership function 1.
            mf2 (MembershipFunction): Membership function 2.
            x2 (np.ndarray): Universe of discourse for membership function 2.

        Returns:
            np.ndarray: Relationship between membership function 1 and membership function 2.
        """

        X1, X2 = np.meshgrid(x1, x2, indexing="ij")  # Create a 2D grid of x1 and x2 values
        combined_values = self.operator.combine(mf1, mf2)(np.vstack([X1.ravel(), X2.ravel()])).reshape(X1.shape)

        return combined_values
