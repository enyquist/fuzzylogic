# standard libraries
from typing import Union

# fuzzy logic libraries
from fuzzylogic.mf.base import MembershipFunction
from fuzzylogic.tconorms.base import TCoNorm
from fuzzylogic.tnorms.base import TNorm


class Composition:
    """
    Composition class for composing membership functions with a t-norm or t-conorm operator.
    """

    def __init__(self, operator: Union[TNorm, TCoNorm]):
        """
        Initializes the composition method.

        Args:
            operator (TNorm): t-norm or t-conorm operator
        """

        if not isinstance(operator, TNorm) and not isinstance(operator, TCoNorm):
            raise ValueError(f"Provided operator is not an instance of TNorm or TCoNorm. Received {type(operator)}")

        self.operator = operator

    def compose(self, mf1: MembershipFunction, mf2: MembershipFunction) -> MembershipFunction:
        """
        Evaluates the composition of two membership functions.

        Args:
            mf1 (MembershipFunction): Membership function 1.
            mf2 (MembershipFunction): Membership function 2.

        Returns:
            MembershipFunction: Combined membership function.
        """

        return self.operator.combine(mf1, mf2)
