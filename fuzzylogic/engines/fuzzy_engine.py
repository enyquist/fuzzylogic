# standard libraries
from functools import reduce
from typing import List, Union

# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.engines.base import DEFUZZ, FuzzyEngine
from fuzzylogic.mf.base import MembershipFunction1D
from fuzzylogic.rules.fuzzy_rule import FuzzyRule
from fuzzylogic.tconorms.tconorms import TCoNorm


class MamdaniFuzzyEngine(FuzzyEngine):
    """
    Base class for fuzzy inference engines.
    """

    def __init__(
        self,
        rules: List[FuzzyRule],
        aggregate_operator: TCoNorm,
        defuzz: str = "centroid",
    ):
        """
        Initialises a fuzzy inference engine.

        Args:
            rules (List[FuzzyRule]): List of fuzzy rules
            aggregate_operator (TCoNorm): Operator to aggregate fuzzy rules
            defuzz (Defuzzification): Defuzzification method
        """
        # Initialize the base class
        super().__init__(rules=rules)

        # Check if the aggregate operator is valid
        if not isinstance(aggregate_operator, TCoNorm):
            raise TypeError(f"The aggregate operator must be a TCoNorm. Got {type(aggregate_operator)}.")

        # Check if the defuzzification method is valid
        if defuzz not in DEFUZZ:
            raise ValueError(f"The defuzzification method is not valid. Got {defuzz}.")

        self.aggregate_operator = aggregate_operator
        self.defuzz = DEFUZZ[defuzz]

    def __repr__(self) -> str:
        """
        Returns the string representation of the fuzzy inference engine.
        """
        rules = self.rules
        op = self.aggregate_operator
        defuzz = self.defuzz.__name__

        return f"FuzzyEngine(rules={rules}, aggregate_operator={op}, defuzz={defuzz})"

    def compose(self, inputs: Union[float, List[float]]) -> MembershipFunction1D:
        """
        Composes the fuzzy inference engine.

        Args:
            x (Union[float, List[float]]): Crips input values
        """

        # Evaluate the rules
        clipped_consequent_mfs = [rule.evaluate(inputs) for rule in self.rules]

        # Aggregate the rules
        self.composed_consequent_mf = reduce(self.aggregate_operator.combine, clipped_consequent_mfs)

        return self.composed_consequent_mf

    def infer(self, x: np.ndarray) -> np.ndarray:
        """
        Performs fuzzy inference.

        Args:
            x (np.ndarray): Universe of discourse

        Returns:
            np.ndarray: Membership values of the output variable
        """

        # Check if the composed consequent membership function exists
        if not hasattr(self, "composed_consequent_mf"):
            raise ValueError("The fuzzy inference engine has not been composed yet. Call the compose method first.")

        # Defuzzify the composed consequent membership function
        defuzzified_value = self.defuzz.defuzz(x, self.composed_consequent_mf)

        return defuzzified_value

    def calculate_fuzzy_control_surface(self, antecedent_ranges: List[np.ndarray]) -> np.ndarray:
        """
        Calculates the fuzzy control surface of the fuzzy inference engine.

        Args:
            antecedent_ranges (List[np.ndarray]): List of numpy arrays containing the antecedent ranges

        Returns:
            np.ndarray: Fuzzy control surface
        """

        # Create a multi-dimensional meshgrid from the antecedent ranges
        meshgrid = np.meshgrid(*antecedent_ranges)

        # Flatten the meshgrid for easier iteration
        flat_meshgrid = [grid.flatten() for grid in meshgrid]

        # Calculate the total number of points
        total_points = len(flat_meshgrid[0])

        # Initialize the fuzzy surface
        outputs = np.zeros(total_points)

        # Iterate over the flattened meshgrid
        for idx, point in enumerate(zip(*flat_meshgrid)):
            # Create a list of coordinates for this point
            coords = list(point)

            # Compose the engine with the current inputs
            self.compose(coords)

            # Get Rule Strength (Degree of Membership) from the rules
            rule_doms = [np.minimum(*rule.dom) for rule in self.rules]

            # Aggregate the rule strengths
            aggregated_rule_dom = np.maximum(*rule_doms) if len(self.rules) > 1 else rule_doms[0]

            # Calculate the output value
            outputs[idx] = self.defuzz.defuzz(aggregated_rule_dom, self.composed_consequent_mf)

        # Reshape the fuzzy surface
        output_shape = tuple(len(range_) for range_ in antecedent_ranges)
        outputs = outputs.reshape(output_shape)

        return outputs
