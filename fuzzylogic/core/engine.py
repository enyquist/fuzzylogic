# standard libraries
from abc import ABC, abstractmethod
from typing import List

# third party libraries
import numpy as np

# fuzzy logic libraries
from fuzzylogic.core.mf import MembershipFunction1D, MembershipFunction2D
from fuzzylogic.fuzzy_rule import FuzzyRule


class FuzzyEngine(ABC):
    """
    Base class for fuzzy inference engines.
    """

    def __init__(
        self,
        rules: List[FuzzyRule],
    ):
        """
        Initialises a fuzzy inference engine.

        Args:
            rules (List[FuzzyRule]): List of fuzzy rules
        """
        # Check if the rules are valid
        if not rules:
            raise ValueError("The rules list cannot be empty.")

        self.rules = rules

    @abstractmethod
    def compose(self, inputs: float | List[float]) -> MembershipFunction1D | MembershipFunction2D:
        """
        Composes the fuzzy infernce engine with the inputs.

        Args:
            inputs (float | List[float]): Crisp Input(s) value(s)

        Returns:
            MembershipFunction1D | MembershipFunction2D: Output membership function
        """
        pass

    @abstractmethod
    def infer(self, x: np.ndarray) -> np.ndarray:
        """
        Perform inference with the fuzzy inference engine.

        Args:
            x (np.ndarray): Universe of discourse

        Returns:
            np.ndarray: Membership values of the output variable
        """
        pass

    @abstractmethod
    def calculate_fuzzy_control_surface(self, antecedent_ranges: List[np.ndarray]) -> np.ndarray:
        """
        Calculates the fuzzy control surface.

        Args:
            antecedent_ranges (List[np.ndarray]): List of numpy arrays containing the antecedent ranges

        Returns:
            np.ndarray: Fuzzy control surface
        """
        pass
