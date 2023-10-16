# standard libraries
from typing import List, Optional, Union

# fuzzy logic libraries
from fuzzylogic.core.mf import MembershipFunction1D
from fuzzylogic.core.tconorm import TCoNorm
from fuzzylogic.core.tnorm import TNorm
from fuzzylogic.mf import ConstantMF, FuzzySingleton


class FuzzyRule:
    """
    Fuzzy rule class
    """

    def __init__(
        self,
        antecedents: List[MembershipFunction1D],
        operators: List[str],
        consequent: MembershipFunction1D,
        dom_operator: TNorm,
        tnorm: TNorm,
        tconorm: TCoNorm,
        implication_operator: Union[TNorm, TCoNorm],
        mf_names: Optional[List[str]] = None,
    ):
        """
        Args:
            antecedents (List[MembershipFunction1D]): antecedents of the rule
            operators (List[str]): operators to combine the antecedents
            consequent (MembershipFunction1D): consequent of the rule
            dom_operator (TNorm): operator to calculate the degree of membership of the antecedents
            tnorm (TNorm): operator to calculate the degree of membership of the consequent aka the AND operator
            tconorm (TCoNorm): operator to calculate the degree of membership of the consequent aka the OR operator
            implication_operator (Union[TNorm, TCoNorm]): operator to calculate the qualified consequent
            mf_names (Optional[List[str]], optional): names of the antecedents and consequent. Defaults to None.
        """

        # Check if antecedents are valid
        if not isinstance(antecedents, list):
            raise TypeError(f"Expected antecedents to be a list, but got {type(antecedents)}")
        if not all(isinstance(antecedent, MembershipFunction1D) for antecedent in antecedents):
            raise TypeError(
                f"Expected antecedents to be a list of MembershipFunction1D, but got {type(antecedents[0])}"
            )

        # Check if operators are valid
        if len(operators) != len(antecedents) - 1:
            raise ValueError(f"Expected operators to have length {len(antecedents) - 1}, but got {len(operators)}")

        # Check if consequent is valid
        if not isinstance(consequent, MembershipFunction1D):
            raise TypeError(f"Expected consequent to be a MembershipFunction1D, but got {type(consequent)}")

        # Check if Degree of Membership operator is valid
        if not isinstance(dom_operator, TNorm):
            raise TypeError(f"Expected dom_operator to be a TNorm, but got {type(dom_operator)}")

        # Check if TNorm operator is valid
        if not isinstance(tnorm, TNorm):
            raise TypeError(f"Expected tnorm to be a TNorm, but got {type(tnorm)}")

        # Check if TConorm operator is valid
        if not isinstance(tconorm, TCoNorm):
            raise TypeError(f"Expected tconorm to be a TCoNorm, but got {type(tconorm)}")

        # Check if operator is valid
        if not isinstance(implication_operator, TNorm) and not isinstance(implication_operator, TCoNorm):
            raise TypeError(f"Expected operator to be a TNorm or TConorm, but got {type(implication_operator)}")

        # Check if mf_names is valid
        if mf_names is not None:
            if not isinstance(mf_names, list):
                raise TypeError(f"Expected mf_names to be a list, but got {type(mf_names)}")
            if not all(isinstance(mf_name, str) for mf_name in mf_names):
                raise TypeError(f"Expected mf_names to be a list of strings, but got {type(mf_names[0])}")
            if len(mf_names) != len(antecedents) + 1:
                raise ValueError(f"Expected mf_names to have length {len(antecedents) + 1}, but got {len(mf_names)}")

        self.antecedents = antecedents
        self.operators = operators
        self.consequent = consequent
        self.dom_operator = dom_operator
        self.tnorm_operator = tnorm
        self.tconorm_operator = tconorm
        self.implication_operator = implication_operator
        self.mf_names = mf_names

    def __repr__(self):
        # Handling the case when there are no antecedents or just one
        if not self.antecedents:
            return "FuzzyRule()"
        elif len(self.antecedents) == 1:
            single_antecedent = self.mf_names[0] if self.mf_names else repr(self.antecedents[0])
            return f"FuzzyRule({single_antecedent}, {self.consequent!r})"

        # Constructing the representation string with mf_names if available
        repr_str = "IF "
        for i, antecedent in enumerate(self.antecedents):
            antecedent_repr = self.mf_names[i] if self.mf_names else repr(antecedent)
            repr_str += f"{antecedent_repr} "
            if i < len(self.operators):
                repr_str += f"{self.operators[i].upper()} "

        consequent_repr = self.mf_names[-1] if self.mf_names else repr(self.consequent)
        repr_str += f"THEN {consequent_repr}"

        return repr_str

    def evaluate(self, x: Union[float, List[float]]) -> MembershipFunction1D:
        """
        Evaluate the rule.

        Args:
            x (Union[float, List[float]]): Crisp input

        Returns:
            MembershipFunction1D: Clipped consequent
        """

        # Get the degree of membership of the antecedents
        self.dom = self.get_rule_strength(x)

        # Cast antecedent_dom to MembershipFunction1D
        antecedent_dom_mfs = [ConstantMF(value=antecedent_dom_i) for antecedent_dom_i in self.dom]

        # Rule firing strength - using reduce to successively combine all membership functions
        rule_firing_strength_mf = self._calculate_rule_firing_strength(antecedent_dom_mfs)

        # Clip consequent using implication operator
        clipped_consequent_mf = self.implication_operator.combine(rule_firing_strength_mf, self.consequent)

        return clipped_consequent_mf

    def _calculate_rule_firing_strength(self, antecedent_dom_mfs: List[MembershipFunction1D]) -> MembershipFunction1D:
        """
        Calculate the rule firing strength.

        Args:
            antecedent_dom_mfs (List[MembershipFunction1D]): List of antecedent degree of membership
                membership functions

        Returns:
            MembershipFunction1D: Combined Antecedent Membership Function
        """

        # start with first antecedent
        rule_firing_strength_mf = antecedent_dom_mfs[0]

        # combine all antecedents
        for idx, operator in enumerate(self.operators):
            if operator == "and":
                rule_firing_strength_mf = self.tnorm_operator.combine(
                    rule_firing_strength_mf, antecedent_dom_mfs[idx + 1]
                )
            elif operator == "or":
                rule_firing_strength_mf = self.tconorm_operator.combine(
                    rule_firing_strength_mf, antecedent_dom_mfs[idx + 1]
                )
            else:
                raise ValueError(f"Expected operator to be 'and' or 'or', but got {operator} at index {idx}")

        return rule_firing_strength_mf

    def get_rule_strength(self, x: Union[float, List[float]]) -> MembershipFunction1D:
        """
        Get the rule strength aka the degree of membership of the antecedents.

        Args:
            x (Union[float, List[float]]): Crisp input

        Returns:
            MembershipFunction1D: Clipped consequent
        """

        # Cast x to list if it is a float
        if isinstance(x, float):
            x = [x]

        # Assert length of x is equal to the number of antecedents
        if len(x) != len(self.antecedents):
            raise ValueError(f"Expected x to have length {len(self.antecedents)}, but got {len(x)}")

        # Fuzzify Crisp input
        fuzzy_x = [FuzzySingleton(value=x_i) for x_i in x]

        # Degree of membership of antecedents
        antecedent_dom = [
            self.dom_operator.combine(fuzzy_x_i, antecedent)(x_i)
            for fuzzy_x_i, antecedent, x_i in zip(fuzzy_x, self.antecedents, x)
        ]

        return antecedent_dom
