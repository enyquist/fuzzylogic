class FuzzySet:
    """
    FuzzySet is a class that represents a fuzzy set.
    """

    def __init__(self):
        """
        Initializes the fuzzy set.
        """
        self.set = {}

    def add(self, x: float, membership_value: float):
        """
        Adds a point to the set.
        """
        self.set[x] = membership_value
