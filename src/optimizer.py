"""
Core optimizer module tying together various optimization methods.
"""
from typing import Dict, List, Optional

from methods.benders_method import BendersOptimizer
from methods.pulp_method import PulpOptimizer
from methods.quantum_methods import QuantumBendersOptimizer


class FoodProductionOptimizer:
    """
    Main optimizer class selecting appropriate method.
    """
    def __init__(self,
                 farms: List[str],
                 foods: Dict[str, Dict[str, float]],
                 food_groups: Dict[str, List[str]],
                 config: Optional[Dict] = None):
        self.farms = farms
        self.foods = foods
        self.food_groups = food_groups
        self.config = config or {}
        # ...existing code for parameter validation and generation...

    def solve_with_benders(self, timeout: Optional[float] = None):
        return BendersOptimizer(self.farms, self.foods, self.food_groups, self.config).solve(timeout)

    def solve_with_pulp(self):
        return PulpOptimizer(self.farms, self.foods, self.food_groups, self.config).solve()

    def solve_with_quantum(self):
        return QuantumBendersOptimizer(self.farms, self.foods, self.food_groups, self.config).solve()  


class SimpleFoodOptimizer(FoodProductionOptimizer):
    """Simplified optimizer for example usage."""
    def __init__(self):
        super().__init__([], {}, {}, {'parameters': {}})
        # ...existing code for loading default data...

    def load_food_data(self):
        # ...existing load_food_data implementation...
        pass

    def calculate_metrics(self, solution):
        # ...existing calculate_metrics implementation...
        pass
