"""
Core optimizer module tying together various optimization methods.
"""
import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    from .methods.benders_method import optimize_with_benders
    from .methods.pulp_method import optimize_with_pulp
    from .methods.quantum_enhanced import optimize_with_quantum_benders
    from .methods.quantum_inspired import optimize_with_quantum_inspired_benders
    from .methods.quantum_enhanced_merge import optimize_with_quantum_benders_merge
    from .data_models import OptimizationObjective, OptimizationResult
except ImportError as e:
    print(f"Error loading optimization methods: {e}", file=sys.stderr)
    # Define placeholders to avoid errors
    BendersOptimizer = PulpOptimizer = QuantumBendersOptimizer = None
    OptimizationResult = dict

class FoodProductionOptimizer:

    def __init__(self, 
                 farms: List[str], 
                 foods: Dict[str, Dict[str, float]], 
                 food_groups: Dict[str, List[str]],
                 config: Optional[Dict] = None):
        """Initialize the multi-objective food production optimization model."""
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.farms = farms
        self.foods = foods
        self.food_groups = food_groups
        self.config = config or {}

        # Validate inputs and generate parameters
        self._validate_inputs()
        self.parameters = self._generate_model_parameters()
        self.results = []

    def _validate_inputs(self):
        """Validate input configurations."""
        if not self.farms:
            raise ValueError("At least one farm must be provided")
        required_keys = [obj.value for obj in OptimizationObjective]
        for food, scores in self.foods.items():
            missing_keys = [key for key in required_keys if key not in scores]
            if missing_keys:
                raise ValueError(f"Missing keys for food {food}: {missing_keys}")
            for key, value in scores.items():
                if not 0 <= value <= 1:
                    raise ValueError(f"Invalid score range for {food}.{key}: {value}")
        for group, foods in self.food_groups.items():
            unknown_foods = set(foods) - set(self.foods.keys())
            if unknown_foods:
                raise ValueError(f"Unknown foods in group {group}: {unknown_foods}")

    from .methods.benders_method import optimize_with_benders
    from .methods.pulp_method import optimize_with_pulp
    from .methods.quantum_enhanced import optimize_with_quantum_benders
    from .methods.quantum_inspired import optimize_with_quantum_inspired_benders
    from .methods.quantum_enhanced_merge import optimize_with_quantum_benders_merge

    def _calculate_metrics(self, solution) -> Dict[str, float]:
        """Calculate optimization metrics."""
        metrics = {}
        
        # Calculate objective contributions
        for obj in OptimizationObjective:
            metrics[obj.value] = sum(
                self.parameters['objective_weights'][obj.value] *
                self.foods[c][obj.value] *
                area
                for (f, c), area in solution.items()
            )
        
        # Calculate total area
        total_area = sum(solution.values())
        metrics['total_area'] = total_area
        
        # Calculate utilization
        for f in self.farms:
            farm_area = sum(
                area for (farm, _), area in solution.items() 
                if farm == f
            )
            metrics[f'utilization_{f}'] = (
                farm_area / self.parameters['land_availability'][f]
            )
        
        return metrics

    

    
class SimpleFoodOptimizer(FoodProductionOptimizer):
    """Simplified version of the FoodProductionOptimizer."""
    
    def __init__(self, complexity_level: str = 'simple'):
        """
        Initialize with empty parameters to be loaded later.
        
        Args:
            complexity_level (str): One of 'simple', 'intermediate', or 'full'
        """
        self.farms = []
        self.foods = {}
        self.food_groups = {}
        self.config = {'parameters': {}}
        self.parameters = {'weights': {}}
        self.logger = logging.getLogger(__name__)
        self.complexity_level = complexity_level
    
    def load_food_data(self) -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
        """
        Load food data based on specified complexity level.
        
        Returns:
            Tuple containing farms, foods, food_groups, and config
        """
        from .scenarios import load_food_data as scenarios_load_food_data
        farms, foods, food_groups, config = scenarios_load_food_data(self.complexity_level)
        
        # Update instance variables
        self.farms = farms
        self.foods = foods
        self.food_groups = food_groups
        self.config = config
        self.parameters = config['parameters']
        
        return farms, foods, food_groups, config

    def calculate_metrics(self, solution):
        """Calculate optimization metrics."""
        metrics = {}
        
        # Calculate objective contributions
        nutritional_value = 0
        nutrient_density = 0
        affordability = 0
        sustainability = 0
        environmental_impact = 0
        
        for (farm, food), area in solution.items():
            nutritional_value += self.foods[food]['nutritional_value'] * area * self.parameters['weights']['nutritional_value']
            nutrient_density += self.foods[food]['nutrient_density'] * area * self.parameters['weights']['nutrient_density']
            affordability += self.foods[food]['affordability'] * area * self.parameters['weights']['affordability']
            sustainability += self.foods[food]['sustainability'] * area * self.parameters['weights']['sustainability']
            environmental_impact += self.foods[food]['environmental_impact'] * area * self.parameters['weights']['environmental_impact']
        
        metrics['nutritional_value'] = nutritional_value
        metrics['nutrient_density'] = nutrient_density
        metrics['affordability'] = affordability
        metrics['sustainability'] = sustainability
        metrics['environmental_impact'] = -environmental_impact  # Negative because we want to minimize this
        
        # Calculate total area and utilization
        total_area = sum(solution.values())
        metrics['total_area'] = total_area
        
        # Calculate farm utilization
        for farm in self.farms:
            farm_area = sum(
                area for (f, _), area in solution.items() 
                if f == farm
            )
            metrics[f'utilization_{farm}'] = farm_area / self.parameters['land_availability'][farm]
        
        return metrics

    def _generate_smart_binary_solution(self, Ny, F, C):
        """
        Generate a smart initial solution with at least one food per farm.
        
        Args:
            Ny: Total number of binary variables
            F: Number of farms
            C: Number of foods
            
        Returns:
            A binary solution array of shape (Ny, 1)
        """
        self.logger.info("Generating smart initial solution")
        y_sol = np.zeros((Ny, 1))
        
        # For each farm, select at least one food with good score
        for farm_idx in range(F):
            # Get the food scores for this farm
            farm = self.farms[farm_idx]
            food_scores = [(food_idx, self.foods[food]['nutritional_value']) 
                         for food_idx, food in enumerate(self.foods)]
            
            # Sort by score (highest first)
            food_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select top 2 foods for this farm
            for i in range(min(2, len(food_scores))):
                food_idx = food_scores[i][0]
                pos = farm_idx * C + food_idx
                y_sol[pos, 0] = 1
                self.logger.info(f"Selected food {list(self.foods.keys())[food_idx]} for farm {farm}")
        
        return y_sol

    def solve(self, method: str, f: np.ndarray, A: np.ndarray, b: np.ndarray, C: np.ndarray, c: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        Solve the optimization problem using the specified method.
        Possible methods are:
        - 'pulp': Uses PuLP solver
        - 'benders': Uses Benders decomposition
        - 'quantum-enhanced': Uses quantum-enhanced Benders decomposition
        - 'quantum-inspired': Uses quantum-inspired Benders decomposition
        - 'quantum-enhanced-merge': Uses quantum-enhanced Benders with advanced merging
        """
        # Choose optimization method
        if method == 'pulp':
            result = optimize_with_pulp(f, A, b, C, c, self.solver_params, debug=debug)
        elif method == 'benders':
            result = optimize_with_benders(f, A, b, C, c, self.solver_params, debug=debug)
        elif method == 'quantum-enhanced':
            result = optimize_with_quantum_benders(f, A, b, C, c, self.solver_params, debug=debug)
        elif method == 'quantum-inspired':
            result = optimize_with_quantum_inspired_benders(f, A, b, C, c, self.solver_params, debug=debug)
        elif method == 'quantum-enhanced-merge':
            result = optimize_with_quantum_benders_merge(f, A, b, C, c, self.solver_params, debug=debug)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        return result