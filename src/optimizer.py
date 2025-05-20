"""
Core optimizer module tying together various optimization methods.
"""
import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    from .methods.benders_method import BendersDecompositionOptimizer
    from .methods.pulp_method import FoodProductionOptimizer as PulpOptimizer
    from .data_models import OptimizationObjective, OptimizationResult
except ImportError as e:
    print(f"Error loading optimization methods: {e}", file=sys.stderr)
    # Define placeholders to avoid errors
    BendersDecompositionOptimizer = PulpOptimizer = None
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
    
    def _generate_model_parameters(self) -> Dict:
        """Generate model parameters with default values."""
        # Get configuration or use defaults
        config = self.config.get('parameters', {})
        
        return {
            'land_availability': config.get('land_availability', {
                farm: np.random.uniform(10, 100)                # L_f: Total land available at farm f (hectares)
                for farm in self.farms
            }),
            'minimum_planting_area': config.get('minimum_planting_area', {
                food: np.random.uniform(1, 5)                  # Use self.foods which now comes from CSV
                for food in self.foods.keys()                  # NEW: Iterate over keys of loaded foods
            }),
            'food_group_constraints': config.get('food_group_constraints', {
                group: {
                    'min_foods': 1,
                    'max_foods': len(foods_in_group)
                }
                for group, foods_in_group in self.food_groups.items()
            }),
            'weights': config.get('weights', {
                obj.value: 1.0 / len(OptimizationObjective)    # w_i: Weights for each objective (sum = 1)
                for obj in OptimizationObjective
            }),
            'max_percentage_per_crop': config.get('max_percentage_per_crop', {
                c: 1.0  # default allow 100% if not specified
                for c in self.foods.keys()
            }),
            'social_benefit': config.get('social_benefit', {
                f: 0.2            # default to 20% if not specified
                for f in self.farms
            }),
            'min_utilization': config.get('min_utilization', 0.2),
            'global_min_different_foods': config.get('global_min_different_foods', 5),
            'min_foods_per_farm': config.get('min_foods_per_farm', 1),
            'max_foods_per_farm': config.get('max_foods_per_farm', 8),
            'min_total_land_usage_percentage': config.get('min_total_land_usage_percentage', 0.5)
        }

    def _calculate_metrics(self, solution) -> Dict[str, float]:
        """Calculate optimization metrics."""
        metrics = {}
        
        # Calculate objective contributions
        for obj in OptimizationObjective:
            metrics[obj.value] = sum(
                self.parameters['weights'][obj.value] *
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
    
    def solve(self, method: str, debug: bool = False) -> OptimizationResult:
        """
        Solve the optimization problem using the specified method.
        Possible methods are:
        - 'pulp': Uses PuLP solver
        - 'benders': Uses Benders decomposition
        """
        # Choose optimization method
        if method == 'pulp':
            # Create PulpOptimizer and call optimize_with_pulp method
            pulp_optimizer = PulpOptimizer(
                farms=self.farms, 
                foods=self.foods, 
                food_groups=self.food_groups, 
                config=self.config
            )
            result = pulp_optimizer.optimize_with_pulp()
        elif method == 'benders':
            # Create BendersDecompositionOptimizer and call solve_with_benders method
            benders_optimizer = BendersDecompositionOptimizer(
                farms=self.farms, 
                foods=self.foods, 
                food_groups=self.food_groups, 
                config=self.config
            )
            result = benders_optimizer.solve_with_benders()
        else:
            raise ValueError(f"Unknown or unsupported optimization method: {method}. " 
                           f"Supported methods are: 'pulp', 'benders'")
        
        return result

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