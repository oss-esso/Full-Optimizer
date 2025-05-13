import numpy as np
import pandas as pd
import pulp
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
import time
import networkx as nx
import sys
import os
import re
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle

#====================================================================
'''Stuff to do:
1. Add imports from excel file
2. figure out how to implement quantum optimizer
3. Add the tree decomposition to the model'''
#====================================================================





# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class OptimizationObjective(Enum):
    """Types of optimization objectives."""
    NUTRITIONAL_VALUE = "nutritional_value"
    NUTRIENT_DENSITY = "nutrient_density"
    ENVIRONMENTAL_IMPACT = "environmental_impact"
    AFFORDABILITY = "affordability"


@dataclass
class OptimizationResult:
    """Class for storing optimization results."""
    problem: pulp.LpProblem
    status: str
    objective_value: float
    solution: Dict[Tuple[str, str], float]
    metrics: Dict[str, float]
    runtime: float

class FoodProductionOptimizer:
    def __init__(self, 
                 farms: List[str], 
                 foods: Dict[str, Dict[str, float]], 
                 food_groups: Dict[str, List[str]],
                 config: Optional[Dict] = None):
        """
        Initialize the multi-objective food production optimization model
        
        Args:
            farms (List[str]): List of farm identifiers
            foods (Dict): Dictionary of foods with their scores
                {
                    'food_name': {
                        'nutritional_value': float,     # N_c: Nutritional value score (higher is better)
                        'nutrient_density': float,      # D_c: Nutrient density score (higher is better)
                        'environmental_impact': float,  # E_c: Environmental impact score (lower is better)
                        'affordability': float,         # P_c: Affordability score (higher is better)
                    }
                }
            food_groups (Dict): Mapping of food groups (G_g) and their constituents
            config (Dict): Optional configuration dictionary
        """
        # Logging configuration
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Core configuration
        self.farms = farms
        self.foods = foods
        self.food_groups = food_groups
        self.config = config or {}

        # Validation
        self._validate_inputs()
        
        # Model parameters
        self.parameters = self._generate_model_parameters()
        
        # Initialize results storage
        self.results = []

    def _validate_inputs(self):
        """Validate input configurations and raise informative errors."""
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
        config = self.config.get('parameters', {})
        
        return {
            'land_availability': config.get('land_availability', {
                farm: np.random.uniform(10, 100)
                for farm in self.farms
            }),
            'minimum_planting_area': config.get('minimum_planting_area', {
                food: np.random.uniform(1, 5)
                for food in self.foods
            }),
            'food_group_constraints': config.get('food_group_constraints', {
                group: {
                    'min_foods': max(1, len(foods) // 2),
                    'max_foods': len(foods)
                }
                for group, foods in self.food_groups.items()
            }),
            'objective_weights': config.get('objective_weights', {
                obj.value: 1.0 / len(OptimizationObjective)
                for obj in OptimizationObjective
            })
        }

    def solve_optimization_problem(self, 
                                 timeout: Optional[float] = None) -> OptimizationResult:
        """Solve the multi-objective food production optimization problem."""
        try:
            start_time = time.time()
            
            # Create the linear programming problem
            prob = pulp.LpProblem("Multi_Objective_Food_Production", pulp.LpMaximize)
            
            # Decision Variables
            area_vars = pulp.LpVariable.dicts(
                "Area", 
                ((f, c) for f in self.farms for c in self.foods), 
                lowBound=0
            )
            
            plant_vars = pulp.LpVariable.dicts(
                "Plant", 
                ((f, c) for f in self.farms for c in self.foods), 
                cat='Binary'
            )

            # Objective Function
            objective = pulp.LpAffineExpression()
            weights = self.parameters['objective_weights']

            for obj in OptimizationObjective:
                score_sum = pulp.lpSum(
                    self.foods[c][obj.value] * area_vars[f, c] 
                    for f in self.farms for c in self.foods
                )
                
                weighted_score = weights[obj.value] * score_sum
                
                if obj == OptimizationObjective.ENVIRONMENTAL_IMPACT:
                    objective -= weighted_score
                else:
                    objective += weighted_score

            prob += objective

            # Add constraints
            self._add_land_availability_constraints(prob, area_vars)
            self._add_planting_area_constraints(prob, area_vars, plant_vars)
            self._add_food_group_constraints(prob, plant_vars)
            
            # Solve classically
            classical_solution, classical_metrics, classical_runtime = self._solve_classical(prob, area_vars, timeout)
            
            # Create result object
            result = OptimizationResult(
                problem=prob,
                status=pulp.LpStatus[prob.status],
                objective_value=pulp.value(prob.objective) if prob.objective else 0.0,
                solution=classical_solution,
                metrics=classical_metrics,
                runtime=classical_runtime
            )
            
            self.results.append(result)
            return result
        
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise
    
    def _add_land_availability_constraints(self, prob, area_vars):
        """Add land availability constraints."""
        for f in self.farms:
            prob += pulp.lpSum(area_vars[f, c] for c in self.foods) <= \
                    self.parameters['land_availability'][f]
    
    def _add_planting_area_constraints(self, prob, area_vars, plant_vars):
        """Add minimum and maximum planting area constraints."""
        for f in self.farms:
            for c in self.foods:
                min_area = self.parameters['minimum_planting_area'][c]
                farm_land = self.parameters['land_availability'][f]
                
                prob += area_vars[f, c] >= min_area * plant_vars[f, c]
                prob += area_vars[f, c] <= farm_land * plant_vars[f, c]
    
    def _add_food_group_constraints(self, prob, plant_vars):
        """Add food group diversity constraints."""
        for g, group_foods in self.food_groups.items():
            constraints = self.parameters['food_group_constraints'][g]
            min_foods = constraints['min_foods']
            max_foods = constraints['max_foods']
            
            group_plant_sum = pulp.lpSum(
                plant_vars[f, c] 
                for f in self.farms 
                for c in group_foods
            )
            prob += group_plant_sum >= min_foods
            prob += group_plant_sum <= max_foods
    
    def _solve_classical(self, prob: pulp.LpProblem, area_vars: Dict, timeout: Optional[float] = None) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float], float]:
        """Solve the problem using classical optimization."""
        start_time = time.time()

        try:
            if timeout:
                prob.solve(pulp.PULP_CBC_CMD(timeLimit=timeout))
            else:
                prob.solve()

            if prob.status != pulp.LpStatusOptimal:
                self.logger.warning(f"Optimization did not reach optimal solution. Status: {pulp.LpStatus[prob.status]}")

            solution = {}
            if prob.status == pulp.LpStatusOptimal:
                solution = {
                    (f, c): area_vars[f, c].varValue
                    for f in self.farms
                    for c in self.foods
                    if area_vars[f, c].varValue is not None and area_vars[f, c].varValue > 1e-6
                }

            metrics = self._calculate_metrics(solution)
            runtime = time.time() - start_time

            return solution, metrics, runtime

        except Exception as e:
            self.logger.error(f"Classical optimization failed: {e}")
            return {}, {}, time.time() - start_time
    
    def _calculate_metrics(self, solution) -> Dict[str, float]:
        """Calculate optimization metrics."""
        metrics = {}
        
        for obj in OptimizationObjective:
            metrics[obj.value] = sum(
                self.parameters['objective_weights'][obj.value] *
                self.foods[c][obj.value] *
                area
                for (f, c), area in solution.items()
            )
        
        total_area = sum(solution.values())
        metrics['total_area'] = total_area
        
        for f in self.farms:
            farm_area = sum(
                area for (farm, _), area in solution.items() 
                if farm == f
            )
            metrics[f'utilization_{f}'] = (
                farm_area / self.parameters['land_availability'][f]
            )
        
        return metrics

# Example usage
def main():
    # Example configuration
    config = {
        'parameters': {
            'land_availability': {
                'Farm1': 50,  # L_f for Farm1
                'Farm2': 75   # L_f for Farm2
            },
            'minimum_planting_area': {
                'Tomatoes': 2,   # A^min_c for Tomatoes
                'Lettuce': 1,    # A^min_c for Lettuce
                'Carrots': 1.5   # A^min_c for Carrots
            },
            'food_group_constraints': {
                'Vegetables': {
                    'min_foods': 2,  # FG^min_g
                    'max_foods': 3   # FG^max_g
                }
            },
            'objective_weights': {
                'nutritional_value': 0.3,  # w_N
                'nutrient_density': 0.2,   # w_D
                'environmental_impact': 0.2, # w_E
                'affordability': 0.15,     # w_P
            }
        }
    }
    
    # Example food and farm configuration
    # Each food has scores corresponding to N_c, D_c, E_c, P_c
    foods = {
        'Tomatoes': {
            'nutritional_value': 0.8,  # N_c
            'nutrient_density': 0.7,   # D_c
            'environmental_impact': 0.3, # E_c
            'affordability': 0.6,      # P_c
        },
        'Lettuce': {
            'nutritional_value': 0.7,
            'nutrient_density': 0.8,
            'environmental_impact': 0.2,
            'affordability': 0.7,
        },
        'Carrots': {
            'nutritional_value': 0.6,
            'nutrient_density': 0.9,
            'environmental_impact': 0.4,
            'affordability': 0.5,
        }
    }
    
    # Food groups G_g
    food_groups = {
        'Vegetables': ['Tomatoes', 'Lettuce', 'Carrots']
    }
    
    print("\n" + "="*80)
    print(" FOOD PRODUCTION OPTIMIZATION")
    print("="*80)
    
    print("\nInitializing the optimization model...")
    # Create optimizer
    optimizer = FoodProductionOptimizer(
        farms=['Farm1', 'Farm2'],
        foods=foods,
        food_groups=food_groups,
        config=config
    )
    
    print("\nStarting the optimization process...")
    print("This will use classical optimization with PuLP.")
    print("Detailed visualizations will be created at each step.")
    
    # Solve optimization problem
    result = optimizer.solve_optimization_problem(timeout=120)
    
    # Generate and print production report
    report = optimizer.generate_production_report(result)
    
    # Generate visualization
    optimizer.plot_solution(result, save_path="solution_visualization.png")
    
    # Print summary of the optimization process
    print("\n" + "="*80)
    print(" OPTIMIZATION RESULTS SUMMARY")
    print("="*80)
    
    # Check if we have a valid solution
    if not result.solution:
        print("\n⚠️ No valid solution found. Check the visualizations for details.")
    else:
        print("\nProduction Allocation Report:")
        print(report)
        
        # Print optimization metrics
        print("\nOptimization Metrics:")
        for metric, value in optimizer._calculate_metrics(result.solution).items():
            print(f"{metric}: {value:.4f}")
    
    # List created visualizations
    visualization_files = [
        "solution_visualization.png"
    ]
    
    print("\n" + "="*80)
    print(" GENERATED VISUALIZATIONS")
    print("="*80)
    print("\nThe following visualization files have been created:")
    
    for i, file in enumerate(visualization_files):
        import os
        if os.path.exists(file):
            print(f"{i+1}. {file} ✓")
        else:
            print(f"{i+1}. {file} ✗ (not created)")
    
    print("\nTo view these visualizations, open the generated PNG files.")
    print("They provide detailed insights into the optimization process.")
    
    print("\n" + "="*80)
    print(" OPTIMIZATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
