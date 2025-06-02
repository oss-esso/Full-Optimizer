import pulp
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import sys
import os
import types

# Add parent directory to path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute import instead of relative import
from data_models import OptimizationResult, OptimizationObjective

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
                        #'profitability': float,         # As of now not implemented
                        'sustainability': float         # As of now not implemented
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
        self.farms = farms  # List of farms (f)
        self.foods = foods  # Dictionary of foods (c) with their scores
        self.food_groups = food_groups  # Food groups (g) and their constituents
        self.config = config or {}

        # Validation
        self._validate_inputs()
        
        # Model parameters
        self.parameters = self._generate_model_parameters()
        
        # Initialize results storage
        self.results = []

    def _validate_inputs(self):
        """Validate input configurations and raise informative errors."""
        # Validate farms (check if farms are not empty)
        if not self.farms:
            raise ValueError("At least one farm must be provided")
        
        # Validate foods (check if foods are not empty)
        required_keys = [obj.value for obj in OptimizationObjective]
        
        for food, scores in self.foods.items():
            missing_keys = [key for key in required_keys if key not in scores]
            if missing_keys:
                raise ValueError(f"Missing keys for food {food}: {missing_keys}")
            
            # Validate score ranges (check if scores are normalized)
            for key, value in scores.items():
                if not 0 <= value <= 1:
                    raise ValueError(f"Invalid score range for {food}.{key}: {value}")
        
        # Validate food groups (check if all foods are in groups)
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
    
    def optimize_with_pulp(self):
        """
        Solves the optimization problem using PuLP directly as a simpler approach
        compared to Benders decomposition.
        """
        # Ensure _print_all_constraints is available for logging
        if not hasattr(self, '_print_all_constraints'):
            self._print_all_constraints = types.MethodType(FoodProductionOptimizer._print_all_constraints, self)

        # Create the optimization model
        model = pulp.LpProblem("Food_Production_Optimization", pulp.LpMaximize)
        
        # Decision variables
        # x_ij: hectares of food j grown on farm i
        x = {}
        # y_ij: binary variable indicating if food j is grown on farm i
        y = {}
        
        # Initialize variables
        for farm in self.farms:
            for food in self.foods:
                x[farm, food] = pulp.LpVariable(f"x_{farm}_{food}", lowBound=0)
                y[farm, food] = pulp.LpVariable(f"y_{farm}_{food}", cat='Binary')
        
        # Get weight parameters
        weights = self.parameters['weights']
        self.logger.info("Building objective function with weights:")
        for key, value in weights.items():
            self.logger.info(f"  {key}: {value}")
        
        # Objective function: maximize weighted sum of food scores
        objective = pulp.lpSum([
            (
                weights['nutritional_value'] * self.foods[food].get('nutritional_value', 0) +
                weights['nutrient_density'] * self.foods[food].get('nutrient_density', 0) +
                weights['affordability'] * self.foods[food].get('affordability', 0) +
                weights['sustainability'] * self.foods[food].get('sustainability', 0) -
                weights['environmental_impact'] * self.foods[food].get('environmental_impact', 0)
            ) * x[farm, food]
            for farm in self.farms
            for food in self.foods
        ])
        
        model += objective
        
        # Constraints
        
        # 1. Land availability constraints
        for farm in self.farms:
            model += pulp.lpSum([x[farm, food] for food in self.foods]) <= self.parameters['land_availability'][farm], f"Land_Constraint_{farm}"
        
        # 2. Food group constraints - enforce proper diversity across all food groups
        if self.food_groups:  # Only add if food groups exist
            food_group_constraints = self.parameters.get('food_group_constraints', {})
            
            # Track foods selected from each group
            group_foods_selected = {}
            
            for group, foods in self.food_groups.items():
                if foods:  # Only add if the group has foods
                    # Get the min/max constraints for this group
                    if group in food_group_constraints and 'min_foods' in food_group_constraints[group]:
                        min_foods = food_group_constraints[group]['min_foods']
                    else:
                        min_foods = self.parameters.get('min_foods_per_group', 1)
                    
                    # Get max_foods if specified, otherwise default to all foods in the group
                    if group in food_group_constraints and 'max_foods' in food_group_constraints[group]:
                        max_foods = food_group_constraints[group]['max_foods']
                    else:
                        max_foods = len(foods)
                    
                    # Enforce min/max bounds on number of distinct foods selected from this group
                    self.logger.info(f"Food group diversity constraint for {group}: {min_foods} to {max_foods} distinct foods must be selected (from {len(foods)} foods)")
                    
                    # Binary variables to indicate if a food is selected at all across farms
                    group_foods_selected[group] = {}
                    for food in foods:
                        food_var = pulp.LpVariable(f"group_food_{group}_{food}", cat='Binary')
                        group_foods_selected[group][food] = food_var
                        
                        # Link to farm-specific binary variables
                        # Food is selected if it's grown on at least one farm
                        for farm in self.farms:
                            model += food_var >= y[farm, food], f"Group_Food_Lower_{group}_{food}_{farm}"
                        
                        # Food is not selected if it's not grown on any farm
                        model += food_var * len(self.farms) <= pulp.lpSum([y[farm, food] for farm in self.farms]), f"Group_Food_Upper_{group}_{food}"
                    
                    # Create diversity constraints for this group using the binary indicators
                    model += pulp.lpSum([group_foods_selected[group][food] for food in foods]) >= min_foods, f"Min_Foods_{group}"
                    model += pulp.lpSum([group_foods_selected[group][food] for food in foods]) <= max_foods, f"Max_Foods_{group}"
                    
                    # Add constraint requiring minimum land area for each selected food
                    for food in foods:
                        # Minimum land area if this food is selected (at least 1 hectare per selected food)
                        model += pulp.lpSum([x[farm, food] for farm in self.farms]) >= 1.0 * group_foods_selected[group][food], f"Min_Area_Selected_{group}_{food}"

        # Add a global constraint to ensure at least a minimum number of different food types are selected
        global_min_foods = self.parameters.get('global_min_different_foods', 5)
        # Create binary indicator variables for whether each food is selected at all
        food_selected = {}
        for food in self.foods:
            food_selected[food] = pulp.LpVariable(f"food_selected_{food}", cat='Binary')
            for farm in self.farms:
                model += food_selected[food] >= y[farm, food], f"Food_Selected_Lower_{food}_{farm}"
            model += food_selected[food] * len(self.farms) <= pulp.lpSum([y[farm, food] for farm in self.farms]), f"Food_Selected_Upper_{food}"
            
        model += pulp.lpSum([food_selected[food] for food in self.foods]) >= global_min_foods, "Global_Min_Different_Foods"
        self.logger.info(f"Added constraint: At least {global_min_foods} different food types must be selected in total")
        
        # 3. Linking constraints - x and y
        for farm in self.farms:
            for food in self.foods:
                # Fetch constraint parameters
                min_area = max(self.parameters.get('minimum_planting_area', {}).get(food, 0.0001), 0.0001)
                max_percentage = self.parameters.get('max_percentage_per_crop', {}).get(food, 0.3)
                self.logger.info(f"Linking constraint for {farm},{food}: min_area={min_area}, max_percentage={max_percentage}")
                
                # If y=0, then x=0; if y=1, then x <= max_percentage * land_availability
                land_availability = self.parameters['land_availability'][farm]
                model += x[farm, food] <= land_availability * max_percentage * y[farm, food], f"Upper_Link_{farm}_{food}"
                
                # Apply minimum planting area constraint when selected
                model += x[farm, food] >= min_area * y[farm, food], f"Lower_Link_{farm}_{food}"
        
        # 4. Farm utilization - social benefit constraints
        for farm in self.farms:
            # Use provided parameter 'social_benefit' or default to 0.2
            min_util = self.parameters.get('social_benefit', {}).get(farm, self.parameters.get('min_utilization', 0.2))
            model += pulp.lpSum([x[farm, food] for food in self.foods]) >= min_util * self.parameters['land_availability'][farm], f"Min_Land_Use_{farm}"
        
        # 5. Food variety constraints per farm
        for farm in self.farms:
            min_foods_farm = self.parameters.get('min_foods_per_farm', 1)
            max_foods_farm = self.parameters.get('max_foods_per_farm', 8)
            model += pulp.lpSum([y[farm, food] for food in self.foods]) >= min_foods_farm, f"Min_Foods_{farm}"
            model += pulp.lpSum([y[farm, food] for food in self.foods]) <= max_foods_farm, f"Max_Foods_{farm}"

        # Stronger food group constraints to ensure each food group is meaningfully represented
        total_land = sum(self.parameters['land_availability'][farm] for farm in self.farms)
        for group, group_foods in self.food_groups.items():
            if group_foods:
                # Ensure at least 10% of total land for each food group
                min_group_area = total_land * 0.10
                model += pulp.lpSum([x[farm, food] for farm in self.farms for food in group_foods]) >= min_group_area, f"Min_Area_Group_{group}"
                
                # Require at least 2 different food types from each group with area > 1 hectare
                # First create binary indicators for foods with significant area
                significant_food = {}
                for food in group_foods:
                    significant_food[food] = pulp.LpVariable(f"significant_{food}", cat='Binary')
                    # Food is significant if total area across all farms is > 1 hectare
                    model += pulp.lpSum([x[farm, food] for farm in self.farms]) >= 1.0 * significant_food[food], f"Significant_Lower_{food}"
                    model += pulp.lpSum([x[farm, food] for farm in self.farms]) <= total_land * significant_food[food], f"Significant_Upper_{food}"
                
                # Require at least 2 significant foods per group (or all foods if less than 2)
                min_significant = min(2, len(group_foods))
                model += pulp.lpSum([significant_food[food] for food in group_foods]) >= min_significant, f"Min_Significant_Foods_{group}"

        # 6. Add constraint for minimum total land utilization
        total_land = sum(self.parameters['land_availability'][farm] for farm in self.farms)
        min_total_percentage = self.parameters.get('min_total_land_usage_percentage', 0)
        min_total_usage = min_total_percentage * total_land
        
        # Only add the constraint if min_total_percentage is greater than 0
        if min_total_percentage > 0:
            model += pulp.lpSum([x[farm, food] for farm in self.farms for food in self.foods]) >= min_total_usage, "Min_Total_Land"
            self.logger.info(f"Added constraint: Total land usage must be at least {min_total_usage:.2f} hectares ({min_total_percentage*100:.0f}% of {total_land:.0f})")
        else:
            self.logger.info("Skipping minimum total land usage constraint (percentage set to 0)")
            
        # Solve the model with increased time limit and relaxed gap
        pulp_time_limit = self.config.get('pulp_time_limit', 120)
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=pulp_time_limit, options=['allowableGap=0.05'])
        start_time = time.time()
        model.solve(solver)
        runtime = time.time() - start_time
        
        # Check status
        status_str = pulp.LpStatus[model.status]
        self.logger.info(f"Optimization status: {status_str}")
        obj_value = pulp.value(model.objective)
        self.logger.info(f"Objective value: {obj_value}")
        
        # If infeasible, print all constraint equations
        if model.status == pulp.LpStatusInfeasible:
            self.logger.warning("Model reported as infeasible. Printing all constraints for debugging.")
            self._print_all_constraints(model)
        
        # Extract solution
        solution = {}
        if model.status == pulp.LpStatusOptimal:
            self.logger.info("Model is optimal. Extracting solution from PuLP model...")
            total_extracted_value = 0.0
            for farm in self.farms:
                farm_total = 0
                for food in self.foods:
                    # Check if the variable has a value and it's significant
                    x_val = x[farm, food].value()
                    y_val = y[farm, food].value()
                    
                    # Log all variable values for debugging
                    if x_val is not None:
                        self.logger.debug(f"  Variable x[{farm}, {food}]: {x_val:.6f}")
                    
                    # Use a much smaller threshold to capture small but meaningful values
                    if x_val is not None and x_val > 1e-6:  # Include if non-negligible area (lowered threshold)
                        # Verify this assignment doesn't exceed land availability
                        if farm_total + x_val <= self.parameters['land_availability'][farm]:
                            solution[(farm, food)] = x_val
                            farm_total += x_val
                            total_extracted_value += x_val
                            self.logger.info(f"  Farm {farm}, Food {food}: {x_val:.6f} hectares (y={y_val})")
                        else:
                            self.logger.warning(f"  Rejecting assignment that would exceed land availability: Farm {farm}, Food {food}: {x_val:.6f} hectares")
                
                # Log total land allocated for this farm
                self.logger.info(f"  Total land allocated for {farm}: {farm_total:.2f} hectares")
                # Double-check that total doesn't exceed land availability
                available = self.parameters['land_availability'][farm]
                if farm_total > available * 1.001:  # Allow for small floating point error
                    self.logger.error(f"  ERROR: Total allocation for {farm} ({farm_total:.2f}) exceeds available land ({available:.2f})")
            
            # Log extraction summary
            self.logger.info(f"Solution extraction complete: {len(solution)} assignments, total area: {total_extracted_value:.6f}")
            if len(solution) == 0:
                self.logger.warning("No solution assignments extracted! This may indicate all variable values are below threshold.")
        elif obj_value is not None and obj_value > 0:
            self.logger.warning("Model is not optimal but has a positive objective value. Solution may not respect all constraints.")
            # For non-optimal solutions, strictly enforce the land availability constraint
            for farm in self.farms:
                farm_total = 0
                available = self.parameters['land_availability'][farm]
                food_allocations = []
                
                # First collect all allocations and sort by objective contribution (best first)
                for food in self.foods:
                    x_val = x[farm, food].value()
                    if x_val is not None and x_val > 0.01:
                        food_score = (
                            weights['nutritional_value'] * self.foods[food].get('nutritional_value', 0) +
                            weights['nutrient_density'] * self.foods[food].get('nutrient_density', 0) +
                            weights['affordability'] * self.foods[food].get('affordability', 0) +
                            weights['sustainability'] * self.foods[food].get('sustainability', 0) -
                            weights['environmental_impact'] * self.foods[food].get('environmental_impact', 0)
                        )
                        food_allocations.append((food, x_val, food_score))
                
                # Sort by score per hectare (highest first)
                food_allocations.sort(key=lambda x: x[2]/x[1], reverse=True)
                
                # Allocate land up to availability - take best foods first
                for food, x_val, _ in food_allocations:
                    # Take as much as we can fit
                    allocation = min(x_val, available - farm_total)
                    if allocation > 0:
                        solution[(farm, food)] = allocation
                        farm_total += allocation
                        self.logger.info(f"  Farm {farm}, Food {food}: {allocation:.2f} hectares (reduced from {x_val:.2f})")
                    
                    # Stop if we've reached capacity
                    if farm_total >= available:
                        break
                
                self.logger.info(f"  Total land allocated for {farm}: {farm_total:.2f} hectares (available: {available:.2f})")
        else:
            # If truly infeasible, log constraint status to identify problematic constraints
            self._check_infeasibility(model, x, y)
        
        # Calculate metrics
        metrics = self._calculate_metrics(solution)
        self.logger.info("Calculated metrics:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        # Check if there's a significant difference between PuLP's objective and our calculation
        calculated_obj = metrics.get('calculated_objective', 0.0)
        if obj_value is not None and abs(obj_value - calculated_obj) > 0.01:
            self.logger.warning(f"Discrepancy between PuLP objective ({obj_value:.4f}) and calculated objective ({calculated_obj:.4f})")
            # Only use calculated objective if it's non-zero and the discrepancy is huge
            if calculated_obj > 0 and abs(obj_value - calculated_obj) > 1.0:
                self.logger.info(f"Using calculated objective ({calculated_obj:.4f}) instead of PuLP value ({obj_value:.4f})")
                obj_value = calculated_obj
        
        # Create a proper result object - preserve PuLP's objective value even if solution is empty
        result = OptimizationResult(
            status="optimal" if solution else "infeasible",
            objective_value=obj_value if obj_value is not None else 0.0,  # Don't check if > 0
            solution=solution,
            metrics=metrics,
            runtime=runtime
        )
        
        # Store the result
        self.results.append(result)
        
        return result
    
    def _check_infeasibility(self, model, x, y):
        """Check constraints to identify sources of infeasibility."""
        self.logger.warning("PuLP model reported as infeasible, checking constraints...")
        
        # Print all constraints first for complete reference
        self._print_all_constraints(model)
        
        # Check land availability constraints
        for farm in self.farms:
            total_allocated = sum(x[farm, food].value() or 0 for food in self.foods)
            available = self.parameters['land_availability'][farm]
            self.logger.info(f"  {farm} land: Allocated={total_allocated:.2f}, Available={available:.2f}")
        
        # Check minimum utilization constraints
        for farm in self.farms:
            total_allocated = sum(x[farm, food].value() or 0 for food in self.foods)
            min_required = self.parameters.get('social_benefit', {}).get(farm, 0.2) * self.parameters['land_availability'][farm]
            self.logger.info(f"  {farm} min utilization: Allocated={total_allocated:.2f}, Required={min_required:.2f}")
        
        # Check food selection constraints
        for farm in self.farms:
            foods_selected = sum(1 for food in self.foods if y[farm, food].value() > 0.5)
            min_foods = self.parameters.get('min_foods_per_farm', 1)
            max_foods = self.parameters.get('max_foods_per_farm', 8)
            self.logger.info(f"  {farm} foods selected: {foods_selected} (range: {min_foods}-{max_foods})")
            
        # Check food group constraints
        if self.food_groups:
            for group, foods in self.food_groups.items():
                if not foods:
                    continue
                food_group_constraints = self.parameters.get('food_group_constraints', {})
                if group in food_group_constraints and 'min_foods' in food_group_constraints[group]:
                    min_foods = food_group_constraints[group]['min_foods']
                else:
                    min_foods = 1
                    
                selected = sum(1 for food in foods if any(y[farm, food].value() > 0.5 for farm in self.farms))
                self.logger.info(f"  Group {group}: {selected} foods selected (required: {min_foods})")
            
        # Add a summary statement
        self.logger.warning("Model infeasibility may be due to conflicts between constraints.")
        self.logger.warning("Consider relaxing some constraints, particularly farm utilization and food variety.")
    
    def _print_all_constraints(self, model):
        """Print all constraints in the model for debugging."""
        self.logger.info("============= ALL MODEL CONSTRAINTS =============")
        
        if not hasattr(model, 'constraints'):
            self.logger.warning("Model has no constraints attribute - can't print constraints")
            return
            
        # Print objective function first
        self.logger.info("OBJECTIVE FUNCTION:")
        self.logger.info(f"  {model.objective}")
        
        # Print all constraints with a counter
        self.logger.info("\nCONSTRAINTS:")
        for i, (name, constraint) in enumerate(model.constraints.items(), 1):
            # Try to format constraint nicely
            try:
                # Get the constraint sense (<, >, or =)
                sense = ""
                if constraint.sense == pulp.LpConstraintLE:
                    sense = " <= "
                elif constraint.sense == pulp.LpConstraintGE:
                    sense = " >= "
                elif constraint.sense == pulp.LpConstraintEQ:
                    sense = " = "
                
                # Format the constraint expression and constant term
                expr = str(constraint.toDict()['constant'])
                if constraint.toDict()['coefficients']:
                    for var, coef in constraint.toDict()['coefficients'].items():
                        if coef == 1:
                            expr = f"{var} + {expr}"
                        elif coef == -1:
                            expr = f"-{var} + {expr}"
                        else:
                            expr = f"{coef}*{var} + {expr}"
                
                # Print the full constraint
                self.logger.info(f"  [{i}] {name}: {constraint.toDict()['expression']}{sense}{constraint.toDict()['constant']}")
            except:
                # Fallback if formatting fails
                self.logger.info(f"  [{i}] {name}: {constraint}")
        
        # Print variable bounds
        self.logger.info("\nVARIABLE BOUNDS:")
        for v in model.variables():
            lb = v.lowBound if v.lowBound is not None else "-inf"
            ub = v.upBound if v.upBound is not None else "+inf"
            self.logger.info(f"  {v.name}: {lb} <= {v.name} <= {ub}, Type: {v.cat}")
        
        self.logger.info("================ END CONSTRAINTS ================")
        
    def _calculate_metrics(self, solution) -> Dict[str, float]:
        """Calculate optimization metrics."""
        metrics = {}
        
        # Dictionary to store component contributions for comparison with objective
        obj_components = {}
        obj_total = 0.0
        
        # Extract weights for clarity
        weights = self.parameters['weights']
        
        # Calculate raw (unweighted) objective contributions by food attribute
        for obj in OptimizationObjective:
            obj_key = obj.value
            obj_weight = weights.get(obj_key, 0.0)
            
            # Calculate raw metric value (unweighted sum)
            raw_value = sum(
                self.foods[c][obj_key] * area
                for (f, c), area in solution.items()
            )
            
            # Store both raw and weighted values
            metrics[obj_key] = raw_value
            
            # Calculate contribution to objective (matching how it's done in the objective function)
            if obj == OptimizationObjective.ENVIRONMENTAL_IMPACT:
                # Environmental impact is negatively weighted in the objective
                obj_components[obj_key] = -1 * obj_weight * raw_value
            else:
                obj_components[obj_key] = obj_weight * raw_value
            
            # Add to objective total
            obj_total += obj_components[obj_key]
        
        # Log the component contributions for debugging
        self.logger.info("Objective function component contributions:")
        for obj_key, contribution in obj_components.items():
            self.logger.info(f"  {obj_key}: {contribution:.4f} (weight: {weights.get(obj_key, 0.0):.4f})")
        self.logger.info(f"  Total calculated objective: {obj_total:.4f}")
        
        # Add the calculated objective to metrics for reference
        metrics['calculated_objective'] = obj_total
        
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
                if self.parameters['land_availability'][f] > 0 else 0
            )
        
        return metrics
        
    def solve_optimization_problem(self, timeout: Optional[float] = None):
        """Legacy method for backward compatibility - calls optimize_with_pulp."""
        return self.optimize_with_pulp()

# Replace the free function with a safer version
def optimize_with_pulp(self):
    """Module-level function that safely calls the optimize_with_pulp instance method."""
    # Ensure required methods are available
    if not hasattr(self, '_check_infeasibility'):
        self._check_infeasibility = types.MethodType(FoodProductionOptimizer._check_infeasibility, self)
    
    if not hasattr(self, '_print_all_constraints'):
        self._print_all_constraints = types.MethodType(FoodProductionOptimizer._print_all_constraints, self)
    
    # Initialize required attributes if they don't exist
    if not hasattr(self, 'results'):
        self.results = []
    
    # If 'weights' exists but 'objective_weights' doesn't, copy weights to objective_weights
    if hasattr(self, 'parameters'):
        if 'weights' in self.parameters and 'objective_weights' not in self.parameters:
            self.parameters['objective_weights'] = self.parameters['weights']
    
    # Call the instance method
    result = FoodProductionOptimizer.optimize_with_pulp(self)
    
    # Don't append the result again - it's already added in the class method
    # Just return it
    return result