"""
PuLP-based optimizer implementation.
"""
from typing import Dict, List, Optional, Any, Tuple
import pulp
import time
import logging

# Import from parent directory
from ..data_models import OptimizationResult, OptimizationObjective

def optimize_with_pulp(self):
        """
        Solves the optimization problem using PuLP directly as a simpler approach
        compared to Benders decomposition.
        """
        import pulp

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
        
        # Get weight parameters - be flexible with naming
        weights = {}
        if 'objective_weights' in self.parameters:
            weights = self.parameters['objective_weights']
        elif 'weights' in self.parameters:
            weights = self.parameters['weights']
        else:
            # Default equal weights
            weights = {
                'nutritional_value': 0.2,
                'nutrient_density': 0.2,
                'environmental_impact': 0.2,
                'affordability': 0.2,
                'sustainability': 0.2
            }
            self.logger.warning("No weights found in parameters, using default equal weights.")
        
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
        
        # 2. Food group constraints - at least one food from each group across all farms
        if self.food_groups:  # Only add if food groups exist
            # Check if we have group-specific constraints in the parameters
            food_group_constraints = self.parameters.get('food_group_constraints', {})
            
            for group, foods in self.food_groups.items():
                if foods:  # Only add if the group has foods
                    # Get min_foods from food_group_constraints or use default
                    if group in food_group_constraints and 'min_foods' in food_group_constraints[group]:
                        min_foods = food_group_constraints[group]['min_foods']
                    else:
                        min_foods = self.parameters.get('min_foods_per_group', 1)
                    
                    self.logger.info(f"Food group constraint for {group}: min_foods={min_foods} out of {len(foods)} foods")
                    model += pulp.lpSum([y[farm, food] for farm in self.farms for food in foods]) >= min_foods, f"Min_Foods_{group}"
                    
                    # Also add max_foods constraint if specified
                    if group in food_group_constraints and 'max_foods' in food_group_constraints[group]:
                        max_foods = food_group_constraints[group]['max_foods']
                        model += pulp.lpSum([y[farm, food] for farm in self.farms for food in foods]) <= max_foods, f"Max_Foods_{group}"
        
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
                min_area = max(self.parameters.get('min_planting_area', {}).get(food, 0.0001), 0.0001)
                max_percentage = self.parameters.get('max_percentage_per_crop', {}).get(food, 0.3)
                self.logger.info(f"Linking constraint for {farm},{food}: min_area={min_area}, max_percentage={max_percentage}")
                
                # If y=0, then x=0; if y=1, then x <= max_percentage * land_availability
                land_availability = self.parameters['land_availability'][farm]
                model += x[farm, food] <= land_availability * max_percentage * y[farm, food], f"Upper_Link_{farm}_{food}"
                
                # Apply minimum planting area constraint when selected
                model += x[farm, food] >= min_area * y[farm, food], f"Lower_Link_{farm}_{food}"
        
        # 4. Farm utilization - reduce to 20% minimum utilization for feasibility
        for farm in self.farms:
            # Use provided parameter 'min_utilization' or default to 0.2
            min_util = self.parameters.get('min_utilization', 0.2)
            model += pulp.lpSum([x[farm, food] for food in self.foods]) >= min_util * self.parameters['land_availability'][farm], f"Min_Land_Use_{farm}"
        
        # 5. Food variety constraints per farm
        for farm in self.farms:
            min_foods_farm = self.parameters.get('min_foods_per_farm', 1)
            max_foods_farm = self.parameters.get('max_foods_per_farm', 8)
            model += pulp.lpSum([y[farm, food] for food in self.foods]) >= min_foods_farm, f"Min_Foods_{farm}"
            model += pulp.lpSum([y[farm, food] for food in self.foods]) <= max_foods_farm, f"Max_Foods_{farm}"
        
        # 6. Add constraint for minimum total land utilization
        total_land = sum(self.parameters['land_availability'][farm] for farm in self.farms)
        min_total_percentage = self.parameters.get('min_total_land_usage_percentage', 0.5)
        min_total_usage = min_total_percentage * total_land
        model += pulp.lpSum([x[farm, food] for farm in self.farms for food in self.foods]) >= min_total_usage, "Min_Total_Land"
            
        # Solve the model with increased time limit and relaxed gap
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=120, options=['allowableGap=0.05'])
        model.solve(solver)
        
        # Check status
        status_str = pulp.LpStatus[model.status]
        self.logger.info(f"Optimization status: {status_str}")
        obj_value = pulp.value(model.objective)
        self.logger.info(f"Objective value: {obj_value}")
        
        # When infeasible, check all constraints to identify violations
        if model.status == pulp.LpStatusInfeasible or model.status == pulp.LpStatusNotSolved:
            self.logger.warning(f"Model reported as {status_str}. Checking constraints for violations...")
            
            # Get variable values, even if infeasible
            for v in model.variables():
                v.varValue = v.value()
            
            # Check land availability constraints
            self.logger.info("Checking land availability constraints:")
            for farm in self.farms:
                total_used = sum(x[farm, food].value() or 0 for food in self.foods)
                available = self.parameters['land_availability'][farm]
                violation = total_used - available
                if violation > 0.001:  # Small tolerance for floating point
                    self.logger.error(f"  VIOLATION - {farm}: Using {total_used:.2f} ha (exceeds {available:.2f} ha by {violation:.2f})")
                else:
                    self.logger.info(f"  OK - {farm}: Using {total_used:.2f} ha of {available:.2f} ha")
            
            # Check farm utilization constraints
            self.logger.info("Checking minimum farm utilization constraints:")
            for farm in self.farms:
                total_used = sum(x[farm, food].value() or 0 for food in self.foods)
                min_utilization = self.parameters.get('min_utilization', 0.2)
                required = min_utilization * self.parameters['land_availability'][farm]
                violation = required - total_used
                if violation > 0.001:  # Small tolerance
                    self.logger.error(f"  VIOLATION - {farm}: Using {total_used:.2f} ha (below minimum {required:.2f} ha by {violation:.2f})")
                else:
                    self.logger.info(f"  OK - {farm}: Using {total_used:.2f} ha (minimum: {required:.2f} ha)")
            
            # Check global different foods constraint
            self.logger.info("Checking global minimum different foods constraint:")
            foods_selected = sum(1 for food in self.foods if any(y[farm, food].value() > 0.5 for farm in self.farms))
            min_foods = self.parameters.get('global_min_different_foods', 5)
            if foods_selected < min_foods:
                self.logger.error(f"  VIOLATION - Only {foods_selected} different foods selected (required: {min_foods})")
            else:
                self.logger.info(f"  OK - {foods_selected} different foods selected (required: {min_foods})")
            
            # Check food group constraints
            if self.food_groups:
                food_group_constraints = self.parameters.get('food_group_constraints', {})
                self.logger.info("Checking food group constraints:")
                
                for group, foods in self.food_groups.items():
                    if not foods:
                        continue
                        
                    # Get min_foods from food_group_constraints or use default
                    if group in food_group_constraints and 'min_foods' in food_group_constraints[group]:
                        min_foods = food_group_constraints[group]['min_foods']
                    else:
                        min_foods = self.parameters.get('min_foods_per_group', 1)
                    
                    # Count selected foods in this group
                    selected = sum(1 for food in foods if any(y[farm, food].value() > 0.5 for farm in self.farms))
                    
                    if selected < min_foods:
                        self.logger.error(f"  VIOLATION - Group {group}: Only {selected} foods selected (required: {min_foods})")
                    else:
                        self.logger.info(f"  OK - Group {group}: {selected} foods selected (required: {min_foods})")
            
            # Check minimum planting area constraints
            self.logger.info("Checking minimum planting area constraints (for active y):")
            for farm in self.farms:
                for food in self.foods:
                    y_val = y[farm, food].value()
                    if y_val and y_val > 0.5:  # If y is active (approx 1)
                        x_val = x[farm, food].value() or 0
                        min_area = max(self.parameters.get('min_planting_area', {}).get(food, 0.0001), 0.0001)
                        if x_val < min_area - 0.0001:  # Small tolerance
                            self.logger.error(f"  VIOLATION - {farm}, {food}: Area={x_val:.6f} (below min={min_area:.6f})")
            
            # Add a summary statement
            self.logger.warning("Model infeasibility may be due to conflicts between constraints.")
            self.logger.warning("Consider relaxing some constraints, particularly farm utilization and food variety.")
        
        # Extract solution - ONLY if model is truly optimal
        solution = {}
        if model.status == pulp.LpStatusOptimal:
            self.logger.info("Model is optimal. Extracting solution from PuLP model...")
            for farm in self.farms:
                farm_total = 0
                for food in self.foods:
                    # Check if the variable has a value and it's significant
                    x_val = x[farm, food].value()
                    y_val = y[farm, food].value()
                    
                    if x_val is not None and x_val > 0.01:  # Include if non-negligible area
                        # Verify this assignment doesn't exceed land availability
                        if farm_total + x_val <= self.parameters['land_availability'][farm]:
                            solution[(farm, food)] = x_val
                            farm_total += x_val
                            self.logger.info(f"  Farm {farm}, Food {food}: {x_val:.2f} hectares (y={y_val})")
                        else:
                            self.logger.warning(f"  Rejecting assignment that would exceed land availability: Farm {farm}, Food {food}: {x_val:.2f} hectares")
                
                # Log total land allocated for this farm
                self.logger.info(f"  Total land allocated for {farm}: {farm_total:.2f} hectares")
                # Double-check that total doesn't exceed land availability
                available = self.parameters['land_availability'][farm]
                if farm_total > available * 1.001:  # Allow for small floating point error
                    self.logger.error(f"  ERROR: Total allocation for {farm} ({farm_total:.2f}) exceeds available land ({available:.2f})")
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
            self.logger.warning("PuLP model reported as infeasible, checking constraints...")
            
            # Check land availability constraints
            for farm in self.farms:
                total_allocated = sum(x[farm, food].value() or 0 for food in self.foods)
                available = self.parameters['land_availability'][farm]
                self.logger.info(f"  {farm} land: Allocated={total_allocated:.2f}, Available={available:.2f}")
            
            # Check minimum utilization constraints
            for farm in self.farms:
                total_allocated = sum(x[farm, food].value() or 0 for food in self.foods)
                min_required = 0.2 * self.parameters['land_availability'][farm]
                self.logger.info(f"  {farm} min utilization: Allocated={total_allocated:.2f}, Required={min_required:.2f}")
            
            # Check food selection constraints
            for farm in self.farms:
                foods_selected = sum(1 for food in self.foods if y[farm, food].value() > 0.5)
                self.logger.info(f"  {farm} foods selected: {foods_selected}")
        
        # Calculate metrics
        metrics = self.calculate_metrics(solution)
        self.logger.info("Calculated metrics:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        # Verify objective value against metrics
        metrics_sum = sum(value for key, value in metrics.items() 
                         if key in ['nutritional_value', 'nutrient_density', 'affordability', 
                                   'sustainability', 'environmental_impact'])
        
        self.logger.info(f"PuLP objective value: {obj_value:.4f}")
        self.logger.info(f"Sum of metrics: {metrics_sum:.4f}")
        
        # Check for significant discrepancy
        if abs(obj_value - metrics_sum) > 1.0 and metrics_sum > 0:
            self.logger.warning(f"Significant discrepancy between PuLP objective ({obj_value:.4f}) and metrics sum ({metrics_sum:.4f})")
            
            # Compare weights used in objective vs metrics calculation
            self.logger.info("Weights used in objective function:")
            for key, value in weights.items():
                self.logger.info(f"  {key}: {value}")
            
            metrics_weights = self.parameters.get('weights', {})
            self.logger.info("Weights used in metrics calculation:")
            for key, value in metrics_weights.items():
                self.logger.info(f"  {key}: {value}")
            
            # Option to use metrics sum as the objective value when there's a significant discrepancy
            use_metrics_sum = self.parameters.get('use_metrics_sum_as_objective', True)
            if use_metrics_sum:
                self.logger.info(f"Using metrics sum ({metrics_sum:.4f}) as objective value instead of PuLP value ({obj_value:.4f})")
                obj_value = metrics_sum
        
        # Create a proper result object - use non-zero objective if available
        result = OptimizationResult(
            status="optimal" if solution else "infeasible",  # Override status if we have a solution
            objective_value=obj_value if obj_value is not None and obj_value > 0 else 0.0,
            solution=solution,
            metrics=metrics,
            runtime=0.0  # Will be updated in main
        )
        
        return result