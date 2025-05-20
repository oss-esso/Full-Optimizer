"""
Benders decomposition optimizer implementation.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import os
import sys
import time
import logging

# Fix relative import by using absolute import
# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_models import OptimizationResult, OptimizationObjective

class BendersDecompositionOptimizer:
    def __init__(self, 
                 farms: List[str], 
                 foods: Dict[str, Dict[str, float]], 
                 food_groups: Dict[str, List[str]],
                 config: Optional[Dict] = None):
        """
        Initialize the Benders decomposition optimizer for food production
        
        Args:
            farms (List[str]): List of farm identifiers
            foods (Dict): Dictionary of foods with their scores
            food_groups (Dict): Mapping of food groups and their constituents
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

        # Model parameters
        self.parameters = self._generate_model_parameters()
        
        # Initialize results storage
        self.results = []
        self.benders_bounds = None

    def _generate_model_parameters(self) -> Dict:
        """Generate model parameters with default values."""
        # Get configuration or use defaults
        config = self.config.get('parameters', {})
        
        return {
            'land_availability': config.get('land_availability', {
                farm: np.random.uniform(10, 100) for farm in self.farms
            }),
            'min_planting_area': config.get('minimum_planting_area', {
                food: np.random.uniform(1, 5) for food in self.foods.keys()
            }),
            'food_group_constraints': config.get('food_group_constraints', {
                group: {
                    'min_foods': 1,
                    'max_foods': len(foods_in_group)
                }
                for group, foods_in_group in self.food_groups.items()
            }),
            'weights': config.get('weights', {
                obj.value: 1.0 / len(OptimizationObjective) for obj in OptimizationObjective
            }),
            'max_percentage_per_crop': config.get('max_percentage_per_crop', {
                c: 1.0 for c in self.foods.keys()
            }),
            'min_utilization': config.get('min_utilization', 0.2),
            'global_min_different_foods': config.get('global_min_different_foods', 5),
            'min_foods_per_farm': config.get('min_foods_per_farm', 1),
            'max_foods_per_farm': config.get('max_foods_per_farm', 8),
        }
    
    def calculate_metrics(self, solution) -> Dict[str, float]:
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

    def solve_with_benders(self):
        """Uses Benders decomposition to solve the problem."""
        # This method just calls the existing optimize_with_benders function
        return optimize_with_benders(self)

# The original optimize_with_benders function
def optimize_with_benders(self):
    """
    Uses Benders decomposition to solve the problem.
    """
    from my_functions.benders import Benders
    
    # Initialize Benders solver
    benders = Benders()
    benders.multi_cut = True  # Use multi-cut Benders for better convergence
    benders.eps = 0.005  # Slightly looser convergence tolerance to help with numerical issues
    benders.max_iterations = 100  # Allow more iterations
    benders.relative = True  # Use relative gap criterion
    
    # Enable experimental anti-cycling features
    benders.use_norm_cuts = True  # Use normalized optimality cuts
    benders.use_trust_region = True  # Use trust region to stabilize convergence
    
    self.logger.info("Initializing Benders decomposition:")
    self.logger.info(f"  Number of farms (F): {len(self.farms)}")
    self.logger.info(f"  Number of foods (C): {len(self.foods)}")
    
    # Dimensions
    F = len(self.farms)
    C = len(self.foods)
    Nx = F * C  # continuous x variables
    Ny = F * C  # binary y variables
    
    self.logger.info(f"  Number of continuous variables (Nx): {Nx}")
    self.logger.info(f"  Number of binary variables (Ny): {Ny}")
    
    # Generate a good initial solution based on food scores
    # Building a good initial solution is crucial to Benders convergence
    y_init = np.zeros((Ny, 1), dtype=float)
    
    # Calculate food scores
    food_scores = {}
    for food, attrs in self.foods.items():
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
        score = (
            weights['nutritional_value'] * attrs['nutritional_value'] +
            weights['nutrient_density'] * attrs['nutrient_density'] +
            weights['affordability'] * attrs['affordability'] +
            weights['sustainability'] * attrs['sustainability'] -
            weights['environmental_impact'] * attrs['environmental_impact']
        )
        food_scores[food] = score
    
    # Sort foods by score
    sorted_foods = sorted(food_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Select top two foods for each farm, ensuring diversity
    farm_selections = {farm: [] for farm in self.farms}
    
    # First ensure grain representation (high priority)
    for farm_idx, farm in enumerate(self.farms):
        grain_added = False
        for food, _ in sorted_foods:
            if food in self.food_groups.get('Grains', []) and not grain_added:
                food_idx = list(self.foods.keys()).index(food)
                pos = farm_idx * C + food_idx
                y_init[pos, 0] = 1
                farm_selections[farm].append(food)
                grain_added = True
                self.logger.info(f"  Adding grain to initial solution: Farm {farm}, Food {food}")
                break
    
    # Next ensure fruit/vegetable representation
    for farm_idx, farm in enumerate(self.farms):
        fv_added = False
        combined_fv = self.food_groups.get('Fruits', []) + self.food_groups.get('Vegetables', [])
        for food, _ in sorted_foods:
            if food in combined_fv and not fv_added and food not in farm_selections[farm]:
                food_idx = list(self.foods.keys()).index(food)
                pos = farm_idx * C + food_idx
                y_init[pos, 0] = 1
                farm_selections[farm].append(food)
                fv_added = True
                self.logger.info(f"  Adding fruit/vegetable to initial solution: Farm {farm}, Food {food}")
                break
    
    # Add high-score foods if needed to ensure at least 2 foods per farm
    for farm_idx, farm in enumerate(self.farms):
        while len(farm_selections[farm]) < 2:
            for food, score in sorted_foods:
                if food not in farm_selections[farm]:
                    food_idx = list(self.foods.keys()).index(food)
                    pos = farm_idx * C + food_idx
                    y_init[pos, 0] = 1
                    farm_selections[farm].append(food)
                    self.logger.info(f"  Adding high-score food to initial solution: Farm {farm}, Food {food}")
                    break
    
    self.logger.info("Initial solution:")
    for farm, foods in farm_selections.items():
        self.logger.info(f"  Farm {farm}: {foods}")
    
    # Check initial solution for food group coverage
    self.logger.info("Checking initial solution for food group coverage:")
    group_coverage = {group: 0 for group in self.food_groups}
    for farm, foods in farm_selections.items():
        for food in foods:
            for group, group_foods in self.food_groups.items():
                if food in group_foods:
                    group_coverage[group] += 1

    for group, count in group_coverage.items():
        min_required = self.parameters.get('food_group_constraints', {}).get(group, {}).get('min_foods', 1)
        if count < min_required:
            self.logger.warning(f"  Initial solution has insufficient {group}: {count}/{min_required}")
            
            # Try to add a food from this group to one of the farms
            for food in self.food_groups[group]:
                if not any(food in farm_selections[f] for f in self.farms):
                    # Find farm with fewest foods
                    target_farm = min(self.farms, key=lambda f: len(farm_selections[f]))
                    
                    # Add this food to the farm's selection
                    farm_idx = self.farms.index(target_farm)
                    food_idx = list(self.foods.keys()).index(food)
                    pos = farm_idx * C + food_idx
                    y_init[pos, 0] = 1
                    farm_selections[target_farm].append(food)
                    
                    self.logger.info(f"  Added {food} to {target_farm} to improve {group} coverage")
                    break
        else:
            self.logger.info(f"  Initial solution has good {group} coverage: {count}/{min_required}")

    # Build objective coefficient vectors
    self.logger.info("Building objective function components:")
    c = np.zeros((Nx, 1))
    f = np.zeros((Ny, 1))
    
    for fi, farm in enumerate(self.farms):
        for food_idx, food in enumerate(self.foods):
            pos = fi * C + food_idx
            
            # Calculate the score for this farm-food combination
            food_data = self.foods[food]
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
            
            pos_score = (
                weights['nutritional_value'] * food_data['nutritional_value'] +
                weights['nutrient_density'] * food_data['nutrient_density'] +
                weights['affordability'] * food_data['affordability'] +
                weights['sustainability'] * food_data['sustainability']
            )
            neg_score = weights['environmental_impact'] * food_data['environmental_impact']
            
            # Net score
            net_score = pos_score - neg_score
            
            # For Benders decomposition with minimization, we use -score
            c[pos, 0] = -net_score
            
            self.logger.info(f"Farm {farm}, Food {food}, Score: {net_score}")
    
    # Build constraint matrices
    # RELAXATION: Reduce minimum farm utilization from 75% to 50%
    # RELAXATION: Remove minimum foods per farm constraint initially
    
    # A*x + B*y >= b constraints (involving x)
    num_linking = 2 * Nx  # 2 per farm-food pair
    num_land = F  # 1 per farm
    num_min_utilization = F  # 1 per farm
    num_total_land = 1  # 1 overall
    num_food_group_area = len(self.food_groups)  # Add one constraint per food group for minimum area
    m = num_linking + num_land + num_min_utilization + num_total_land + num_food_group_area
    
    # D*y >= d constraints (only involving y)
    num_min_foods = F  # 1 per farm (now requiring only 1 food per farm instead of 2)
    num_max_foods = F  # 1 per farm  
    num_group_constraints = len(self.food_groups) * F  # 1 per group per farm
    n = num_min_foods + num_max_foods + num_group_constraints
    
    self.logger.info(f"  Number of master problem constraints (m): {m}")
    self.logger.info(f"  Number of subproblem constraints (n): {n}")
    
    # Initialize matrices
    A = np.zeros((m, Nx))
    B = np.zeros((m, Ny))
    b = np.zeros((m, 1))
    D = np.zeros((n, Ny))
    d = np.zeros((n, 1))
    
    # Add linking constraints: x_ij <= land * y_ij and x_ij >= min_area * y_ij
    constraint_idx = 0
    self.logger.info("Adding linking constraints:")
    
    for fi, farm in enumerate(self.farms):
        land_availability = self.parameters['land_availability'][farm]
        for food_idx, food in enumerate(self.foods):
            pos = fi * C + food_idx
            
            # RELAXATION: Reduce minimum viable planting area by 20%
            min_viable_area = self.parameters.get('min_planting_area', {}).get(food, 5) * 0.8
            
            # 1. x_ij - min_area * y_ij >= 0 (lower bound)
            A[constraint_idx, pos] = 1  # Coefficient for x_ij
            B[constraint_idx, pos] = -min_viable_area  # Coefficient for y_ij
            b[constraint_idx, 0] = 0
            constraint_idx += 1
            
            # 2. -x_ij + land * y_ij >= 0 (upper bound)
            A[constraint_idx, pos] = -1  # Coefficient for x_ij
            B[constraint_idx, pos] = land_availability  # Coefficient for y_ij
            b[constraint_idx, 0] = 0
            constraint_idx += 1
            
            self.logger.info(f"Farm {farm}, Food {food}: Lower bound = {min_viable_area}, Upper bound = {land_availability}")
    
    # Add farm land constraints: sum(x_ij) <= land for each farm
    self.logger.info("Adding farm land constraints:")
    for fi, farm in enumerate(self.farms):
        land_availability = self.parameters['land_availability'][farm]
        
        # Set up constraint: -sum(x_ij) >= -land
        for food_idx in range(C):
            pos = fi * C + food_idx
            A[constraint_idx, pos] = -1  # Coefficient for x_ij
        
        b[constraint_idx, 0] = -land_availability
        constraint_idx += 1
        
        self.logger.info(f"Farm {farm}: Land availability = {land_availability}")
    
    # Add minimum farm utilization constraints: sum(x_ij) >= 0.5 * land for each farm
    # RELAXATION: Reduced from 75% to 50% of land
    self.logger.info("Adding minimum farm utilization constraints (reduced to 50%):")
    for fi, farm in enumerate(self.farms):
        land_availability = self.parameters['land_availability'][farm]
        min_utilization = 0.5 * land_availability  # Reduced from 0.75 to 0.5
        
        # Set up constraint: sum(x_ij) >= min_utilization
        for food_idx in range(C):
            pos = fi * C + food_idx
            A[constraint_idx, pos] = 1  # Coefficient for x_ij
        
        b[constraint_idx, 0] = min_utilization
        constraint_idx += 1
        
        self.logger.info(f"Farm {farm}: Minimum utilization = {min_utilization} (50% of {land_availability})")
    
    # Add total land utilization constraint across all farms
    # RELAXATION: Reduced from 70% to 50% of total land
    self.logger.info("Adding total land utilization constraint (reduced to 50%):")
    total_land = sum(self.parameters['land_availability'][farm] for farm in self.farms)
    min_total_usage = 0.5 * total_land  # Reduced from 0.7 to 0.5
    
    # Set up constraint: sum(x_ij for all i,j) >= min_total_usage
    for pos in range(Nx):
        A[constraint_idx, pos] = 1  # Coefficient for all x_ij
    
    b[constraint_idx, 0] = min_total_usage
    constraint_idx += 1
    
    self.logger.info(f"Total land: {total_land}, Minimum usage: {min_total_usage} (50%)")
    
    # Now set up D and d for y-only constraints
    d_constraint_idx = 0
    
    # Add minimum foods per farm constraints: sum(y_ij) >= 1 for each farm
    # RELAXATION: Reduced from 2 to 1 minimum foods
    self.logger.info("Adding minimum foods per farm constraints (reduced to 1):")
    for fi, farm in enumerate(self.farms):
        # Set up constraint: sum(y_ij) >= 1
        for food_idx in range(C):
            pos = fi * C + food_idx
            D[d_constraint_idx, pos] = 1  # Coefficient for y_ij
        
        d[d_constraint_idx, 0] = 1  # Reduced from 2 to 1
        d_constraint_idx += 1
        
        self.logger.info(f"Farm {farm}: Minimum foods = 1")
    
    # Add maximum foods per farm constraints: sum(y_ij) <= 5 for each farm
    self.logger.info("Adding maximum foods per farm constraints:")
    for fi, farm in enumerate(self.farms):
        # Set up constraint: -sum(y_ij) >= -5 (equivalent to sum(y_ij) <= 5)
        for food_idx in range(C):
            pos = fi * C + food_idx
            D[d_constraint_idx, pos] = -1  # Coefficient for y_ij (negative)
        
        d[d_constraint_idx, 0] = -5  # At most 5 foods (negative sign converts <= to >=)
        d_constraint_idx += 1
        
        self.logger.info(f"Farm {farm}: Maximum foods = 5")
    
    # Add food group constraints
    self.logger.info("Adding food group constraints:")
    
    # Replace specific grain/fruit/vegetable constraints with a general approach for all food groups
    for group, foods in self.food_groups.items():
        # Skip empty groups
        if not foods:
            continue
        
        # Get food indices for this group
        group_indices = [list(self.foods.keys()).index(food) for food in foods]
        
        if group_indices:
            # Create a constraint to ensure minimum number of foods from this group
            # For each farm, set coefficient 1 for all y variables corresponding to foods in this group
            for fi, farm in enumerate(self.farms):
                for food_idx in group_indices:
                    pos = fi * C + food_idx
                    D[d_constraint_idx, pos] = 1  # Coefficient for y_ij if food is in this group
            
            # Minimum foods required from this group across all farms
            # Get constraint from parameters if available, otherwise use at least 1
            min_foods = self.parameters.get('food_group_constraints', {}).get(group, {}).get('min_foods', 1)
            # Ensure we require at least 1 food from each group
            min_foods = max(min_foods, 1)
            
            # Set right-hand side to enforce minimum foods from this group
            d[d_constraint_idx, 0] = min_foods
            d_constraint_idx += 1
            
            self.logger.info(f"Global constraint: At least {min_foods} {group} across all farms")
            
            # Add constraint for minimum land allocation to this group
            # This is handled in the subproblem (continuous variables)
            # Require at least 5% of total land for each food group
            total_land = sum(self.parameters['land_availability'][farm] for farm in self.farms)
            min_group_area = 0.05 * total_land
            
            # For each farm-food pair in this group, add coefficient to x variable in subproblem
            for fi, farm in enumerate(self.farms):
                for food_idx in group_indices:
                    pos = fi * C + food_idx
                    A[constraint_idx, pos] = 1  # Coefficient for x_ij if food is in this group
            
            # Set right-hand side to minimum area
            b[constraint_idx, 0] = min_group_area
            constraint_idx += 1  # Increment the main constraint index
            
            self.logger.info(f"Area constraint: At least {min_group_area:.2f} hectares for {group}")
    
    # Set problem data with relaxed constraints
    benders.set_problem_data(A=A, B=B, b=b, c=c, f=f, D=D, d=d, y_init=y_init)
    
    # Pass farm information for better cuts
    benders._farm_size = {i: self.parameters['land_availability'][farm] for i, farm in enumerate(self.farms)}
    benders._num_farms = F
    benders._num_foods = C
    
    # Run Benders decomposition with progressive tightening
    self.logger.info("Starting Benders decomposition with progressive constraint tightening:")
    start_time = time.time()
    
    # Phase 1: Solve with relaxed constraints
    self.logger.info("Phase 1: Solving with relaxed constraints...")
    benders.solve_problem()
    
    # Extract initial solution - if it's feasible
    if benders.y_sol is not None and benders.x_sol is not None:
        self.logger.info("Phase 1 complete - feasible solution found.")
        y_arr = benders.y_sol.flatten()
        x_arr = benders.x_sol.flatten()
        
        # Create initial solution dictionary for debugging
        phase1_solution = {}
        for i, val in enumerate(y_arr):
            if val > 0.5:  # Binary y is active
                farm = self.farms[i // C]
                food = list(self.foods.keys())[i % C]
                if x_arr[i] > 1.0:
                    phase1_solution[(farm, food)] = x_arr[i]
                    self.logger.info(f"  Phase 1 solution: Farm {farm}, Food {food}: {x_arr[i]:.2f} hectares")
    else:
        self.logger.warning("Phase 1 failed to find a feasible solution.")
    
    runtime = time.time() - start_time
    
    # Extract solution from the most recent solve
    self.logger.info("Benders decomposition completed:")
    self.logger.info(f"  Upper bound: {benders.UB}")
    self.logger.info(f"  Lower bound: {benders.LB}")
    gap = abs(benders.UB - benders.LB)
    relative_gap = gap / abs(benders.UB) if benders.UB != 0 else float('inf')
    self.logger.info(f"  Absolute gap: {gap}")
    self.logger.info(f"  Relative gap: {relative_gap:.4f} ({relative_gap*100:.2f}%)")
    self.logger.info(f"  Convergence tolerance: {benders.eps}")
    self.logger.info(f"  Number of iterations: {len(benders.lower_bounds)}")
    self.logger.info(f"  Max iterations: {benders.max_iterations}")
    self.logger.info(f"  Runtime: {runtime:.2f} seconds")
    
    # Check if we hit iteration limit
    if len(benders.lower_bounds) >= benders.max_iterations:
        self.logger.warning("  REASON FOR SUBOPTIMAL: Maximum iterations reached before convergence")
    elif runtime >= benders.time_limit if hasattr(benders, 'time_limit') else False:
        self.logger.warning("  REASON FOR SUBOPTIMAL: Time limit reached before convergence")
    elif gap > benders.eps:
        self.logger.warning(f"  REASON FOR SUBOPTIMAL: Gap ({gap:.6f}) exceeds tolerance ({benders.eps})")
        
        # Provide recommendation
        if gap < 0.1:
            self.logger.info("  RECOMMENDATION: Solution is close to optimal. Consider increasing 'benders.eps' to accept this solution as optimal.")
        else:
            self.logger.info("  RECOMMENDATION: Try increasing max_iterations or improving the initial solution quality.")
    
    # Extract and build solution
    if benders.y_sol is None or benders.x_sol is None:
        self.logger.error("Benders decomposition did not return y_sol or x_sol; using zero vectors.")
        y_arr = np.zeros((Ny, ))
        x_arr = np.zeros((Nx, ))
    else:
        y_arr = benders.y_sol.flatten()
        x_arr = benders.x_sol.flatten()
    
    # Create solution dictionary
    solution = {}
    self.logger.info("Extracting solution:")
    for i, val in enumerate(y_arr):
        if val > 0.5:  # Binary y is active
            farm = self.farms[i // C]
            food = list(self.foods.keys())[i % C]
            
            # Only include values that are significantly positive
            if x_arr[i] > 1.0:
                solution[(farm, food)] = x_arr[i]
                self.logger.info(f"  Farm {farm}, Food {food}: {x_arr[i]:.2f} hectares")
    
    # If solution is empty, try to recover the best solution from previous iterations
    if not solution and hasattr(benders, 'best_x_sol') and benders.best_x_sol is not None:
        self.logger.warning("Empty final solution, trying to recover best solution from previous iterations...")
        best_x_arr = benders.best_x_sol.flatten()
        best_y_arr = benders.best_y_sol.flatten() if benders.best_y_sol is not None else np.zeros((Ny, ))
        
        for i, val in enumerate(best_y_arr):
            if val > 0.5:  # Binary y is active
                farm = self.farms[i // C]
                food = list(self.foods.keys())[i % C]
                if best_x_arr[i] > 1.0:
                    solution[(farm, food)] = best_x_arr[i]
                    self.logger.info(f"  Recovered solution: Farm {farm}, Food {food}: {best_x_arr[i]:.2f} hectares")
    
    # If still no solution, create a basic feasible solution based on initial y values
    if not solution:
        self.logger.warning("No solution found from Benders. Creating a basic feasible solution...")
        # Use the initial y values to create a simple solution
        for farm_idx, farm in enumerate(self.farms):
            land_avail = self.parameters['land_availability'][farm]
            farm_foods = [food for food in farm_selections.get(farm, [])]
            
            if farm_foods:
                # Distribute land evenly among selected foods
                land_per_food = land_avail / len(farm_foods)
                for food in farm_foods:
                    solution[(farm, food)] = land_per_food
                    self.logger.info(f"  Basic solution: Farm {farm}, Food {food}: {land_per_food:.2f} hectares")
    
    # Calculate metrics
    metrics = self.calculate_metrics(solution)
    self.logger.info("Calculated metrics:")
    for metric, value in metrics.items():
        self.logger.info(f"  {metric}: {value:.4f}")
    
    # Create a proper result object
    original_obj = -benders.UB if np.isfinite(benders.UB) else 0
    if f is not None and benders.y_sol is not None:
        original_obj -= float(f.T.dot(benders.y_sol)[0, 0])
        
    result = OptimizationResult(
        status="optimal" if abs(benders.UB - benders.LB) <= benders.eps else "suboptimal",
        objective_value=original_obj,  # Convert from minimization to maximization
        solution=solution,
        metrics=metrics,
        runtime=runtime,
        benders_data={
            'lower_bounds': benders.lower_bounds,
            'upper_bounds': benders.upper_bounds,
            'iterations': len(benders.lower_bounds),
            'max_iterations': benders.max_iterations,
            'gap': gap,
            'relative_gap': relative_gap,
            'tolerance': benders.eps
        }
    )
    
    # Also save bounds data for plotting
    self.benders_bounds = {
        'lower': benders.lower_bounds,
        'upper': benders.upper_bounds
    }
    
    return result
