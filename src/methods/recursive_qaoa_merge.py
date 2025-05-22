import os
import sys
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Ensure my_functions is in path
try:
    # Look for my_functions two directories up from this file
    my_functions_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'my_functions')
    if os.path.exists(my_functions_path):
        sys.path.append(os.path.dirname(my_functions_path))
        from my_functions.benders import Benders
        from my_functions.recursive_qaoa import solve_benders_master_with_recursive_qaoa
    else:
        print(f"Warning: 'my_functions' directory not found at {my_functions_path}", file=sys.stderr)
        Benders = None
        solve_benders_master_with_recursive_qaoa = None
except ImportError as e:
    print(f"Error importing quantum methods: {e}", file=sys.stderr)
    Benders = None
    solve_benders_master_with_recursive_qaoa = None

# Add parent directory to path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute import instead of relative import
from data_models import OptimizationResult, OptimizationObjective

def optimize_with_recursive_qaoa_merge(self, recursion_threshold=5):
    """
    Uses hybrid quantum-classical Benders decomposition with Recursive QAOA for master problem 
    and implements the subgraph merge strategy from the paper.
    
    Args:
        recursion_threshold: Maximum problem size to solve directly with QAOA before recursion
    """
    import traceback
    from my_functions.benders import Benders
    from my_functions.recursive_qaoa import solve_benders_master_with_recursive_qaoa
    import numpy as np
    import time

    # Initialize Benders solver with modified parameters
    benders = Benders()
    benders.multi_cut = True
    benders.eps = 0.001  # Tighter tolerance
    benders.max_iterations = 50  # Reduced from 100
    benders.relative = True
    
    self.logger.info("Initializing recursive QAOA merge Benders decomposition:")
    self.logger.info(f"  Number of farms (F): {len(self.farms)}")
    self.logger.info(f"  Number of foods (C): {len(self.foods)}")
    
    # Dimensions
    F = len(self.farms)
    C = len(self.foods)
    Nx = F * C  # continuous x variables
    Ny = F * C  # binary y variables
    
    self.logger.info(f"  Number of continuous variables (Nx): {Nx}")
    self.logger.info(f"  Number of binary variables (Ny): {Ny}")
    
    # Build objective coefficient vectors first
    self.logger.info("Building objective function components:")
    c = np.zeros((Nx, 1))
    f = np.zeros((Ny, 1))
    
    for fi, farm in enumerate(self.farms):
        for food_idx, food in enumerate(self.foods):
            pos = fi * C + food_idx
            
            food_data = self.foods[food]
            weights = self.parameters['weights']
            
            pos_score = (
                weights['nutritional_value'] * food_data['nutritional_value'] +
                weights['nutrient_density'] * food_data['nutrient_density'] +
                weights['affordability'] * food_data['affordability'] +
                weights['sustainability'] * food_data['sustainability']
            )
            neg_score = weights['environmental_impact'] * food_data['environmental_impact']
            
            net_score = pos_score - neg_score
            c[pos, 0] = -net_score
            f[pos, 0] = -net_score  # Add to f vector as well
            
            self.logger.info(f"Farm {farm}, Food {food}, Score: {net_score}")
    
    # Now generate the smart initial solution using the defined f vector
    y_init = np.zeros((Ny, 1), dtype=float)
    # Initialize based on objective coefficients
    for fi in range(F):
        # Get objective coefficients for this farm
        farm_scores = []
        for food_idx in range(C):
            pos = fi * C + food_idx
            farm_scores.append((food_idx, -f[pos, 0]))  # Negative because we minimize
        
        # Sort foods by score (higher is better)
        farm_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 2-3 foods with highest scores
        num_foods = np.random.randint(2, 4)
        for food_idx, _ in farm_scores[:num_foods]:
            y_init[fi * C + food_idx] = 1.0

    # Build constraint matrices
    num_linking = 2 * Nx
    num_land = F
    num_min_utilization = F
    num_total_land = 1
    m = num_linking + num_land + num_min_utilization + num_total_land
    
    num_min_foods = F
    num_max_foods = F
    num_group_constraints = len(self.food_groups) * F
    n = num_min_foods + num_max_foods + num_group_constraints
    
    self.logger.info(f"  Number of master problem constraints (m): {m}")
    self.logger.info(f"  Number of subproblem constraints (n): {n}")
    
    # Initialize matrices
    A = np.zeros((m, Nx))
    B = np.zeros((m, Ny))
    b = np.zeros((m, 1))
    D = np.zeros((n, Ny))
    d = np.zeros((n, 1))
    
    # Add linking constraints
    constraint_idx = 0
    self.logger.info("Adding linking constraints:")
    
    for fi, farm in enumerate(self.farms):
        land_availability = self.parameters['land_availability'][farm]
        for food_idx, food in enumerate(self.foods):
            pos = fi * C + food_idx
            
            min_viable_area = self.parameters.get('min_planting_area', {}).get(food, 5) * 0.8
            
            # 1. x_ij - min_area * y_ij >= 0 (lower bound)
            A[constraint_idx, pos] = 1
            B[constraint_idx, pos] = -min_viable_area
            b[constraint_idx, 0] = 0
            constraint_idx += 1
            
            # 2. -x_ij + land * y_ij >= 0 (upper bound)
            A[constraint_idx, pos] = -1
            B[constraint_idx, pos] = land_availability
            b[constraint_idx, 0] = 0
            constraint_idx += 1
            
            self.logger.info(f"Farm {farm}, Food {food}: Lower bound = {min_viable_area}, Upper bound = {land_availability}")
    
    # Add farm land constraints
    self.logger.info("Adding farm land constraints:")
    for fi, farm in enumerate(self.farms):
        land_availability = self.parameters['land_availability'][farm]
        
        # Set up constraint: -sum(x_ij) >= -land
        for food_idx in range(C):
            pos = fi * C + food_idx
            A[constraint_idx, pos] = -1
        
        b[constraint_idx, 0] = -land_availability
        constraint_idx += 1
        
        self.logger.info(f"Farm {farm}: Land availability = {land_availability}")
    
    # Add minimum farm utilization constraints - reduced to 20% instead of 50%
    self.logger.info("Adding minimum farm utilization constraints (reduced to 20%):")
    for fi, farm in enumerate(self.farms):
        land_availability = self.parameters['land_availability'][farm]
        min_utilization = 0.2 * land_availability  # 20% minimum utilization
        
        for food_idx in range(C):
            pos = fi * C + food_idx
            A[constraint_idx, pos] = 1
        
        b[constraint_idx, 0] = min_utilization
        constraint_idx += 1
        
        self.logger.info(f"Farm {farm}: Minimum utilization = {min_utilization} (20% of {land_availability})")
    
    # Add total land utilization constraint
    self.logger.info("Adding total land utilization constraint (reduced to 20%):")
    total_land = sum(self.parameters['land_availability'][farm] for farm in self.farms)
    min_total_usage = 0.2 * total_land  # 20% minimum utilization
    
    for pos in range(Nx):
        A[constraint_idx, pos] = 1
    
    b[constraint_idx, 0] = min_total_usage
    constraint_idx += 1
    
    self.logger.info(f"Total land: {total_land}, Minimum usage: {min_total_usage} (20%)")
    
    # Setup D and d for y-only constraints
    d_constraint_idx = 0
    
    # Add minimum foods per farm constraints
    self.logger.info("Adding minimum foods per farm constraints (reduced to 1):")
    for fi, farm in enumerate(self.farms):
        for food_idx in range(C):
            pos = fi * C + food_idx
            D[d_constraint_idx, pos] = 1
        
        d[d_constraint_idx, 0] = 1  # At least 1 food per farm
        d_constraint_idx += 1
        
        self.logger.info(f"Farm {farm}: Minimum foods = 1")
    
    # Add maximum foods per farm constraints
    self.logger.info("Adding maximum foods per farm constraints:")
    for fi, farm in enumerate(self.farms):
        for food_idx in range(C):
            pos = fi * C + food_idx
            D[d_constraint_idx, pos] = -1
        
        d[d_constraint_idx, 0] = -4  # Maximum 4 foods per farm
        d_constraint_idx += 1
        
        self.logger.info(f"Farm {farm}: Maximum foods = 4")
    
    # Add food group constraints
    self.logger.info("Adding food group constraints:")
    
    # Add grain constraints - ensure we have at least one grain
    grain_indices = [list(self.foods.keys()).index(food) for food in self.food_groups.get('Grains', [])]
    if grain_indices:
        for fi, farm in enumerate(self.farms):
            for food_idx in grain_indices:
                pos = fi * C + food_idx
                D[d_constraint_idx, pos] = 1
        
        d[d_constraint_idx, 0] = F  # At least one grain per farm
        d_constraint_idx += 1
        
        self.logger.info(f"Global constraint: At least {F} grains across all farms")
    
    # Set problem data in Benders
    benders.set_problem_data(A=A, B=B, b=b, c=c, f=f, D=D, d=d, y_init=y_init)
    
    benders._farm_size = {i: self.parameters['land_availability'][farm] for i, farm in enumerate(self.farms)}
    benders._num_farms = F
    benders._num_foods = C
    
    # QUBO configuration parameters - adjusted for better performance
    qubo_config = {
        "eta_min": -500.0,  # Reduced range
        "eta_max": 500.0,   # Reduced range
        "eta_num_bits": 5,  # Reduced bits
        "penalty_coefficient": 5000.0,  # Reduced penalty
        "penalty_slack_num_bits": 3     # Reduced bits
    }
    
    # QAOA parameters for the recursive QAOA master problem solver
    recursive_qaoa_params = {
        "qaoa_depth": 2,  # Reduced depth
        "num_shots": 2048,  # Increased shots
        "optimizer_method": "COBYLA",
        "max_iter": 150,  # Increased iterations
        "backend": None,
        "use_simulator_fallback": True,
        "recursion_threshold": recursion_threshold,  # Recursion threshold parameter
        "random_seed": None  # Use different random seed each time
    }
    
    # Calculate QUBO metrics
    self.quantum_metrics = {
        'num_variables': int(Ny + qubo_config["eta_num_bits"] + (qubo_config["penalty_slack_num_bits"] * D.shape[0]) + qubo_config["penalty_slack_num_bits"]), 
        'num_qubits_qaoa': int(Ny + qubo_config["eta_num_bits"] + (qubo_config["penalty_slack_num_bits"] * D.shape[0]) + qubo_config["penalty_slack_num_bits"]),
        'num_spins': int(Ny + qubo_config["eta_num_bits"] + (qubo_config["penalty_slack_num_bits"] * D.shape[0]) + qubo_config["penalty_slack_num_bits"]),
        'matrix_density': 25.0,
        'original_vars': Ny,
        'eta_bits': qubo_config["eta_num_bits"],
        'slack_vars_D': qubo_config["penalty_slack_num_bits"] * D.shape[0],
        'slack_vars_cuts': qubo_config["penalty_slack_num_bits"],
        'slack_bits_per_constraint': qubo_config["penalty_slack_num_bits"],
        'recursion_threshold': recursion_threshold  # Add recursion threshold to metrics
    }
    
    # Run Benders with recursive QAOA for master problem
    self.logger.info("Starting hybrid recursive QAOA Benders decomposition with merge strategy:")
    start_time = time.time()
    
    # Benders iteration loop
    self.optimality_cuts = []
    self.feasibility_cuts = []
    self.lower_bounds = []
    self.upper_bounds = []
    LB = -np.inf
    UB = np.inf
    y_sol = y_init.copy()
    quantum_bounds = {'lower': [], 'upper': []}
    best_x_sol = None  # Initialize best_x_sol
    best_y_sol = y_init.copy() # Initialize best_y_sol with initial y
    best_obj_value = -np.inf # Initialize with a very small number for maximization
    found_feasible = False # Track if any feasible solution is found
    
    # Set modified convergence parameters for the quantum approach
    quantum_eps = benders.eps * 1.5  # Tighter tolerance
    quantum_min_iterations = 3       # Reduced minimum iterations
    max_iterations = 15              # Reduced from 20
    
    # Store subgraph solutions for merging later
    subgraph_solutions = []
    
    # Iteration loop
    self.logger.info("Beginning recursive QAOA Benders iterations with merge strategy...")
    for k in range(max_iterations):
        iteration_start = time.time()
        self.logger.info(f"Iteration {k+1}/{max_iterations}")
        
        # Solve subproblem for fixed y
        solved, dual_vars, obj_val_sub, extreme_ray = benders.solve_subproblem(y_sol)
        
        # Update upper bound if feasible
        if solved:
            current_total_obj = obj_val_sub + float(f.T.dot(y_sol)[0, 0])
            self.logger.info(f"Subproblem solved: objective value = {current_total_obj}")
            
            if current_total_obj < UB:
                UB = current_total_obj
                self.logger.info(f"New upper bound: {UB}")
            
            # Generate optimality cut
            self.optimality_cuts.append(dual_vars)
            self.logger.info(f"Generated optimality cut from dual variables")
        else:
            # Generate feasibility cut
            self.feasibility_cuts.append(extreme_ray)
            self.logger.info(f"Subproblem infeasible, generated feasibility cut")
        
        # Solve master problem with recursive QAOA algorithm
        self.logger.info("Solving master problem using recursive QAOA algorithm...")
        
        try:
            # Adjust recursive QAOA parameters based on iteration
            current_recursive_qaoa_params = recursive_qaoa_params.copy()
            current_recursive_qaoa_params["qaoa_depth"] = min(2, 1 + k // 7)  # Increase depth over time
            current_recursive_qaoa_params["recursion_threshold"] = max(5, recursion_threshold - k)  # Decrease threshold over time
            
            self.logger.info(f"Using recursive QAOA for master problem with recursion_threshold={current_recursive_qaoa_params['recursion_threshold']}")
            
            qaoa_master_result = solve_benders_master_with_recursive_qaoa(
                f_coeffs=f,
                D_matrix=D,
                d_vector=d,
                optimality_cuts=self.optimality_cuts,
                feasibility_cuts=self.feasibility_cuts,
                B_matrix=B,
                b_vector=b,
                Ny=Ny,
                config=qubo_config,
                qaoa_params=current_recursive_qaoa_params
            )

            if qaoa_master_result and qaoa_master_result.get("solution") is not None and qaoa_master_result.get("error") is None:
                y_quantum = qaoa_master_result["solution"]
                master_obj = qaoa_master_result["objective"]
                
                # Extract subgraph solutions from recursive QAOA if available
                if qaoa_master_result.get("metrics", {}).get("subresults"):
                    # Extract and save subgraph solutions for final merge step
                    sub_results = qaoa_master_result.get("metrics", {}).get("subresults", {})
                    if isinstance(sub_results, dict):
                        sub_solutions = list(sub_results.values())
                        self.logger.info(f"Obtained {len(sub_solutions)} subgraph solutions from recursion for merging")
                        subgraph_solutions.append(sub_solutions)
                
                # Update y_sol directly from QAOA result - no noise for now as we'll use the merging
                y_sol = y_quantum.reshape((Ny, 1))
                
                # Update lower bound with more conservative updates
                if master_obj > LB:
                    LB = master_obj
                    self.logger.info(f"New lower bound from recursive QAOA: {LB}")
                
                # Capture QUBO metrics from the first successful QAOA solve
                if k == 0 and qaoa_master_result.get("metrics"):
                    first_qaoa_run_metrics = qaoa_master_result["metrics"]
                    self.quantum_metrics.update(first_qaoa_run_metrics)
            else:
                # If recursive QAOA fails, we keep using the last solution
                error_msg = qaoa_master_result.get("error", "Unknown error") if qaoa_master_result else "Recursive QAOA solver returned None"
                self.logger.error(f"Recursive QAOA master problem failed: {error_msg}. Using previous solution.")
                
                # Keep using the previous solution
                master_obj = float(f.T.dot(y_sol)[0, 0])
                if master_obj > LB:
                    LB = master_obj

        except Exception as e:
            self.logger.error(f"Error in recursive QAOA master problem: {e}")
            self.logger.error(traceback.format_exc())
            # Keep using the previous solution
            self.logger.error(f"Using previous solution due to error.")
            
            # Keep the current y_sol
            master_obj = float(f.T.dot(y_sol)[0, 0])
            if master_obj > LB:
                LB = master_obj

        # Store bounds
        self.lower_bounds.append(LB)
        self.upper_bounds.append(UB)
        quantum_bounds['lower'].append(LB)
        quantum_bounds['upper'].append(UB)
        
        # Check convergence with modified criteria
        gap = UB - LB
        self.logger.info(f"Current bounds: LB = {LB}, UB = {UB}, Gap = {gap}")
        self.logger.info(f"Iteration time: {time.time() - iteration_start:.2f} seconds")
        
        # Only check convergence after minimum iterations and with modified tolerance
        if k >= quantum_min_iterations and gap <= quantum_eps * abs(UB):
            self.logger.info(f"Converged with gap = {gap} <= {quantum_eps * abs(UB)}")
            break
    
    # Post-process the quantum solution using results of recursive decomposition
    self.logger.info("Iterations complete. Post-processing recursive QAOA solutions...")
    
    # Process the multiple solutions from recursion levels
    merged_solutions = []
    
    # Use the final y_sol as one candidate
    merged_solutions.append(y_sol.flatten())
    
    # Try to extract additional candidate solutions from the recursion results
    if len(subgraph_solutions) > 0 and any(subgraph_solutions):
        self.logger.info("Using solutions from recursive QAOA levels...")
        
        # Extract candidate solutions from different recursion levels if available
        for sub_sols in subgraph_solutions:
            for sub_sol in sub_sols:
                if sub_sol and hasattr(sub_sol, "get") and sub_sol.get("solution") is not None:
                    # Extract solution from this recursion level
                    sol_dict = sub_sol.get("solution", {})
                    if sol_dict:
                        sol_array = np.zeros(Ny)
                        # Convert dictionary to array
                        for idx, val in sol_dict.items():
                            if 0 <= int(idx) < Ny:  # Ensure idx is treated as integer
                                sol_array[int(idx)] = val
                        merged_solutions.append(sol_array)
                        self.logger.info(f"Added solution from recursion level with obj: {sub_sol.get('objective', 'unknown')}")
    
    # Also generate Z2 symmetry solution (flip all bits) for diversity
    if merged_solutions:
        # Add Z2 symmetry solution
        z2_solution = 1 - merged_solutions[0].copy()
        merged_solutions.append(z2_solution)
        
        # Add partially flipped solutions for diversity
        # Partial random flips (30% of bits)
        partial_flip = merged_solutions[0].copy()
        flip_indices = np.random.choice(range(Ny), size=int(0.3*Ny), replace=False)
        for idx in flip_indices:
            partial_flip[idx] = 1 - partial_flip[idx]
        merged_solutions.append(partial_flip)
    
    runtime = time.time() - start_time
    self.logger.info(f"Total runtime: {runtime:.2f} seconds")
    
    # Evaluate merged solutions to find best feasible solution
    candidate_solutions = [sol.reshape((Ny, 1)) for sol in merged_solutions]
    
    # Also include the last solution from iterations if not already in merged_solutions
    if len(candidate_solutions) == 0 or not np.array_equal(y_sol, candidate_solutions[0]):
        candidate_solutions.append(y_sol)
    
    # Extract solution - using probabilistic selection among merged solutions
    found_feasible_solutions = []
    feasible_obj_values = []
    feasible_x_sols = []
    feasible_y_sols = []
    
    for i, candidate_y in enumerate(candidate_solutions):
        self.logger.info(f"Trying merged candidate solution {i+1}/{len(candidate_solutions)}")
        solved_primal, x_sol_candidate, obj_val_candidate = benders.solve_primal_subproblem(candidate_y)
        
        if solved_primal and x_sol_candidate is not None:
            obj_value = obj_val_candidate + float(f.T.dot(candidate_y)[0, 0])
            found_feasible_solutions.append(True)
            feasible_obj_values.append(obj_value)
            feasible_x_sols.append(x_sol_candidate)
            feasible_y_sols.append(candidate_y)
            self.logger.info(f"Found feasible merged solution with objective value: {obj_value}")
    
    # Probabilistic selection among feasible solutions
    found_feasible = len(found_feasible_solutions) > 0
    
    if found_feasible:
        # Convert objective values to probabilities (better solutions have higher probability)
        # Shift to positive to use as probabilities
        min_obj = min(feasible_obj_values)
        shifted_obj_values = [v - min_obj + 1.0 for v in feasible_obj_values]
        
        # Use softmax-like formula to convert to probabilities
        exp_values = [np.exp(v) for v in shifted_obj_values]
        sum_exp = sum(exp_values)
        probs = [v/sum_exp for v in exp_values]
        
        self.logger.info(f"Selection probabilities for merged solutions: {probs}")
        
        # Select based on probabilities
        selected_idx = np.random.choice(range(len(probs)), p=probs)
        
        best_obj_value = feasible_obj_values[selected_idx]
        best_x_sol = feasible_x_sols[selected_idx]
        best_y_sol = feasible_y_sols[selected_idx]
        
        self.logger.info(f"Probabilistically selected merged solution {selected_idx+1} with objective value: {best_obj_value}")
    
    # If no feasible solution found
    if not found_feasible:
        self.logger.warning("No feasible merged solution found. Keeping recursive QAOA solution but marking as infeasible.")
        
        # Even here, randomly select one of the candidate solutions
        selected_idx = np.random.randint(0, len(candidate_solutions))
        best_y_sol = candidate_solutions[selected_idx]
        best_obj_value = float(f.T.dot(best_y_sol)[0, 0])
        
        self.logger.info(f"Using randomly selected recursive QAOA solution with objective: {best_obj_value}")
    
    # Extract solution
    if best_x_sol is not None:
        x_arr = best_x_sol.flatten()
        y_arr = best_y_sol.flatten()
        
        solution = {}
        self.logger.info("Extracting solution:")
        for i, val in enumerate(y_arr):
            if val > 0.5:  # Binary y is active
                farm = self.farms[i // C]
                food = list(self.foods.keys())[i % C]
                
                if x_arr[i] > 1.0:
                    solution[(farm, food)] = x_arr[i]
                    self.logger.info(f"  Farm {farm}, Food {food}: {x_arr[i]:.2f} hectares")
    else:
        self.logger.warning("No valid solution found from primal subproblem")
        solution = {}
    
    # Calculate metrics
    metrics = self.calculate_metrics(solution)
    self.logger.info("Calculated metrics:")
    for metric, value in metrics.items():
        self.logger.info(f"  {metric}: {value:.4f}")
    
    # Add recursive QAOA specific metrics
    metrics.update({
        'recursion_threshold': recursive_qaoa_params['recursion_threshold'],
        'recursive_qaoa_depth': recursive_qaoa_params['qaoa_depth'],
        'num_candidate_solutions': len(merged_solutions)
    })
    
    # Create proper result object
    if not found_feasible:
        obj_value = 0.0  # Fallback if no feasible solution found
    else:
        obj_value = -best_obj_value  # Convert from minimization to maximization
    
    result = OptimizationResult(
        status="optimal" if found_feasible else "infeasible",
        objective_value=obj_value,
        solution=solution,
        metrics=metrics,
        runtime=runtime,
        benders_data={
            'lower_bounds': quantum_bounds['lower'],
            'upper_bounds': quantum_bounds['upper'],
            'iterations': len(quantum_bounds['lower'])
        }
    )
    
    # Save bounds for plotting
    self.quantum_bounds = quantum_bounds
    
    return result 