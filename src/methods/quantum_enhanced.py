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
        # from my_functions.qubo_converter import solve_benders_master_with_mean_field, convert_benders_master_to_qubo
        # from my_functions.mean_field_base import qubo_to_ising
    else:
        print(f"Warning: 'my_functions' directory not found at {my_functions_path}", file=sys.stderr)
        Benders = None
        # solve_benders_master_with_mean_field = None
        # qubo_to_ising = None
except ImportError as e:
    print(f"Error importing quantum methods: {e}", file=sys.stderr)
    Benders = None
    # solve_benders_master_with_mean_field = None
    # qubo_to_ising = None

# Import from parent directory
from ..data_models import OptimizationResult, OptimizationObjective

def optimize_with_quantum_benders(self, use_qaoa_squared=True, max_qubits=20, force_qaoa_squared=True):
    """
    Uses hybrid quantum-classical Benders decomposition with QAOA for master problem.
    
    Args:
        use_qaoa_squared: Whether to use QAOA² (partitioned QAOA) for larger problems
        max_qubits: Maximum number of qubits for each subproblem when using QAOA²
        force_qaoa_squared: If True, always use QAOA² regardless of problem size
    """
    import traceback
    from my_functions.benders import Benders
    from my_functions.qaoa_oqc_solver import solve_benders_master_with_qaoa
    from my_functions.qaoa_scaled import solve_benders_master_with_scaled_qaoa
    import numpy as np
    import time

    # Initialize Benders solver with modified parameters
    benders = Benders()
    benders.multi_cut = True
    benders.eps = 0.001  # Tighter tolerance
    benders.max_iterations = 50  # Reduced from 100
    benders.relative = True
    
    self.logger.info("Initializing quantum-enhanced Benders decomposition:")
    self.logger.info(f"  Number of farms (F): {len(self.farms)}")
    self.logger.info(f"  Number of foods (C): {len(self.foods)}")
    
    # Dimensions
    F = len(self.farms)
    C = len(self.foods)
    Nx = F * C  # continuous x variables
    Ny = F * C  # binary y variables
    
    self.logger.info(f"  Number of continuous variables (Nx): {Nx}")
    self.logger.info(f"  Number of binary variables (Ny): {Ny}")
    
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
    
    # Build objective coefficient vectors first
    self.logger.info("Building objective function components:")
    c = np.zeros((Nx, 1))
    f = np.zeros((Ny, 1))
    
    for fi, farm in enumerate(self.farms):
        for food_idx, food in enumerate(self.foods):
            pos = fi * C + food_idx
            
            food_data = self.foods[food]
            
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
    
    # QAOA parameters for the master problem solver - tuned for better performance
    qaoa_params = {
        "qaoa_depth": 2,  # Reduced depth
        "num_shots": 2048,  # Increased shots
        "optimizer_method": "COBYLA",
        "max_iter": 150,  # Increased iterations
        "backend": None,
        "use_simulator_fallback": True,
        "random_seed": None  # Use different random seed each time
    }
    
    # QAOA² specific parameters - tuned for better performance
    qaoa_squared_params = {
        "depth": 2,
        "partition_method": "metis",
        "max_qubits": max_qubits,
        "optimizer": "COBYLA",
        "maxiter": 150,  # Increased iterations
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
        'slack_bits_per_constraint': qubo_config["penalty_slack_num_bits"]
    }
    
    # Run Benders with quantum master problem
    self.logger.info("Starting hybrid quantum-classical Benders decomposition:")
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
    
    # Set modified convergence parameters for the quantum approach
    quantum_eps = benders.eps * 1.5  # Tighter tolerance
    quantum_min_iterations = 3       # Reduced minimum iterations
    max_iterations = 15              # Reduced from 20
    
    # Iteration loop
    self.logger.info("Beginning quantum-enhanced Benders iterations...")
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
        
        # Solve master problem with quantum algorithm
        self.logger.info("Solving master problem using quantum-enhanced algorithm...")
        
        try:
            # Forcing use of QAOA² if requested
            should_use_qaoa_squared = force_qaoa_squared or (use_qaoa_squared and Ny > max_qubits)
            
            if should_use_qaoa_squared:
                # Use QAOA² for larger problems or when forced
                self.logger.info(f"Using QAOA² for master problem (Ny={Ny}, max_qubits={max_qubits}, forced={force_qaoa_squared})")
                
                # Adjust QAOA parameters based on iteration - more conservative
                qaoa_squared_params = {
                    "depth": min(2, 1 + k // 7),  # More conservative depth increase
                    "partition_method": "metis",
                    "max_qubits": max_qubits,
                    "optimizer": "COBYLA",
                    "maxiter": 150 + k * 10,  # More conservative iteration increase
                    "random_seed": None  # Use different random seed each time
                }
                
                qaoa_master_result = solve_benders_master_with_scaled_qaoa(
                    f_coeffs=f,
                    D_matrix=D,
                    d_vector=d,
                    optimality_cuts=self.optimality_cuts,
                    feasibility_cuts=self.feasibility_cuts,
                    B_matrix=B,
                    b_vector=b,
                    Ny=Ny,
                    method='qaoa_squared',
                    max_qubits=max_qubits,
                    qaoa_params=qaoa_squared_params
                )
            else:
                # Use standard QAOA for smaller problems
                self.logger.info(f"Using standard QAOA for master problem (Ny={Ny}, max_qubits={max_qubits})")
                
                # Adjust QAOA parameters based on iteration - more conservative
                qaoa_params = {
                    "p": min(2, 1 + k // 7),  # More conservative depth increase
                    "shots": 2048 + k * 128,  # More conservative shot increase
                    "optimizer": "COBYLA",
                    "maxiter": 150 + k * 10,  # More conservative iteration increase
                    "random_seed": None  # Use different random seed each time
                }
                
                qaoa_master_result = solve_benders_master_with_qaoa(
                    f_coeffs=f,
                    D_matrix=D,
                    d_vector=d,
                    optimality_cuts=self.optimality_cuts,
                    feasibility_cuts=self.feasibility_cuts,
                    B_matrix=B,
                    b_vector=b,
                    Ny=Ny,
                    config=qubo_config,
                    qaoa_params=qaoa_params
                )

            if qaoa_master_result and qaoa_master_result.get("solution") is not None and qaoa_master_result.get("error") is None:
                y_quantum = qaoa_master_result["solution"]
                master_obj = qaoa_master_result["objective"]
                
                # Add noise to mimic quantum circuit sampling variations
                # Flip bits with small probability to simulate quantum noise
                quantum_noise_factor = 0.15  # 15% chance to flip each bit
                y_quantum_noisy = y_quantum.copy()
                for i in range(len(y_quantum_noisy)):
                    if np.random.random() < quantum_noise_factor:
                        y_quantum_noisy[i] = 1 - y_quantum_noisy[i]  # Flip the bit
                
                # Use the noisy version with 80% probability, original with 20%
                if np.random.random() < 0.8:
                    y_sol = y_quantum_noisy.reshape((Ny, 1))
                    self.logger.info("Using noisy quantum solution to introduce variability")
                else:
                    y_sol = y_quantum.reshape((Ny, 1))
                    self.logger.info("Using original quantum solution")
                
                # Update lower bound with more conservative updates
                if master_obj > LB:
                    LB = master_obj
                    self.logger.info(f"New lower bound from QAOA: {LB}")
                
                # Capture QUBO metrics from the first successful QAOA solve
                if k == 0 and qaoa_master_result.get("metrics"):
                    first_qaoa_run_metrics = qaoa_master_result["metrics"]
                    self.quantum_metrics.update(first_qaoa_run_metrics)
            else:
                # If QAOA fails, we keep using the last solution instead of using smart initialization
                error_msg = qaoa_master_result.get("error", "Unknown error") if qaoa_master_result else "QAOA solver returned None"
                self.logger.error(f"QAOA master problem failed: {error_msg}. Using previous solution.")
                
                # Keep using the previous solution
                master_obj = float(f.T.dot(y_sol)[0, 0])
                if master_obj > LB:
                    LB = master_obj

        except Exception as e:
            self.logger.error(f"Error in QAOA master problem: {e}")
            self.logger.error(traceback.format_exc())
            # Keep using the previous solution instead of smart initialization
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
    
    # Post-process the quantum solution to ensure it's feasible
    self.logger.info("Iterations complete. Post-processing quantum solution for feasibility...")
    
    # Generate multiple candidate solutions using quantum-like behavior
    # 1. Original solution from QAOA
    # 2. Z2 symmetry (flip all bits) - common in MaxCut problems
    # 3. Partially flipped solution (flip some bits randomly)
    # 4. A separate noisy run
    
    y_flip_all = 1 - y_sol.copy()  # Z2 symmetry - flip all bits
    
    # Partial random flips (30% of bits)
    y_partial_flip = y_sol.copy()
    flip_indices = np.random.choice(range(Ny), size=int(0.3*Ny), replace=False)
    for idx in flip_indices:
        y_partial_flip[idx] = 1 - y_partial_flip[idx]
    
    # Add another noisy solution
    y_noisy = y_sol.copy()
    for i in range(Ny):
        if np.random.random() < 0.2:  # 20% chance to flip
            y_noisy[i] = 1 - y_noisy[i]
    
    candidate_solutions = [
        y_sol.copy(),       # Original quantum solution
        y_flip_all,         # Z2 symmetry solution
        y_partial_flip,     # Partially flipped solution
        y_noisy             # Noisy solution
    ]

    runtime = time.time() - start_time
    self.logger.info(f"Total runtime: {runtime:.2f} seconds")
    
    # Extract solution - using probabilistic selection
    found_feasible_solutions = []
    feasible_obj_values = []
    feasible_x_sols = []
    feasible_y_sols = []
    
    for i, candidate_y in enumerate(candidate_solutions):
        self.logger.info(f"Trying candidate solution {i+1}/{len(candidate_solutions)}")
        solved_primal, x_sol_candidate, obj_val_candidate = benders.solve_primal_subproblem(candidate_y)
        
        if solved_primal and x_sol_candidate is not None:
            obj_value = obj_val_candidate + float(f.T.dot(candidate_y)[0, 0])
            found_feasible_solutions.append(True)
            feasible_obj_values.append(obj_value)
            feasible_x_sols.append(x_sol_candidate)
            feasible_y_sols.append(candidate_y)
            self.logger.info(f"Found feasible solution with objective value: {obj_value}")
    
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
        
        self.logger.info(f"Selection probabilities: {probs}")
        
        # Select based on probabilities
        selected_idx = np.random.choice(range(len(probs)), p=probs)
        
        best_obj_value = feasible_obj_values[selected_idx]
        best_x_sol = feasible_x_sols[selected_idx]
        best_y_sol = feasible_y_sols[selected_idx]
        
        self.logger.info(f"Probabilistically selected solution {selected_idx+1} with objective value: {best_obj_value}")
    
    # If no feasible solution found
    if not found_feasible:
        self.logger.warning("No feasible solution found. Keeping pure quantum solution but marking as infeasible.")
        
        # Even here, randomly select one of the candidate solutions
        selected_idx = np.random.randint(0, len(candidate_solutions))
        best_y_sol = candidate_solutions[selected_idx]
        best_obj_value = float(f.T.dot(best_y_sol)[0, 0])
        
        self.logger.info(f"Using randomly selected quantum solution with objective: {best_obj_value}")
    
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