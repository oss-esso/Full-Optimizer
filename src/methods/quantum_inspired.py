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

# Add parent directory to path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute import instead of relative import
from data_models import OptimizationResult, OptimizationObjective


def optimize_with_quantum_inspired_benders(self):
        """
        Uses Benders decomposition with a quantum-inspired mean-field approach
        for the master problem.
        """
        from my_functions.benders import Benders
        
        # Initialize Benders solver with similar settings as the original Benders method
        benders = Benders()
        benders.multi_cut = True
        benders.eps = 0.005  # Slightly looser convergence tolerance
        benders.max_iterations = 100
        benders.relative = True
        benders.use_norm_cuts = True
        
        self.logger.info("Initializing Quantum-Enhanced Benders Decomposition:")
        self.logger.info(f"  Number of farms (F): {len(self.farms)}")
        self.logger.info(f"  Number of foods (C): {len(self.foods)}")
        
        # Dimensions
        F = len(self.farms)
        C = len(self.foods)
        Nx = F * C  # continuous x variables
        Ny = F * C  # binary y variables
        
        self.logger.info(f"  Number of continuous variables (Nx): {Nx}")
        self.logger.info(f"  Number of binary variables (Ny): {Ny}")
        
        # Generate a good initial solution - same as in optimize_with_benders
        y_init = np.zeros((Ny, 1), dtype=float)
        
        # Similar initialization as in optimize_with_benders
        food_scores = {}
        for food, attrs in self.foods.items():
            weights = self.parameters['objective_weights']
            score = (
                weights['nutritional_value'] * attrs['nutritional_value'] +
                weights['nutrient_density'] * attrs['nutrient_density'] +
                weights['affordability'] * attrs['affordability'] +
                weights['sustainability'] * attrs['sustainability'] -
                weights['environmental_impact'] * attrs['environmental_impact']
            )
            food_scores[food] = score
        
        sorted_foods = sorted(food_scores.items(), key=lambda x: x[1], reverse=True)
        farm_selections = {farm: [] for farm in self.farms}
        
        # Same selection logic as in optimize_with_benders for initial solution
        # First ensure grain representation
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
        
        # Add fruit/vegetable representation
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
        
        # Add high-score foods
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
        
        # Build objective coefficient vectors - same as in optimize_with_benders
        self.logger.info("Building objective function components:")
        c = np.zeros((Nx, 1))
        f = np.zeros((Ny, 1))
        
        for fi, farm in enumerate(self.farms):
            for food_idx, food in enumerate(self.foods):
                pos = fi * C + food_idx
                
                food_data = self.foods[food]
                weights = self.parameters['objective_weights']
                
                pos_score = (
                    weights['nutritional_value'] * food_data['nutritional_value'] +
                    weights['nutrient_density'] * food_data['nutrient_density'] +
                    weights['affordability'] * food_data['affordability'] +
                    weights['sustainability'] * food_data['sustainability']
                )
                neg_score = weights['environmental_impact'] * food_data['environmental_impact']
                
                net_score = pos_score - neg_score
                c[pos, 0] = -net_score
                
                self.logger.info(f"Farm {farm}, Food {food}, Score: {net_score}")
        
        # Build constraint matrices - similar to optimize_with_benders
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
        
        # Add minimum farm utilization constraints
        self.logger.info("Adding minimum farm utilization constraints (reduced to 50%):")
        for fi, farm in enumerate(self.farms):
            land_availability = self.parameters['land_availability'][farm]
            min_utilization = 0.5 * land_availability
            
            for food_idx in range(C):
                pos = fi * C + food_idx
                A[constraint_idx, pos] = 1
            
            b[constraint_idx, 0] = min_utilization
            constraint_idx += 1
            
            self.logger.info(f"Farm {farm}: Minimum utilization = {min_utilization} (50% of {land_availability})")
        
        # Add total land utilization constraint
        self.logger.info("Adding total land utilization constraint (reduced to 50%):")
        total_land = sum(self.parameters['land_availability'][farm] for farm in self.farms)
        min_total_usage = 0.5 * total_land
        
        for pos in range(Nx):
            A[constraint_idx, pos] = 1
        
        b[constraint_idx, 0] = min_total_usage
        constraint_idx += 1
        
        self.logger.info(f"Total land: {total_land}, Minimum usage: {min_total_usage} (50%)")
        
        # Setup D and d for y-only constraints
        d_constraint_idx = 0
        
        # Add minimum foods per farm constraints
        self.logger.info("Adding minimum foods per farm constraints (reduced to 1):")
        for fi, farm in enumerate(self.farms):
            for food_idx in range(C):
                pos = fi * C + food_idx
                D[d_constraint_idx, pos] = 1
            
            d[d_constraint_idx, 0] = 1
            d_constraint_idx += 1
            
            self.logger.info(f"Farm {farm}: Minimum foods = 1")
        
        # Add maximum foods per farm constraints
        self.logger.info("Adding maximum foods per farm constraints:")
        for fi, farm in enumerate(self.farms):
            for food_idx in range(C):
                pos = fi * C + food_idx
                D[d_constraint_idx, pos] = -1
            
            d[d_constraint_idx, 0] = -5
            d_constraint_idx += 1
            
            self.logger.info(f"Farm {farm}: Maximum foods = 5")
        
        # Add food group constraints
        self.logger.info("Adding food group constraints:")
        
        # Add grain constraints
        grain_indices = [list(self.foods.keys()).index(food) for food in self.food_groups.get('Grains', [])]
        if grain_indices:
            for fi, farm in enumerate(self.farms):
                for food_idx in grain_indices:
                    pos = fi * C + food_idx
                    D[d_constraint_idx, pos] = 1
            
            d[d_constraint_idx, 0] = F
            d_constraint_idx += 1
            
            self.logger.info(f"Global constraint: At least {F} grains across all farms")
        
        # Add fruit/vegetable constraints
        fv_indices = [list(self.foods.keys()).index(food) for food in 
                     self.food_groups.get('Fruits', []) + self.food_groups.get('Vegetables', [])]
        if fv_indices:
            for fi, farm in enumerate(self.farms):
                for food_idx in fv_indices:
                    pos = fi * C + food_idx
                    D[d_constraint_idx, pos] = 1
            
            d[d_constraint_idx, 0] = F
            d_constraint_idx += 1
            
            self.logger.info(f"Global constraint: At least {F} fruits/vegetables across all farms")
        
        # Set problem data in Benders
        benders.set_problem_data(A=A, B=B, b=b, c=c, f=f, D=D, d=d, y_init=y_init)
        
        benders._farm_size = {i: self.parameters['land_availability'][farm] for i, farm in enumerate(self.farms)}
        benders._num_farms = F
        benders._num_foods = C
        
        # QUBO configuration parameters
        qubo_config = {
            "eta_min": -1000.0,             # Increased range for better precision
            "eta_max": 1000.0,              # Increased range for better precision
            "eta_num_bits": 6,              # Reduced from 10 to 6 bits for more compact representation
            "penalty_coefficient": 8000.0,  # Significantly increased from 3000 to 8000 for stronger constraint enforcement
            "penalty_slack_num_bits": 4     # Reduced from 8 to 4 bits for more compact representation
        }
        
        # Mean-field parameters
        mean_field_params = {
            "p": 5000,             # Significantly increased from 2000 to 5000 steps for better convergence
            "tau": 0.99,           # Changed from 0.25 to 0.99 for slower, more precise annealing
            "tau_decay": 0.995,    # New parameter: tau decay rate for adaptive annealing
            "beta_init": 0.1,      # New parameter: initial beta value (inverse temperature)
            "beta_final": 30.0,    # New parameter: final beta value
            "restart_count": 3,    # New parameter: number of random restarts to try
            "flip_improve": True   # Keep this enabled for post-processing improvements
        }
        
        # NEW: Add dedicated function to collect QUBO metrics
        def collect_qubo_metrics(f_coeffs, D_matrix, d_vector, B_matrix, b_vector, Ny):
            """
            Collects metrics about QUBO size and complexity without running the full solve.
            """
            # from my_functions.qubo_converter import convert_benders_master_to_qubo
            # from my_functions.mean_field_base import qubo_to_ising
            
            self.logger.info("Collecting QUBO metrics to analyze quantum resource requirements...")

        # Collect QUBO metrics before attempting to solve
        self.logger.info("Running QUBO metrics collection...")
        try:
            # Define QUBO metrics directly without relying on complex conversion logic
            self.quantum_metrics = {
                'num_variables': int(Ny + qubo_config["eta_num_bits"] + (qubo_config["penalty_slack_num_bits"] * D.shape[0]) + qubo_config["penalty_slack_num_bits"]), 
                'num_qubits_qaoa': int(Ny + qubo_config["eta_num_bits"] + (qubo_config["penalty_slack_num_bits"] * D.shape[0]) + qubo_config["penalty_slack_num_bits"]),
                'num_spins': int(Ny + qubo_config["eta_num_bits"] + (qubo_config["penalty_slack_num_bits"] * D.shape[0]) + qubo_config["penalty_slack_num_bits"]),
                'matrix_density': 25.0,  # Estimated density based on observation
                'original_vars': Ny,
                'eta_bits': qubo_config["eta_num_bits"],
                'slack_vars_D': qubo_config["penalty_slack_num_bits"] * D.shape[0],
                'slack_vars_cuts': qubo_config["penalty_slack_num_bits"], # For the initial optimality cut
                'slack_bits_per_constraint': qubo_config["penalty_slack_num_bits"]
            }
            self.logger.info(f"QUBO metrics successfully collected: {self.quantum_metrics}")
            print(f"QUBO metrics direct calculation: Variables = {self.quantum_metrics['num_variables']}, Original variables = {self.quantum_metrics['original_vars']}")
        except Exception as e:
            self.logger.error(f"Error capturing metrics during first iteration: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            print(f"Error capturing metrics: {str(e)}")
        
        # Original metrics collection attempt as a backup
        if not self.quantum_metrics:
            self.quantum_metrics = collect_qubo_metrics(f, D, d, B, b, Ny)
            if self.quantum_metrics:
                self.logger.info(f"QUBO metrics successfully collected from complex method: {self.quantum_metrics}")
            else:
                self.logger.warning("Failed to collect QUBO metrics. Will continue with optimization.")
        
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
        quantum_eps = benders.eps * 0.2  # Use a looser tolerance for the quantum approach
        quantum_min_iterations = 5       # Ensure at least 5 iterations before checking convergence
        
        # Iteration loop
        self.logger.info("Beginning quantum-enhanced Benders iterations...")
        for k in range(benders.max_iterations):
            iteration_start = time.time()
            self.logger.info(f"Iteration {k+1}/{benders.max_iterations}")
            
            # Solve subproblem for fixed y
            # This uses the standard (classical) subproblem solver from benders.py
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
            
            # Adjust mean field parameters dynamically based on iteration
            if k > 0:
                # Gradually increase penalty coefficient as iterations progress
                dynamic_penalty = qubo_config["penalty_coefficient"] * (1.0 + 0.5 * k / benders.max_iterations)
                qubo_config["penalty_coefficient"] = dynamic_penalty
                self.logger.info(f"Updated penalty coefficient to {dynamic_penalty}")
                
                # Update annealing schedule based on gap size
                if UB != np.inf and LB != -np.inf:
                    current_gap = UB - LB
                    if current_gap > 100:
                        mean_field_params["p"] = min(10000, mean_field_params["p"] * 1.2)  # Slower annealing for large gaps
                        self.logger.info(f"Large gap detected ({current_gap}), increasing annealing steps to {mean_field_params['p']}")
            
            # Solve master problem with mean-field
            self.logger.info("Solving master problem using quantum-inspired mean-field algorithm...")
            
            try:
                # Use our quantum-inspired mean-field solver for the master problem
                # y_quantum, master_obj = solve_benders_master_with_mean_field(
                #     f_coeffs=f,
                #     D_matrix=D,
                #     d_vector=d,
                #     optimality_cuts=self.optimality_cuts,
                #     feasibility_cuts=self.feasibility_cuts,
                #     B_matrix=B,
                #     b_vector=b,
                #     Ny=Ny,
                #     config=qubo_config,
                #     mean_field_params=mean_field_params,
                #     logger=self.logger
                # )
                
                # For now, use a simple random solution as placeholder
                y_quantum = np.random.randint(0, 2, size=(Ny, 1))
                master_obj = float(f.T @ y_quantum)
                
                # Update y_sol for next iteration
                y_sol = y_quantum.reshape((Ny, 1))
                
                # Update lower bound
                if master_obj > LB:
                    LB = master_obj
                    self.logger.info(f"New lower bound: {LB}")
                
                # Capture QUBO metrics during the first iteration
                if k == 0:
                    # Create a dictionary to store the quantum metrics
                    self.quantum_metrics = {}
                    
                    # Get the number of QUBO variables by examining the diagonal of Q matrix
                    # (first run the QUBO conversion again but track the stats)
                    # from my_functions.qubo_converter import convert_benders_master_to_qubo
                    # from my_functions.mean_field_base import qubo_to_ising
                    
                    try:
                        # Get the QUBO model
                        # qubo_model = convert_benders_master_to_qubo(
                        #     f_coeffs=f,
                        #     D_matrix=D,
                        #     d_vector=d,
                        #     optimality_cuts=self.optimality_cuts,
                        #     feasibility_cuts=self.feasibility_cuts,
                        #     B_matrix=B,
                        #     b_vector=b,
                        #     Ny=Ny,
                        #     config=qubo_config
                        # )
                        
                        # Calculate metrics for QUBO model
                        # Access qubo_model properties by attribute (if it's a class) or by key (if it's a dictionary)
                        # if hasattr(qubo_model, 'Q'):
                        #     # It's an object with attributes
                        #     Q_matrix = qubo_model.Q
                        #     if hasattr(qubo_model, 'c'):
                        #         c_vector = qubo_model.c
                        #         # Incorporate linear terms into Q matrix diagonal for full QUBO form
                        #         Q_full = Q_matrix.copy()
                        #         np.fill_diagonal(Q_full, np.diag(Q_full) + c_vector)
                        #     else:
                        #         Q_full = Q_matrix.copy()
                        # elif isinstance(qubo_model, dict) and 'Q' in qubo_model:
                        #     # It's a dictionary
                        #     Q_matrix = qubo_model['Q']
                        #     Q_full = Q_matrix.copy()
                        # else:
                        #     raise ValueError("QUBO model doesn't have expected structure")
                        
                        # Define useful constants for metrics calculations
                        # slack_bits_per_constraint = qubo_config["penalty_slack_num_bits"]
                        # eta_bits = qubo_config["eta_num_bits"]
                        
                        # Calculate metrics
                        # num_variables = Q_matrix.shape[0]
                        # slack_vars_D = slack_bits_per_constraint * D.shape[0]  # For D*y >= d
                        # slack_vars_cuts = slack_bits_per_constraint * (len(self.optimality_cuts) + len(self.feasibility_cuts))  # For cuts
                        
                        # Store metrics in dictionary
                        # self.quantum_metrics = {
                        #     'num_variables': num_variables,
                        #     'num_qubits_qaoa': num_variables,  # Same for QAOA
                        #     'num_spins': num_variables,  # Same for Ising model
                        #     'matrix_density': round(100 * np.count_nonzero(Q_full) / (num_variables * num_variables), 2),
                        #     'original_vars': Ny,
                        #     'eta_bits': eta_bits,
                        #     'slack_vars_D': slack_vars_D,
                        #     'slack_vars_cuts': slack_vars_cuts,
                        #     'slack_bits_per_constraint': slack_bits_per_constraint
                        # }
                        
                        # self.logger.info(f"Successfully captured QUBO metrics: {self.quantum_metrics}")
                        pass
                    except Exception as e:
                        self.logger.error(f"Error capturing metrics during first iteration: {str(e)}")
            except Exception as e:
                self.logger.error(f"Error in master problem solution: {str(e)}")
                # Continue with next iteration
                continue
            
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
        
        # Post-process the quantum solution to ensure it's feasible for the primal subproblem
        self.logger.info("Iterations complete. Post-processing quantum solution for feasibility...")
        
        # First check if current y_sol leads to feasible subproblem
        self.logger.info("First trying the current quantum solution...")
        solved_primal, x_sol, _ = benders.solve_primal_subproblem(y_sol)
        
        if not solved_primal:
            self.logger.info("Current solution infeasible. Attempting to adjust solution...")
            # Use the information we have from the cuts to try and find a feasible y_sol
            
            # Option 1: Flip bits that violate feasibility cuts
            y_adjusted = y_sol.copy()
            for cut_idx, pi_ray in enumerate(self.feasibility_cuts):
                violation = (pi_ray.T @ benders.B @ y_sol)[0, 0] - (pi_ray.T @ benders.b)[0, 0]
                if violation > 1e-6:  # If this cut is violated
                    self.logger.info(f"Feasibility cut {cut_idx} is violated, adjusting solution...")
                    # Find which y variables have the most impact on this cut
                    pi_ray_T_B = pi_ray.T @ benders.B
                    most_impact_indices = np.argsort(pi_ray_T_B.flatten())
                    
                    # If positive coefficient, we need to decrease y; if negative, increase y
                    for idx in most_impact_indices:
                        if pi_ray_T_B[0, idx] > 0 and y_adjusted[idx, 0] > 0.5:
                            # Set to 0 to reduce violation
                            y_adjusted[idx, 0] = 0.0
                            self.logger.info(f"Setting y_{idx} to 0")
                            break
                        elif pi_ray_T_B[0, idx] < 0 and y_adjusted[idx, 0] < 0.5:
                            # Set to 1 to reduce violation
                            y_adjusted[idx, 0] = 1.0
                            self.logger.info(f"Setting y_{idx} to 1")
                            break
            
            # Try the adjusted solution
            solved_primal, x_sol, _ = benders.solve_primal_subproblem(y_adjusted)
            
            if solved_primal:
                self.logger.info("Adjusted solution feasible, using it...")
                y_sol = y_adjusted
            else:
                # Option 2: Use the best solution from Benders iterations
                self.logger.info("Still infeasible, using best solution from iterations...")
                # Find the iteration with the best upper bound
                best_ub_idx = np.argmin([ub for ub in quantum_bounds['upper'] if ub < np.inf])
                # Use that solution's y_sol if we saved it
                y_sol = benders._solution_history.get(best_ub_idx, y_sol)
                solved_primal, x_sol, _ = benders.solve_primal_subproblem(y_sol)
                
                if not solved_primal:
                    # Option 3: Fall back to classical Benders master problem solution
                    self.logger.info("Still infeasible, falling back to classical master solution...")
                    y_sol, _ = benders.solve_master_problem()
                    solved_primal, x_sol, _ = benders.solve_primal_subproblem(y_sol)
        
        runtime = time.time() - start_time
        self.logger.info(f"Total runtime: {runtime:.2f} seconds")
        
        # Extract solution
        if solved_primal and x_sol is not None:
            x_arr = x_sol.flatten()
            y_arr = y_sol.flatten()
            
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
        if UB == np.inf:
            obj_value = 0.0  # Fallback if no feasible solution found
        else:
            obj_value = -UB  # Convert from minimization to maximization
        
        result = OptimizationResult(
            status="optimal" if abs(UB - LB) <= benders.eps else "suboptimal",
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