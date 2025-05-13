"""
Solves MILPs using the Benders decomposition method.

Author: Edoardo Spigarolo
Date: Spring 2025
"""

import cvxpy as cp
import numpy as np
import sys
import time  # for timing subproblem/master solves
import pulp  # use PuLP for Master problem
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray
import logging


class Benders:
    """
    Solves MILPs of the following form using Benders decomposition:

    min c_T*x + f_T*y     over x,y
    s.t   A*x + B*y >= b (m constraints)
          D*y >= d       (n constraints)
          x >= 0         (Nx-dimensional vector)
          y >= 0         (Ny-dimensional vector)
          y integer
    """
    def __init__(self):
        # Data - will be set by the caller
        self.Nx = None  # Size of vector x
        self.Ny = None  # Size of vector y
        self.m = None   # Number of complicating constraints A*x + B*y >= b
        self.n = None   # Number of constraints D*y >= d
        self.c = None   # Linear objective coefficients for x
        self.f = None   # Linear objective coefficients for y
        self.A = None   # Constraint matrix for x
        self.B = None   # Constraint matrix for y
        self.b = None   # Right-hand side for complicating constraints
        self.D = None   # Constraint matrix for y-only constraints
        self.d = None   # Right-hand side for y-only constraints

        self.y_init = None  # Initial feasible guess (will be set by caller)

        self.eps = 1e-3                                    # Convergence value
        self.max_iterations = 20                           # Number of maximum iterations
        self.LB = -np.inf  # Lower bound of objective function
        self.UB = np.inf   # Upper bound of objective function
        self.optimality_cuts = []
        self.feasibility_cuts = []
        self.optimality_cut_counts = []      # Track number of optimality cuts per iteration
        self.feasibility_cut_counts = []     # Track number of feasibility cuts per iteration
        # Enable multi-cut and timing instrumentation
        self.multi_cut = False
        self.subproblem_times = []
        self.master_times = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.relative = False  # Whether to use relative gap criterion
        
        # Anti-cycling features
        self.use_norm_cuts = False  # Use normalized optimality cuts
        self.use_trust_region = False  # Use trust region to stabilize convergence
        self.best_y_sol = None  # Best y solution found so far
        self.best_x_sol = None  # Best x solution found so far
        self.best_obj = np.inf  # Best objective value found so far
        self.converged = False # Track convergence status
        self.gap = np.inf      # Track final gap

    def set_problem_data(self, A, B, b, c, f, D, d, y_init=None):
        """Set the problem data."""
        self.A = np.array(A)
        self.B = np.array(B)
        self.b = np.array(b).reshape((-1, 1))
        self.c = np.array(c).reshape((-1, 1))
        self.f = np.array(f).reshape((-1, 1))
        self.D = np.array(D)
        self.d = np.array(d).reshape((-1, 1))
        
        # Set dimensions
        self.Nx = self.A.shape[1]
        self.Ny = self.B.shape[1]
        self.m = self.A.shape[0]
        self.n = self.D.shape[0]
        
        # Set initial guess
        if y_init is not None:
            self.y_init = np.array(y_init, dtype=int).reshape((self.Ny, 1))
        else:
            # Default to all zeros
            self.y_init = np.zeros((self.Ny, 1), dtype=int)

    def solve_problem(self):
        """Run Benders decomposition: CVXPY subproblem + PuLP master solves."""
        if any(x is None for x in [self.Nx, self.Ny, self.m, self.n, self.c, self.f, self.A, self.B, self.b, self.D, self.d, self.y_init]):
            raise ValueError("Problem data not set. Call set_problem_data() first.")
            
        # Reset cuts and histories
        self.optimality_cuts = []
        self.feasibility_cuts = []
        self.optimality_cut_counts = []
        self.feasibility_cut_counts = []
        self.subproblem_times = []
        self.master_times = []
        self.lower_bounds = []
        self.upper_bounds = []
        self._previous_solutions = []
        
        # Initialize solution and bounds
        y_sol = self.y_init.copy() # Start with the initial guess
        self.LB = -np.inf
        self.UB = np.inf
        self.best_y_sol = None # Track best y found across iterations
        self.best_x_sol = None  # Track corresponding x for best y
        self.best_obj = np.inf  # Track best objective (minimization) found
        self.converged = False # Reset convergence status
        self.gap = np.inf      # Reset gap
        
        # Prepare parameters for multi-stage approach
        # Start with looser convergence criteria and tighten over iterations
        starting_eps = self.eps * 5
        current_eps = starting_eps
        eps_reduction_factor = 0.5
        
        # Minimum iterations (must run at least this many even if converged)
        min_iterations = min(5, self.max_iterations)
        
        # For reporting
        best_obj = np.inf
        best_y = None
        best_x = None
        last_improvement_iter = 0
        
        # Set trust region parameters if enabled
        trust_radius = 0.5 * self.Ny if self.use_trust_region else None
        
        # Iteration starts
        print("Starting Benders decomposition iterations:")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Convergence tolerance: {self.eps}")
        print(f"Minimum iterations: {min_iterations}")
        print(f"Using trust region: {self.use_trust_region}")
        print(f"Using normalized cuts: {self.use_norm_cuts}")
        
        for k in range(self.max_iterations):
            k_str = f"Iteration {k+1}/{self.max_iterations}"
            print(f"\n{k_str}")
            print("-" * len(k_str))
            
            # Add current y_sol to previous solutions to prevent cycling
            if y_sol is not None and np.sum(y_sol) > 0:
                self._previous_solutions.append(y_sol.copy())
            
            # Solve subproblem for fixed CURRENT y_sol
            t_sub_start = time.time()
            solved, dual_vars, obj_val_sub, extreme_ray = self.solve_subproblem(y_sol)
            t_sub = time.time() - t_sub_start
            self.subproblem_times.append(t_sub)
            
            # Update upper bound (UB) if feasible
            if solved:
                # Calculate objective of original problem if y_sol were used
                current_total_obj_min = obj_val_sub + float(self.f.T.dot(y_sol)) 
                print(f"Subproblem solved. Subproblem obj (max pT(b-By)): {obj_val_sub}, Full obj for current y (min cTx+fTy): {current_total_obj_min}")
                
                # Update UB (upper bound on the MINIMIZATION problem)
                if current_total_obj_min < self.UB:
                    self.UB = current_total_obj_min
                    
                # Track overall best y solution found SO FAR based on feasible subproblems
                if current_total_obj_min < self.best_obj:
                    self.best_obj = current_total_obj_min
                    self.best_y_sol = y_sol.copy() # Store the y that gave the best objective
                    # Note: We don't have the corresponding x here, only the dual vars
                    print(f"New best feasible solution objective found (minimization): {self.best_obj}")
                
                # Generate optimality cut using dual variables
                pi_star = dual_vars 
                # Normalize cut if requested
                if self.use_norm_cuts and np.linalg.norm(pi_star) > 1e-10:
                    pi_star = pi_star / np.linalg.norm(pi_star)
                    
                self.optimality_cuts.append(pi_star)
                self.optimality_cut_counts.append(1)
                self.feasibility_cut_counts.append(0)
                print(f"Generated 1 optimality cut")
                
            else:
                # Generate feasibility cut using extreme ray
                self.feasibility_cuts.append(extreme_ray)
                self.optimality_cut_counts.append(0)
                self.feasibility_cut_counts.append(1)
                print(f"Subproblem infeasible. Generated 1 feasibility cut")
            
            # Solve master problem, with trust region if enabled
            t_master_start = time.time()
            # Trust region should center around the y that yielded the best objective found so far
            trust_center_y = self.best_y_sol if self.use_trust_region and self.best_y_sol is not None else None 
                
            # Pass the CURRENT y_sol (from previous master solve) to master problem features if needed (e.g. trust region ADJUSTMENT)
            # but the trust region CENTER should be based on best_y_sol
            y_sol_master, lb = self.solve_master_problem(
                excluded_solutions=self._previous_solutions[-10:] if len(self._previous_solutions) > 10 else self._previous_solutions,
                trust_center=trust_center_y, # Center on best known y
                trust_radius=trust_radius
            )
                
            # Adjust trust radius based on progress (using obj_val_sub, which is from current y_sol)
            if trust_center_y is not None:
                if solved and current_total_obj_min < self.best_obj: # Check if current y improved best obj
                    trust_radius = min(trust_radius * 1.5, self.Ny) # Expand
                else:
                    trust_radius = max(trust_radius * 0.5, 3) # Shrink
                
            t_master = time.time() - t_master_start
            self.master_times.append(t_master)
            
            # Update the current y solution for the next iteration
            y_sol = y_sol_master 
            
            # Update lower bound from master problem objective
            if lb > self.LB:
                self.LB = lb
            
            # Store bounds history
            self.lower_bounds.append(self.LB)
            self.upper_bounds.append(self.UB)
            
            # Print current bounds (for the minimization problem)
            print(f"Current bounds (minimization): LB = {self.LB}, UB = {self.UB}")
            
            # Gap computation (for the minimization problem)
            gap = np.inf
            if self.UB < np.inf and self.LB > -np.inf:
                if self.relative:
                    if abs(self.UB) > 1e-10: 
                        gap = abs((self.UB - self.LB) / (abs(self.UB) + 1e-10))
                        print(f"Relative gap: {gap*100:.4f}%")
                    else:
                        print("Cannot compute relative gap: UB too close to zero")
                else:
                    gap = abs(self.UB - self.LB)
                    print(f"Absolute gap: {gap}")
            
            self.gap = gap # Store current gap
            
            # Check convergence
            if k >= min_iterations - 1 and gap <= current_eps:
                if current_eps > self.eps:
                    current_eps *= eps_reduction_factor
                    print(f"Reducing epsilon to {current_eps}")
                else:
                    print(f"Converged: gap = {gap} < eps = {self.eps}")
                    self.converged = True 
                    break
            
            # Check stagnation
            stagnation_limit = min(10, self.max_iterations // 5)
            # Need to track iteration of last UB improvement
            if solved and current_total_obj_min < self.UB: # Need a way to track last UB improvement iter
                 # last_improvement_iter = k # Reinstate this logic if needed
                 pass 
            # if k - last_improvement_iter > stagnation_limit and k >= min_iterations - 1:
            #     print(f"Warning: No improvement in {stagnation_limit} iterations. Using best solution found.")
            #     y_sol = self.best_y_sol.copy() # Use best y found
            #     break
            
            # Track solution
            print(f"Current solution y: sum(y) = {np.sum(y_sol)}, y[y>0.5] indices: {np.where(y_sol > 0.5)[0]}")
            
            # Track timing
            print(f"Times: Subproblem = {t_sub:.2f}s, Master = {t_master:.2f}s")
        
        # Use the best y solution found during iterations for final primal solve
        final_y = self.best_y_sol if self.best_y_sol is not None else self.y_init
            
        # Solve subproblem one more time with best y to get best x (dual variables and obj_val for original_obj)
        # This dual solve is primarily to ensure UB is consistent if best_y changed.
        solved_dual, p_val_final, obj_val_dual_final, _ = self.solve_subproblem(final_y)

        # NOW, solve the PRIMAL subproblem to get the actual x values for the best y
        solved_primal, x_primal_final, obj_val_primal_final = self.solve_primal_subproblem(final_y)

        best_x_to_store = np.zeros((self.Nx, 1)) # Default fallback
        final_obj_for_minimization = np.inf

        if solved_primal and x_primal_final is not None:
            best_x_to_store = x_primal_final
            # The objective value for the original minimization problem (cTx + fTy)
            final_obj_for_minimization = obj_val_primal_final + float(self.f.T.dot(final_y))
            print(f"Final primal subproblem solved. Primal Obj (cTx): {obj_val_primal_final}, fTy: {float(self.f.T.dot(final_y))}, Total Minimized Obj: {final_obj_for_minimization}")
        elif solved_dual:
            print("Warning: Final primal subproblem solve failed! Using dual subproblem objective for bounds, but x values are a fallback.")
            final_obj_for_minimization = obj_val_dual_final + float(self.f.T.dot(final_y))
        else:
            print("Warning: Both final primal and dual subproblem solves failed or y was None. Using fallback values.")
            if final_y is not None: 
                 final_obj_for_minimization = self.best_obj if self.best_obj != np.inf else np.inf

        # Update overall objective value for the class (minimization form)
        self.obj_val = final_obj_for_minimization
        
        # Final UB should be consistent with the final solution found
        self.UB = min(self.UB, self.obj_val) # Ensure UB reflects the best feasible solution found

        print(f"\nBenders decomposition completed.")
        print(f"Final bounds (minimization): LB = {self.LB}, UB = {self.UB}")
        print(f"Final objective value (for minimization, cTx+fTy): {self.obj_val}") 
        # Iterations completed is k+1 if loop finished, or k if break/stagnation
        iterations_completed = k + 1 if k < self.max_iterations -1 and not self.converged else self.max_iterations # Approximation
        print(f"Iterations: {iterations_completed} (out of maximum {self.max_iterations})")
        print(f"Total optimality cuts: {sum(self.optimality_cut_counts)}")
        print(f"Total feasibility cuts: {sum(self.feasibility_cut_counts)}")
        
        # Store and return solution
        self.x_sol = best_x_to_store # Primal X for best Y found
        self.y_sol = final_y       # Best Y found
        
        return self.x_sol, self.y_sol, self.obj_val # obj_val is the one for minimization cTx + fTy

    def show_results(self, i, obj_value_master, x_sol, y_sol):
        """Displays the results of the optimization problem.

        Inputs:
          - i(int):                  The index of last iteration
          - obj_value_master(float): The objective function value of the master problem
          - x_sol(np.ndarray):       The optimal x solution [Nx,1]
          - y_sol(np.ndarray):       The optimal y solution [Ny,1]
        """

        if i > self.max_iterations:
            print("\nThe algorithm did not converge in the given iterations.")
        else:
            print("\n*** Optimal solution to the MILP problem found. ***")
        print("The optimal value is: {}".format(obj_value_master))

        if x_sol is not None:
            print("The optimal solution is x*={}, y*={}".format(*x_sol, *y_sol))
        else:
            print("\nThe algorithm did not find the optimal solution. Please try another initial feasible guess y_init!")

    def solve_subproblem(self, y):
        """Solves the primal-dual subproblem for fixed y."""
        # Convert y to flattened form for easier handling
        y_flat = y.flatten()
        
        # Setup and solve dual subproblem (p^T (b - By))
        p = cp.Variable((self.m, 1))  # dual variable
        
        # Right-hand side: b - B*y
        rhs = self.b - self.B @ y
        
        # Objective: maximize p^T(b - By) s.t. p^T A <= c^T
        obj = cp.Maximize(cp.sum(cp.multiply(rhs, p)))
        
        # Dual constraints: p^T A <= c^T
        constraints = [p >= 0]
        for j in range(self.Nx):
            a_j = self.A[:, j].reshape(-1, 1)
            c_j = self.c[j, 0]
            constraints.append(cp.sum(cp.multiply(a_j, p)) <= c_j)
        
        # Build and solve problem
        prob = cp.Problem(obj, constraints)
        try:
            sub_obj = prob.solve(solver=cp.ECOS, verbose=False)
            if p.value is None:
                # Try with another solver
                print("ECOS failed, trying SCS...")
                sub_obj = prob.solve(solver=cp.SCS, verbose=False)
            
            # Check solved status
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # Problem solved successfully
                return True, p.value, sub_obj, None
            elif prob.status in [cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE]:
                # Extract extreme ray 
                # Since we can't easily get the extreme ray directly from CVXPY,
                # we can solve a modified problem to find a ray
                extreme_ray = self.find_extreme_ray(y_flat)
                return False, None, np.inf, extreme_ray
            else:
                # Other failure modes
                print(f"Subproblem failed with status: {prob.status}")
                # Fall back to solving the feasibility problem
                extreme_ray = self.find_extreme_ray(y_flat)
                return False, None, np.inf, extreme_ray
                
        except cp.error.SolverError:
            print("Solver error in subproblem. Trying fallback solver.")
            try:
                # Try with a different solver
                sub_obj = prob.solve(solver=cp.SCS, verbose=False)
                if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    return True, p.value, sub_obj, None
                else:
                    print(f"Fallback solver also failed with status: {prob.status}")
                    extreme_ray = self.find_extreme_ray(y_flat)
                    return False, None, np.inf, extreme_ray
            except:
                print("All solvers failed. Using feasibility subproblem.")
                extreme_ray = self.find_extreme_ray(y_flat)
                return False, None, np.inf, extreme_ray
    
    def find_extreme_ray(self, y_flat):
        """Find an extreme ray when the dual subproblem is unbounded."""
        # Setup feasibility problem to find extreme ray
        p = cp.Variable((self.m, 1))
        
        # Right-hand side: b - B*y
        rhs = self.b - self.B @ y_flat.reshape(-1, 1)
        
        # Create a normalization constraint to bound the problem
        norm_constraint = cp.sum(p) == 1
        
        # Dual constraints from original problem
        constraints = [p >= 0, norm_constraint]
        for j in range(self.Nx):
            a_j = self.A[:, j].reshape(-1, 1)
            c_j = self.c[j, 0]
            constraints.append(cp.sum(cp.multiply(a_j, p)) <= c_j)
        
        # Objective: maximize p^T(b - By)
        # If this is positive, we've found a ray that proves infeasibility
        obj = cp.Maximize(cp.sum(cp.multiply(rhs, p)))
        
        # Build and solve
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            if p.value is not None and prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return p.value
            else:
                # Fallback to SCS
                prob.solve(solver=cp.SCS, verbose=False)
                if p.value is not None:
                    return p.value
                else:
                    # Last resort - just return a simple ray pointing in the direction of the most violated constraint
                    ray = np.zeros((self.m, 1))
                    # Find the most violated constraint in rhs
                    most_violated_idx = np.argmin(rhs)
                    ray[most_violated_idx] = 1.0
                    return ray
        except:
            # Even feasibility problem failed, create a basic ray
            ray = np.zeros((self.m, 1))
            # Identify the most violated constraint
            most_violated_idx = np.argmin(self.b - self.B @ y_flat.reshape(-1, 1))
            ray[most_violated_idx] = 1.0
            return ray

    def solve_modified_subproblem(self, y):
        """Solves the modified sub-problem in the dual form to find the extreme ray."""
        p = cp.Variable((self.m, 1))
        constraints = [p.T@(self.b-self.B@y) == 1, p.T@self.A <= np.zeros((1, self.Nx)), p >= np.zeros((self.m, 1))]
        prob = cp.Problem(cp.Maximize(0), constraints)
        # Use ECOS for reliability
        prob.solve(solver=cp.ECOS, verbose=False)
        return p.value if prob.status.lower() == 'optimal' else None

    def solve_master_problem(self, optimality_cuts=None, feasibility_cuts=None, excluded_solutions=None, log_level=logging.INFO, trust_center=None, trust_radius=None):
        """
        Solve the master problem with the current cuts.
        
        Args:
            optimality_cuts: List of optimality cuts to add
            feasibility_cuts: List of feasibility cuts to add
            excluded_solutions: List of solutions to exclude
            log_level: Logging level
            trust_center: Center point for trust region constraint (y_k)
            trust_radius: Radius of trust region (max Hamming distance)
        
        Returns:
            Tuple (solution, objective_value)
        """
        if optimality_cuts is None:
            optimality_cuts = self.optimality_cuts
        if feasibility_cuts is None:
            feasibility_cuts = self.feasibility_cuts
        if excluded_solutions is None:
            excluded_solutions = []
        
        # Create a PuLP problem
        prob = pulp.LpProblem("Master_Problem", pulp.LpMinimize)
        
        # Create variables
        y_vars = [pulp.LpVariable(f"y_{i}", cat='Binary') for i in range(self.Ny)]
        eta = pulp.LpVariable("eta", lowBound=-1e10, upBound=1e10, cat='Continuous')
        
        # Add objective function (minimize negative of original objective)
        obj_expr = eta
        if self.f is not None:
            # Add f^T y term if it exists
            for i in range(self.Ny):
                obj_expr += self.f[i, 0] * y_vars[i]
        
        prob += obj_expr, "Objective_Function"
        
        # Add y-only constraints: D*y >= d
        for i in range(self.n):
            constraint_expr = 0
            for j in range(self.Ny):
                constraint_expr += self.D[i, j] * y_vars[j]
            
            # Add the constraint: D*y >= d
            prob += constraint_expr >= self.d[i, 0], f"Y_Constraint_{i}"
        
        # Add optimality cuts: eta >= c^T x* + (b - B*y)^T pi*
        for i, cut in enumerate(optimality_cuts):
            pi = cut  # dual solution
            b_By_pi = 0
            
            # Calculate b^T pi
            b_pi = float(self.b.T @ pi)
            
            # Calculate (B*y)^T pi term
            for j in range(self.Ny):
                b_By_pi -= float(pi.T @ self.B[:, j].reshape(-1, 1)) * y_vars[j]
            
            # Add the optimality cut: eta >= b^T pi - (B*y)^T pi
            prob += eta >= b_pi + b_By_pi, f"Optimality_Cut_{i}"
        
        # Add feasibility cuts: 0 >= (b - B*y)^T pi
        for i, cut in enumerate(feasibility_cuts):
            pi = cut  # dual ray
            b_By_pi = 0
            
            # Calculate b^T pi
            b_pi = float(self.b.T @ pi)
            
            # Calculate (B*y)^T pi term
            for j in range(self.Ny):
                b_By_pi -= float(pi.T @ self.B[:, j].reshape(-1, 1)) * y_vars[j]
            
            # Add the feasibility cut: 0 >= b^T pi - (B*y)^T pi
            prob += 0 >= b_pi + b_By_pi, f"Feasibility_Cut_{i}"
        
        # Add trust region constraint if requested
        if trust_center is not None and trust_radius is not None:
            # Create a Hamming distance constraint: sum(y_i != trust_center_i) <= trust_radius
            hamming_dist = 0
            for i in range(self.Ny):
                if trust_center[i, 0] > 0.5:  # trust_center[i] = 1
                    hamming_dist += (1 - y_vars[i])  # counts if y_i = 0
                else:  # trust_center[i] = 0
                    hamming_dist += y_vars[i]  # counts if y_i = 1
            
            prob += hamming_dist <= trust_radius, "Trust_Region"
            print(f"Added trust region constraint with radius {trust_radius}")
        
        # Anti-cycling constraints: prevent previous solutions from reoccurring
        if excluded_solutions:
            for ex_idx, excluded_y in enumerate(excluded_solutions):
                ex_y_flat = excluded_y.flatten()
                
                # Create a constraint that guarantees the new solution differs from the excluded one
                # by at least one binary variable
                hamming_dist = 0
                for i in range(self.Ny):
                    if ex_y_flat[i] > 0.5:  # excluded_y[i] = 1
                        hamming_dist += (1 - y_vars[i])  # counts if y_i = 0
                    else:  # excluded_y[i] = 0
                        hamming_dist += y_vars[i]  # counts if y_i = 1
                
                prob += hamming_dist >= 1, f"Anti_Cycling_{ex_idx}"
            
            print(f"Added {len(excluded_solutions)} anti-cycling constraints")
        
        # Extract F (number of farms) and C (number of foods) to add farming-specific constraints
        # Only if we have farm-specific information
        if hasattr(self, '_farm_size') and hasattr(self, '_num_farms') and hasattr(self, '_num_foods'):
            num_farms = self._num_farms
            num_foods = self._num_foods
            
            # Add farm-specific constraints
            for farm in range(num_farms):
                # Count the number of foods used in this farm
                farm_foods_used = pulp.lpSum([y_vars[farm * num_foods + f] for f in range(num_foods)])
                
                # Minimum and maximum number of foods per farm
                prob += farm_foods_used >= 1, f"Min_Foods_Farm_{farm}"  # At least 1 food
                prob += farm_foods_used <= 4, f"Max_Foods_Farm_{farm}"  # At most 4 foods
        
        # Solve master problem
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
        prob.solve(solver)
        
        # Check status
        status = pulp.LpStatus[prob.status]
        if status != 'Optimal':
            print(f"Warning: Master problem not solved to optimality. Status: {status}")
            if status == 'Infeasible':
                print("Master problem is infeasible. This might be due to conflicts between constraints.")
                
                # If the problem is infeasible, try relaxing anti-cycling constraints
                if excluded_solutions:
                    print("Retrying without anti-cycling constraints...")
                    return self.solve_master_problem(
                        optimality_cuts=optimality_cuts,
                        feasibility_cuts=feasibility_cuts,
                        excluded_solutions=None,  # Remove excluded solutions constraints
                        log_level=log_level,
                        trust_center=None,  # Remove trust region
                        trust_radius=None
                    )
        
        # Extract solution
        y_sol = np.zeros((self.Ny, 1))
        for i, var in enumerate(y_vars):
            y_sol[i, 0] = var.value() if var.value() is not None else 0
        
        # Extract lower bound
        lb_val = pulp.value(prob.objective) if prob.status == pulp.LpStatusOptimal else self.LB
        
        # Print solution info
        y_active = np.where(y_sol > 0.5)[0]
        print(f"Master problem solution: sum(y) = {np.sum(y_sol)}, activated y indices: {y_active}")
        
        return y_sol, lb_val

    def solve_primal_subproblem(self, y):
        """Solves the primal subproblem for fixed y to get x values."""
        if y is None:
            print("Warning: y is None in solve_primal_subproblem. Cannot solve.")
            return False, np.zeros((self.Nx, 1)), np.inf
            
        x_primal = cp.Variable((self.Nx, 1))
        
        # Objective: minimize c^T x
        obj = cp.Minimize(self.c.T @ x_primal)
        
        # Constraints: A*x >= b - B*y
        #              x >= 0
        constraints = [self.A @ x_primal >= (self.b - self.B @ y),
                       x_primal >= 0]
        
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return True, x_primal.value, prob.value
            else:
                # Try SCS if ECOS fails
                print(f"Primal subproblem failed with ECOS (status: {prob.status}), trying SCS...")
                prob.solve(solver=cp.SCS, verbose=False)
                if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    return True, x_primal.value, prob.value
                else:
                    print(f"Primal subproblem failed with SCS also (status: {prob.status}) for y: {y.flatten()}")
                    return False, None, np.inf
        except Exception as e:
            print(f"Error in primal subproblem solve: {e}")
            return False, None, np.inf