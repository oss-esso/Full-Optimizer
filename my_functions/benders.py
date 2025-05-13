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
        
        # Solution history to track y_sol throughout iterations
        self._solution_history = {}  # Maps iteration number to y_sol

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
        y_sol = self.y_init.copy()
        self.LB = -np.inf
        self.UB = np.inf
        self.best_y_sol = None
        self.best_x_sol = None
        self.best_obj = np.inf
        
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
            # Only if it's a valid solution
            if y_sol is not None and np.sum(y_sol) > 0:
                self._previous_solutions.append(y_sol.copy())
            
            # Solve subproblem for fixed y_sol
            t_sub_start = time.time()
            solved, x_sol, obj_val, extreme_ray = self.solve_subproblem(y_sol)
            t_sub = time.time() - t_sub_start
            self.subproblem_times.append(t_sub)
            
            # Update upper bound (UB) if feasible
            if solved:
                # Since we're maximizing the original problem, we negate when updating bounds
                obj_original = -obj_val - float(self.f.T.dot(y_sol))  # Convert back to original objective
                print(f"Subproblem solved. Objective: {obj_val}, Original objective (maximization): {obj_original}")
                
                # Update UB if better than current
                if obj_val < self.UB:
                    self.UB = obj_val
                    # Store this solution in the history
                    self._solution_history[k] = y_sol.copy()
                    
                # Track best solution found
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_y = y_sol.copy()
                    best_x = x_sol.copy()
                    last_improvement_iter = k
                    print(f"New best solution found! Objective: {best_obj}, Original objective: {-best_obj - float(self.f.T.dot(best_y))}")
                
                # Store best solution overall for use if the algo doesn't converge
                if obj_val < self.best_obj:
                    self.best_obj = obj_val
                    self.best_y_sol = y_sol.copy()
                    self.best_x_sol = x_sol.copy()
                
                # Generate optimality cut
                dual_var = x_sol  # Actually the dual variable
                
                # Normalize cut if requested (helps numerical stability)
                if self.use_norm_cuts and np.linalg.norm(dual_var) > 1e-10:
                    # Normalize the cut by dividing by its norm
                    dual_var = dual_var / np.linalg.norm(dual_var)
                    
                self.optimality_cuts.append(dual_var)
                self.optimality_cut_counts.append(1)
                self.feasibility_cut_counts.append(0)
                print(f"Generated 1 optimality cut")
                
            else:
                # Generate feasibility cut
                self.feasibility_cuts.append(extreme_ray)
                self.optimality_cut_counts.append(0)
                self.feasibility_cut_counts.append(1)
                print(f"Subproblem infeasible. Generated 1 feasibility cut")
            
            # Solve master problem, with trust region if enabled
            t_master_start = time.time()
            if self.use_trust_region and k > 0 and best_y is not None:
                # Use trust region around best solution found so far
                previous_center = best_y
                y_sol, lb = self.solve_master_problem(
                    excluded_solutions=self._previous_solutions[-10:] if len(self._previous_solutions) > 10 else self._previous_solutions,
                    trust_center=previous_center,
                    trust_radius=trust_radius
                )
                
                # Adjust trust radius based on progress
                if solved and obj_val < best_obj:
                    # Good progress - expand trust region
                    trust_radius = min(trust_radius * 1.5, self.Ny)
                else:
                    # Poor progress - shrink trust region
                    trust_radius = max(trust_radius * 0.5, 3)  # Don't go below 3 changes
            else:
                # Solve without trust region
                y_sol, lb = self.solve_master_problem(
                    excluded_solutions=self._previous_solutions[-10:] if len(self._previous_solutions) > 10 else self._previous_solutions
                )
                
            t_master = time.time() - t_master_start
            self.master_times.append(t_master)
            
            # Update lower bound
            if lb > self.LB:
                self.LB = lb
            
            # Store bounds history
            self.lower_bounds.append(self.LB)
            self.upper_bounds.append(self.UB)
            
            # Print current bounds
            # Since we're maximizing the original problem, we need to negate the bounds
            print(f"Current bounds (minimization): LB = {self.LB}, UB = {self.UB}")
            print(f"Current bounds (maximization): LB = {-self.UB}, UB = {-self.LB}")
            if solved:
                print(f"Current objective value (for maximization): {-obj_val - float(self.f.T.dot(y_sol))}")
            
            # Gap computation
            if self.UB < np.inf and self.LB > -np.inf:
                if self.relative:
                    # Calculate relative gap - we need to be careful about signs
                    # since we're minimizing the negative of what we want to maximize
                    if abs(self.UB) > 1e-10:  # Avoid division by zero
                        # This gap is normalized by UB (which is negative)
                        gap = abs((self.UB - self.LB) / (abs(self.UB) + 1e-10))
                        print(f"Relative gap: {gap*100:.4f}%")
                    else:
                        gap = np.inf
                        print("Cannot compute relative gap: UB too close to zero")
                else:
                    # Absolute gap
                    gap = abs(self.UB - self.LB)
                    print(f"Absolute gap: {gap}")
                
                # Check if gap is small enough after minimum iterations
                if k >= min_iterations - 1 and gap <= current_eps:
                    # If we've done the absolute minimum iterations and the gap is small enough
                    # with the current tolerance, reduce the tolerance for stricter convergence
                    if current_eps > self.eps:
                        current_eps *= eps_reduction_factor
                        print(f"Reducing epsilon to {current_eps}")
                        # Continue iterations with tighter tolerance
                    else:
                        # We've reached the original target tolerance
                        print(f"Converged: gap = {gap} < eps = {self.eps}")
                        break
            
            # Check stagnation - if no improvement in many iterations
            stagnation_limit = min(10, self.max_iterations // 5)
            if k - last_improvement_iter > stagnation_limit and k >= min_iterations - 1:
                print(f"Warning: No improvement in {stagnation_limit} iterations. Using best solution found.")
                y_sol = best_y.copy()
                break
            
            # Track solution
            print(f"Current solution: sum(y) = {np.sum(y_sol)}, y[y>0.5] indices: {np.where(y_sol > 0.5)[0]}")
            
            # Track timing
            print(f"Times: Subproblem = {t_sub:.2f}s, Master = {t_master:.2f}s")
        
        # If best_y is None, use best solution found so far
        if best_y is None:
            best_y = self.best_y_sol if self.best_y_sol is not None else y_sol
            
        # Solve subproblem one more time with best y to get best x (dual variables and obj_val for original_obj)
        # This dual solve is primarily to ensure UB is consistent if best_y changed.
        solved_dual, p_val_final, obj_val_dual_final, _ = self.solve_subproblem(best_y)

        # NOW, solve the PRIMAL subproblem to get the actual x values
        solved_primal, x_primal_final, obj_val_primal_final = self.solve_primal_subproblem(best_y)

        best_x_to_store = np.zeros((self.Nx, 1)) # Default fallback
        final_obj_for_minimization = np.inf

        if solved_primal and x_primal_final is not None:
            best_x_to_store = x_primal_final
            # The objective value for the original minimization problem (cTx + fTy)
            final_obj_for_minimization = obj_val_primal_final + float(self.f.T.dot(best_y))
            print(f"Final primal subproblem solved. Primal Obj (cTx): {obj_val_primal_final}, fTy: {float(self.f.T.dot(best_y))}, Total Minimized Obj: {final_obj_for_minimization}")
        elif solved_dual:
            print("Warning: Final primal subproblem solve failed! Using dual subproblem objective for bounds, but x values are a fallback.")
            # If primal failed, but dual was solved, use dual objective. x values are just a placeholder.
            final_obj_for_minimization = obj_val_dual_final + float(self.f.T.dot(best_y))
            # best_x_to_store remains zeros
        else:
            print("Warning: Both final primal and dual subproblem solves failed or y was None. Using fallback values.")
            # If best_y itself was None or both subproblems failed, obj_val remains from iterations or np.inf
            # best_x_to_store remains zeros
            # final_obj_for_minimization remains np.inf
            if best_y is not None: # If y was valid, try to use the best obj from iterations
                 final_obj_for_minimization = self.best_obj + float(self.f.T.dot(best_y)) if self.best_obj != np.inf else np.inf

        # Update overall objective value for the class (minimization form)
        self.obj_val = final_obj_for_minimization
        # For a maximization problem, the user of the class will typically negate this.
        self.original_obj = -self.obj_val if self.obj_val != np.inf else -np.inf
        
        # Final UB should be consistent with the final solution found, if primal subproblem was solved
        # self.UB is obj_val from dual, which should ideally match obj_val_primal_final
        if solved_primal:
            # Update UB to reflect the cost of the primal solution if it's worse than current UB from duals
            # This shouldn't happen if strong duality holds and everything is correct
            # self.UB = min(self.UB, obj_val_primal_final) # obj_val_primal_final is only cTx
            pass # self.UB is based on dual subproblems throughout iterations

        print(f"\nBenders decomposition completed.")
        print(f"Final bounds (minimization): LB = {self.LB}, UB = {self.UB}")
        # For maximization, the user will interpret these negated
        print(f"Final objective value (for minimization, cTx+fTy): {self.obj_val}") 
        print(f"Iterations: {k+1} (out of maximum {self.max_iterations})")
        print(f"Total optimality cuts: {sum(self.optimality_cut_counts)}")
        print(f"Total feasibility cuts: {sum(self.feasibility_cut_counts)}")
        
        # Store and return solution
        self.x_sol = best_x_to_store # THIS IS NOW THE PRIMAL X
        self.y_sol = best_y
        # self.obj_val and self.original_obj are set above.
        
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