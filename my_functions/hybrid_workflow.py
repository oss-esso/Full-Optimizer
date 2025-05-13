import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Set, Any
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass

from my_functions.tree_decomposition import TreeDecomposer
from my_functions.qaoa_solver import QAOACircuit

@dataclass
class OptimizationResult:
    """Class for storing optimization results."""
    solution: np.ndarray
    objective_value: float
    runtime: float
    method: str
    subproblem_results: Optional[Dict[int, Any]] = None

class HybridOptimizer:
    """Main class for hybrid quantum-classical optimization."""
    
    def __init__(self, 
                 tree_decomposer: TreeDecomposer,
                 classical_optimizer: Any,
                 quantum_optimizer: QAOACircuit,
                 combine_solutions: Callable[[Dict[int, np.ndarray], Dict[int, Set[int]]], np.ndarray],
                 is_critical: Callable[[Set[int]], bool] = lambda _: False):
        """Initialize the hybrid optimizer."""
        self.tree_decomposer = tree_decomposer
        self.classical_optimizer = classical_optimizer
        self.quantum_optimizer = quantum_optimizer
        self.combine_solutions = combine_solutions
        self.is_critical = is_critical
        
    def optimize(self, qubo: np.ndarray, decomp_method: str = 'min_fill', timeout: Optional[float] = None) -> OptimizationResult:
        """Run hybrid optimization workflow."""
        start_time = time.time()
        try:
            # Phase 1: Decompose problem
            print("Phase 1: Tree Decomposition")
            tree_decomp = self.tree_decomposer.decompose(decomp_method)
            bags = tree_decomp.bags
            tree = tree_decomp.tree
            
            # Phase 2: Solve subproblems
            print("Phase 2: Solving subproblems")
            subproblem_solutions = {}
            subproblem_results = {}
            
            for bag_id, variables in bags.items():
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    raise TimeoutError("Optimization timed out during subproblem solving")
                
                print(f"Solving subproblem {bag_id} with {len(variables)} variables...")
                
                # Extract subproblem QUBO
                sub_qubo = self._extract_subproblem_qubo(qubo, variables)
                
                try:
                    # Solve subproblem
                    if self.is_critical(variables):
                        print(f"Using quantum solver for critical subproblem {bag_id}")
                        # Use quantum solver for critical subproblems
                        result = self.quantum_optimizer.solve_qubo(sub_qubo)
                        if result is None:
                            raise ValueError("Quantum solver failed to find a solution")
                        solution, energy = result
                        method = 'quantum'
                    else:
                        print(f"Using classical solver for subproblem {bag_id}")
                        # Use classical solver for non-critical subproblems
                        result = self.classical_optimizer.solve_qubo(sub_qubo)
                        if result is None:
                            raise ValueError("Classical solver failed to find a solution")
                        solution, energy = result
                        method = 'classical'
                    
                    # Map solution back to original variables
                    subproblem_solutions[bag_id] = {
                        var: sol for var, sol in zip(sorted(variables), solution)
                    }
                    
                    subproblem_results[bag_id] = {
                        'method': method,
                        'objective': energy,
                        'solution': solution,
                        'variables': sorted(variables)
                    }
                    
                except Exception as e:
                    print(f"Error solving subproblem {bag_id}: {e}")
                    # Continue with other subproblems
                    continue
            
            # Phase 3: Combine solutions
            print("Phase 3: Combining solutions")
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Optimization timed out during solution combination")
            
            combined_solution = self.combine_solutions(
                subproblem_solutions,
                bags,
                conflict_resolution='majority'
            )
            
            # Calculate final objective value
            final_energy = combined_solution.T @ qubo @ combined_solution
            
            runtime = time.time() - start_time
            
            return OptimizationResult(
                solution=combined_solution,
                objective_value=final_energy,
                runtime=runtime,
                method='hybrid',
                subproblem_results=subproblem_results
            )
            
        except TimeoutError as e:
            print(f"Optimization timed out: {e}")
            # Return best solution found so far
            if 'combined_solution' in locals():
                return OptimizationResult(
                    solution=combined_solution,
                    objective_value=final_energy,
                    runtime=timeout,
                    method='hybrid',
                    subproblem_results=subproblem_results
                )
            raise
        except Exception as e:
            raise RuntimeError(f"Hybrid optimization failed: {e}")

    def _extract_subproblem_qubo(self, qubo: np.ndarray, variables: Set[int]) -> np.ndarray:
        """Extract QUBO matrix for a subproblem."""
        # Sort variables to maintain consistent ordering
        sorted_vars = sorted(variables)
        n = len(sorted_vars)
        
        # Create subproblem QUBO
        sub_qubo = np.zeros((n, n))
        for i, var_i in enumerate(sorted_vars):
            for j, var_j in enumerate(sorted_vars):
                sub_qubo[i, j] = qubo[var_i, var_j]
        
        return sub_qubo
