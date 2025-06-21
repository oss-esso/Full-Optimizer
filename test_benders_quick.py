#!/usr/bin/env python3
"""
Quick test of Benders decomposition implementation.
"""
import sys
import os
sys.path.append('.')

from my_functions.benders import BendersDecomposition
from my_functions.dwave_qpu_adapter import DWaveQPUAdapter, DWaveConfig
from src.scenarios import load_food_data
import numpy as np
import time
import logging

def create_test_qubo(farms, foods, config):
    """Create a simple test QUBO matrix."""
    num_farms = len(farms)
    num_foods = len(foods)
    problem_size = num_farms * num_foods
    
    # Create random symmetric QUBO matrix
    Q = np.random.random((problem_size, problem_size)) * 0.1
    Q = (Q + Q.T) / 2  # Make symmetric
    np.fill_diagonal(Q, np.random.random(problem_size) * 2 - 1)
    return Q

def main():
    print('Testing Benders decomposition...')
    
    # Load data
    farms, foods, food_groups, config = load_food_data('full')
    problem_size = len(farms) * len(foods)
    print(f'Problem size: {problem_size} variables')
    
    # Create QUBO matrix
    qubo_matrix = create_test_qubo(farms, foods, config)
    print(f'QUBO matrix shape: {qubo_matrix.shape}')
    
    # Test Benders decomposition
    benders = BendersDecomposition(qubo_matrix, max_iterations=3, verbose=True)
    
    # Create quantum solver
    dwave_config = DWaveConfig(solver_type='simulator', num_reads=50)
    logger = logging.getLogger(__name__)
    adapter = DWaveQPUAdapter(config=dwave_config, logger=logger)
    
    # Solve
    start_time = time.time()
    result = benders.solve(adapter)
    solve_time = time.time() - start_time
    
    print(f'\\nBenders solve completed in {solve_time:.3f}s')
    print(f'Result success: {bool(result and "solution" in result)}')
    
    if result and 'subproblem_stats' in result:
        stats = result['subproblem_stats']
        print(f'Subproblems solved: {stats.get("num_subproblems", "N/A")}')
        print(f'Avg subproblem size: {stats.get("avg_subproblem_size", "N/A"):.1f}')
        print(f'Total quantum time: {stats.get("total_quantum_time", "N/A"):.3f}s')
        print(f'Iterations: {result.get("iterations", "N/A")}')
        print(f'Converged: {result.get("converged", "N/A")}')
    
    return result is not None and 'solution' in result

if __name__ == "__main__":
    success = main()
    print(f'\\nTest {"PASSED" if success else "FAILED"}')
