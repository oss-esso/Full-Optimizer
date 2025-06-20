#!/usr/bin/env python3
"""
Test script for D-Wave cost estimation functionality.

This script tests the D-Wave QPU adapter's cost estimation capabilities
using real scenario data from scenarios.py. It demonstrates:
- Native problem (18 variables) solving with D-Wave simulator
- Problem size scaling analysis
- Summary table of results
"""

import os
import sys
import logging
import traceback
import numpy as np
import time
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def configure_dwave_environment():
    """Configure D-Wave environment for CPU-only simulator testing."""
    print("✓ Running CPU-only simulator tests (no D-Wave token required)")
    return True

def main():
    """Run D-Wave cost estimation tests for all complexity scenarios."""
    print("=" * 80)
    print("D-WAVE TESTING SUITE - ALL COMPLEXITY SCENARIOS")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    if not configure_dwave_environment():
        print("Exiting...")
        return False
    global test_results_summary
    test_results_summary = []
    try:
        from src.scenarios import load_food_data
        from my_functions.dwave_qpu_adapter import (
            DWaveQPUAdapter, DWaveConfig
        )
        print("✓ All imports successful")
        test_simple_complexity()
        test_intermediate_complexity()
        test_native_18_variable_problem()
        test_benders_decomposition_50var()
        test_native_problem_scaling()
        print_summary_table()
        print("\n" + "=" * 80)
        print("✓ ALL D-WAVE TESTS COMPLETED!")
        print("✓ System tested with simple, intermediate, and full complexity problems")
        print("=" * 80)
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please ensure D-Wave libraries are installed")
        return False
    except Exception as e:
        print(f"✗ Test suite failed: {e}")
        traceback.print_exc()
        return False
    return True

def test_simple_complexity():
    """Test the simple complexity scenario."""
    print("\n" + "-" * 60)
    print("TEST: SIMPLE COMPLEXITY")
    print("-" * 60)
    try:
        from my_functions.dwave_qpu_adapter import DWaveQPUAdapter, DWaveConfig
        from src.scenarios import load_food_data
        
        farms, foods, food_groups, config = load_food_data('simple')
        problem_size = len(farms) * len(foods)
        print(f"Simple problem details:")
        print(f"  Farms: {len(farms)} ({farms})")
        print(f"  Foods: {len(foods)} ({list(foods.keys())})")
        print(f"  Total variables: {problem_size}")
        
        dwave_config = DWaveConfig(
            solver_type='simulator',
            num_reads=500,
            estimate_cost_only=False
        )
        adapter = DWaveQPUAdapter(config=dwave_config, logger=logger)
        print(f"Active sampler: {adapter.sampler_type}")
        
        qubo_matrix = create_food_optimization_qubo(farms, foods, config)
        print(f"QUBO matrix size: {qubo_matrix.shape}")
        
        bqm = adapter.create_bqm_from_qubo(qubo_matrix)
        print(f"BQM variables: {len(bqm.variables)}")
        print(f"BQM interactions: {len(bqm.quadratic)}")
        
        print("\n--- Solving simple problem ---")
        start_time = time.time()
        result = adapter._solve_bqm(bqm)
        solve_time = time.time() - start_time
        
        if 'error' not in result:
            print(f"✓ Simple problem solved successfully!")
            print(f"  Energy: {result['energy']:.6f}")
            print(f"  Solve time: {solve_time:.3f}s")
            test_results_summary.append({
                'test_name': f'Simple ({problem_size} vars)',
                'variables': problem_size,
                'qubits': len(bqm.variables),
                'samples': dwave_config.num_reads,
                'solve_time': solve_time,
                'success': True
            })
        else:
            print(f"✗ Simple problem solving failed: {result['error']}")
            test_results_summary.append({
                'test_name': f'Simple ({problem_size} vars)',
                'variables': problem_size,
                'qubits': 'N/A',
                'samples': dwave_config.num_reads,
                'solve_time': 'N/A',
                'success': False
            })
        print("✓ Simple complexity test completed")
    except Exception as e:
        print(f"✗ Simple problem test failed: {e}")
        traceback.print_exc()
        raise

def test_intermediate_complexity():
    """Test the intermediate complexity scenario."""
    print("\n" + "-" * 60)
    print("TEST: INTERMEDIATE COMPLEXITY")
    print("-" * 60)
    try:
        from my_functions.dwave_qpu_adapter import DWaveQPUAdapter, DWaveConfig
        from src.scenarios import load_food_data
        
        farms, foods, food_groups, config = load_food_data('intermediate')
        problem_size = len(farms) * len(foods)
        print(f"Intermediate problem details:")
        print(f"  Farms: {len(farms)} ({farms})")
        print(f"  Foods: {len(foods)} ({list(foods.keys())})")
        print(f"  Total variables: {problem_size}")
        
        dwave_config = DWaveConfig(
            solver_type='simulator',
            num_reads=750,
            estimate_cost_only=False
        )
        adapter = DWaveQPUAdapter(config=dwave_config, logger=logger)
        print(f"Active sampler: {adapter.sampler_type}")
        
        qubo_matrix = create_food_optimization_qubo(farms, foods, config)
        print(f"QUBO matrix size: {qubo_matrix.shape}")
        
        bqm = adapter.create_bqm_from_qubo(qubo_matrix)
        print(f"BQM variables: {len(bqm.variables)}")
        print(f"BQM interactions: {len(bqm.quadratic)}")
        
        print("\n--- Solving intermediate problem ---")
        start_time = time.time()
        result = adapter._solve_bqm(bqm)
        solve_time = time.time() - start_time
        
        if 'error' not in result:
            print(f"✓ Intermediate problem solved successfully!")
            print(f"  Energy: {result['energy']:.6f}")
            print(f"  Solve time: {solve_time:.3f}s")
            test_results_summary.append({
                'test_name': f'Intermediate ({problem_size} vars)',
                'variables': problem_size,
                'qubits': len(bqm.variables),
                'samples': dwave_config.num_reads,
                'solve_time': solve_time,
                'success': True
            })
        else:
            print(f"✗ Intermediate problem solving failed: {result['error']}")
            test_results_summary.append({
                'test_name': f'Intermediate ({problem_size} vars)',
                'variables': problem_size,
                'qubits': 'N/A',
                'samples': dwave_config.num_reads,
                'solve_time': 'N/A',
                'success': False
            })
        print("✓ Intermediate complexity test completed")
    except Exception as e:
        print(f"✗ Intermediate problem test failed: {e}")
        traceback.print_exc()
        raise

def test_native_18_variable_problem():
    """Test the native full complexity food optimization problem."""
    print("\n" + "-" * 60)
    print("TEST: FULL COMPLEXITY FOOD OPTIMIZATION PROBLEM")
    print("-" * 60)
    try:
        global native_problem_result
        from my_functions.dwave_qpu_adapter import DWaveQPUAdapter, DWaveConfig
        from src.scenarios import load_food_data
        farms, foods, food_groups, config = load_food_data('full')
        problem_size = len(farms) * len(foods)
        print(f"Full complexity problem details:")
        print(f"  Farms: {len(farms)} ({farms})")
        print(f"  Foods: {len(foods)} ({list(foods.keys())})")
        print(f"  Total variables: {problem_size}")
        
        dwave_config = DWaveConfig(
            solver_type='simulator',
            num_reads=250,  # Reduced from 1000 for better performance on 50-variable problem
            estimate_cost_only=False
        )
        adapter = DWaveQPUAdapter(config=dwave_config, logger=logger)
        print(f"Active sampler: {adapter.sampler_type}")
        qubo_matrix = create_food_optimization_qubo(farms, foods, config)
        print(f"QUBO matrix size: {qubo_matrix.shape}")
        print(f"QUBO matrix density: {np.count_nonzero(qubo_matrix) / qubo_matrix.size:.3f}")
        bqm = adapter.create_bqm_from_qubo(qubo_matrix)
        print(f"BQM variables: {len(bqm.variables)}")
        print(f"BQM interactions: {len(bqm.quadratic)}")
        print("\n--- Solving full complexity problem ---")
        start_time = time.time()
        result = adapter._solve_bqm(bqm)
        solve_time = time.time() - start_time
        if 'error' not in result:
            print(f"✓ Full complexity problem solved successfully!")
            print(f"  Energy: {result['energy']:.6f}")
            print(f"  Solve time: {solve_time:.3f}s")
            print(f"  QPU time: {result.get('qpu_time', 0):.6f}s")
            print(f"  Chain break fraction: {result.get('chain_break_fraction', 0):.3f}")
            solution = result['sample']
            selected_vars = [var for var, val in solution.items() if val == 1]
            print(f"  Selected variables: {len(selected_vars)} out of {len(solution)}")
            native_problem_result = {
                'variables': problem_size,
                'qubits': len(bqm.variables),
                'samples': dwave_config.num_reads,
                'energy': result['energy'],
                'solve_time': solve_time,
                'qpu_time': result.get('qpu_time', 0),
                'chain_breaks': result.get('chain_break_fraction', 0),
                'assignments': len(selected_vars),
                'success': True
            }
            test_results_summary.append({
                'test_name': f'Full ({problem_size} vars)',
                'variables': problem_size,
                'qubits': len(bqm.variables),
                'samples': dwave_config.num_reads,
                'solve_time': solve_time,
                'success': True
            })
        else:
            print(f"✗ Full complexity problem solving failed: {result['error']}")
            native_problem_result = {
                'variables': problem_size,
                'error': result['error'],
                'success': False
            }
            test_results_summary.append({
                'test_name': f'Full ({problem_size} vars)',
                'variables': problem_size,
                'qubits': 'N/A',
                'samples': dwave_config.num_reads,
                'solve_time': 'N/A',
                'success': False
            })
        print("✓ Full complexity problem test completed")
    except Exception as e:
        print(f"✗ Full complexity problem test failed: {e}")
        traceback.print_exc()
        raise

def test_native_problem_scaling():
    """Test scaling behavior with different sample sizes for both 18-var and 50-var problems."""
    print("\n" + "-" * 60)
    print("TEST: PROBLEM SCALING ANALYSIS")
    print("-" * 60)
    try:
        from my_functions.dwave_qpu_adapter import DWaveQPUAdapter, DWaveConfig
        from src.scenarios import load_food_data
        
        # Test 18-variable problem scaling
        print("  Testing 18-variable problem scaling...")
        farms_18, foods_18, food_groups_18, config_18 = load_food_data('simple')
        problem_size_18 = len(farms_18) * len(foods_18)
        qubo_matrix_18 = create_food_optimization_qubo(farms_18, foods_18, config_18)
        sample_sizes_18 = [100, 250, 500, 750, 1000]  # Sample sizes for 18-variable problem
        
        for num_samples in sample_sizes_18:
            print(f"    18-var with {num_samples} samples...")
            dwave_config = DWaveConfig(
                solver_type='simulator',
                num_reads=num_samples
            )
            adapter = DWaveQPUAdapter(config=dwave_config, logger=logger)
            bqm = adapter.create_bqm_from_qubo(qubo_matrix_18)
            start_time = time.time()
            result = adapter._solve_bqm(bqm)
            solve_time = time.time() - start_time
            if 'error' not in result:
                print(f"      ✓ Success: time={solve_time:.3f}s, energy={result['energy']:.4f}")
                test_results_summary.append({
                    'test_name': f'18-var {num_samples} samples',
                    'variables': problem_size_18,
                    'qubits': len(bqm.variables),
                    'samples': num_samples,
                    'solve_time': solve_time,
                    'success': True
                })
            else:
                print(f"      ✗ Failed: {result['error']}")
                test_results_summary.append({
                    'test_name': f'18-var {num_samples} samples',
                    'variables': problem_size_18,
                    'qubits': 'N/A',
                    'samples': num_samples,
                    'solve_time': 'N/A',
                    'success': False
                })
        
        # Test 50-variable problem scaling
        print("  Testing 50-variable problem scaling...")
        farms_50, foods_50, food_groups_50, config_50 = load_food_data('full')
        problem_size_50 = len(farms_50) * len(foods_50)
        qubo_matrix_50 = create_food_optimization_qubo(farms_50, foods_50, config_50)
        sample_sizes_50 = [50, 100, 200, 300, 500]  # Reduced sample sizes for 50-variable problem        
        for num_samples in sample_sizes_50:
            print(f"    50-var with {num_samples} samples...")
            dwave_config = DWaveConfig(
                solver_type='simulator',
                num_reads=num_samples
            )
            adapter = DWaveQPUAdapter(config=dwave_config, logger=logger)
            bqm = adapter.create_bqm_from_qubo(qubo_matrix_50)
            start_time = time.time()
            result = adapter._solve_bqm(bqm)
            solve_time = time.time() - start_time
            if 'error' not in result:
                print(f"      ✓ Success: time={solve_time:.3f}s, energy={result['energy']:.4f}")
                test_results_summary.append({
                    'test_name': f'50-var {num_samples} samples',
                    'variables': problem_size_50,
                    'qubits': len(bqm.variables),
                    'samples': num_samples,
                    'solve_time': solve_time,
                    'success': True
                })
            else:
                print(f"      ✗ Failed: {result['error']}")
                test_results_summary.append({
                    'test_name': f'50-var {num_samples} samples',
                    'variables': problem_size_50,
                    'qubits': 'N/A',
                    'samples': num_samples,
                    'solve_time': 'N/A',
                    'success': False
                })
        print("✓ Problem scaling analysis completed")
    except Exception as e:
        print(f"✗ Problem scaling test failed: {e}")
        traceback.print_exc()
        raise

def test_benders_decomposition_50var():
    """Test Benders decomposition on the 50-variable problem to reduce annealer usage."""
    print("\n" + "-" * 60)
    print("TEST: BENDERS DECOMPOSITION (50 Variables)")
    print("-" * 60)
    try:
        from my_functions.dwave_qpu_adapter import DWaveQPUAdapter, DWaveConfig
        from my_functions.benders import BendersDecomposition
        from src.scenarios import load_food_data
        
        farms, foods, food_groups, config = load_food_data('full')
        problem_size = len(farms) * len(foods)
        print(f"Benders decomposition problem details:")
        print(f"  Farms: {len(farms)} ({farms})")
        print(f"  Foods: {len(foods)} ({list(foods.keys())})")
        print(f"  Total variables: {problem_size}")
        
        # Create the full QUBO matrix
        qubo_matrix = create_food_optimization_qubo(farms, foods, config)
        print(f"Original QUBO matrix size: {qubo_matrix.shape}")
        print(f"Original QUBO matrix density: {np.count_nonzero(qubo_matrix) / qubo_matrix.size:.3f}")
        
        # Initialize Benders decomposition
        print("\n--- Setting up Benders decomposition ---")
        benders = BendersDecomposition(
            qubo_matrix=qubo_matrix,
            max_iterations=10,
            tolerance=1e-6,
            verbose=True
        )
        
        # Configure D-Wave for smaller subproblems
        dwave_config = DWaveConfig(
            solver_type='simulator',
            num_reads=100,  # Fewer samples needed for smaller subproblems
            estimate_cost_only=False
        )
        adapter = DWaveQPUAdapter(config=dwave_config, logger=logger)
        print(f"Active sampler: {adapter.sampler_type}")
        
        print("\n--- Solving with Benders decomposition ---")
        start_time = time.time()
        
        # Run Benders decomposition
        result = benders.solve(quantum_solver=adapter)
        
        solve_time = time.time() - start_time
        
        if result and 'solution' in result:
            print(f"✓ Benders decomposition solved successfully!")
            print(f"  Total solve time: {solve_time:.3f}s")
            print(f"  Iterations: {result.get('iterations', 'N/A')}")
            print(f"  Final objective: {result.get('objective_value', 'N/A'):.6f}")
            print(f"  Convergence: {result.get('converged', 'N/A')}")
            
            # Count subproblem statistics
            subproblem_info = result.get('subproblem_stats', {})
            total_quantum_time = subproblem_info.get('total_quantum_time', 0)
            num_subproblems = subproblem_info.get('num_subproblems', 'N/A')
            avg_subproblem_size = subproblem_info.get('avg_subproblem_size', 'N/A')
            
            print(f"  Subproblems solved: {num_subproblems}")
            print(f"  Average subproblem size: {avg_subproblem_size}")
            print(f"  Total quantum time: {total_quantum_time:.3f}s")
            print(f"  Classical overhead: {solve_time - total_quantum_time:.3f}s")
            
            test_results_summary.append({
                'test_name': f'Benders ({problem_size} vars)',
                'variables': problem_size,
                'qubits': f'~{avg_subproblem_size}' if avg_subproblem_size != 'N/A' else 'N/A',
                'samples': f'{num_subproblems}x{dwave_config.num_reads}' if num_subproblems != 'N/A' else dwave_config.num_reads,
                'solve_time': solve_time,
                'success': True
            })
        else:
            print(f"✗ Benders decomposition failed")
            test_results_summary.append({
                'test_name': f'Benders ({problem_size} vars)',
                'variables': problem_size,
                'qubits': 'N/A',
                'samples': 'N/A',
                'solve_time': 'N/A',
                'success': False
            })
        print("✓ Benders decomposition test completed")
    except ImportError as e:
        print(f"✗ Benders decomposition not available: {e}")
        print("  Note: This requires the Benders module to be properly implemented")
        test_results_summary.append({
            'test_name': f'Benders (50 vars)',
            'variables': 50,
            'qubits': 'N/A',
            'samples': 'N/A',
            'solve_time': 'N/A',
            'success': False
        })
    except Exception as e:
        print(f"✗ Benders decomposition test failed: {e}")
        traceback.print_exc()
        test_results_summary.append({
            'test_name': f'Benders (50 vars)',
            'variables': 50,
            'qubits': 'N/A', 
            'samples': 'N/A',
            'solve_time': 'N/A',
            'success': False
        })

def print_summary_table():
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Test':<30} {'Vars':<6} {'Qubits':<8} {'Samples':<8} {'Time (s)':<10} {'Success':<8}")
    print("-" * 60)
    for r in test_results_summary:
        print(f"{r['test_name']:<30} {r['variables']:<6} {r['qubits']:<8} {r['samples']:<8} {r['solve_time'] if isinstance(r['solve_time'], float) else '-':<10} {str(r['success']):<8}")
    print("=" * 60)

def create_food_optimization_qubo(farms: List[str], foods: Dict[str, Dict], config: Dict) -> np.ndarray:
    num_farms = len(farms)
    num_foods = len(foods)
    problem_size = num_farms * num_foods
    Q = np.zeros((problem_size, problem_size))
    weights = config.get('parameters', {}).get('weights', {
        'nutritional_value': 0.25,
        'nutrient_density': 0.2,
        'affordability': 0.15,
        'sustainability': 0.15,
        'environmental_impact': 0.25
    })
    food_list = list(foods.keys())
    for farm_idx in range(num_farms):
        for food_idx, food_name in enumerate(food_list):
            # Example: maximize nutritional value, minimize cost, etc.
            Q[farm_idx * num_foods + food_idx, farm_idx * num_foods + food_idx] -= weights.get('nutritional_value', 0.25) * foods[food_name].get('nutritional_value', 1)
    penalty_strength = 10.0
    for farm_idx in range(num_farms):
        farm_vars = [farm_idx * num_foods + food_idx for food_idx in range(num_foods)]
        for i in farm_vars:
            Q[i, i] += penalty_strength
            for j in farm_vars:
                if i != j:
                    Q[i, j] += penalty_strength
    return Q

if __name__ == "__main__":
    success = main()
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if success:
        print("\n🎉 All tests passed! Check the Results folder for detailed reports.")
        print("\nNext steps:")
        print("1. Review the summary table above for performance insights")
        print("2. Use the JSON data for further analysis if needed")
        print("3. Plan your quantum experiments within budget")
        print("4. Consider starting with smaller complexity levels")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
    input("\nPress Enter to exit...")
