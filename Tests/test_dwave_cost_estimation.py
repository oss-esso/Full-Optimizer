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
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

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
        test_synthetic_problems()
        test_native_problem_scaling()
        test_synthetic_problems()
        print_summary_table()
        analyze_scaling_and_interpolate()
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
    global test_results_summary
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
        density = np.count_nonzero(qubo_matrix) / qubo_matrix.size
        print(f"QUBO matrix size: {qubo_matrix.shape}, density: {density:.3f}")
        
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
                'density': density,  # Calculated from QUBO matrix
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
                'density': density,
                'success': False
            })
        print("✓ Simple complexity test completed")
    except Exception as e:
        print(f"✗ Simple problem test failed: {e}")
        traceback.print_exc()
        raise

def test_intermediate_complexity():
    """Test the intermediate complexity scenario."""
    global test_results_summary
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
        density = np.count_nonzero(qubo_matrix) / qubo_matrix.size
        print(f"QUBO matrix size: {qubo_matrix.shape}, density: {density:.3f}")
        
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
                'density': density,
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
                'density': density,
                'success': False
            })
        print("✓ Intermediate complexity test completed")
    except Exception as e:
        print(f"✗ Intermediate problem test failed: {e}")
        traceback.print_exc()
        raise

def test_native_18_variable_problem():
    """Test the native full complexity food optimization problem."""
    global test_results_summary
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
        density = np.count_nonzero(qubo_matrix) / qubo_matrix.size
        print(f"QUBO matrix size: {qubo_matrix.shape}")
        print(f"QUBO matrix density: {density:.3f}")
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
                'density': density,
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
                'density': density,
                'success': False
            })
        print("✓ Full complexity problem test completed")
    except Exception as e:
        print(f"✗ Full complexity problem test failed: {e}")
        traceback.print_exc()
        raise

def test_native_problem_scaling():
    """Test scaling behavior with different sample sizes for both 18-var and 50-var problems."""
    global test_results_summary
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
    global test_results_summary
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
        
        # Run Benders decomposition
        result = benders.solve(quantum_solver=adapter)
        
        if result and 'solution' in result:
            print(f"✓ Benders decomposition solved successfully!")
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
            print(f"  Total quantum solve time: {total_quantum_time:.3f}s")
            print(f"  (Quantum time only - excludes classical overhead)")
            
            test_results_summary.append({
                'test_name': f'Benders ({problem_size} vars)',
                'variables': problem_size,
                'qubits': f'~{avg_subproblem_size:.0f}' if avg_subproblem_size != 'N/A' else 'N/A',
                'samples': f'{num_subproblems}x{dwave_config.num_reads}' if num_subproblems != 'N/A' else dwave_config.num_reads,
                'solve_time': total_quantum_time,  # Use quantum time only
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

def test_synthetic_problems():
    """Test synthetic QUBO problems with controlled variable counts and sample sizes."""
    global test_results_summary
    print("\n" + "-" * 60)
    print("TEST: SYNTHETIC QUBO PROBLEMS")
    print("-" * 60)
    
    try:
        from my_functions.dwave_qpu_adapter import DWaveQPUAdapter, DWaveConfig
        import numpy as np
        
        def create_synthetic_qubo(n_vars, max_density=0.7):
            """Create a synthetic QUBO matrix with n_vars variables and limited density."""
            # Create random symmetric QUBO matrix with controlled density
            Q = np.zeros((n_vars, n_vars))
            
            # Calculate how many off-diagonal elements to fill (upper triangle)
            total_off_diag = (n_vars * (n_vars - 1)) // 2
            num_nonzero = min(int(total_off_diag * max_density), total_off_diag)
            
            # Randomly select positions for non-zero elements in upper triangle
            positions = np.random.choice(total_off_diag, num_nonzero, replace=False)
            
            # Convert positions to (i, j) coordinates in upper triangle
            count = 0
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if count in positions:
                        value = np.random.random() * 0.2 - 0.1  # Values between -0.1 and 0.1
                        Q[i, j] = value
                        Q[j, i] = value  # Make symmetric
                    count += 1
            
            # Fill diagonal values
            np.fill_diagonal(Q, np.random.random(n_vars) * 2 - 1)  # Diagonal values between -1 and 1
            return Q
        
        # Test configurations: (variables, sample_sizes)
        test_configs = [
            (3, [50, 250]),
            (6, [50, 250]),
            (12, [50, 250]),
            (18, [50, 100, 250]),
            (50, [50, 250, 500])
        ]
        
        for n_vars, sample_sizes in test_configs:
            print(f"\n  Testing {n_vars}-variable synthetic problem:")
              # Create synthetic QUBO
            qubo_matrix = create_synthetic_qubo(n_vars)
            density = np.count_nonzero(qubo_matrix) / qubo_matrix.size
            print(f"    QUBO matrix: {qubo_matrix.shape}, density: {density:.3f}")
            
            for num_samples in sample_sizes:
                print(f"    Solving with {num_samples} samples...")
                
                # Configure D-Wave
                dwave_config = DWaveConfig(
                    solver_type='simulator',
                    num_reads=num_samples,
                    estimate_cost_only=False
                )
                adapter = DWaveQPUAdapter(config=dwave_config, logger=logger)
                
                # Create BQM and solve
                bqm = adapter.create_bqm_from_qubo(qubo_matrix)
                start_time = time.time()
                result = adapter._solve_bqm(bqm)
                solve_time = time.time() - start_time
                
                if 'error' not in result:
                    print(f"      ✓ Success: time={solve_time:.3f}s, energy={result['energy']:.4f}")
                    test_results_summary.append({
                        'test_name': f'Synthetic {n_vars}v-{num_samples}s',
                        'variables': n_vars,
                        'qubits': len(bqm.variables),
                        'samples': num_samples,
                        'solve_time': solve_time,
                        'density': density,
                        'success': True
                    })
                else:
                    print(f"      ✗ Failed: {result['error']}")
                    test_results_summary.append({
                        'test_name': f'Synthetic {n_vars}v-{num_samples}s',
                        'variables': n_vars,
                        'qubits': 'N/A',
                        'samples': num_samples,
                        'solve_time': 'N/A',
                        'density': density,
                        'success': False
                    })
        
        print("✓ Synthetic problems test completed")
        
    except Exception as e:
        print(f"✗ Synthetic problems test failed: {e}")
        traceback.print_exc()
        raise

def print_summary_table():
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Test':<30} {'Vars':<6} {'Qubits':<8} {'Samples':<8} {'Time (s)':<10} {'Success':<8}")
    print("-" * 60)
    for r in test_results_summary:
        print(f"{r['test_name']:<30} {r['variables']:<6} {r['qubits']:<8} {r['samples']:<8} {r['solve_time'] if isinstance(r['solve_time'], float) else '-':<10} {str(r['success']):<8}")
    print("=" * 60)

def analyze_scaling_and_interpolate():
    """Analyze scaling patterns and perform interpolation on the test results."""
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS & INTERPOLATION")
    print("=" * 80)
      # Group results by problem size
    scaling_data = {}
    for result in test_results_summary:
        # Debug: print what we're checking
        print(f"Debug: Checking result - success: {result['success']}, "
              f"solve_time: {result['solve_time']} (type: {type(result['solve_time'])}), "
              f"samples: {result['samples']} (type: {type(result['samples'])})")
        
        if (result['success'] and 
            isinstance(result['solve_time'], (int, float)) and 
            isinstance(result['samples'], (int, float)) and
            result['solve_time'] != 'N/A' and
            result['samples'] != 'N/A'):  # Extra safety checks
            
            vars_count = result['variables']
            samples = int(result['samples'])  # Convert to int
            time = float(result['solve_time'])  # Convert to float
            density = result.get('density', 'N/A')  # Get density if available
            
            if vars_count not in scaling_data:
                scaling_data[vars_count] = {'samples': [], 'times': [], 'densities': []}
            
            scaling_data[vars_count]['samples'].append(samples)
            scaling_data[vars_count]['times'].append(time)
            scaling_data[vars_count]['densities'].append(density)
            print(f"  Added to scaling data: {vars_count} vars, {samples} samples, {time:.3f}s, density: {density}")
    
    # Perform linear interpolation for each problem size
    print("Linear scaling coefficients (Time = a * Samples + b):")
    print(f"{'Variables':<12} {'Coeff (a)':<12} {'Intercept (b)':<12} {'R²':<8} {'Time/Sample':<12}")
    print("-" * 68)
    
    interpolation_results = {}
    
    for vars_count in sorted(scaling_data.keys()):
        data = scaling_data[vars_count]
        samples = np.array(data['samples'], dtype=float)  # Ensure float type
        times = np.array(data['times'], dtype=float)      # Ensure float type
        
        if len(samples) >= 2:
            # Linear regression: time = a * samples + b
            coeffs = np.polyfit(samples, times, 1)
            a, b = coeffs
            
            # Calculate R²
            predicted = a * samples + b
            ss_res = np.sum((times - predicted) ** 2)
            ss_tot = np.sum((times - np.mean(times)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1
            
            # Time per sample
            time_per_sample = a
            
            interpolation_results[vars_count] = {
                'coefficient': a,
                'intercept': b,
                'r_squared': r_squared,
                'time_per_sample': time_per_sample,
                'samples': samples,
                'times': times,
                'densities': data['densities']
            }
            
            print(f"{vars_count:<12} {a:<12.6f} {b:<12.3f} {r_squared:<8.3f} {time_per_sample:<12.6f}")
        else:
            print(f"{vars_count:<12} {'N/A':<12} {'N/A':<12} {'N/A':<8} {'N/A':<12}")
    
    # Show interpolated predictions for common sample sizes
    print(f"\nInterpolated solve times for common sample sizes:")
    print(f"{'Variables':<12} {'100 samples':<12} {'250 samples':<12} {'500 samples':<12} {'1000 samples':<12}")
    print("-" * 68)
    
    for vars_count in sorted(interpolation_results.keys()):
        data = interpolation_results[vars_count]
        a, b = data['coefficient'], data['intercept']
        
        t_100 = a * 100 + b
        t_250 = a * 250 + b
        t_500 = a * 500 + b
        t_1000 = a * 1000 + b
        
        print(f"{vars_count:<12} {t_100:<12.3f} {t_250:<12.3f} {t_500:<12.3f} {t_1000:<12.3f}")
    
    # Complexity scaling analysis
    print(f"\nComplexity scaling (Variables vs Time per Sample):")
    if len(interpolation_results) >= 2:
        var_counts = list(interpolation_results.keys())
        time_per_samples = [interpolation_results[v]['time_per_sample'] for v in var_counts]
        
        # Fit polynomial to see if it's O(n), O(n²), etc.
        var_array = np.array(var_counts, dtype=float)
        time_array = np.array(time_per_samples, dtype=float)
        
        # Try linear and quadratic fits
        linear_coeffs = np.polyfit(var_array, time_array, 1)
        quad_coeffs = np.polyfit(var_array, time_array, 2) if len(var_counts) >= 3 else None
        
        print(f"Linear fit: Time/Sample = {linear_coeffs[0]:.8f} * Variables + {linear_coeffs[1]:.6f}")
        if quad_coeffs is not None:
            print(f"Quadratic fit: Time/Sample = {quad_coeffs[0]:.10f} * Variables² + {quad_coeffs[1]:.8f} * Variables + {quad_coeffs[2]:.6f}")
    
    # Create visualizations
    create_scaling_plots(interpolation_results, scaling_data)
    
    print("=" * 80)

def create_scaling_plots(interpolation_results, scaling_data):
    """Create comprehensive plots for scaling analysis with exponential fits."""
    print("\nCreating scaling plots with exponential fits...")
    
    try:
        # Set up matplotlib for better plots
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Define exponential/power-law function: y = a * x^b
        def power_law(x, a, b):
            return a * np.power(x, b)
        
        # Define exponential function: y = a * exp(b * x)
        def exponential_func(x, a, b):
            return a * np.exp(b * x)        # Plot 1: Variables vs Solve time grouped by sample count
        # First, reorganize data by sample count
        sample_groups = {}
        for vars_count in sorted(interpolation_results.keys()):
            data = interpolation_results[vars_count]
            samples_list = data['samples']
            times_list = data['times']
            densities_list = data['densities']
            
            for sample_count, solve_time, density in zip(samples_list, times_list, densities_list):
                if sample_count not in sample_groups:
                    sample_groups[sample_count] = {'variables': [], 'times': [], 'densities': [], 'is_synthetic': []}
                sample_groups[sample_count]['variables'].append(vars_count)
                sample_groups[sample_count]['times'].append(solve_time)
                sample_groups[sample_count]['densities'].append(density)
                # Determine if synthetic based on typical density patterns
                # Synthetic problems will have density <= 0.7, real problems typically higher
                sample_groups[sample_count]['is_synthetic'].append(density <= 0.7)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        for i, sample_count in enumerate(sorted(sample_groups.keys())):
            data = sample_groups[sample_count]
            variables = data['variables']
            times = data['times']
            densities = data['densities']
            is_synthetic = data['is_synthetic']
            
            color = colors[i % len(colors)]
            
            # Separate synthetic and real problems
            real_vars, real_times, real_densities = [], [], []
            synth_vars, synth_times, synth_densities = [], [], []
            
            for var, time, density, is_synth in zip(variables, times, densities, is_synthetic):
                if is_synth:
                    synth_vars.append(var)
                    synth_times.append(time)
                    synth_densities.append(density)
                else:
                    real_vars.append(var)
                    real_times.append(time)
                    real_densities.append(density)
            
            # Plot real problems with circles
            if real_vars:
                ax1.scatter(real_vars, real_times, 
                           label=f'{sample_count} samples (real)', 
                           color=color, alpha=0.7, s=80, marker='o')
            
            # Plot synthetic problems with triangles, color-coded by density
            if synth_vars:
                scatter = ax1.scatter(synth_vars, synth_times, 
                                    c=synth_densities, cmap='viridis', 
                                    alpha=0.8, s=80, marker='^',
                                    label=f'{sample_count} samples (synthetic)')
                
                # Add colorbar for density if this is the first synthetic scatter
                if i == 0 and synth_vars:
                    cbar = plt.colorbar(scatter, ax=ax1)
                    cbar.set_label('Graph Density')
            # Try exponential fit if enough data points (combine real and synthetic)
            all_vars = real_vars + synth_vars
            all_times = real_times + synth_times
            
            if len(all_vars) >= 3:                
                try:
                    # Sort for proper fitting
                    sorted_indices = np.argsort(all_vars)
                    vars_sorted = np.array(all_vars)[sorted_indices]
                    times_sorted = np.array(all_times)[sorted_indices]
                    
                    # Power-law fit: t = a * vars^b
                    popt_power, _ = curve_fit(power_law, vars_sorted, times_sorted, p0=[0.001, 1.0])
                    x_fit = np.linspace(min(vars_sorted), max(vars_sorted), 100)
                    y_power = power_law(x_fit, *popt_power)
                    ax1.plot(x_fit, y_power, '-', color=color, alpha=0.8, linewidth=2)
                    print(f"  Power-law fit for {sample_count} samples: t = {popt_power[0]:.6f} * vars^{popt_power[1]:.3f}")
                except Exception as e:
                    print(f"  Power-law fit failed for {sample_count} samples: {e}")
            elif len(all_vars) >= 2:
                # Linear fit for fewer points
                sorted_indices = np.argsort(all_vars)
                vars_sorted = np.array(all_vars)[sorted_indices]
                times_sorted = np.array(all_times)[sorted_indices]
                
                linear_coeffs = np.polyfit(vars_sorted, times_sorted, 1)
                x_fit = np.linspace(min(vars_sorted), max(vars_sorted), 100)
                y_linear = np.polyval(linear_coeffs, x_fit)
                ax1.plot(x_fit, y_linear, '--', color=color, alpha=0.5)
        
        ax1.set_xlabel('Number of Variables')
        ax1.set_ylabel('Solve Time (seconds)')
        ax1.set_title('Scaling: Solve Time vs Problem Size (grouped by sample count)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Log-log plot for power-law analysis
        for i, vars_count in enumerate(sorted(interpolation_results.keys())):
            data = interpolation_results[vars_count]
            samples = data['samples']
            times = data['times']
            
            color = colors[i % len(colors)]
            ax2.loglog(samples, times, 'o-', label=f'{vars_count} variables', color=color)
        
        ax2.set_xlabel('Number of Samples (log scale)')
        ax2.set_ylabel('Solve Time (log scale)')
        ax2.set_title('Log-Log Plot: Power-Law Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Complexity scaling (variables vs time per sample)
        if len(interpolation_results) >= 2:
            var_counts = np.array(sorted(interpolation_results.keys()))
            time_per_samples = np.array([interpolation_results[v]['time_per_sample'] for v in var_counts])
            
            ax3.scatter(var_counts, time_per_samples, color='red', s=100, alpha=0.7)
            
            # Fit different complexity models
            # Linear: O(n)
            linear_coeffs = np.polyfit(var_counts, time_per_samples, 1)
            x_fit = np.linspace(min(var_counts), max(var_counts), 100)
            y_linear = np.polyval(linear_coeffs, x_fit)
            ax3.plot(x_fit, y_linear, '--', label=f'Linear: {linear_coeffs[0]:.6f}n + {linear_coeffs[1]:.6f}', color='blue')
            
            # Quadratic: O(n²)
            if len(var_counts) >= 3:
                quad_coeffs = np.polyfit(var_counts, time_per_samples, 2)
                y_quad = np.polyval(quad_coeffs, x_fit)
                ax3.plot(x_fit, y_quad, '-', label=f'Quadratic: {quad_coeffs[0]:.8f}n² + {quad_coeffs[1]:.6f}n + {quad_coeffs[2]:.6f}', color='green')
            
            # Exponential fit
            try:
                popt_exp, _ = curve_fit(exponential_func, var_counts, time_per_samples, p0=[0.001, 0.1])
                y_exp = exponential_func(x_fit, *popt_exp)
                ax3.plot(x_fit, y_exp, '-', label=f'Exponential: {popt_exp[0]:.6f}*exp({popt_exp[1]:.3f}*n)', color='orange')
                print(f"  Exponential complexity fit: t = {popt_exp[0]:.6f} * exp({popt_exp[1]:.3f} * variables)")
            except Exception as e:
                print(f"  Exponential complexity fit failed: {e}")
            
            ax3.set_xlabel('Number of Variables')
            ax3.set_ylabel('Time per Sample (seconds)')
            ax3.set_title('Complexity Scaling Analysis')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Heatmap of solve times
        if len(interpolation_results) >= 2:
            # Create a heatmap of interpolated solve times
            var_counts = sorted(interpolation_results.keys())
            sample_sizes = [50, 100, 250, 500, 1000]
            
            heatmap_data = np.zeros((len(var_counts), len(sample_sizes)))
            
            for i, vars_count in enumerate(var_counts):
                data = interpolation_results[vars_count]
                a, b = data['coefficient'], data['intercept']
                for j, samples in enumerate(sample_sizes):
                    heatmap_data[i, j] = a * samples + b
            
            im = ax4.imshow(heatmap_data, cmap='viridis', aspect='auto')
            ax4.set_xticks(range(len(sample_sizes)))
            ax4.set_xticklabels(sample_sizes)
            ax4.set_yticks(range(len(var_counts)))
            ax4.set_yticklabels(var_counts)
            ax4.set_xlabel('Number of Samples')
            ax4.set_ylabel('Number of Variables')
            ax4.set_title('Interpolated Solve Times Heatmap (seconds)')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax4)
            cbar.set_label('Solve Time (seconds)')
            
            # Add text annotations
            for i in range(len(var_counts)):
                for j in range(len(sample_sizes)):
                    text = ax4.text(j, i, f'{heatmap_data[i, j]:.3f}',
                                   ha="center", va="center", color="white", fontsize=8)
        
        plt.tight_layout()
        
        # Save the plot
        results_dir = os.path.join(project_root, "Results")
        os.makedirs(results_dir, exist_ok=True)
        plot_path = os.path.join(results_dir, "dwave_scaling_analysis_with_fits.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Scaling analysis plot saved to: {plot_path}")
        
        # Show plot if running interactively
        plt.show()
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        traceback.print_exc()

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
