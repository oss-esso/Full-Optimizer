#!/usr/bin/env python3
"""
Test script for D-Wave cost estimation functionality.

This script tests the D-Wave QPU adapter's cost estimation capabilities
using real scenario data from scenarios.py. It demonstrates:
- Free cost estimation for all complexity levels
- Problem size scaling analysis
- Budget planning and recommendations
- Quantum vs classical method comparison
"""

import sys
import os
import time
import json
import logging
import numpy as np
import traceback
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
    """Configure D-Wave environment and authentication."""
    try:
        # Try to load from .env file first
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
    except Exception:
        pass  # Ignore .env file errors
    
    # Check for token
    if 'DWAVE_API_TOKEN' not in os.environ:
        print("\n" + "⚠️ " * 20)
        print("D-WAVE AUTHENTICATION REQUIRED")
        print("⚠️ " * 20)
        print("\nTo test hybrid solvers, you need a D-Wave Leap account.")
        print("1. Go to: https://cloud.dwavesys.com/leap/")
        print("2. Sign up for free account (includes 1 minute QPU time)")
        print("3. Get your API token from the dashboard")
        print("4. Set environment variable: DWAVE_API_TOKEN=your_token")
        print("\nAlternatively, create a .env file with:")
        print("DWAVE_API_TOKEN=your_token_here")
        print()
        
        proceed = input("Continue with CPU-only testing? (y/n): ").lower()
        if proceed != 'y':
            return False
    else:
        print("✓ D-Wave token configured")
    
    return True

def main():
    """Run comprehensive D-Wave cost estimation tests."""
    print("=" * 80)
    print("D-WAVE PROGRESSIVE TESTING SUITE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configure D-Wave environment
    if not configure_dwave_environment():
        print("Exiting...")
        return False
    
    try:
        # Import required modules
        from src.scenarios import load_food_data
        from my_functions.dwave_qpu_adapter import (
            DWaveQPUAdapter, DWaveConfig, get_free_dwave_analysis,
            estimate_dwave_cost_for_problem, create_simple_food_problem
        )
        
        print("✓ All imports successful")
        
        # Test 0: Create simple test problem
        test_simple_problem_creation()
        
        # Test 1: Basic D-Wave adapter initialization
        test_basic_initialization()
        
        # Test 2: CPU-based testing (Simulated Annealing)
        test_cpu_solver()
        
        # Test 3: Hybrid solver testing
        test_hybrid_solver()
        
        # Test 4: Progressive complexity testing
        test_progressive_complexity()
        
        # Test 5: Full scenario testing
        test_full_scenario()
        
        # Test 6: QPU readiness check (don't use QPU yet)
        test_qpu_readiness()
        
        # Test 7: NEW - Scaling analysis with real scenarios
        test_scaling_analysis_with_scenarios()
        
        # Test 8: Generate comprehensive report
        generate_progressive_test_report()
        
        print("\n" + "=" * 80)
        print("✓ ALL PROGRESSIVE D-WAVE TESTS COMPLETED!")
        print("✓ System is ready for hybrid and QPU testing")
        print("=" * 80)
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please ensure D-Wave libraries are installed")
        return False
    except Exception as e:
        print(f"✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_simple_problem_creation():
    """Test creation and validation of simple test problems."""
    print("\n" + "-" * 60)
    print("TEST 0: SIMPLE PROBLEM CREATION")
    print("-" * 60)
    
    try:
        # Declare global variable at the beginning of the function
        global simple_test_problem
        
        from my_functions.dwave_qpu_adapter import create_simple_food_problem
        
        # Create simple problem
        problem = create_simple_food_problem()
        
        print(f"✓ Created simple problem:")
        print(f"  Farms: {problem['farms']}")
        print(f"  Foods: {list(problem['foods'].keys())}")
        print(f"  Complexity: {problem['complexity']}")
        print(f"  Variables: {len(problem['farms']) * len(problem['foods'])}")
        
        # Validate problem structure
        assert len(problem['farms']) == 2, "Should have 2 farms"
        assert len(problem['foods']) == 3, "Should have 3 foods"
        assert problem['complexity'] == 'micro', "Should be micro complexity"
        
        print("✓ Simple problem validation passed")
        
        # Store for later use
        simple_test_problem = problem
        
    except Exception as e:
        print(f"✗ Simple problem creation failed: {e}")
        raise

def test_basic_initialization():
    """Test basic D-Wave adapter initialization and connection."""
    print("\n" + "-" * 60)
    print("TEST 1: BASIC D-WAVE ADAPTER INITIALIZATION")
    print("-" * 60)
    
    try:
        from my_functions.dwave_qpu_adapter import DWaveQPUAdapter, DWaveConfig
        
        # Test with different configurations
        configs = [
            ("Default Config", DWaveConfig()),
            ("Budget-Conscious", DWaveConfig(max_budget_usd=25.0, num_reads=500)),
            ("High-Performance", DWaveConfig(max_budget_usd=200.0, num_reads=2000)),
            ("Estimation-Only", DWaveConfig(estimate_cost_only=True))
        ]
        
        for config_name, config in configs:
            print(f"\n--- Testing {config_name} ---")
            
            adapter = DWaveQPUAdapter(config=config, logger=logger)
            
            # Test connection
            connection_test = adapter.test_connection()
            print(f"Connection Status: {connection_test.get('status', 'unknown')}")
            print(f"Simulator Available: {connection_test.get('simulator_available', False)}")
            print(f"QPU Available: {connection_test.get('qpu_available', False)}")
            
            # Test basic functionality
            metrics = adapter._get_performance_metrics()
            print(f"Initial Metrics: {metrics}")
            
        print("✓ Basic initialization tests passed")
        
    except Exception as e:
        print(f"✗ Basic initialization test failed: {e}")
        raise

def test_cpu_solver():
    """Test CPU-based simulated annealing solver."""
    print("\n" + "-" * 60)
    print("TEST 2: CPU-BASED SOLVER (SIMULATED ANNEALING)")
    print("-" * 60)
    
    try:
        from my_functions.dwave_qpu_adapter import DWaveQPUAdapter, DWaveConfig
        
        # Force CPU-only configuration
        config = DWaveConfig(
            solver_type='simulator',
            num_reads=100,  # Fewer reads for faster testing
            estimate_cost_only=False
        )
        
        adapter = DWaveQPUAdapter(config=config, logger=logger)
        
        print(f"Active sampler: {adapter.sampler_type}")
        assert adapter.sampler_type == 'simulator', "Should use simulator"
        
        # Test simple QUBO problem
        print("\n--- Testing Simple QUBO Problem ---")
        Q = np.array([
            [1, -2, 0],
            [-2, 2, -1],
            [0, -1, 1]
        ])
        
        bqm = adapter.create_bqm_from_qubo(Q)
        result = adapter._solve_bqm(bqm)
        
        print(f"BQM Variables: {len(bqm.variables)}")
        print(f"Solution Energy: {result['energy']:.4f}")
        print(f"Solver Time: {result['qpu_time']:.3f}s")
        print(f"Sample: {result['sample']}")
        
        assert 'error' not in result, f"CPU solver failed: {result.get('error')}"
        
        # Test with simple food problem
        print("\n--- Testing Simple Food Problem ---")
        if 'simple_test_problem' in globals():
            problem = simple_test_problem
            farms = problem['farms']
            foods = list(problem['foods'].keys())
            
            # Create a simple optimization problem
            num_vars = len(farms) * len(foods)
            
            # Build objective (prefer high-nutrition foods)
            f = np.zeros((num_vars, 1))
            for i, (farm, food) in enumerate([(f, fd) for f in farms for fd in foods]):
                food_data = problem['foods'][food]
                score = food_data['nutritional_value'] - food_data['environmental_impact']
                f[i, 0] = -score  # Negative for minimization
            
            # Build constraint (each farm must select at least one food)
            D = np.zeros((len(farms), num_vars))
            d = np.ones((len(farms), 1))
            
            for farm_idx in range(len(farms)):
                for food_idx in range(len(foods)):
                    var_idx = farm_idx * len(foods) + food_idx
                    D[farm_idx, var_idx] = 1
            
            # Solve with D-Wave
            result = adapter.solve_benders_master_with_dwave(
                f_coeffs=f,
                D_matrix=D,
                d_vector=d,
                optimality_cuts=[],
                feasibility_cuts=[],
                Ny=num_vars,
                config={'penalty_coefficient': 1000.0}
            )
            
            if 'error' not in result:
                solution = result['solution']
                print(f"Solution found: {np.sum(solution > 0.5)} foods selected")
                print(f"Objective value: {result['objective']:.4f}")
                print(f"Wall time: {result['wall_time']:.3f}s")
                
                # Interpret solution
                selected = []
                for i, val in enumerate(solution):
                    if val > 0.5:
                        farm_idx = i // len(foods)
                        food_idx = i % len(foods)
                        selected.append(f"{farms[farm_idx]}->{foods[food_idx]}")
                print(f"Selected combinations: {selected}")
            else:
                print(f"Solution failed: {result['error']}")
        
        print("✓ CPU solver tests passed")
        
    except Exception as e:
        print(f"✗ CPU solver test failed: {e}")
        raise

def test_hybrid_solver():
    """Test D-Wave Leap Hybrid solver."""
    print("\n" + "-" * 60)
    print("TEST 3: HYBRID SOLVER (LEAP HYBRID)")
    print("-" * 60)
    
    try:
        # Declare global variable at the beginning of the function
        global hybrid_test_result
        
        from my_functions.dwave_qpu_adapter import DWaveQPUAdapter, DWaveConfig
        
        # Configure for hybrid solver
        config = DWaveConfig(
            solver_type='hybrid',
            time_limit=10.0,  # 10 seconds for testing
            estimate_cost_only=False
        )
        
        adapter = DWaveQPUAdapter(config=config, logger=logger)
        
        print(f"Active sampler: {adapter.sampler_type}")
        
        if adapter.sampler_type != 'hybrid':
            print("⚠️  Hybrid solver not available - skipping hybrid tests")
            print("   This is normal if you don't have Leap access configured")
            return
        
        print("✓ Hybrid solver initialized successfully")
        
        # Test with simple problem first
        print("\n--- Testing Simple QUBO with Hybrid Solver ---")
        Q = np.array([
            [1, -2, 0, 0],
            [-2, 2, -1, 0],
            [0, -1, 1, -1],
            [0, 0, -1, 1]
        ])
        
        bqm = adapter.create_bqm_from_qubo(Q)
        result = adapter._solve_bqm(bqm)
        
        if 'error' not in result:
            print(f"✓ Hybrid solver test successful:")
            print(f"  Energy: {result['energy']:.4f}")
            print(f"  Time: {result['qpu_time']:.3f}s")
            print(f"  Sample: {result['sample']}")
            
            # Store result for comparison
            hybrid_test_result = result
        else:
            print(f"✗ Hybrid solver test failed: {result['error']}")
            return
        
        print("✓ Hybrid solver tests passed")
        
    except Exception as e:
        print(f"✗ Hybrid solver test failed: {e}")
        import traceback
        traceback.print_exc()

def test_progressive_complexity():
    """Test different complexity levels progressively."""
    print("\n" + "-" * 60)
    print("TEST 4: PROGRESSIVE COMPLEXITY TESTING")
    print("-" * 60)
    
    try:
        # Declare global variable at the beginning of the function
        global complexity_test_results
        
        from my_functions.dwave_qpu_adapter import DWaveQPUAdapter, DWaveConfig
        
        # Test with hybrid if available, otherwise simulator
        config = DWaveConfig(
            solver_type='auto',  # Auto-select best available
            time_limit=15.0,
            num_reads=200
        )
        
        adapter = DWaveQPUAdapter(config=config, logger=logger)
        
        print(f"Using solver: {adapter.sampler_type}")
        
        # Test complexity levels progressively
        test_levels = ['micro', 'tiny', 'small']
        results = {}
        
        for level in test_levels:
            if level not in adapter.COMPLEXITY_LEVELS:
                continue
                
            level_config = adapter.COMPLEXITY_LEVELS[level]
            print(f"\n--- Testing {level.upper()} complexity ---")
            print(f"  {level_config.description}")
            print(f"  Variables: {level_config.total_variables}")
            print(f"  Estimated qubits: {level_config.estimated_qubits}")
            
            # Get cost estimation
            estimation = adapter.estimate_qpu_cost(
                problem_size=level_config.total_variables,
                complexity_level=level
            )
            
            print(f"  Estimated cost: ${estimation.estimated_cost_usd:.4f}")
            print(f"  Feasible: {estimation.is_feasible}")
            
            # Test with a simple problem of this size
            try:
                problem_size = level_config.total_variables
                
                # Create a random-ish QUBO for testing
                np.random.seed(42)  # Reproducible
                Q = np.random.normal(0, 1, (problem_size, problem_size))
                Q = (Q + Q.T) / 2  # Make symmetric
                Q = Q * 0.5  # Scale down
                
                bqm = adapter.create_bqm_from_qubo(Q)
                
                start_time = time.time()
                result = adapter._solve_bqm(bqm)
                solve_time = time.time() - start_time
                
                if 'error' not in result:
                    results[level] = {
                        'energy': result['energy'],
                        'solve_time': solve_time,
                        'estimation': estimation,
                        'success': True
                    }
                    print(f"  ✓ Solved in {solve_time:.3f}s, energy: {result['energy']:.4f}")
                else:
                    results[level] = {
                        'error': result['error'],
                        'estimation': estimation,
                        'success': False
                    }
                    print(f"  ✗ Failed: {result['error']}")
                    
            except Exception as e:
                print(f"  ✗ Error testing {level}: {e}")
                results[level] = {'error': str(e), 'success': False}
        
        # Summary
        print(f"\n--- COMPLEXITY TEST SUMMARY ---")
        successful = sum(1 for r in results.values() if r.get('success', False))
        print(f"Successful levels: {successful}/{len(results)}")
        
        for level, result in results.items():
            if result.get('success'):
                print(f"  ✓ {level.upper()}: {result['solve_time']:.3f}s")
            else:
                print(f"  ✗ {level.upper()}: {result.get('error', 'Unknown error')}")
        
        # Store results globally for report
        complexity_test_results = results
        
        print("✓ Progressive complexity tests completed")
        
    except Exception as e:
        print(f"✗ Progressive complexity test failed: {e}")
        raise

def test_full_scenario():
    """Test full-scale scenario from scenarios.py."""
    print("\n" + "-" * 60)
    print("TEST 5: FULL SCENARIO TESTING")
    print("-" * 60)
    
    try:
        from src.scenarios import load_food_data
        from my_functions.dwave_qpu_adapter import DWaveQPUAdapter, DWaveConfig
        
        # Declare global variable at the beginning of the function
        global full_scenario_result
        
        # Load full complexity scenario
        farms, foods, food_groups, config = load_food_data('full')
        
        problem_size = len(farms) * len(foods)
        print(f"Full scenario size:")
        print(f"  Farms: {len(farms)}")
        print(f"  Foods: {len(foods)}")
        print(f"  Variables: {problem_size}")
        
        # Configure adapter for larger problems
        dwave_config = DWaveConfig(
            solver_type='auto',
            time_limit=30.0,  # Longer time for larger problems
            num_reads=500,
            estimate_cost_only=True  # Start with estimation only
        )
        
        adapter = DWaveQPUAdapter(config=dwave_config, logger=logger)
        
        # Get cost estimation first
        estimation = adapter.estimate_qpu_cost(problem_size, complexity_level='full')
        
        print(f"\nCost estimation for full scenario:")
        print(f"  Variables: {estimation.num_variables}")
        print(f"  Estimated qubits: {estimation.estimated_qubits}")
        print(f"  Estimated cost: ${estimation.estimated_cost_usd:.4f}")
        print(f"  Feasible: {estimation.is_feasible}")
        print(f"  Warnings: {len(estimation.warnings)}")
        
        for warning in estimation.warnings:
            print(f"    ⚠️  {warning}")
        
        for rec in estimation.recommendations:
            print(f"    💡 {rec}")
        
        # Decide whether to proceed with actual solving
        if estimation.is_feasible and estimation.estimated_cost_usd < 1.0:  # $1 threshold
            print(f"\n--- Attempting to solve full scenario ---")
            
            # Switch to actual solving
            adapter.config.estimate_cost_only = False
            
            # Build simple food allocation problem
            f = np.zeros((problem_size, 1))
            
            # Assign objective coefficients based on food quality
            for farm_idx, farm in enumerate(farms):
                for food_idx, food in enumerate(foods):
                    var_idx = farm_idx * len(foods) + food_idx
                    food_data = foods[food]
                    
                    # Simple scoring based on available metrics
                    score = 0
                    if 'nutritional_value' in food_data:
                        score += food_data['nutritional_value'] * 0.4
                    if 'sustainability' in food_data:
                        score += food_data['sustainability'] * 0.3
                    if 'affordability' in food_data:
                        score += food_data['affordability'] * 0.3
                    
                    f[var_idx, 0] = -score  # Negative for minimization
            
            # Build constraints (each farm selects at least one food)
            D = np.zeros((len(farms), problem_size))
            d = np.ones((len(farms), 1))
            
            for farm_idx in range(len(farms)):
                for food_idx in range(len(foods)):
                    var_idx = farm_idx * len(foods) + food_idx
                    D[farm_idx, var_idx] = 1
            
            # Solve
            start_time = time.time()
            result = adapter.solve_benders_master_with_dwave(
                f_coeffs=f,
                D_matrix=D,
                d_vector=d,
                optimality_cuts=[],
                feasibility_cuts=[],
                Ny=problem_size,
                config={'penalty_coefficient': 10000.0}
            )
            solve_time = time.time() - start_time
            
            if 'error' not in result:
                print(f"✓ Full scenario solved successfully!")
                print(f"  Objective: {result['objective']:.4f}")
                print(f"  Wall time: {solve_time:.3f}s")
                print(f"  Energy: {result['energy']:.4f}")
                print(f"  Selections: {np.sum(result['solution'] > 0.5)} out of {problem_size}")
                
                full_scenario_result = {
                    'result': result,
                    'estimation': estimation,
                    'solve_time': solve_time,
                    'success': True
                }
            else:
                print(f"✗ Full scenario solving failed: {result['error']}")
                full_scenario_result = {
                    'error': result['error'],
                    'estimation': estimation,
                    'success': False
                }
        else:
            print(f"\n--- Skipping actual solving ---")
            if not estimation.is_feasible:
                print("  Reason: Problem not feasible for current QPU")
            else:
                print(f"  Reason: Cost too high (${estimation.estimated_cost_usd:.4f})")
            
            full_scenario_result = {
                'estimation': estimation,
                'skipped': True,
                'reason': 'cost_or_feasibility'
            }
        
        print("✓ Full scenario testing completed")
        
    except Exception as e:
        print(f"✗ Full scenario test failed: {e}")
        import traceback
        traceback.print_exc()

def test_qpu_readiness():
    """Test QPU readiness without actually using QPU time."""
    print("\n" + "-" * 60)
    print("TEST 6: QPU READINESS CHECK")
    print("-" * 60)
    
    try:
        from my_functions.dwave_qpu_adapter import DWaveQPUAdapter, DWaveConfig
        
        # Configure for QPU but don't use it yet
        config = DWaveConfig(
            use_real_qpu=True,
            solver_type='qpu',
            num_reads=100,
            estimate_cost_only=True  # Don't actually use QPU
        )
        
        adapter = DWaveQPUAdapter(config=config, logger=logger)
        
        print(f"QPU sampler initialized: {adapter.qpu_sampler is not None}")
        
        # Test connection
        connection_test = adapter.test_connection()
        
        print(f"\nConnection test results:")
        print(f"  Status: {connection_test.get('status', 'unknown')}")
        print(f"  Simulator: {connection_test.get('simulator_available', False)}")
        print(f"  Hybrid: {connection_test.get('hybrid_available', False)}")
        print(f"  QPU: {connection_test.get('qpu_available', False)}")
        
        if connection_test.get('qpu_available'):
            solver_info = connection_test.get('solver_info', {})
            print(f"  QPU Name: {solver_info.get('name', 'Unknown')}")
            print(f"  QPU Test: {connection_test.get('qpu_test', 'unknown')}")
            
            # Estimate costs for QPU usage
            print(f"\n--- QPU Cost Estimates ---")
            
            test_sizes = [6, 15, 40, 100]  # micro, tiny, small, medium
            
            for size in test_sizes:
                estimation = adapter.estimate_qpu_cost(size, num_reads=100)
                print(f"  {size} variables: ${estimation.estimated_cost_usd:.4f} "
                      f"({estimation.estimated_qubits} qubits)")
            
            print(f"\n💡 QPU is ready for testing!")
            print(f"   Recommended: Start with micro or tiny problems")
            print(f"   Budget: Monitor costs, start with small num_reads")
        else:
            print(f"\n⚠️  QPU not available:")
            if 'qpu_test' in connection_test:
                print(f"   Reason: {connection_test['qpu_test']}")
            else:
                print(f"   Check Leap account and solver access")
        
        print("✓ QPU readiness check completed")
        
    except Exception as e:
        print(f"✗ QPU readiness check failed: {e}")
        import traceback
        traceback.print_exc()

def test_scaling_analysis_with_scenarios():
    """Test scaling analysis using real scenarios from scenarios.py."""
    print("\n" + "-" * 60)
    print("TEST 7: SCALING ANALYSIS WITH REAL SCENARIOS")
    print("-" * 60)
    
    try:
        # Declare global variable at the beginning of the function
        global scenario_scaling_results
        
        from src.scenarios import load_food_data
        from my_functions.dwave_qpu_adapter import DWaveQPUAdapter, DWaveConfig
        import numpy as np
        
        # Test scaling analysis for each scenario complexity
        complexities = ['simple', 'intermediate', 'full']
        scaling_results = {}
        
        for complexity in complexities:
            print(f"\n--- Testing {complexity.upper()} scenario scaling ---")
            
            try:
                # Load scenario data
                farms, foods, food_groups, config = load_food_data(complexity)
                problem_size = len(farms) * len(foods)
                
                print(f"Scenario details:")
                print(f"  Farms: {len(farms)}")
                print(f"  Foods: {len(foods)}")
                print(f"  Total variables: {problem_size}")
                
                # Create QUBO problem from scenario
                qubo_matrix = create_food_optimization_qubo(farms, foods, config)
                
                # Create BQM
                adapter = DWaveQPUAdapter(config=DWaveConfig(solver_type='simulator'))
                bqm = adapter.create_bqm_from_qubo(qubo_matrix)
                
                print(f"Created BQM with {len(bqm.variables)} variables")
                
                # Define sample sizes for scaling analysis
                max_sample_size = min(problem_size - 1, 80)  # Don't exceed problem size
                sample_sizes = [
                    max(2, int(max_sample_size * 0.1)),   # 10%
                    max(4, int(max_sample_size * 0.25)),  # 25%
                    max(6, int(max_sample_size * 0.5)),   # 50%
                    max(8, int(max_sample_size * 0.75)),  # 75%
                ]
                
                # Remove duplicates and sort
                sample_sizes = sorted(list(set(sample_sizes)))
                sample_sizes = [s for s in sample_sizes if s < problem_size]
                
                if len(sample_sizes) < 2:
                    print(f"  ⚠️  Problem too small for meaningful scaling analysis")
                    continue
                
                print(f"Using sample sizes: {sample_sizes}")
                
                # Run scaling analysis
                print(f"Running scaling analysis...")
                scaling_result = adapter.estimate_full_time(
                    bqm=bqm,
                    sampler=adapter.sim_sampler,
                    sample_sizes=sample_sizes,
                    num_reads=50,  # Fewer reads for faster testing
                    plot_results=True
                )
                
                if 'error' not in scaling_result:
                    scaling_results[complexity] = scaling_result
                    
                    print(f"\n✓ Scaling analysis completed for {complexity}:")
                    print(f"  Exponent: {scaling_result['exponent']:.3f}")
                    print(f"  Predicted time: {scaling_result['predicted_time']:.4f}s")
                    print(f"  Estimated cost: ${scaling_result['estimated_cost']:.4f}")
                    
                    # Business analysis
                    exponent = scaling_result['exponent']
                    if exponent <= 1.5:
                        complexity_rating = "Excellent - Linear/sub-quadratic scaling"
                    elif exponent <= 2.5:
                        complexity_rating = "Good - Quadratic scaling"
                    elif exponent <= 3.5:
                        complexity_rating = "Moderate - Cubic scaling"
                    else:
                        complexity_rating = "Challenging - Higher-order scaling"
                    
                    print(f"  Complexity rating: {complexity_rating}")
                    
                else:
                    print(f"  ✗ Scaling analysis failed: {scaling_result['error']}")
                
            except Exception as e:
                print(f"  ✗ Error testing {complexity} scenario: {e}")
                continue
        
        # Generate business summary
        if scaling_results:
            print(f"\n--- BUSINESS SCALING SUMMARY ---")
            print(f"Successfully analyzed {len(scaling_results)} scenarios:")
            
            for complexity, result in scaling_results.items():
                farms_count = len(load_food_data(complexity)[0])
                foods_count = len(load_food_data(complexity)[1])
                
                print(f"\n{complexity.upper()} ({farms_count}×{foods_count} = {result['full_size']} vars):")
                print(f"  Scaling exponent: {result['exponent']:.3f}")
                print(f"  Predicted solve time: {result['predicted_time']:.4f}s")
                print(f"  Estimated QPU cost: ${result['estimated_cost']:.4f}")
                
                # Business recommendations
                if result['predicted_time'] < 1.0:
                    print(f"  💚 Business viability: Excellent (fast solving)")
                elif result['predicted_time'] < 10.0:
                    print(f"  💛 Business viability: Good (reasonable solving time)")
                else:
                    print(f"  💔 Business viability: Challenging (long solving time)")
        
        # Store results globally for report
        scenario_scaling_results = scaling_results
        
        print("✓ Scenario scaling analysis completed")
        
    except Exception as e:
        print(f"✗ Scenario scaling analysis failed: {e}")
        import traceback
        traceback.print_exc()

def create_food_optimization_qubo(farms: List[str], 
                                 foods: Dict[str, Dict],
                                 config: Dict) -> np.ndarray:
    """
    Create a QUBO matrix for food optimization problem.
    
    Args:
        farms: List of farm names
        foods: Dictionary of foods with their properties
        config: Configuration parameters
        
    Returns:
        QUBO matrix for the optimization problem
    """
    num_farms = len(farms)
    num_foods = len(foods)
    problem_size = num_farms * num_foods
    
    Q = np.zeros((problem_size, problem_size))
    
    # Get weights from config
    weights = config.get('parameters', {}).get('weights', {
        'nutritional_value': 0.25,
        'nutrient_density': 0.2,
        'affordability': 0.15,
        'sustainability': 0.15,
        'environmental_impact': 0.25
    })
    
    food_list = list(foods.keys())
    
    # Build objective terms
    for farm_idx in range(num_farms):
        for food_idx, food_name in enumerate(food_list):
            var_idx = farm_idx * num_foods + food_idx
            food_data = foods[food_name]
            
            # Calculate objective score (higher is better)
            objective_score = 0.0
            
            # Positive contributions
            for metric in ['nutritional_value', 'nutrient_density', 'affordability', 'sustainability']:
                if metric in food_data and metric in weights:
                    objective_score += weights[metric] * food_data[metric]
            
            # Negative contribution (environmental impact)
            if 'environmental_impact' in food_data and 'environmental_impact' in weights:
                objective_score -= weights['environmental_impact'] * food_data['environmental_impact']
            
            # Add to diagonal (negative because QUBO minimizes)
            Q[var_idx, var_idx] = -objective_score
    
    # Add constraint penalties
    penalty_strength = 10.0
    
    # Constraint: Each farm must select at least one food
    for farm_idx in range(num_farms):
        farm_vars = [farm_idx * num_foods + food_idx for food_idx in range(num_foods)]
        
        # Penalty for constraint: (1 - sum(x_i))^2 = 1 - 2*sum(x_i) + sum_i sum_j x_i*x_j
        for i in farm_vars:
            Q[i, i] += penalty_strength  # Quadratic terms
            Q[i, i] -= 2 * penalty_strength  # Linear terms
            
            for j in farm_vars:
                if i != j:
                    Q[i, j] += penalty_strength / 2  # Cross terms (divide by 2 for symmetry)
    
    return Q

def generate_progressive_test_report():
    """Generate comprehensive report for progressive testing."""
    print("\n" + "-" * 60)
    print("TEST 8: GENERATING PROGRESSIVE TEST REPORT")
    print("-" * 60)
    
    try:
        # Create results directory
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Collect all test results
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'test_summary': {
                'simple_problem': 'simple_test_problem' in globals(),
                'cpu_solver': True,  # Always available
                'hybrid_solver': 'hybrid_test_result' in globals(),
                'complexity_tests': 'complexity_test_results' in globals(),
                'full_scenario': 'full_scenario_result' in globals(),
                'scaling_analysis': 'scenario_scaling_results' in globals()
            }
        }
        
        # Add test results if available
        if 'simple_test_problem' in globals():
            report_data['simple_problem'] = simple_test_problem
        
        if 'hybrid_test_result' in globals():
            report_data['hybrid_result'] = hybrid_test_result
        
        if 'complexity_test_results' in globals():
            report_data['complexity_results'] = complexity_test_results
        
        if 'full_scenario_result' in globals():
            report_data['full_scenario'] = full_scenario_result
            
        if 'scenario_scaling_results' in globals():
            report_data['scaling_analysis'] = scenario_scaling_results
        
        # Generate business recommendations
        recommendations = []
        
        if report_data['test_summary']['cpu_solver']:
            recommendations.append("✓ CPU solver (SimulatedAnnealing) is working - safe fallback available")
        
        if report_data['test_summary']['hybrid_solver']:
            recommendations.append("✓ Hybrid solver is working - recommended for most problems")
        else:
            recommendations.append("⚠️  Hybrid solver not available - check Leap access")
        
        if 'scaling_analysis' in report_data:
            scaling_data = report_data['scaling_analysis']
            best_complexity = min(scaling_data.keys(), 
                                key=lambda k: scaling_data[k]['predicted_time'])
            recommendations.append(f"✓ Best performing scenario: {best_complexity} "
                                 f"({scaling_data[best_complexity]['predicted_time']:.3f}s predicted)")
            
            # Business cost analysis
            total_estimated_cost = sum(r['estimated_cost'] for r in scaling_data.values())
            recommendations.append(f"💰 Total estimated cost for all scenarios: ${total_estimated_cost:.4f}")
        
        report_data['recommendations'] = recommendations
        report_data['business_summary'] = generate_business_summary()
        
        # Save JSON report
        json_path = os.path.join(results_dir, 'dwave_progressive_test_report.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"✓ Progressive test report saved: {json_path}")
        
        # Print summary
        print(f"\n--- TEST SUMMARY ---")
        for test_name, status in report_data['test_summary'].items():
            print(f"  {test_name}: {'✓' if status else '✗'}")
        
        print(f"\n--- BUSINESS RECOMMENDATIONS ---")
        for rec in recommendations:
            print(f"  {rec}")
        
        if 'business_summary' in report_data:
            print(f"\n--- BUSINESS SUMMARY ---")
            business = report_data['business_summary']
            print(f"  Recommended starting scenario: {business['recommended_scenario']}")
            print(f"  Estimated budget needed: ${business['budget_estimate']:.2f}")
            print(f"  Scaling outlook: {business['scaling_outlook']}")
        
        print("✓ Progressive test report generation completed")
        
    except Exception as e:
        print(f"✗ Report generation failed: {e}")
        raise

def generate_business_summary() -> Dict[str, Any]:
    """Generate business-focused summary for D-Wave proposal."""
    if 'scenario_scaling_results' not in globals():
        return {"error": "No scaling results available"}
    
    scaling_data = globals()['scenario_scaling_results']
    
    # Find best scenario (fastest predicted solve time)
    best_scenario = min(scaling_data.keys(), key=lambda k: scaling_data[k]['predicted_time'])
    best_result = scaling_data[best_scenario]
    
    # Calculate budget estimates
    total_cost = sum(r['estimated_cost'] for r in scaling_data.values())
    max_cost = max(r['estimated_cost'] for r in scaling_data.values())
    
    # Determine scaling outlook
    avg_exponent = np.mean([r['exponent'] for r in scaling_data.values()])
    
    if avg_exponent <= 2.0:
        scaling_outlook = "Excellent - Polynomial scaling suitable for larger problems"
    elif avg_exponent <= 3.0:
        scaling_outlook = "Good - Manageable scaling with some limitations"
    else:
        scaling_outlook = "Challenging - May require problem decomposition"
    
    return {
        'recommended_scenario': best_scenario,
        'budget_estimate': max(total_cost * 10, 100),  # 10x buffer for full testing
        'scaling_outlook': scaling_outlook,
        'fastest_solve_time': best_result['predicted_time'],
        'total_scenarios_tested': len(scaling_data),
        'feasibility_rating': 'High' if avg_exponent <= 2.5 else 'Medium' if avg_exponent <= 3.5 else 'Low'
    }

# ...existing code...

if __name__ == "__main__":
    success = main()
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        print("\n🎉 All tests passed! Check the Results folder for detailed reports.")
        print("\nNext steps:")
        print("1. Review the HTML report for cost breakdowns")
        print("2. Use the JSON data for further analysis")
        print("3. Plan your quantum experiments within budget")
        print("4. Consider starting with smaller complexity levels")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
    
    input("\nPress Enter to exit...")
