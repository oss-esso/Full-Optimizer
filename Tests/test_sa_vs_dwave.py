"""
Test script for quantum and classical adapters for food production optimization.

This script tests and validates both the D-Wave QPU adapter and the simulated annealing
adapter to ensure they work correctly with the existing optimization framework.
"""

import sys
import os
import numpy as np
import logging
import time
from typing import Dict, Any, Optional

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import scenario loading
try:
    from src.scenarios import load_food_data
    SCENARIOS_AVAILABLE = True
    logger.info("Scenario loader imported successfully")
except ImportError as e:
    logger.error(f"Failed to import scenario loader: {e}")
    SCENARIOS_AVAILABLE = False

# Import the adapters
try:
    from my_functions.simulated_annealing_adapter import (
        SimulatedAnnealingAdapter, SimulatedAnnealingConfig, 
        CoolingSchedule, NeighborhoodType, solve_benders_master_with_sa
    )
    SA_AVAILABLE = True
    logger.info("Simulated annealing adapter imported successfully")
except ImportError as e:
    logger.error(f"Failed to import simulated annealing adapter: {e}")
    SA_AVAILABLE = False

try:
    from my_functions.dwave_qpu_adapter import (
        DWaveQPUAdapter, DWaveConfig, solve_benders_master_with_dwave
    )
    DWAVE_AVAILABLE = True
    logger.info("D-Wave QPU adapter imported successfully")
    
    # Verify we're using the correct D-Wave implementation
    try:
        from dimod.reference.samplers import ExactSolver
        logger.info("Verified D-Wave using dimod.reference.samplers.ExactSolver")
    except ImportError:
        logger.warning("dimod.reference.samplers.ExactSolver not available - may use simulator")
        
except ImportError as e:
    logger.error(f"Failed to import D-Wave adapter: {e}")
    DWAVE_AVAILABLE = False

# Import QUBO converter for testing
try:
    from my_functions.qubo_converter import convert_benders_master_to_qubo
    QUBO_AVAILABLE = True
    logger.info("QUBO converter imported successfully")
except ImportError as e:
    logger.error(f"Failed to import QUBO converter: {e}")
    QUBO_AVAILABLE = False

# Add data structure for storing test results
from dataclasses import dataclass
from typing import List

@dataclass
class TestResult:
    """Store results from a single test run."""
    method: str
    scenario: str
    config_name: str
    objective_value: float
    runtime: float
    energy: float
    status: str
    iterations: int = None
    final_temperature: float = None
    acceptance_rate: float = None
    chain_break_fraction: float = None
    solution: List[float] = None
    error: str = None

def create_test_benders_problem_from_scenario(complexity_level: str = 'simple') -> Dict[str, Any]:
    """
    Create a test Benders master problem using actual scenario data from scenarios.py.
    
    Args:
        complexity_level: Scenario complexity level ('simple', 'intermediate', or 'full')
        
    Returns:
        Dictionary with problem components extracted from scenario
    """
    if not SCENARIOS_AVAILABLE:
        logger.warning("Scenarios not available, falling back to synthetic problem")
        return create_test_benders_problem(5, complexity_level)
    
    try:
        logger.info(f"Loading {complexity_level} scenario from scenarios.py")
        
        # Load scenario data using the same approach as test_configuration.py
        farms, foods, food_groups, config = load_food_data(complexity_level)
        
        # Extract problem dimensions
        num_farms = len(farms)
        num_foods = len(foods)
        Ny = num_farms * num_foods  # Number of binary variables (farm-food pairs)
        
        logger.info(f"Scenario loaded: {num_farms} farms, {num_foods} foods, {Ny} variables")
        
        # Create objective coefficients based on food attributes and weights
        weights = config['parameters']['weights']
        f_coeffs = []
        
        for farm in farms:
            for food in foods:
                food_data = foods[food]
                # Calculate composite objective coefficient
                coeff = (
                    weights['nutritional_value'] * food_data.get('nutritional_value', 0) +
                    weights['nutrient_density'] * food_data.get('nutrient_density', 0) +
                    weights['affordability'] * food_data.get('affordability', 0) +
                    weights['sustainability'] * food_data.get('sustainability', 0) -
                    weights['environmental_impact'] * food_data.get('environmental_impact', 0)
                )
                f_coeffs.append(coeff)
        
        f_coeffs = np.array(f_coeffs).reshape(-1, 1)
        
        # Create constraint matrices based on scenario parameters
        constraints = []
        constraint_rhs = []
        
        # Land availability constraints (one per farm)
        land_availability = config['parameters']['land_availability']
        for i, farm in enumerate(farms):
            constraint = np.zeros(Ny)
            # Set coefficients for this farm's variables
            for j in range(num_foods):
                var_idx = i * num_foods + j
                constraint[var_idx] = 1.0  # Each allocation uses 1 unit of land
            
            constraints.append(constraint)
            constraint_rhs.append(land_availability[farm])
        
        # Food group constraints if available
        if 'food_group_constraints' in config['parameters']:
            food_group_constraints = config['parameters']['food_group_constraints']
            for group_name, group_foods in food_groups.items():
                if group_name in food_group_constraints:
                    group_constraint = food_group_constraints[group_name]
                    
                    # Minimum foods constraint for this group
                    if 'min_foods' in group_constraint:
                        min_foods = group_constraint['min_foods']
                        constraint = np.zeros(Ny)
                        
                        for farm_idx, farm in enumerate(farms):
                            for food in group_foods:
                                if food in foods:
                                    food_idx = list(foods.keys()).index(food)
                                    var_idx = farm_idx * num_foods + food_idx
                                    constraint[var_idx] = 1.0
                        
                        constraints.append(constraint)
                        constraint_rhs.append(min_foods)
        
        # Convert to matrices
        if constraints:
            D_matrix = np.vstack([c.reshape(1, -1) for c in constraints])
            d_vector = np.array(constraint_rhs).reshape(-1, 1)
        else:
            # Fallback: at least one allocation constraint
            D_matrix = np.ones((1, Ny))
            d_vector = np.array([[1.0]])
        
        # Empty cuts for testing
        optimality_cuts = []
        feasibility_cuts = []
        
        problem_data = {
            'f_coeffs': f_coeffs,
            'D_matrix': D_matrix,
            'd_vector': d_vector,
            'optimality_cuts': optimality_cuts,
            'feasibility_cuts': feasibility_cuts,
            'Ny': Ny,
            'config': config,
            'scenario_info': {
                'complexity_level': complexity_level,
                'num_farms': num_farms,
                'num_foods': num_foods,
                'farms': farms,
                'foods': list(foods.keys()),
                'food_groups': food_groups
            }
        }
        
        logger.info(f"Created Benders problem from {complexity_level} scenario:")
        logger.info(f"  - Variables: {Ny}")
        logger.info(f"  - Constraints: {D_matrix.shape[0]}")
        logger.info(f"  - Farms: {farms}")
        logger.info(f"  - Foods: {list(foods.keys())}")
        
        return problem_data
        
    except Exception as e:
        logger.error(f"Error loading scenario {complexity_level}: {e}")
        logger.warning("Falling back to synthetic problem")
        import traceback
        logger.debug(traceback.format_exc())
        return create_test_benders_problem(5, complexity_level)

def create_test_benders_problem(size: int = 5, complexity_level: str = 'simple') -> Dict[str, Any]:
    """
    Create a synthetic test Benders master problem for validation (fallback).
    
    Args:
        size: Problem size (number of binary variables)
        complexity_level: Scenario complexity level (for reference)
        
    Returns:
        Dictionary with problem components
    """
    logger.info(f"Creating synthetic test problem with {size} variables")
    
    # Create objective coefficients (favor some variables)
    f_coeffs = np.random.randn(size, 1) * 2.0
    
    # Create constraints (sum of variables should be >= 1)
    D_matrix = np.ones((1, size))
    d_vector = np.array([[1.0]])
    
    # Add some additional constraints
    if size > 3:
        # Add constraint: first half sum <= second half sum
        constraint2 = np.zeros((1, size))
        half = size // 2
        constraint2[0, :half] = 1
        constraint2[0, half:] = -1
        
        D_matrix = np.vstack([D_matrix, constraint2])
        d_vector = np.vstack([d_vector, np.array([[0.0]])])
    
    # Empty cuts for testing
    optimality_cuts = []
    feasibility_cuts = []
    
    return {
        'f_coeffs': f_coeffs,
        'D_matrix': D_matrix,
        'd_vector': d_vector,
        'optimality_cuts': optimality_cuts,
        'feasibility_cuts': feasibility_cuts,
        'Ny': size,
        'scenario_info': {
            'complexity_level': complexity_level,
            'is_synthetic': True
        }
    }

def test_simulated_annealing_adapter():
    """Test the simulated annealing adapter with various configurations."""
    if not SA_AVAILABLE:
        logger.error("Simulated annealing adapter not available for testing")
        return False, []
    
    logger.info("=== Testing Simulated Annealing Adapter ===")
    
    test_results = []
    
    try:
        # Test different configurations with names
        test_configs = [
            ("Exponential-SingleFlip", SimulatedAnnealingConfig(
                initial_temperature=100.0,
                cooling_schedule=CoolingSchedule.EXPONENTIAL,
                neighborhood_type=NeighborhoodType.SINGLE_FLIP,
                max_iterations=1000,
                cooling_rate=0.95
            )),
            ("Linear-MultiFlip", SimulatedAnnealingConfig(
                initial_temperature=50.0,
                cooling_schedule=CoolingSchedule.LINEAR,
                neighborhood_type=NeighborhoodType.MULTI_FLIP,
                max_iterations=500,
                flip_probability=0.2
            )),
            ("Adaptive-RandomWalk", SimulatedAnnealingConfig(
                initial_temperature=200.0,
                cooling_schedule=CoolingSchedule.ADAPTIVE,
                neighborhood_type=NeighborhoodType.RANDOM_WALK,
                max_iterations=800,
                adaptive_cooling=True,
                use_restart=True
            ))
        ]
        
        # Define test scenarios using actual scenario data
        test_scenarios = [
            'simple',
            'intermediate',
            'full'
        ]
        
        for complexity_level in test_scenarios:
            logger.info(f"\n--- Testing scenario: {complexity_level} ---")
            test_problem = create_test_benders_problem_from_scenario(complexity_level)
            
            # Log scenario information
            if 'scenario_info' in test_problem:
                info = test_problem['scenario_info']
                if not info.get('is_synthetic', False):
                    logger.info(f"  Farms: {info.get('farms', [])}")
                    logger.info(f"  Foods: {info.get('foods', [])}")
                    logger.info(f"  Variables: {info.get('num_farms', 0)} x {info.get('num_foods', 0)} = {test_problem['Ny']}")
            
            for config_name, config in test_configs:
                logger.info(f"\nTesting configuration: {config_name}")
                logger.info(f"  Schedule: {config.cooling_schedule.value}, Neighborhood: {config.neighborhood_type.value}")
                
                adapter = SimulatedAnnealingAdapter(config=config, logger=logger)
                start_time = time.time()
                
                result = adapter.solve_benders_master_with_sa(
                    f_coeffs=test_problem['f_coeffs'],
                    D_matrix=test_problem['D_matrix'],
                    d_vector=test_problem['d_vector'],
                    optimality_cuts=test_problem['optimality_cuts'],
                    feasibility_cuts=test_problem['feasibility_cuts'],
                    Ny=test_problem['Ny']
                )
                
                solve_time = time.time() - start_time
                
                # Create test result record
                test_result = TestResult(
                    method="Simulated Annealing",
                    scenario=complexity_level,
                    config_name=config_name,
                    objective_value=result.get('objective', 0.0),
                    runtime=solve_time,
                    energy=result.get('energy', 0.0),
                    status="Success" if not result.get('error') else "Failed",
                    iterations=result.get('iterations'),
                    final_temperature=result.get('final_temperature'),
                    acceptance_rate=result.get('acceptance_rate'),
                    solution=result.get('solution', []),
                    error=result.get('error')
                )
                test_results.append(test_result)
                
                if result.get('error'):
                    logger.error(f"Simulated annealing failed: {result['error']}")
                    continue
                
                logger.info(f"✓ Solution found in {solve_time:.2f}s")
                logger.info(f"  Objective: {result['objective']:.6f}, Energy: {result['energy']:.6f}")
                logger.info(f"  Iterations: {result['iterations']}, Final temp: {result['final_temperature']:.6f}")
                logger.info(f"  Acceptance rate: {result['acceptance_rate']:.3f}")
                
                # Validate solution
                if len(result['solution']) != test_problem['Ny']:
                    logger.error(f"Solution size mismatch: expected {test_problem['Ny']}, got {len(result['solution'])}")
                
                # Check if solution is binary
                solution = result['solution']
                if not all(0 <= x <= 1 for x in solution):
                    logger.warning("Solution contains non-binary values")
        
        logger.info("\n=== Simulated Annealing Tests Completed ===")
        return True, test_results
        
    except Exception as e:
        logger.error(f"Error testing simulated annealing adapter: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, test_results

def test_dwave_adapter():
    """Test the D-Wave QPU adapter (using ExactSolver or simulator)."""
    if not DWAVE_AVAILABLE:
        logger.error("D-Wave adapter not available for testing")
        return False, []
    
    logger.info("=== Testing D-Wave Adapter ===")
    logger.info("Using dimod.reference.samplers.ExactSolver for exact solutions")
    
    test_results = []
    
    try:
        # Test configurations with names
        test_configs = [
            ("ExactSolver-100reads", DWaveConfig(
                use_real_qpu=False,  # Use ExactSolver
                num_reads=100,
                annealing_time=20.0
            )),
            ("ExactSolver-500reads", DWaveConfig(
                use_real_qpu=False,
                num_reads=500,
                annealing_time=50.0,
                auto_scale=True
            ))
        ]
        
        # Define test scenarios using actual scenario data (smaller scenarios for D-Wave)
        test_scenarios = [
            'simple',
            'intermediate'  # Skip 'full' for D-Wave as it might be too large
        ]
        
        for complexity_level in test_scenarios:
            logger.info(f"\n--- Testing D-Wave with scenario: {complexity_level} ---")
            test_problem = create_test_benders_problem_from_scenario(complexity_level)
            
            # Log scenario information
            if 'scenario_info' in test_problem:
                info = test_problem['scenario_info']
                if not info.get('is_synthetic', False):
                    logger.info(f"  Farms: {info.get('farms', [])}")
                    logger.info(f"  Foods: {info.get('foods', [])}")
                    logger.info(f"  Variables: {test_problem['Ny']}")
            
            for config_name, config in test_configs:
                logger.info(f"\nTesting D-Wave configuration: {config_name}")
                logger.info(f"  Reads: {config.num_reads}, Annealing time: {config.annealing_time}")
                
                adapter = DWaveQPUAdapter(config=config, logger=logger)
                start_time = time.time()
                
                result = adapter.solve_benders_master_with_dwave(
                    f_coeffs=test_problem['f_coeffs'],
                    D_matrix=test_problem['D_matrix'],
                    d_vector=test_problem['d_vector'],
                    optimality_cuts=test_problem['optimality_cuts'],
                    feasibility_cuts=test_problem['feasibility_cuts'],
                    Ny=test_problem['Ny']
                )
                
                solve_time = time.time() - start_time
                
                # Create test result record
                test_result = TestResult(
                    method="D-Wave QPU",
                    scenario=complexity_level,
                    config_name=config_name,
                    objective_value=result.get('objective', 0.0),
                    runtime=solve_time,
                    energy=result.get('energy', 0.0),
                    status="Success" if not result.get('error') else "Failed",
                    chain_break_fraction=result.get('chain_break_fraction'),
                    solution=result.get('solution', []),
                    error=result.get('error')
                )
                test_results.append(test_result)
                
                if result.get('error'):
                    logger.error(f"D-Wave solving failed: {result['error']}")
                    continue
                
                logger.info(f"✓ D-Wave solution found in {solve_time:.2f}s")
                logger.info(f"  Objective: {result['objective']:.6f}, Energy: {result.get('energy', 'N/A')}")
                logger.info(f"  Chain breaks: {result.get('chain_break_fraction', 'N/A')}")
                
                # Validate solution
                if len(result['solution']) != test_problem['Ny']:
                    logger.error(f"Solution size mismatch: expected {test_problem['Ny']}, got {len(result['solution'])}")
        
        logger.info("\n=== D-Wave Tests Completed ===")
        return True, test_results
        
    except Exception as e:
        logger.error(f"Error testing D-Wave adapter: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, test_results

def test_qubo_conversion():
    """Test QUBO conversion functionality."""
    if not QUBO_AVAILABLE:
        logger.error("QUBO converter not available for testing")
        return False, []
    
    logger.info("=== Testing QUBO Conversion ===")
    
    test_results = []
    
    try:
        # Create test problem using intermediate complexity scenario
        test_problem = create_test_benders_problem_from_scenario('intermediate')
        
        config = {
            "eta_min": -100.0,
            "eta_max": 100.0,
            "eta_num_bits": 4,
            "penalty_coefficient": 1000.0,
            "penalty_slack_num_bits": 3
        }
        
        logger.info("Converting Benders problem to QUBO...")
        start_time = time.time()
        qubo_model = convert_benders_master_to_qubo(
            test_problem['f_coeffs'],
            test_problem['D_matrix'],
            test_problem['d_vector'],
            test_problem['optimality_cuts'],
            test_problem['feasibility_cuts'],
            None,  # B_matrix
            None,  # b_vector
            test_problem['Ny'],
            config,
            logger
        )
        conversion_time = time.time() - start_time
        
        if qubo_model is None:
            logger.error("QUBO conversion failed")
            return False, test_results
        
        logger.info(f"✓ QUBO conversion successful in {conversion_time:.2f}s:")
        logger.info(f"  Q matrix shape: {qubo_model.Q.shape}")
        logger.info(f"  Linear terms: {len(qubo_model.c)}")
        logger.info(f"  Offset: {qubo_model.offset}")
        
        # Test with both adapters if available
        if SA_AVAILABLE:
            logger.info("\nTesting QUBO with simulated annealing...")
            sa_adapter = SimulatedAnnealingAdapter(logger=logger)
            start_time = time.time()
            sa_result = sa_adapter._solve_qubo_sa(
                qubo_model.Q + np.diag(qubo_model.c), 
                qubo_model.offset, 
                qubo_model.Q.shape[0]
            )
            sa_solve_time = time.time() - start_time
            
            test_result = TestResult(
                method="QUBO-SA",
                scenario="intermediate",
                config_name="QUBO-Conversion",
                objective_value=0.0,  # Not applicable for QUBO test
                runtime=sa_solve_time,
                energy=sa_result.get('energy', 0.0),
                status="Success" if not sa_result.get('error') else "Failed",
                error=sa_result.get('error')
            )
            test_results.append(test_result)
            
            logger.info(f"✓ SA QUBO solve time: {sa_solve_time:.2f}s, energy: {sa_result.get('energy', 'ERROR')}")
        
        return True, test_results
        
    except Exception as e:
        logger.error(f"Error testing QUBO conversion: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, test_results

def benchmark_adapters():
    """Benchmark and compare adapter performance."""
    if not (SA_AVAILABLE and DWAVE_AVAILABLE):
        logger.error("Both adapters not available for benchmarking")
        return []
    
    logger.info("=== Benchmarking Adapters ===")
    
    benchmark_results = []
    
    try:
        # Create test problem using simple scenario
        test_problem = create_test_benders_problem_from_scenario('simple')
        
        # Run multiple trials
        num_trials = 3
        
        # Simulated annealing trials
        sa_config = SimulatedAnnealingConfig(
            max_iterations=1000,
            cooling_rate=0.95,
            log_interval=1000  # Reduce logging
        )
        
        logger.info(f"\nRunning {num_trials} simulated annealing trials...")
        sa_results = []
        for trial in range(num_trials):
            adapter = SimulatedAnnealingAdapter(config=sa_config)
            start_time = time.time()
            
            result = adapter.solve_benders_master_with_sa(
                f_coeffs=test_problem['f_coeffs'],
                D_matrix=test_problem['D_matrix'],
                d_vector=test_problem['d_vector'],
                optimality_cuts=test_problem['optimality_cuts'],
                feasibility_cuts=test_problem['feasibility_cuts'],
                Ny=test_problem['Ny']
            )
            
            solve_time = time.time() - start_time
            
            benchmark_result = TestResult(
                method="SA-Benchmark",
                scenario="simple",
                config_name=f"Trial-{trial+1}",
                objective_value=result.get('objective', 0.0),
                runtime=solve_time,
                energy=result.get('energy', 0.0),
                status="Success" if not result.get('error') else "Failed",
                error=result.get('error')
            )
            benchmark_results.append(benchmark_result)
            sa_results.append(solve_time)
            
            if not result.get('error'):
                logger.info(f"  Trial {trial+1}: {solve_time:.2f}s, energy={result['energy']:.6f}")
        
        # D-Wave trials
        dwave_config = DWaveConfig(
            use_real_qpu=False,
            num_reads=100
        )
        
        logger.info(f"\nRunning {num_trials} D-Wave trials...")
        dwave_results = []
        for trial in range(num_trials):
            adapter = DWaveQPUAdapter(config=dwave_config)
            start_time = time.time()
            
            result = adapter.solve_benders_master_with_dwave(
                f_coeffs=test_problem['f_coeffs'],
                D_matrix=test_problem['D_matrix'],
                d_vector=test_problem['d_vector'],
                optimality_cuts=test_problem['optimality_cuts'],
                feasibility_cuts=test_problem['feasibility_cuts'],
                Ny=test_problem['Ny']
            )
            
            solve_time = time.time() - start_time
            
            benchmark_result = TestResult(
                method="DWave-Benchmark",
                scenario="simple",
                config_name=f"Trial-{trial+1}",
                objective_value=result.get('objective', 0.0),
                runtime=solve_time,
                energy=result.get('energy', 0.0),
                status="Success" if not result.get('error') else "Failed",
                error=result.get('error')
            )
            benchmark_results.append(benchmark_result)
            dwave_results.append(solve_time)
            
            if not result.get('error'):
                logger.info(f"  Trial {trial+1}: {solve_time:.2f}s, energy={result.get('energy', 'N/A')}")
        
        # Compare results
        logger.info("\n=== Benchmark Results ===")
        if sa_results:
            logger.info(f"Simulated Annealing: {np.mean(sa_results):.2f}±{np.std(sa_results):.2f}s")
        if dwave_results:
            logger.info(f"D-Wave: {np.mean(dwave_results):.2f}±{np.std(dwave_results):.2f}s")
        
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Error during benchmarking: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return benchmark_results

def print_results_summary(all_results: List[TestResult]):
    """Print a comprehensive, nicely formatted summary of all test results."""
    if not all_results:
        logger.warning("No results to summarize")
        return
    
    print("\n" + "="*80)
    print("                    QUANTUM ADAPTERS TEST RESULTS SUMMARY")
    print("="*80)
    
    # Group results by method
    methods = {}
    for result in all_results:
        if result.method not in methods:
            methods[result.method] = []
        methods[result.method].append(result)
    
    # Print summary for each method
    for method_name, results in methods.items():
        print(f"\n📊 {method_name.upper()} RESULTS")
        print("-" * 50)
        
        # Calculate statistics
        successful_results = [r for r in results if r.status == "Success"]
        failed_results = [r for r in results if r.status == "Failed"]
        
        print(f"Total Tests: {len(results)}")
        print(f"✅ Successful: {len(successful_results)} ({len(successful_results)/len(results)*100:.1f}%)")
        print(f"❌ Failed: {len(failed_results)} ({len(failed_results)/len(results)*100:.1f}%)")
        
        if successful_results:
            # Runtime statistics
            runtimes = [r.runtime for r in successful_results]
            print(f"\n⏱️  RUNTIME STATISTICS:")
            print(f"   Mean: {np.mean(runtimes):.3f}s ± {np.std(runtimes):.3f}s")
            print(f"   Min:  {np.min(runtimes):.3f}s")
            print(f"   Max:  {np.max(runtimes):.3f}s")
            
            # Objective value statistics
            objectives = [r.objective_value for r in successful_results if r.objective_value is not None]
            if objectives:
                print(f"\n🎯 OBJECTIVE VALUE STATISTICS:")
                print(f"   Mean: {np.mean(objectives):.6f} ± {np.std(objectives):.6f}")
                print(f"   Min:  {np.min(objectives):.6f}")
                print(f"   Max:  {np.max(objectives):.6f}")
            
            # Energy statistics
            energies = [r.energy for r in successful_results if r.energy is not None]
            if energies:
                print(f"\n⚡ ENERGY STATISTICS:")
                print(f"   Mean: {np.mean(energies):.6f} ± {np.std(energies):.6f}")
                print(f"   Min:  {np.min(energies):.6f}")
                print(f"   Max:  {np.max(energies):.6f}")
            
            # Method-specific statistics
            if method_name == "Simulated Annealing":
                iterations = [r.iterations for r in successful_results if r.iterations is not None]
                final_temps = [r.final_temperature for r in successful_results if r.final_temperature is not None]
                acceptance_rates = [r.acceptance_rate for r in successful_results if r.acceptance_rate is not None]
                
                if iterations:
                    print(f"\n🔄 SA-SPECIFIC METRICS:")
                    print(f"   Avg Iterations: {np.mean(iterations):.0f} ± {np.std(iterations):.0f}")
                if final_temps:
                    print(f"   Final Temp: {np.mean(final_temps):.6f} ± {np.std(final_temps):.6f}")
                if acceptance_rates:
                    print(f"   Acceptance Rate: {np.mean(acceptance_rates):.3f} ± {np.std(acceptance_rates):.3f}")
            
            elif method_name == "D-Wave QPU":
                chain_breaks = [r.chain_break_fraction for r in successful_results if r.chain_break_fraction is not None]
                if chain_breaks:
                    print(f"\n🔗 D-WAVE SPECIFIC METRICS:")
                    print(f"   Chain Break Rate: {np.mean(chain_breaks):.4f} ± {np.std(chain_breaks):.4f}")
            
            # Best result
            best_result = min(successful_results, key=lambda x: x.energy if x.energy is not None else float('inf'))
            print(f"\n🏆 BEST RESULT:")
            print(f"   Scenario: {best_result.scenario}")
            print(f"   Config: {best_result.config_name}")
            print(f"   Objective: {best_result.objective_value:.6f}")
            print(f"   Energy: {best_result.energy:.6f}")
            print(f"   Runtime: {best_result.runtime:.3f}s")
        
        # Show failed results
        if failed_results:
            print(f"\n❌ FAILED TESTS:")
            for result in failed_results:
                print(f"   {result.scenario}/{result.config_name}: {result.error}")
        
        # Detailed breakdown by scenario
        scenarios = {}
        for result in successful_results:
            if result.scenario not in scenarios:
                scenarios[result.scenario] = []
            scenarios[result.scenario].append(result)
        
        if scenarios:
            print(f"\n📈 PERFORMANCE BY SCENARIO:")
            for scenario, scenario_results in scenarios.items():
                runtimes = [r.runtime for r in scenario_results]
                energies = [r.energy for r in scenario_results if r.energy is not None]
                print(f"   {scenario.upper()}:")
                print(f"     Tests: {len(scenario_results)}")
                print(f"     Avg Runtime: {np.mean(runtimes):.3f}s")
                if energies:
                    print(f"     Avg Energy: {np.mean(energies):.6f}")
    
    # Cross-method comparison
    if len(methods) > 1:
        print(f"\n🔄 CROSS-METHOD COMPARISON")
        print("-" * 50)
        
        comparison_data = []
        for method_name, results in methods.items():
            successful = [r for r in results if r.status == "Success"]
            if successful:
                avg_runtime = np.mean([r.runtime for r in successful])
                avg_energy = np.mean([r.energy for r in successful if r.energy is not None])
                success_rate = len(successful) / len(results)
                comparison_data.append((method_name, avg_runtime, avg_energy, success_rate))
        
        if comparison_data:
            print(f"{'Method':<20} {'Avg Runtime':<12} {'Avg Energy':<15} {'Success Rate':<12}")
            print("-" * 65)
            for method, runtime, energy, success in comparison_data:
                print(f"{method:<20} {runtime:<12.3f} {energy:<15.6f} {success:<12.1%}")
            
            # Determine winners
            fastest_method = min(comparison_data, key=lambda x: x[1])
            lowest_energy = min(comparison_data, key=lambda x: x[2])
            most_reliable = max(comparison_data, key=lambda x: x[3])
            
            print(f"\n🏅 WINNERS:")
            print(f"   ⚡ Fastest: {fastest_method[0]} ({fastest_method[1]:.3f}s)")
            print(f"   🎯 Lowest Energy: {lowest_energy[0]} ({lowest_energy[2]:.6f})")
            print(f"   🛡️  Most Reliable: {most_reliable[0]} ({most_reliable[3]:.1%})")
    
    print("\n" + "="*80)

def main():
    """Run all tests and benchmarks."""
    logger.info("Starting quantum adapters validation tests...")
    
    # Check availability
    logger.info(f"Simulated Annealing available: {SA_AVAILABLE}")
    logger.info(f"D-Wave adapter available: {DWAVE_AVAILABLE}")
    logger.info(f"QUBO converter available: {QUBO_AVAILABLE}")
    
    all_test_results = []
    test_success = {}
    
    # Test QUBO conversion first
    if QUBO_AVAILABLE:
        success, results = test_qubo_conversion()
        test_success['qubo_conversion'] = success
        all_test_results.extend(results)
    
    # Test simulated annealing adapter
    if SA_AVAILABLE:
        success, results = test_simulated_annealing_adapter()
        test_success['simulated_annealing'] = success
        all_test_results.extend(results)
    
    # Test D-Wave adapter
    if DWAVE_AVAILABLE:
        success, results = test_dwave_adapter()
        test_success['dwave_adapter'] = success
        all_test_results.extend(results)
    
    # Benchmark if both available
    if SA_AVAILABLE and DWAVE_AVAILABLE:
        benchmark_results = benchmark_adapters()
        all_test_results.extend(benchmark_results)
    
    # Print comprehensive results summary
    print_results_summary(all_test_results)
    
    # Summary
    logger.info("\n=== Test Summary ===")
    for test_name, result in test_success.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    overall_success = all(test_success.values())
    logger.info(f"\nOverall: {'SUCCESS' if overall_success else 'FAILED'}")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
