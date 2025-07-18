"""
Column Generation Integration Utilities

This module provides utilities for integrating the Column Generation approach
with the existing EPDT system, test runner, and user interfaces.

Features:
- Easy integration with existing test scenarios
- Performance comparison between heuristic and optimality bounds
- Configuration management for different solution approaches
- Results analysis and reporting
- Benchmark generation and evaluation
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import json
import logging
from pathlib import Path

# Import EPDT components
from .epdt_data_structures import Order, Vehicle, Route, Solution
from .column_generation import (
    ColumnGenerationSolver, ColumnGenerationConfig, 
    solve_epdt_with_column_generation, ColumnGenerationResult
)
from .column_generation_test import run_column_generation_tests

# Try to import heuristic for comparison
try:
    from .first_level import l1_heuristic
    HEURISTIC_AVAILABLE = True
except ImportError:
    HEURISTIC_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from benchmarking different solution approaches."""
    
    problem_name: str
    problem_size: Dict[str, int]  # orders, vehicles, tasks
    
    # Column Generation results
    cg_feasible: bool
    cg_lower_bound: float
    cg_upper_bound: float
    cg_optimality_gap: float
    cg_solve_time: float
    cg_iterations: int
    cg_columns_generated: int
    
    # Heuristic results (if available)
    heuristic_feasible: bool = False
    heuristic_objective: float = float('inf')
    heuristic_solve_time: float = 0.0
    heuristic_vehicles_used: int = 0
    heuristic_unassigned_orders: int = 0
    
    # Quality comparison
    optimality_gap_vs_heuristic: float = float('inf')
    heuristic_quality_ratio: float = 0.0  # heuristic_obj / lower_bound
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'problem_name': self.problem_name,
            'problem_size': self.problem_size,
            'column_generation': {
                'feasible': self.cg_feasible,
                'lower_bound': self.cg_lower_bound,
                'upper_bound': self.cg_upper_bound,
                'optimality_gap': self.cg_optimality_gap,
                'solve_time': self.cg_solve_time,
                'iterations': self.cg_iterations,
                'columns_generated': self.cg_columns_generated
            },
            'heuristic': {
                'feasible': self.heuristic_feasible,
                'objective': self.heuristic_objective,
                'solve_time': self.heuristic_solve_time,
                'vehicles_used': self.heuristic_vehicles_used,
                'unassigned_orders': self.heuristic_unassigned_orders
            },
            'comparison': {
                'optimality_gap_vs_heuristic': self.optimality_gap_vs_heuristic,
                'heuristic_quality_ratio': self.heuristic_quality_ratio
            }
        }


class EPDTBenchmarkSuite:
    """
    Comprehensive benchmarking suite for EPDT solution approaches.
    
    This class coordinates testing and comparison between the Column Generation
    approach and the EPDT heuristic on various problem instances.
    """
    
    def __init__(self, config: ColumnGenerationConfig = None):
        """
        Initialize benchmark suite.
        
        Args:
            config: Configuration for Column Generation algorithm
        """
        self.config = config or ColumnGenerationConfig()
        self.results: List[BenchmarkResult] = []
    
    def benchmark_problem(self, problem_name: str, orders: List[Order], 
                         vehicles: List[Vehicle]) -> BenchmarkResult:
        """
        Benchmark both approaches on a specific problem.
        
        Args:
            problem_name: Name identifier for the problem
            orders: List of orders to schedule
            vehicles: List of available vehicles
            
        Returns:
            BenchmarkResult with performance comparison
        """
        logger.info(f"🔬 Benchmarking problem: {problem_name}")
        
        # Problem size metrics
        total_tasks = sum(len(order.get_all_tasks()) for order in orders)
        problem_size = {
            'orders': len(orders),
            'vehicles': len(vehicles),
            'tasks': total_tasks
        }
        
        # Initialize result
        result = BenchmarkResult(
            problem_name=problem_name,
            problem_size=problem_size,
            cg_feasible=False,
            cg_lower_bound=float('inf'),
            cg_upper_bound=float('inf'),
            cg_optimality_gap=float('inf'),
            cg_solve_time=0.0,
            cg_iterations=0,
            cg_columns_generated=0
        )
        
        # Run Column Generation
        logger.info("   🔍 Running Column Generation...")
        cg_start_time = time.time()
        try:
            cg_result = solve_epdt_with_column_generation(orders, vehicles, self.config)
            result.cg_solve_time = time.time() - cg_start_time
            
            result.cg_feasible = cg_result.feasible
            result.cg_lower_bound = cg_result.lower_bound
            result.cg_upper_bound = cg_result.upper_bound
            result.cg_optimality_gap = cg_result.optimality_gap
            
            if cg_result.statistics:
                result.cg_iterations = cg_result.statistics.iterations
                result.cg_columns_generated = cg_result.statistics.total_columns_generated
            
            logger.info(f"      ✅ CG completed: feasible={cg_result.feasible}")
            if cg_result.feasible:
                logger.info(f"         Lower bound: {cg_result.lower_bound:.2f}")
                logger.info(f"         Upper bound: {cg_result.upper_bound:.2f}")
                logger.info(f"         Gap: {cg_result.optimality_gap:.6f}")
            
        except Exception as e:
            logger.error(f"      ❌ Column Generation failed: {str(e)}")
            result.cg_solve_time = time.time() - cg_start_time
        
        # Run Heuristic if available
        if HEURISTIC_AVAILABLE:
            logger.info("   🚀 Running EPDT Heuristic...")
            heuristic_start_time = time.time()
            try:
                heuristic_params = {
                    'tabu_tenure': 10,
                    'M1': 50,
                    'M2': 200,
                    'exploration_strategy': 'vnd'
                }
                
                heuristic_solution = l1_heuristic(orders, vehicles, heuristic_params)
                result.heuristic_solve_time = time.time() - heuristic_start_time
                
                if heuristic_solution:
                    result.heuristic_feasible = True
                    
                    # Calculate objective value (simplified)
                    from .second_level import calculate_z2_score
                    total_cost = 0.0
                    vehicles_used = 0
                    
                    for vehicle_id, route in heuristic_solution.routes.items():
                        if route.tasks:
                            vehicles_used += 1
                            total_cost += calculate_z2_score(route)
                    
                    result.heuristic_objective = total_cost
                    result.heuristic_vehicles_used = vehicles_used
                    
                    # Count unassigned orders
                    if hasattr(heuristic_solution, 'unassigned_orders'):
                        result.heuristic_unassigned_orders = len(heuristic_solution.unassigned_orders)
                    
                    logger.info(f"      ✅ Heuristic completed: objective={total_cost:.2f}")
                    logger.info(f"         Vehicles used: {vehicles_used}")
                    logger.info(f"         Unassigned orders: {result.heuristic_unassigned_orders}")
                
            except Exception as e:
                logger.error(f"      ❌ Heuristic failed: {str(e)}")
                result.heuristic_solve_time = time.time() - heuristic_start_time
        else:
            logger.info("   ⚠️  EPDT Heuristic not available")
        
        # Calculate comparison metrics
        if result.cg_feasible and result.heuristic_feasible:
            if result.cg_lower_bound > 0:
                result.heuristic_quality_ratio = result.heuristic_objective / result.cg_lower_bound
            
            if result.heuristic_objective != float('inf') and result.cg_upper_bound != float('inf'):
                result.optimality_gap_vs_heuristic = abs(
                    result.heuristic_objective - result.cg_upper_bound
                ) / max(result.heuristic_objective, 1e-10)
        
        self.results.append(result)
        return result
    
    def benchmark_scenario(self, scenario_name: str) -> BenchmarkResult:
        """
        Benchmark a predefined scenario.
        
        Args:
            scenario_name: Name of scenario to benchmark
            
        Returns:
            BenchmarkResult for the scenario
        """
        logger.info(f"🎯 Benchmarking scenario: {scenario_name}")
        
        # Load scenario
        if scenario_name == "furgoni":
            orders, vehicles = self._load_furgoni_scenario()
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        return self.benchmark_problem(scenario_name, orders, vehicles)
    
    def _load_furgoni_scenario(self) -> Tuple[List[Order], List[Vehicle]]:
        """Load the furgoni scenario for benchmarking."""
        try:
            # Try to import scenario
            from ...src.moda_scenarios import create_furgoni_scenario
            from ..data_adapter import convert_instance_to_epdt_input
            
            # Create scenario
            instance = create_furgoni_scenario()
            
            # Convert to EPDT format
            orders, vehicles, _ = convert_instance_to_epdt_input(instance)
            
            return orders, vehicles
            
        except ImportError:
            logger.error("❌ Could not load furgoni scenario")
            # Return simple test problem
            return self._create_simple_test_problem()
    
    def _create_simple_test_problem(self) -> Tuple[List[Order], List[Vehicle]]:
        """Create a simple test problem for demonstration."""
        from .epdt_data_structures import Task, TaskType
        
        # Create simple orders
        orders = []
        for i in range(5):
            pickup_task = Task(
                id=f"P{i}",
                location_id=f"LOC_P{i}",
                task_type=TaskType.PICKUP,
                order_id=f"O{i}",
                lat=float(i),
                lon=0.0,
                service_time=10.0,
                demand=100.0,
                volume=1.0
            )
            
            delivery_task = Task(
                id=f"D{i}",
                location_id=f"LOC_D{i}",
                task_type=TaskType.DELIVERY,
                order_id=f"O{i}",
                lat=float(i),
                lon=1.0,
                service_time=10.0,
                demand=-100.0,
                volume=-1.0
            )
            
            order = Order(
                id=f"O{i}",
                pickup_tasks=[pickup_task],
                delivery_tasks=[delivery_task],
                is_mandatory=True
            )
            orders.append(order)
        
        # Create vehicles
        vehicles = []
        for i in range(3):
            vehicle = Vehicle(
                id=f"V{i}",
                depot_id="DEPOT",
                weight_capacity=1000.0,
                volume_capacity=10.0,
                cost_per_km=1.0
            )
            vehicles.append(vehicle)
        
        return orders, vehicles
    
    def save_results(self, filename: str):
        """Save benchmark results to JSON file."""
        results_data = [result.to_dict() for result in self.results]
        
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'benchmark_results': results_data,
                'summary': self._generate_summary()
            }, f, indent=2)
        
        logger.info(f"💾 Results saved to {output_path}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from all results."""
        if not self.results:
            return {}
        
        cg_successful = sum(1 for r in self.results if r.cg_feasible)
        heuristic_successful = sum(1 for r in self.results if r.heuristic_feasible)
        
        avg_cg_time = sum(r.cg_solve_time for r in self.results) / len(self.results)
        avg_heuristic_time = sum(r.heuristic_solve_time for r in self.results) / len(self.results)
        
        quality_ratios = [r.heuristic_quality_ratio for r in self.results 
                         if r.heuristic_quality_ratio > 0 and r.heuristic_quality_ratio != float('inf')]
        avg_quality_ratio = sum(quality_ratios) / len(quality_ratios) if quality_ratios else 0.0
        
        return {
            'total_problems': len(self.results),
            'column_generation_success_rate': cg_successful / len(self.results),
            'heuristic_success_rate': heuristic_successful / len(self.results),
            'average_cg_solve_time': avg_cg_time,
            'average_heuristic_solve_time': avg_heuristic_time,
            'average_heuristic_quality_ratio': avg_quality_ratio,
            'speedup_factor': avg_cg_time / avg_heuristic_time if avg_heuristic_time > 0 else 0.0
        }
    
    def print_summary(self):
        """Print a comprehensive summary of benchmark results."""
        if not self.results:
            logger.info("📊 No benchmark results to summarize")
            return
        
        logger.info("\n" + "="*60)
        logger.info("📊 EPDT BENCHMARK SUMMARY")
        logger.info("="*60)
        
        summary = self._generate_summary()
        
        logger.info(f"🎯 Total problems benchmarked: {summary['total_problems']}")
        logger.info(f"✅ Column Generation success rate: {summary['column_generation_success_rate']:.1%}")
        logger.info(f"✅ Heuristic success rate: {summary['heuristic_success_rate']:.1%}")
        
        logger.info(f"\n⏱️  Performance Comparison:")
        logger.info(f"   Average CG solve time: {summary['average_cg_solve_time']:.2f} seconds")
        logger.info(f"   Average Heuristic solve time: {summary['average_heuristic_solve_time']:.2f} seconds")
        
        if summary['speedup_factor'] > 0:
            if summary['speedup_factor'] < 1:
                logger.info(f"   🚀 Heuristic is {1/summary['speedup_factor']:.1f}x faster")
            else:
                logger.info(f"   🐌 Column Generation is {summary['speedup_factor']:.1f}x slower")
        
        if summary['average_heuristic_quality_ratio'] > 0:
            logger.info(f"\n📈 Solution Quality:")
            logger.info(f"   Average heuristic quality ratio: {summary['average_heuristic_quality_ratio']:.3f}")
            if summary['average_heuristic_quality_ratio'] > 1.1:
                logger.info(f"   📊 Heuristic solutions are on average {summary['average_heuristic_quality_ratio']:.1f}x the optimal lower bound")
            else:
                logger.info(f"   🎯 Heuristic solutions are very close to optimal")
        
        logger.info(f"\n📋 Individual Results:")
        for result in self.results:
            logger.info(f"   {result.problem_name}:")
            if result.cg_feasible:
                logger.info(f"      CG: LB={result.cg_lower_bound:.1f}, UB={result.cg_upper_bound:.1f}, Gap={result.cg_optimality_gap:.4f}")
            if result.heuristic_feasible:
                logger.info(f"      Heuristic: Obj={result.heuristic_objective:.1f}, Vehicles={result.heuristic_vehicles_used}")


def run_comprehensive_benchmark(scenarios: List[str] = None, 
                              config: ColumnGenerationConfig = None,
                              save_results: bool = True) -> List[BenchmarkResult]:
    """
    Run comprehensive benchmark comparing Column Generation and heuristic.
    
    Args:
        scenarios: List of scenario names to benchmark (default: ["furgoni"])
        config: Column Generation configuration
        save_results: Whether to save results to file
        
    Returns:
        List of benchmark results
    """
    if scenarios is None:
        scenarios = ["furgoni"]
    
    if config is None:
        config = ColumnGenerationConfig(
            max_iterations=20,
            optimality_gap_tolerance=1e-4,
            time_limit_seconds=300.0,
            verbose=True
        )
    
    logger.info("🚀 Starting Comprehensive EPDT Benchmark")
    logger.info("="*60)
    
    benchmark_suite = EPDTBenchmarkSuite(config)
    
    # Run benchmarks
    for scenario in scenarios:
        try:
            result = benchmark_suite.benchmark_scenario(scenario)
            logger.info(f"✅ Completed benchmark for {scenario}")
        except Exception as e:
            logger.error(f"❌ Failed to benchmark {scenario}: {str(e)}")
    
    # Print summary
    benchmark_suite.print_summary()
    
    # Save results
    if save_results:
        timestamp = int(time.time())
        filename = f"results/epdt_benchmark_results_{timestamp}.json"
        benchmark_suite.save_results(filename)
    
    return benchmark_suite.results


def get_optimality_bounds(orders: List[Order], vehicles: List[Vehicle], 
                         config: ColumnGenerationConfig = None) -> Tuple[float, float]:
    """
    Get optimality bounds for an EPDT problem instance.
    
    Args:
        orders: List of orders to schedule
        vehicles: List of available vehicles
        config: Column Generation configuration
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if config is None:
        config = ColumnGenerationConfig(verbose=False)
    
    result = solve_epdt_with_column_generation(orders, vehicles, config)
    
    if result.feasible:
        return result.lower_bound, result.upper_bound
    else:
        return float('inf'), float('inf')


if __name__ == "__main__":
    # Run comprehensive benchmark
    results = run_comprehensive_benchmark(
        scenarios=["furgoni"],
        save_results=True
    )
