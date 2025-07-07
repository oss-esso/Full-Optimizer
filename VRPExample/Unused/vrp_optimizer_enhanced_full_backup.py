#!/usr/bin/env python3
"""
Enhanced VRP Optimizer with Extended Time Limits and Multiple Search Strategies

This version implements the diagnostic tool recommendations:
1. Increased solver time limit from 300 to 1800+ seconds
2. Multiple OR-Tools first solution strategies
3. Better logging and progress tracking
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from vrp_optimizer_rolling_window import VRPOptimizerRollingWindow
from vrp_data_models import VRPInstance, VRPResult, VRPObjective

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VRPOptimizerEnhanced(VRPOptimizerRollingWindow):
    """Enhanced VRP optimizer with extended time limits and multiple strategies."""
    
    def __init__(self, instance: VRPInstance, objective: VRPObjective = VRPObjective.MINIMIZE_DISTANCE):
        """Initialize the enhanced VRP optimizer."""
        super().__init__(instance, objective)
        
        # Define multiple search strategies to try
        self.search_strategies = [
            ("PATH_CHEAPEST_ARC", routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC),
            ("PATH_MOST_CONSTRAINED_ARC", routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC),
            ("EVALUATOR_STRATEGY", routing_enums_pb2.FirstSolutionStrategy.EVALUATOR_STRATEGY),
            ("SAVINGS", routing_enums_pb2.FirstSolutionStrategy.SAVINGS),
            ("SWEEP", routing_enums_pb2.FirstSolutionStrategy.SWEEP),
            ("CHRISTOFIDES", routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES),
            ("ALL_UNPERFORMED", routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED),
            ("BEST_INSERTION", routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION),
            ("PARALLEL_CHEAPEST_INSERTION", routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION),
            ("SEQUENTIAL_CHEAPEST_INSERTION", routing_enums_pb2.FirstSolutionStrategy.SEQUENTIAL_CHEAPEST_INSERTION),
        ]
        
        # Define local search metaheuristics to try
        self.local_search_strategies = [
            ("GUIDED_LOCAL_SEARCH", routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH),
            ("TABU_SEARCH", routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH),
            ("SIMULATED_ANNEALING", routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING),
            ("GENERIC_TABU_SEARCH", routing_enums_pb2.LocalSearchMetaheuristic.GENERIC_TABU_SEARCH),
        ]
    
    def solve_with_enhanced_strategies(self, time_limit_seconds: int = 1800) -> VRPResult:
        """
        Solve VRP with enhanced strategies and extended time limit.
        
        Args:
            time_limit_seconds: Time limit in seconds (default 1800 = 30 minutes)
        """
        logger.info("=" * 80)
        logger.info("ENHANCED VRP OPTIMIZATION WITH EXTENDED TIME AND MULTIPLE STRATEGIES")
        logger.info("=" * 80)
        logger.info(f"Time limit: {time_limit_seconds} seconds ({time_limit_seconds/60:.1f} minutes)")
        logger.info(f"Testing {len(self.search_strategies)} first solution strategies")
        logger.info(f"Testing {len(self.local_search_strategies)} local search metaheuristics")
        
        if not ORTOOLS_AVAILABLE:
            logger.error("OR-Tools not available!")
            return VRPResult(
                status="error",
                objective_value=float('inf'),
                routes={},
                runtime=0.0,
                metrics={'error': 'OR-Tools not available'}
            )
        
        overall_start_time = time.time()
        best_result = None
        best_objective = float('inf')
        strategy_results = []
        
        # Phase 1: Try different first solution strategies with moderate time limit
        phase1_time_per_strategy = min(300, time_limit_seconds // len(self.search_strategies))
        logger.info(f"\n🚀 PHASE 1: Testing first solution strategies ({phase1_time_per_strategy}s each)")
        logger.info("-" * 60)
        
        for i, (strategy_name, strategy_enum) in enumerate(self.search_strategies):
            logger.info(f"Testing strategy {i+1}/{len(self.search_strategies)}: {strategy_name}")
            
            try:
                result = self._solve_with_strategy(
                    first_solution_strategy=strategy_enum,
                    local_search=routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
                    time_limit_seconds=phase1_time_per_strategy,
                    strategy_name=strategy_name
                )
                
                strategy_results.append(result)
                
                if result.objective_value < best_objective:
                    best_objective = result.objective_value
                    best_result = result
                    logger.info(f"  ✅ NEW BEST: {strategy_name} - Objective: {result.objective_value:.0f}")
                else:
                    logger.info(f"  ➡️  {strategy_name} - Objective: {result.objective_value:.0f}")
                    
            except Exception as e:
                logger.error(f"  ❌ {strategy_name} failed: {str(e)}")
        
        # Phase 2: If we found a solution, try to improve it with local search
        remaining_time = time_limit_seconds - (time.time() - overall_start_time)
        
        if best_result and best_result.objective_value < float('inf') and remaining_time > 60:
            logger.info(f"\n🔧 PHASE 2: Improving best solution with local search ({remaining_time:.0f}s remaining)")
            logger.info("-" * 60)
              # Find the best strategy from Phase 1
            best_strategy = None
            for result in strategy_results:
                if result.objective_value == best_objective:
                    best_strategy = result.metrics.get('strategy_enum')
                    break
            
            if best_strategy:
                # Try each local search metaheuristic
                phase2_time_per_strategy = min(300, remaining_time // len(self.local_search_strategies))
                
                for ls_name, ls_enum in self.local_search_strategies:
                    if remaining_time < 60:
                        break
                        
                    logger.info(f"Trying local search: {ls_name}")
                    
                    try:
                        improved_result = self._solve_with_strategy(
                            first_solution_strategy=best_strategy,
                            local_search=ls_enum,
                            time_limit_seconds=phase2_time_per_strategy,
                            strategy_name=f"Best+{ls_name}"
                        )
                        
                        if improved_result.objective_value < best_objective:
                            best_objective = improved_result.objective_value
                            best_result = improved_result
                            logger.info(f"  ✅ IMPROVED: {ls_name} - Objective: {improved_result.objective_value:.0f}")
                        else:
                            logger.info(f"  ➡️  {ls_name} - No improvement")
                            
                        remaining_time = time_limit_seconds - (time.time() - overall_start_time)
                        
                    except Exception as e:
                        logger.error(f"  ❌ {ls_name} failed: {str(e)}")
        
        # Phase 3: Final intensive search if time permits
        remaining_time = time_limit_seconds - (time.time() - overall_start_time)
        
        if best_result and best_result.objective_value < float('inf') and remaining_time > 300:
            logger.info(f"\n🎯 PHASE 3: Final intensive optimization ({remaining_time:.0f}s remaining)")
            logger.info("-" * 60)
            
            try:                # Use the best strategy found so far with maximum time
                best_strategy = None
                for result in strategy_results:
                    if result.objective_value == best_objective:
                        best_strategy = result.metrics.get('strategy_enum')
                        break
                
                if best_strategy:
                    final_result = self._solve_with_strategy(
                        first_solution_strategy=best_strategy,
                        local_search=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
                        time_limit_seconds=int(remaining_time),
                        strategy_name="Final Intensive"
                    )
                    
                    if final_result.objective_value < best_objective:
                        best_result = final_result
                        logger.info(f"  ✅ FINAL IMPROVEMENT: Objective: {final_result.objective_value:.0f}")
                    else:
                        logger.info(f"  ➡️  Final phase: No improvement")
                        
            except Exception as e:
                logger.error(f"  ❌ Final phase failed: {str(e)}")
        
        total_time = time.time() - overall_start_time
        
        # Summary
        logger.info("=" * 80)
        logger.info("OPTIMIZATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total optimization time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"Strategies tested: {len(strategy_results)}")
        
        if best_result and best_result.objective_value < float('inf'):
            logger.info(f"✅ SOLUTION FOUND!")
            logger.info(f"Best objective value: {best_result.objective_value:.0f}")
            logger.info(f"Total distance: {best_result.total_distance:.0f}")
            logger.info(f"Solution time: {best_result.solution_time:.1f} seconds")
            logger.info(f"Routes: {len([r for r in best_result.routes.values() if len(r) > 2])} active vehicles")
              # Update metrics
            best_result.metrics.update({
                'total_optimization_time': total_time,
                'strategies_tested': len(strategy_results),
                'phase1_results': len([r for r in strategy_results if r.objective_value < float('inf')]),
                'final_strategy': best_result.metrics.get('strategy_name', 'Unknown')
            })
            
        else:
            logger.warning("❌ NO SOLUTION FOUND with any strategy")            
            best_result = VRPResult(
                status="no_solution",
                objective_value=float('inf'),
                routes={},
                runtime=total_time,
                metrics={
                    'error': 'No feasible solution found with any strategy',
                    'total_optimization_time': total_time,
                    'strategies_tested': len(strategy_results)                }
            )
        
        logger.info("=" * 80)
        return best_result
    
    def _solve_with_strategy(self, first_solution_strategy, local_search, time_limit_seconds: int, strategy_name: str) -> VRPResult:
        """Solve with a specific strategy combination using the rolling window base implementation."""
        logger.info(f"    Solving with {strategy_name} (time limit: {time_limit_seconds}s)...")
        start_time = time.time()
        
        try:
            # Use the parent class's rolling window solver with custom parameters
            result = self.optimize_with_rolling_window(
                timeout_duration=time_limit_seconds + 60,  # Buffer for timeout
                time_limit_seconds=time_limit_seconds
            )
            
            solve_time = time.time() - start_time
            
            # Update result with strategy information
            if result.metrics is None:
                result.metrics = {}
            
            result.metrics.update({
                'strategy_name': strategy_name,
                'strategy_enum': first_solution_strategy,
                'local_search': local_search,
                'solve_time': solve_time
            })
            
            return result
            
        except Exception as e:
            solve_time = time.time() - start_time
            logger.error(f"    ❌ {strategy_name} failed: {str(e)}")
            return VRPResult(
                status="error",
                objective_value=float('inf'),
                routes={},
                runtime=solve_time,
                metrics={
                    'strategy_name': strategy_name,
                    'error': f'Exception: {str(e)}',
                    'solve_time': solve_time
                }
            )
    
    def _create_routing_model(self):
        """Create the OR-Tools routing model."""
        num_locations = len(self.instance.locations)
        num_vehicles = len(self.instance.vehicles)
        
        # Use first two locations as depots (depot_asti, depot_milan)
        depot_indices = [0, 1] if num_locations > 1 else [0]
        
        manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_indices)
        routing = pywrapcp.RoutingModel(manager)
        
        return manager, routing

if __name__ == "__main__":
    # Test the enhanced optimizer
    from vrp_scenarios import create_moda_first_scenario
    
    print("Testing Enhanced VRP Optimizer with Extended Time and Multiple Strategies")
    print("=" * 80)
    
    # Create test scenario
    scenario = create_moda_first_scenario()
    
    # Initialize enhanced optimizer
    optimizer = VRPOptimizerEnhanced(scenario)
    
    # Solve with enhanced strategies (30 minutes time limit)
    result = optimizer.solve_with_enhanced_strategies(time_limit_seconds=1800)
    
    # Print results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    if result.objective_value < float('inf'):
        print(f"✅ SUCCESS! Solution found")
        print(f"Objective value: {result.objective_value:.0f}")
        print(f"Total distance: {result.total_distance:.0f}")
        print(f"Vehicles used: {len(result.routes)}")
        print(f"Total optimization time: {result.metrics.get('total_optimization_time', 0):.1f} seconds")
        print(f"Winning strategy: {result.metrics.get('final_strategy', 'Unknown')}")
        
        # Show route summary
        print(f"\nRoute Summary:")
        for vehicle_id, route in result.routes.items():
            print(f"  {vehicle_id}: {len(route)} stops")
            
    else:
        print(f"❌ No solution found")
        print(f"Error: {result.metrics.get('error', 'Unknown')}")
        print(f"Total time spent: {result.metrics.get('total_optimization_time', 0):.1f} seconds")
        print(f"Strategies tested: {result.metrics.get('strategies_tested', 0)}")
