#!/usr/bin/env python3
"""
Test Enhanced Optimization Strategies for MODA_first

This script tests the MODA_first scenario with extended time limits 
and multiple OR-Tools strategies to find a feasible solution.
"""

import logging
import time
from vrp_optimizer_rolling_window import VRPOptimizerRollingWindow
from vrp_scenarios import create_moda_first_scenario

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_extended_strategies(scenario, time_limit_seconds=1800):
    """Test MODA_first with extended time and multiple strategies."""
    
    logger.info("=" * 80)
    logger.info("TESTING MODA_first WITH EXTENDED TIME AND MULTIPLE STRATEGIES")
    logger.info("=" * 80)
    logger.info(f"Time limit: {time_limit_seconds} seconds ({time_limit_seconds/60:.1f} minutes)")
    
    if not ORTOOLS_AVAILABLE:
        logger.error("OR-Tools not available!")
        return None
    
    # Define strategies to test
    strategies = [
        ("PATH_CHEAPEST_ARC", routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC),
        ("PATH_MOST_CONSTRAINED_ARC", routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC),
        ("EVALUATOR_STRATEGY", routing_enums_pb2.FirstSolutionStrategy.EVALUATOR_STRATEGY),
        ("SAVINGS", routing_enums_pb2.FirstSolutionStrategy.SAVINGS),
        ("SWEEP", routing_enums_pb2.FirstSolutionStrategy.SWEEP),
        ("CHRISTOFIDES", routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES),
        ("BEST_INSERTION", routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION),
        ("PARALLEL_CHEAPEST_INSERTION", routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION),
    ]
    
    time_per_strategy = time_limit_seconds // len(strategies)
    logger.info(f"Testing {len(strategies)} strategies with {time_per_strategy}s each")
    
    best_result = None
    best_objective = float('inf')
    results = []
    
    # Test each strategy
    for i, (strategy_name, strategy_enum) in enumerate(strategies):
        logger.info(f"\n🔍 Testing strategy {i+1}/{len(strategies)}: {strategy_name}")
        logger.info("-" * 60)
        
        try:
            # Create optimizer
            optimizer = VRPOptimizerRollingWindow(scenario)
            
            # Override the solve method to use our custom parameters
            result = test_single_strategy(optimizer, strategy_enum, strategy_name, time_per_strategy)
            
            results.append({
                'strategy': strategy_name,
                'result': result,
                'success': result.status == 'feasible' if hasattr(result, 'status') else 'routes' in result and len(result['routes']) > 0
            })
            
            # Check if this is the best result so far
            objective = getattr(result, 'objective_value', float('inf'))
            if hasattr(result, 'routes') and len(result.get('routes', {})) > 0:
                objective = getattr(result, 'total_distance', objective)
            
            if objective < best_objective:
                best_objective = objective
                best_result = result
                logger.info(f"  ✅ NEW BEST: {strategy_name} - Objective: {objective:.0f}")
            else:
                logger.info(f"  ➡️  {strategy_name} - No solution or worse than current best")
                
        except Exception as e:
            logger.error(f"  ❌ {strategy_name} failed with error: {str(e)}")
            results.append({
                'strategy': strategy_name,
                'result': None,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("STRATEGY TEST SUMMARY")
    logger.info("=" * 80)
    
    successful_strategies = [r for r in results if r['success']]
    logger.info(f"Strategies tested: {len(results)}")
    logger.info(f"Successful strategies: {len(successful_strategies)}")
    
    if successful_strategies:
        logger.info("✅ SOLUTION FOUND!")
        logger.info(f"Best strategy: {best_result}")
        for result in successful_strategies:
            logger.info(f"  - {result['strategy']}: Success")
    else:
        logger.warning("❌ NO SOLUTION FOUND with any strategy")
    
    return best_result

def test_single_strategy(optimizer, strategy_enum, strategy_name, time_limit):
    """Test a single strategy with the given time limit."""
    
    logger.info(f"    Setting up {strategy_name} with {time_limit}s time limit...")
    
    try:
        # Use the existing optimizer but with custom OR-Tools parameters
        result = optimizer._solve_with_ortools_rolling_window_custom(strategy_enum, time_limit)
        return result
        
    except Exception as e:
        logger.error(f"    Strategy {strategy_name} failed: {str(e)}")
        return {'status': 'failed', 'error': str(e)}

# Monkey patch the rolling window optimizer to add custom strategy method
def _solve_with_ortools_rolling_window_custom(self, first_solution_strategy, time_limit_seconds):
    """Custom solver with specific strategy and time limit."""
    
    if not ORTOOLS_AVAILABLE:
        return {'status': 'error', 'error': 'OR-Tools not available'}
    
    try:
        # Create routing model
        num_locations = len(self.instance.locations)
        num_vehicles = len(self.instance.vehicles)
        depot_indices = [0, 1] if num_locations > 1 else [0]
        
        manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_indices)
        routing = pywrapcp.RoutingModel(manager)
        
        # Add constraints (reuse existing methods)
        distance_callback_index = self._create_distance_callback(manager)
        routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)
        
        # Add capacity constraints
        demand_callback_index = self._create_demand_callback(manager)
        capacity_constraint = self._add_capacity_constraint(routing, demand_callback_index)
        
        # Add time constraints
        time_callback_index = self._create_time_callback(manager)
        time_constraint = self._add_time_constraint(routing, time_callback_index, manager)
        
        # Add pickup-delivery constraints
        pickup_delivery_constraint = self._add_pickup_delivery_constraints(routing, manager)
        
        # Set search parameters with custom strategy
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = first_solution_strategy
        search_parameters.time_limit.FromSeconds(time_limit_seconds)
        search_parameters.log_search = True
        
        # Solve
        logger.info(f"      Solving with OR-Tools...")
        assignment = routing.SolveWithParameters(search_parameters)
        
        if assignment:
            logger.info(f"      ✅ Solution found!")
            
            # Extract routes
            routes = {}
            for vehicle_id in range(routing.vehicles()):
                route = []
                index = routing.Start(vehicle_id)
                
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    location_id = list(self.instance.locations.keys())[node_index]
                    route.append(location_id)
                    index = assignment.Value(routing.NextVar(index))
                
                # Add depot at end
                end_node = manager.IndexToNode(routing.End(vehicle_id))
                end_location_id = list(self.instance.locations.keys())[end_node]
                route.append(end_location_id)
                
                if len(route) > 2:  # Only include non-empty routes
                    routes[f"vehicle_{vehicle_id}"] = route
            
            return {
                'status': 'feasible',
                'routes': routes,
                'objective_value': assignment.ObjectiveValue(),
                'total_distance': assignment.ObjectiveValue(),
                'vehicles_used': len(routes)
            }
        else:
            logger.info(f"      ❌ No solution found")
            return {'status': 'infeasible', 'error': 'No solution found'}
            
    except Exception as e:
        logger.error(f"      ❌ Exception: {str(e)}")
        return {'status': 'error', 'error': str(e)}

# Add the custom method to the optimizer class
VRPOptimizerRollingWindow._solve_with_ortools_rolling_window_custom = _solve_with_ortools_rolling_window_custom

if __name__ == "__main__":
    print("Testing MODA_first with Extended Time and Multiple Strategies")
    print("=" * 80)
    
    # Create scenario
    scenario = create_moda_first_scenario()
    
    # Test with 30-minute time limit
    result = test_extended_strategies(scenario, time_limit_seconds=1800)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    if result and result.get('status') == 'feasible':
        print("🎉 SUCCESS! Found a feasible solution!")
        print(f"Objective value: {result.get('objective_value', 'N/A')}")
        print(f"Total distance: {result.get('total_distance', 'N/A')}")
        print(f"Vehicles used: {result.get('vehicles_used', 'N/A')}")
        print(f"Routes found: {len(result.get('routes', {}))}")
    else:
        print("❌ No feasible solution found")
        if result:
            print(f"Final status: {result.get('status', 'unknown')}")
            print(f"Error: {result.get('error', 'unknown')}")
