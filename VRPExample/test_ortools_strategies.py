#!/usr/bin/env python3
"""
Test script to compare different OR-Tools optimization strategies for vehicle distribution
"""
import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from vrp_scenarios import get_all_scenarios
from vrp_optimizer_fixed import VRPQuantumOptimizer
from vrp_data_models import VRPObjective
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import time


def test_ortools_strategies():
    """Test different OR-Tools first solution and local search strategies"""
    print("Testing OR-Tools optimization strategies for vehicle distribution")
    print("=" * 80)
    
    scenarios = get_all_scenarios()
    moda_small = scenarios['MODA_small']
    
    print(f'MODA_small: {len(moda_small.vehicles)} vehicles, {len(moda_small.ride_requests)} requests')
    print(f'Total passengers: {sum(req.passengers for req in moda_small.ride_requests)}')
    print()
    
    # First solution strategies to test
    first_solution_strategies = {
        'PATH_CHEAPEST_ARC': routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        'PATH_MOST_CONSTRAINED_ARC': routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
        'EVALUATOR_STRATEGY': routing_enums_pb2.FirstSolutionStrategy.EVALUATOR_STRATEGY,
        'SAVINGS': routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        'SWEEP': routing_enums_pb2.FirstSolutionStrategy.SWEEP,
        'CHRISTOFIDES': routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
        'ALL_UNPERFORMED': routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED,
        'BEST_INSERTION': routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION,
        'PARALLEL_CHEAPEST_INSERTION': routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        'SEQUENTIAL_CHEAPEST_INSERTION': routing_enums_pb2.FirstSolutionStrategy.SEQUENTIAL_CHEAPEST_INSERTION,
        'LOCAL_CHEAPEST_INSERTION': routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
        'GLOBAL_CHEAPEST_ARC': routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
        'LOCAL_CHEAPEST_ARC': routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC,
        'FIRST_UNBOUND_MIN_VALUE': routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE,
    }
    
    # Local search metaheuristics to test
    local_search_strategies = {
        'AUTOMATIC': routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
        'GREEDY_DESCENT': routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
        'GUIDED_LOCAL_SEARCH': routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        'SIMULATED_ANNEALING': routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
        'TABU_SEARCH': routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
        'GENERIC_TABU_SEARCH': routing_enums_pb2.LocalSearchMetaheuristic.GENERIC_TABU_SEARCH,
    }
    
    results = []
    
    # Test different first solution strategies with the current local search
    print("Testing First Solution Strategies:")
    print("-" * 50)
    
    for strategy_name, strategy_enum in first_solution_strategies.items():
        print(f"Testing {strategy_name}...")
        try:
            result = run_ortools_with_strategy(
                moda_small, 
                first_solution=strategy_enum,
                local_search=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
                time_limit=10
            )
            
            vehicles_used = len([r for r in result['routes'].values() if len(r) > 2])
            results.append({
                'type': 'First Solution',
                'strategy': strategy_name,
                'vehicles_used': vehicles_used,
                'total_vehicles': len(moda_small.vehicles),
                'distance': result['distance'],
                'objective': result['objective'],
                'runtime': result['runtime'],
                'routes': result['routes']
            })
            
            print(f"  ✓ {strategy_name}: {vehicles_used}/{len(moda_small.vehicles)} vehicles, "
                  f"distance: {result['distance']:.2f}, runtime: {result['runtime']:.0f}ms")
                  
        except Exception as e:
            print(f"  ✗ {strategy_name}: ERROR - {str(e)}")
            results.append({
                'type': 'First Solution',
                'strategy': strategy_name,
                'vehicles_used': 0,
                'total_vehicles': len(moda_small.vehicles),
                'distance': float('inf'),
                'objective': float('inf'),
                'runtime': 0,
                'error': str(e)
            })
    
    print()
    print("Testing Local Search Metaheuristics:")
    print("-" * 50)
    
    # Test different local search strategies with the best first solution from above
    best_first_solution = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    
    for strategy_name, strategy_enum in local_search_strategies.items():
        print(f"Testing {strategy_name}...")
        try:
            result = run_ortools_with_strategy(
                moda_small, 
                first_solution=best_first_solution,
                local_search=strategy_enum,
                time_limit=10
            )
            
            vehicles_used = len([r for r in result['routes'].values() if len(r) > 2])
            results.append({
                'type': 'Local Search',
                'strategy': strategy_name,
                'vehicles_used': vehicles_used,
                'total_vehicles': len(moda_small.vehicles),
                'distance': result['distance'],
                'objective': result['objective'],
                'runtime': result['runtime'],
                'routes': result['routes']
            })
            
            print(f"  ✓ {strategy_name}: {vehicles_used}/{len(moda_small.vehicles)} vehicles, "
                  f"distance: {result['distance']:.2f}, runtime: {result['runtime']:.0f}ms")
                  
        except Exception as e:
            print(f"  ✗ {strategy_name}: ERROR - {str(e)}")
            results.append({
                'type': 'Local Search',
                'strategy': strategy_name,
                'vehicles_used': 0,
                'total_vehicles': len(moda_small.vehicles),
                'distance': float('inf'),
                'objective': float('inf'),
                'runtime': 0,
                'error': str(e)
            })
    
    # Analyze results
    print()
    print("RESULTS ANALYSIS:")
    print("=" * 80)
    
    # Sort by vehicles used (descending) then by distance (ascending)
    valid_results = [r for r in results if 'error' not in r]
    valid_results.sort(key=lambda x: (-x['vehicles_used'], x['distance']))
    
    print("Best strategies for maximizing vehicle usage:")
    print("Strategy Type".ljust(15) + "Strategy Name".ljust(30) + "Vehicles".ljust(10) + "Distance".ljust(10) + "Runtime")
    print("-" * 80)
    
    for result in valid_results[:10]:  # Show top 10
        vehicles_str = f"{result['vehicles_used']}/{result['total_vehicles']}"
        print(f"{result['type']:<15}{result['strategy']:<30}{vehicles_str:<10}{result['distance']:<10.2f}{result['runtime']:<10.0f}ms")
    
    # Show detailed route analysis for the best strategies
    print()
    print("DETAILED ROUTE ANALYSIS FOR TOP 3 STRATEGIES:")
    print("=" * 80)
    
    for i, result in enumerate(valid_results[:3]):
        print(f"\n{i+1}. {result['strategy']} ({result['type']}):")
        print(f"   Vehicles used: {result['vehicles_used']}/{result['total_vehicles']}")
        print(f"   Total distance: {result['distance']:.2f}")
        print(f"   Routes:")
        
        for vehicle_id, route in result['routes'].items():
            if len(route) > 2:  # Skip empty routes
                requests = (len(route) - 2) // 2  # Estimate ride requests
                print(f"     {vehicle_id}: {len(route)} stops ({requests} requests)")


def run_ortools_with_strategy(instance, first_solution, local_search, time_limit=10):
    """Run OR-Tools with specific strategies"""
    start_time = time.time()
    
    # Build location list for OR-Tools (similar to the main implementation)
    is_vrppd = bool(instance.ride_requests)
    
    if is_vrppd:
        # Multi-depot VRPPD setup
        depot_locations = [loc_id for loc_id in instance.location_ids if 'depot' in loc_id]
        
        # Create location list
        location_list = depot_locations.copy()
        location_to_index = {loc: idx for idx, loc in enumerate(depot_locations)}
        
        # Add pickup and dropoff locations
        for request in instance.ride_requests:
            if request.pickup_location not in location_to_index:
                location_to_index[request.pickup_location] = len(location_list)
                location_list.append(request.pickup_location)
            if request.dropoff_location not in location_to_index:
                location_to_index[request.dropoff_location] = len(location_list)
                location_list.append(request.dropoff_location)
        
        # Create vehicle start/end depot mappings
        vehicle_starts = []
        vehicle_ends = []
        depot_indices = {depot: location_to_index[depot] for depot in depot_locations}
        
        for vehicle_id, vehicle in instance.vehicles.items():
            depot_idx = depot_indices.get(vehicle.depot_id, 0)
            vehicle_starts.append(depot_idx)
            vehicle_ends.append(depot_idx)
        
        num_locations = len(location_list)
        num_vehicles = len(instance.vehicles)
        
        # Create routing index manager with multi-depot
        manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, vehicle_starts, vehicle_ends)
    else:
        # Single depot setup (simplified for now)
        location_list = instance.location_ids
        num_locations = len(location_list)
        num_vehicles = len(instance.vehicles)
        depot_index = 0
        manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_index)
    
    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)
    
    # Create distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        from_location_id = location_list[from_node]
        to_location_id = location_list[to_node]
        return int(instance.get_distance(from_location_id, to_location_id) * 1000)  # Convert to mm
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Add capacity constraints for VRPPD
    if is_vrppd:
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            from_location_id = location_list[from_node]
            
            # Find which request this location belongs to
            for request in instance.ride_requests:
                if from_location_id == request.pickup_location:
                    return request.passengers  # Pick up passengers
                elif from_location_id == request.dropoff_location:
                    return -request.passengers  # Drop off passengers
            return 0  # Depot or other location
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        
        # Add capacity dimension
        vehicle_capacities = [vehicle.capacity for vehicle in instance.vehicles.values()]
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # no slack
            vehicle_capacities,  # vehicle capacities
            True,  # start cumul to zero
            'Capacity'
        )
        
        # Add pickup and delivery constraints
        pickup_deliveries = []
        for request in instance.ride_requests:
            pickup_idx = location_list.index(request.pickup_location)
            delivery_idx = location_list.index(request.dropoff_location)
            pickup_deliveries.append((pickup_idx, delivery_idx))
        
        for pickup_idx, delivery_idx in pickup_deliveries:
            pickup_node = manager.NodeToIndex(pickup_idx)
            delivery_node = manager.NodeToIndex(delivery_idx)
            
            routing.AddPickupAndDelivery(pickup_node, delivery_node)
            routing.solver().Add(
                routing.VehicleVar(pickup_node) == routing.VehicleVar(delivery_node)
            )
    
    # Set search parameters with the specified strategies
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = first_solution
    search_parameters.local_search_metaheuristic = local_search
    search_parameters.time_limit.FromSeconds(time_limit)
      # Solve
    solution = routing.SolveWithParameters(search_parameters)
    runtime = (time.time() - start_time) * 1000
    
    if solution:
        # Extract routes
        routes = {}
        vehicle_ids = sorted(instance.vehicles.keys())
        
        for vehicle_idx in range(num_vehicles):
            if vehicle_idx < len(vehicle_ids):
                vehicle_id = vehicle_ids[vehicle_idx]
                route = []
                
                index = routing.Start(vehicle_idx)
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    route.append(location_list[node_index])
                    index = solution.Value(routing.NextVar(index))
                
                # Add the end depot
                final_node_index = manager.IndexToNode(index)
                route.append(location_list[final_node_index])
                
                routes[vehicle_id] = route
        
        # Calculate total distance using instance's distance method
        total_distance = 0.0
        for vehicle_id, route in routes.items():
            if len(route) > 1:  # Only count routes with actual movement
                for i in range(len(route) - 1):
                    distance = instance.get_distance(route[i], route[i + 1])
                    total_distance += distance
        
        return {
            'routes': routes,
            'objective': solution.ObjectiveValue() / 1000.0,  # Convert back from mm
            'distance': total_distance,  # Already in km from instance.get_distance
            'runtime': runtime
        }
    else:
        raise Exception("OR-Tools could not find a solution")


if __name__ == "__main__":
    test_ortools_strategies()
