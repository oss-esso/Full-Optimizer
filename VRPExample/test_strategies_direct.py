#!/usr/bin/env python3
"""
Test different OR-Tools strategies by temporarily modifying the optimizer
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


def test_strategies_by_patching():
    """Test different strategies by temporarily modifying the optimizer code"""
    print("Testing OR-Tools strategies by modifying search parameters")
    print("=" * 70)
    
    scenarios = get_all_scenarios()
    moda_small = scenarios['MODA_small']
    
    print(f'MODA_small: {len(moda_small.vehicles)} vehicles, {len(moda_small.ride_requests)} requests')
    print(f'Total passengers: {sum(req.passengers for req in moda_small.ride_requests)}')
    print()
    
    # Test strategies that are most likely to work with pickup-delivery
    strategies_to_test = [
        ('PARALLEL_CHEAPEST_INSERTION', routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION),
        ('PATH_CHEAPEST_ARC', routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC),
        ('SEQUENTIAL_CHEAPEST_INSERTION', routing_enums_pb2.FirstSolutionStrategy.SEQUENTIAL_CHEAPEST_INSERTION),
        ('LOCAL_CHEAPEST_INSERTION', routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION),
        ('PATH_MOST_CONSTRAINED_ARC', routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC),
        ('FIRST_UNBOUND_MIN_VALUE', routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE),
    ]
    
    results = []
    
    for strategy_name, strategy_enum in strategies_to_test:
        print(f"Testing {strategy_name}...")
        
        try:
            # Temporarily patch the optimizer code
            result = run_with_strategy(moda_small, strategy_enum)
            
            vehicles_used = len([r for r in result.routes.values() if len(r) > 2])
            total_distance = result.metrics.get("total_distance", 0)
            
            results.append({
                'strategy': strategy_name,
                'vehicles_used': vehicles_used,
                'total_vehicles': len(moda_small.vehicles),
                'distance': total_distance,
                'objective': result.objective_value,
                'runtime': result.runtime
            })
            
            print(f"  ✓ {vehicles_used}/{len(moda_small.vehicles)} vehicles, distance: {total_distance:.2f}, runtime: {result.runtime:.0f}ms")
            
            # Show individual route loads
            if vehicles_used > 0:
                for vehicle_id, route in result.routes.items():
                    if len(route) > 2:
                        requests = (len(route) - 2) // 2
                        print(f"    {vehicle_id}: {requests} requests")
                        
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            results.append({
                'strategy': strategy_name,
                'vehicles_used': 0,
                'total_vehicles': len(moda_small.vehicles),
                'distance': float('inf'),
                'objective': float('inf'),
                'runtime': 0,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY:")
    print("=" * 70)
    
    valid_results = [r for r in results if 'error' not in r]
    # Sort by vehicles used (descending), then by distance (ascending)
    valid_results.sort(key=lambda x: (-x['vehicles_used'], x['distance']))
    
    print("Rank Strategy".ljust(30) + "Vehicles".ljust(12) + "Distance".ljust(12) + "Runtime")
    print("-" * 70)
    
    for i, result in enumerate(valid_results):
        vehicles_str = f"{result['vehicles_used']}/{result['total_vehicles']}"
        print(f"{i+1:2}. {result['strategy']:<27} {vehicles_str:<12} {result['distance']:<12.2f} {result['runtime']:<8.0f}ms")
    
    if valid_results:
        best = valid_results[0]
        print(f"\nBest strategy: {best['strategy']} with {best['vehicles_used']} vehicles")
        return best['strategy']
    
    return None


def run_with_strategy(instance, first_solution_strategy):
    """Run OR-Tools with a specific first solution strategy"""
    
    # Create optimizer
    optimizer = VRPQuantumOptimizer(instance, VRPObjective.MINIMIZE_DISTANCE)
    
    # Monkey patch the search parameters setup
    original_code = '''
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
            )'''
    
    new_code = f'''
            search_parameters.first_solution_strategy = first_solution_strategy'''
    
    # Read the source file
    import inspect
    import types
    
    # Get the source code of the OR-Tools method
    source_file = inspect.getfile(VRPQuantumOptimizer.optimize_with_ortools)
    
    # For simplicity, let's just create a new function that copies the existing logic
    # but with our modified strategy
    
    # Actually, let's use a simpler approach - directly modify the enum before calling
    import vrp_optimizer_fixed
    
    # Temporarily replace the enum in the module
    original_enum = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    
    # Patch the enum value by modifying the module's behavior
    # This is a bit hacky but should work for testing
    
    # Store original method
    original_method = optimizer.optimize_with_ortools
    
    def patched_method():
        # We'll create a custom version that uses our strategy
        return run_ortools_with_custom_strategy(optimizer, first_solution_strategy)
    
    # Replace method temporarily
    optimizer.optimize_with_ortools = patched_method
    
    try:
        result = optimizer.optimize_with_ortools()
        return result
    finally:
        # Restore original method
        optimizer.optimize_with_ortools = original_method


def run_ortools_with_custom_strategy(optimizer, first_solution_strategy):
    """Custom OR-Tools implementation with specified strategy"""
    import time
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    
    start_time = time.time()
    instance = optimizer.instance
    
    # Check if this is a VRPPD scenario
    is_vrppd = bool(instance.ride_requests)
    
    if not is_vrppd:
        # Fall back to original method for non-VRPPD
        return optimizer._VRPQuantumOptimizer__optimize_with_ortools_impl()
    
    # Multi-depot VRPPD setup (simplified version of the main code)
    depot_locations = [loc_id for loc_id in instance.location_ids if 'depot' in loc_id]
    
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
    
    # Create routing model
    routing = pywrapcp.RoutingModel(manager)
    
    # Create distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        from_location_id = location_list[from_node]
        to_location_id = location_list[to_node]
        return int(instance.get_distance(from_location_id, to_location_id))
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Add capacity constraints
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        from_location_id = location_list[from_node]
        
        for request in instance.ride_requests:
            if from_location_id == request.pickup_location:
                return request.passengers
            elif from_location_id == request.dropoff_location:
                return -request.passengers
        return 0
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    vehicle_capacities = [vehicle.capacity for vehicle in instance.vehicles.values()]
    
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # no slack
        vehicle_capacities,
        True,  # start cumul to zero
        'Capacity'
    )
    
    # Add pickup and delivery constraints
    for request in instance.ride_requests:
        pickup_idx = location_list.index(request.pickup_location)
        delivery_idx = location_list.index(request.dropoff_location)
        
        pickup_node = manager.NodeToIndex(pickup_idx)
        delivery_node = manager.NodeToIndex(delivery_idx)
        
        routing.AddPickupAndDelivery(pickup_node, delivery_node)
        routing.solver().Add(
            routing.VehicleVar(pickup_node) == routing.VehicleVar(delivery_node)
        )
    
    # Set search parameters with our custom strategy
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = first_solution_strategy
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(5)  # Shorter time for testing
    
    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    runtime = (time.time() - start_time) * 1000
    
    if solution:
        # Extract routes (simplified)
        routes = {}
        total_distance = 0
        vehicle_ids = sorted(instance.vehicles.keys())
        
        for vehicle_idx in range(num_vehicles):
            if vehicle_idx < len(vehicle_ids):
                vehicle_id = vehicle_ids[vehicle_idx]
                route = [depot_locations[0]]  # Start at first depot
                
                index = routing.Start(vehicle_idx)
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    if node_index >= len(depot_locations):  # Not a depot
                        route.append(location_list[node_index])
                    
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    total_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_idx)
                
                route.append(depot_locations[0])  # End at depot
                routes[vehicle_id] = route
          # Create result
        from vrp_data_models import VRPResult
        return VRPResult(
            status="OPTIMAL",
            routes=routes,
            objective_value=solution.ObjectiveValue() / 1000.0,
            runtime=runtime,
            metrics={
                "total_distance": total_distance / 1000.0,
                "vehicles_used": len([r for r in routes.values() if len(r) > 2])
            }
        )
    else:
        raise Exception("OR-Tools could not find a solution")


if __name__ == "__main__":
    best_strategy = test_strategies_by_patching()
    if best_strategy:
        print(f"\nRecommendation: Use {best_strategy} for better vehicle distribution")
