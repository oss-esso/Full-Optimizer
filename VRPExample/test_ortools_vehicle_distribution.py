#!/usr/bin/env python3
"""
Test script to experiment with OR-Tools vehicle distribution
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


def test_with_vehicle_fixed_cost():
    """Test OR-Tools with different fixed cost per vehicle configurations"""
    print("Testing OR-Tools with vehicle fixed costs")
    print("=" * 60)
    
    scenarios = get_all_scenarios()
    moda_small = scenarios['MODA_small']
    
    print(f'MODA_small: {len(moda_small.vehicles)} vehicles, {len(moda_small.ride_requests)} requests')
    
    # Test different fixed costs
    fixed_costs = [0, 1000, 5000, 10000, 20000]
    
    for fixed_cost in fixed_costs:
        print(f'\n--- Testing with fixed cost per vehicle: {fixed_cost} ---')
        
        # Temporarily modify the optimizer to add fixed cost
        optimizer = VRPQuantumOptimizer(moda_small, VRPObjective.MINIMIZE_DISTANCE)
        
        # Monkey patch the optimize_with_ortools method to add fixed cost
        original_method = optimizer.optimize_with_ortools
        
        def modified_ortools_with_fixed_cost():
            return run_ortools_with_fixed_cost(optimizer, fixed_cost)
        
        optimizer.optimize_with_ortools = modified_ortools_with_fixed_cost
        
        try:
            result = optimizer.optimize_with_ortools()
            
            vehicles_used = len([r for r in result.routes.values() if len(r) > 2])
            print(f'  Vehicles used: {vehicles_used}/{len(moda_small.vehicles)}')
            print(f'  Distance: {result.metrics.get("total_distance", 0):.2f}')
            print(f'  Objective: {result.objective_value:.2f}')
            print(f'  Runtime: {result.runtime:.2f}ms')
            
            # Show route distribution
            for vehicle_id, route in result.routes.items():
                if len(route) > 2:
                    print(f'    {vehicle_id}: {len(route)-2} requests')
                    
        except Exception as e:
            print(f'  ERROR: {str(e)}')


def run_ortools_with_fixed_cost(optimizer, fixed_cost):
    """Run OR-Tools with a specified fixed cost per vehicle"""
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    import time
    
    print(f"    Starting OR-Tools with fixed cost: {fixed_cost}")
    start_time = time.time()
    
    # Get instance
    instance = optimizer.instance
    
    # Build location list for OR-Tools
    is_vrppd = bool(instance.ride_requests)
    
    if is_vrppd:
        # Multi-depot VRPPD setup
        depot_locations = [loc_id for loc_id in instance.location_ids if 'depot' in loc_id]
        pickup_locations = [loc_id for loc_id in instance.location_ids if 'pickup' in loc_id]
        dropoff_locations = [loc_id for loc_id in instance.location_ids if 'dropoff' in loc_id]
        
        # Build location list: depots first, then pickups, then dropoffs
        location_list = depot_locations + pickup_locations + dropoff_locations
    else:
        location_list = instance.location_ids
    
    num_vehicles = len(instance.vehicles)
    
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(len(location_list), num_vehicles, 
                                         list(range(len(depot_locations))))
    
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
    
    # Add fixed cost per vehicle if specified
    if fixed_cost > 0:
        print(f"    Setting fixed cost per vehicle: {fixed_cost}")
        for vehicle_id in range(num_vehicles):
            routing.SetFixedCostOfVehicle(fixed_cost, vehicle_id)
    
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
    
    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(5)
    
    # Solve
    solution = routing.SolveWithParameters(search_parameters)
    runtime = (time.time() - start_time) * 1000
    
    if solution:
        # Extract routes
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
                    if node_index >= len(depot_locations):  # Skip depot nodes in middle
                        route.append(location_list[node_index])
                    
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    total_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_idx)
                
                # End at closest depot
                closest_depot = depot_locations[0]  # Simplified
                route.append(closest_depot)
                routes[vehicle_id] = route
        
        # Create result
        from vrp_data_models import VRPResult
        
        result = VRPResult(
            routes=routes,
            objective_value=solution.ObjectiveValue() / 1000000.0,  # Convert back from mm
            runtime=runtime,
            method="OR-Tools",
            metrics={
                "total_distance": total_distance / 1000000.0,  # Convert back to km
                "vehicles_used": len([r for r in routes.values() if len(r) > 2])
            }
        )
        
        return result
    else:
        raise Exception("OR-Tools could not find a solution")


if __name__ == "__main__":
    test_with_vehicle_fixed_cost()
