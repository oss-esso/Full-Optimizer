#!/usr/bin/env python3
"""
Debug script for OR-Tools MODA scenarios - step by step debugging
"""
import sys
import os
import time

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from vrp_scenarios import get_all_scenarios
from vrp_data_models import VRPObjective

# Import OR-Tools directly for debugging
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
    print("OR-Tools successfully imported for debugging")
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("Error: OR-Tools not available")
    exit(1)

def debug_moda_ortools():
    """Debug OR-Tools on MODA scenario with detailed output"""
    print("Debugging OR-Tools MODA Implementation")
    print("=" * 60)
    
    # Get MODA_small for easier debugging
    scenarios = get_all_scenarios()
    instance = scenarios['MODA_small']  # Start with smaller scenario
    
    print(f'\nMODA_small Details:')
    print(f'  Total locations: {len(instance.locations)}')
    print(f'  Location IDs: {instance.location_ids[:10]}...')  # First 10
    print(f'  Vehicles: {len(instance.vehicles)}')
    print(f'  Vehicle IDs: {list(instance.vehicles.keys())}')
    print(f'  Ride requests: {len(instance.ride_requests)}')
    
    # Print vehicle depot assignments
    print(f'\nVehicle-Depot assignments:')
    for vid, vehicle in instance.vehicles.items():
        print(f'  {vid}: depot={vehicle.depot_id}')
    
    # Print some ride requests
    print(f'\nSample ride requests:')
    for i, req in enumerate(list(instance.ride_requests)[:5]):
        print(f'  Request {i+1}: {req.pickup_location} -> {req.dropoff_location}')
    
    # Find depots
    depot_locations = [loc_id for loc_id in instance.location_ids if loc_id.startswith("depot")]
    print(f'\nDepots found: {depot_locations}')
    
    print(f'\n' + '='*60)
    print('Starting OR-Tools Implementation Debug')
    print('='*60)
    
    start_time = time.time()
    
    try:
        # Step 1: Check if this is ride pooling
        is_ride_pooling = bool(instance.ride_requests)
        print(f'1. Ride pooling detected: {is_ride_pooling}')
        
        if not is_ride_pooling:
            print('ERROR: This should be a ride pooling scenario!')
            return
        
        # Step 2: Build location list for ride pooling
        print(f'\n2. Building location list for pickup-delivery...')
        
        # Check for multi-depot
        depot_locations = [loc_id for loc_id in instance.location_ids if loc_id.startswith("depot")]
        is_multi_depot = len(depot_locations) > 1
        print(f'   Multi-depot scenario: {is_multi_depot}')
        print(f'   Depots: {depot_locations}')
        
        if is_multi_depot:
            # Multi-depot: include all depot locations
            location_list = depot_locations.copy()
            location_to_index = {loc: idx for idx, loc in enumerate(depot_locations)}
            
            # Group vehicles by depot
            vehicles_by_depot = {}
            for vid, vehicle in instance.vehicles.items():
                depot = vehicle.depot_id
                if depot not in vehicles_by_depot:
                    vehicles_by_depot[depot] = []
                vehicles_by_depot[depot].append(vid)
            
            print(f'   Vehicles by depot: {vehicles_by_depot}')
        else:
            # Single depot
            depot_id = depot_locations[0]
            location_list = [depot_id]
            location_to_index = {depot_id: 0}
        
        # Add pickup and dropoff locations
        for request in instance.ride_requests:
            if request.pickup_location not in location_to_index:
                location_to_index[request.pickup_location] = len(location_list)
                location_list.append(request.pickup_location)
            if request.dropoff_location not in location_to_index:
                location_to_index[request.dropoff_location] = len(location_list)
                location_list.append(request.dropoff_location)
        
        num_locations = len(location_list)
        print(f'   Total locations in model: {num_locations}')
        print(f'   Location list (first 10): {location_list[:10]}')
        
        # Step 3: Create pickup-delivery pairs
        print(f'\n3. Creating pickup-delivery pairs...')
        pickup_deliveries = []
        for i, request in enumerate(instance.ride_requests):
            pickup_idx = location_to_index[request.pickup_location]
            dropoff_idx = location_to_index[request.dropoff_location]
            pickup_deliveries.append([pickup_idx, dropoff_idx])
            if i < 5:  # Show first 5
                print(f'   Pair {i+1}: {request.pickup_location}({pickup_idx}) -> {request.dropoff_location}({dropoff_idx})')
        
        print(f'   Total pickup-delivery pairs: {len(pickup_deliveries)}')
        
        # Step 4: Setup OR-Tools manager
        print(f'\n4. Setting up OR-Tools routing model...')
        
        num_vehicles = len(instance.vehicles)
        if is_multi_depot:
            # For multi-depot, we need depot indices for each vehicle
            depot_indices = []
            vehicle_ids = sorted(instance.vehicles.keys())
            for vid in vehicle_ids:
                depot_id = instance.vehicles[vid].depot_id
                depot_idx = location_to_index[depot_id]
                depot_indices.append(depot_idx)
            print(f'   Vehicle depot mapping: {list(zip(vehicle_ids[:5], depot_indices[:5]))}...')  # First 5
            
            manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_indices, depot_indices)
        else:
            depot_index = location_to_index[depot_locations[0]]
            manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_index)
        
        routing = pywrapcp.RoutingModel(manager)
        print(f'   Routing model created: {num_locations} locations, {num_vehicles} vehicles')
          # Step 5: Distance callback
        print(f'\n5. Setting up distance matrix...')
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            from_loc = location_list[from_node]
            to_loc = location_list[to_node]
            distance = instance.get_distance(from_loc, to_loc)
            # Debug: check actual distance calculation
            if from_node < 5 and to_node < 5:  # Debug first few
                print(f'     Distance calc: {from_loc} -> {to_loc} = {distance}')
            return int(distance * 100)  # Scale up to avoid floating point issues
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Test a few distance calculations
        print(f'   Sample distances:')
        for i in range(min(3, num_locations-1)):
            dist = distance_callback(manager.NodeToIndex(i), manager.NodeToIndex(i+1))
            print(f'     {location_list[i]} -> {location_list[i+1]}: {dist}')
        
        # Step 6: Pickup-delivery constraints
        print(f'\n6. Adding pickup-delivery constraints...')
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            1000000,  # maximum distance per vehicle
            True,  # start cumul to zero
            "Distance"
        )
        distance_dimension = routing.GetDimensionOrDie("Distance")
        
        for pickup_index, delivery_index in pickup_deliveries:
            pickup_node_index = manager.NodeToIndex(pickup_index)
            delivery_node_index = manager.NodeToIndex(delivery_index)
            routing.AddPickupAndDelivery(pickup_node_index, delivery_node_index)
            routing.solver().Add(
                routing.VehicleVar(pickup_node_index) == routing.VehicleVar(delivery_node_index)
            )
            routing.solver().Add(
                distance_dimension.CumulVar(pickup_node_index) <= distance_dimension.CumulVar(delivery_node_index)
            )
        
        print(f'   Added {len(pickup_deliveries)} pickup-delivery constraints')
        
        # Step 7: Solve
        print(f'\n7. Solving with OR-Tools...')
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.FromSeconds(5)  # Short time limit for testing
        
        solution = routing.SolveWithParameters(search_parameters)
        
        # Step 8: Analyze solution
        print(f'\n8. Solution Analysis:')
        if solution:
            print(f'   Solution found!')
            print(f'   Objective value: {solution.ObjectiveValue()}')
            
            # Extract routes
            print(f'\n   Extracting routes...')
            routes = {}
            vehicle_ids = sorted(instance.vehicles.keys())
            total_distance = 0
            vehicles_used = 0
            
            for vehicle_idx in range(num_vehicles):
                if vehicle_idx < len(vehicle_ids):
                    vehicle_id = vehicle_ids[vehicle_idx]
                    
                    index = routing.Start(vehicle_idx)
                    route = []
                    route_distance = 0
                    
                    print(f'   \n   Vehicle {vehicle_idx} ({vehicle_id}):')
                    print(f'     Start index: {index}')
                    
                    while not routing.IsEnd(index):
                        node_index = manager.IndexToNode(index)
                        location = location_list[node_index]
                        route.append(location)
                        print(f'     -> {location} (node {node_index}, index {index})')
                        
                        previous_index = index
                        index = solution.Value(routing.NextVar(index))
                        route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_idx)
                    
                    # Add end depot
                    end_node_index = manager.IndexToNode(index)
                    end_location = location_list[end_node_index]
                    route.append(end_location)
                    print(f'     -> {end_location} (END node {end_node_index}, index {index})')
                    
                    if len(route) > 2:  # More than just start->end depot
                        routes[vehicle_id] = route
                        total_distance += route_distance
                        vehicles_used += 1
                        print(f'     Route distance: {route_distance}')
                        print(f'     Route: {" -> ".join(route)}')
                    else:
                        print(f'     Empty route (only depot)')
            
            print(f'\n   Final Results:')
            print(f'   Total routes: {len(routes)}')
            print(f'   Vehicles used: {vehicles_used}')
            print(f'   Total distance: {total_distance}')
            print(f'   OR-Tools objective: {solution.ObjectiveValue()}')
            
        else:
            print(f'   No solution found!')
        
        runtime = time.time() - start_time
        print(f'\n   Total runtime: {runtime:.2f}s')
        
    except Exception as e:
        print(f'ERROR: {str(e)}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_moda_ortools()
