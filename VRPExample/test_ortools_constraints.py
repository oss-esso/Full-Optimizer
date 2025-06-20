#!/usr/bin/env python3
"""
Test script to specifically debug time windows and fractional cargo issues in OR-Tools.
"""

import sys
import os
import time
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from vrp_scenarios import get_all_scenarios
from vrp_optimizer_fixed import VRPQuantumOptimizer
from vrp_data_models import VRPObjective

def test_moda_small_with_debug():
    """Focus on the MODA_small scenario with detailed debug output."""
    print("\n🔍 Testing MODA_small with detailed OR-Tools debugging")
    print("=" * 70)
    
    # Load the scenario
    scenarios = get_all_scenarios()
    scenario = scenarios["MODA_small"]
    
    # Print basic info
    print(f"\nScenario: {scenario.name}")
    print(f"Locations: {len(scenario.locations)}")
    print(f"Vehicles: {len(scenario.vehicles)}")
    print(f"Ride requests: {len(scenario.ride_requests)}")
    
    # Check vehicle constraints
    vehicle = next(iter(scenario.vehicles.values()))
    print(f"\nVehicle constraints:")
    print(f"- Max time: {vehicle.max_time} minutes ({vehicle.max_time / 60} hours)")
    print(f"- Capacity: {vehicle.capacity}")
    
    # Check all time windows
    locations_with_tw = 0
    invalid_tw = 0
    tight_tw = 0
    tw_min_size = float('inf')
    tw_max_size = 0
    
    print("\nAnalyzing time windows:")
    for loc_id, loc in scenario.locations.items():
        if hasattr(loc, 'time_window_start') and loc.time_window_start is not None:
            locations_with_tw += 1
            start = loc.time_window_start
            end = loc.time_window_end
            size = end - start
            
            tw_min_size = min(tw_min_size, size)
            tw_max_size = max(tw_max_size, size)
            
            if size <= 0:
                invalid_tw += 1
                print(f"  ❌ Invalid time window at {loc_id}: [{start}, {end}] (size: {size})")
            elif size < 30:  # Less than 30 minutes is tight
                tight_tw += 1
                print(f"  ⚠️ Tight time window at {loc_id}: [{start}, {end}] (size: {size} minutes)")
    
    print(f"\nTime window summary:")
    print(f"- Locations with time windows: {locations_with_tw}/{len(scenario.locations)}")
    print(f"- Invalid time windows: {invalid_tw}")
    print(f"- Tight time windows: {tight_tw}")
    print(f"- Min window size: {tw_min_size if tw_min_size != float('inf') else 'N/A'} minutes")
    print(f"- Max window size: {tw_max_size} minutes")
    
    # Check fractional passenger counts
    fractional_requests = 0
    total_passengers = 0
    
    print("\nAnalyzing passenger counts:")
    for i, req in enumerate(scenario.ride_requests):
        total_passengers += req.passengers
        if req.passengers != int(req.passengers):
            fractional_requests += 1
            print(f"  Fractional passengers in request {i}: {req.passengers}")
    
    print(f"\nPassenger count summary:")
    print(f"- Total requests: {len(scenario.ride_requests)}")
    print(f"- Requests with fractional passengers: {fractional_requests}")
    print(f"- Total passenger count: {total_passengers}")
    print(f"- Average passengers per request: {total_passengers / len(scenario.ride_requests):.2f}")
    
    # Capacity constraint check
    total_capacity = sum(v.capacity for v in scenario.vehicles.values())
    print(f"\nCapacity check:")
    print(f"- Total capacity: {total_capacity}")
    print(f"- Total demand: {total_passengers}")
    print(f"- Utilization: {total_passengers / total_capacity:.2%}")
    
    if total_passengers > total_capacity:
        print("  ❌ ERROR: Total passenger count exceeds total vehicle capacity!")
    elif total_passengers / total_capacity > 0.9:
        print("  ⚠️ WARNING: High capacity utilization may make finding feasible solutions difficult")
    
    # Debug OR-Tools specific issues
    print("\n🔧 Running modified OR-Tools solver with debug output...")
    
    # Create a custom optimization run that outputs detailed constraint info
    run_debug_ortools_solver(scenario)

def run_debug_ortools_solver(instance):
    """Run OR-Tools solver with detailed debug output for constraints."""
    print("\nDebug OR-Tools Solver")
    print("-" * 50)
    
    start_time = time.time()
    is_ride_pooling = bool(instance.ride_requests)
    
    try:
        # Create location mapping for pickup-delivery scenario
        depot_locations = [loc_id for loc_id in instance.location_ids if loc_id.startswith("depot")]
        depot_id = depot_locations[0] if depot_locations else instance.location_ids[0]
        
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
        num_vehicles = len(instance.vehicles)
        depot_index = 0  # Single depot at index 0
        
        # Create pickup-delivery pairs for constraints
        pickup_deliveries = []
        for request in instance.ride_requests:
            pickup_idx = location_to_index[request.pickup_location]
            dropoff_idx = location_to_index[request.dropoff_location]
            pickup_deliveries.append([pickup_idx, dropoff_idx])
            print(f"Request: {request.id}, Pickup: {request.pickup_location} (idx {pickup_idx}), "
                  f"Dropoff: {request.dropoff_location} (idx {dropoff_idx}), Passengers: {request.passengers}")
        
        # Create routing index manager and model
        manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_index)
        routing = pywrapcp.RoutingModel(manager)
        
        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            
            from_location_id = location_list[from_node]
            to_location_id = location_list[to_node]
            
            return int(instance.get_distance(from_location_id, to_location_id))
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add distance dimension
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            100000,  # vehicle maximum travel distance (high limit)
            True,  # start cumul to zero
            'Distance'
        )
        distance_dimension = routing.GetDimensionOrDie('Distance')
        distance_dimension.SetGlobalSpanCostCoefficient(100)
        
        # Get vehicle capacities
        vehicle_capacities = []
        for vehicle_id in sorted(instance.vehicles.keys()):
            vehicle = instance.vehicles[vehicle_id]
            capacity = getattr(vehicle, 'capacity', 100)
            vehicle_capacities.append(capacity)
        
        print(f"\nVehicle capacities: {vehicle_capacities}")
        
        # Debug demand callback
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            location_id = location_list[from_node]
            
            demand = 0
            if location_id.startswith("depot"):
                demand = 0
            elif is_ride_pooling:
                # For ride pooling: pickups add passengers, dropoffs remove them
                if location_id.startswith("pickup"):
                    # Find the request for this pickup
                    for request in instance.ride_requests:
                        if request.pickup_location == location_id:
                            demand = int(request.passengers)  # TEST: round to integer
                            # Print first few demand values
                            if from_index < 10:
                                print(f"Demand at pickup {location_id}: {demand} (original: {request.passengers})")
                            break
                elif location_id.startswith("dropoff"):
                    # Dropoffs have negative demand
                    for request in instance.ride_requests:
                        if request.dropoff_location == location_id:
                            demand = -int(request.passengers)  # TEST: round to integer
                            # Print first few demand values
                            if from_index < 10:
                                print(f"Demand at dropoff {location_id}: {demand} (original: {-request.passengers})")
                            break
            
            return demand
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            vehicle_capacities,  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity'
        )
        
        capacity_dimension = routing.GetDimensionOrDie('Capacity')
        
        # Add pickup and delivery constraints
        for pickup_idx, delivery_idx in pickup_deliveries:
            pickup_node = manager.NodeToIndex(pickup_idx)
            delivery_node = manager.NodeToIndex(delivery_idx)
            routing.AddPickupAndDelivery(pickup_node, delivery_node)
            
            # Ensure pickup and delivery are on the same route
            routing.solver().Add(
                routing.VehicleVar(pickup_node) == routing.VehicleVar(delivery_node)
            )
            
            # Ensure pickup comes before delivery
            routing.solver().Add(
                capacity_dimension.CumulVar(pickup_node) <= 
                capacity_dimension.CumulVar(delivery_node)
            )
        
        # Add time windows constraints
        print("\nSetting up time dimension with time windows:")
        
        # Use 50 km/h average speed for time estimation
        vehicle_time = getattr(next(iter(instance.vehicles.values())), "max_time", None)
        max_time_seconds = int(vehicle_time * 60) if vehicle_time else 36000  # Default to 10 hours
        print(f"Max time per vehicle: {max_time_seconds/60:.1f} minutes ({max_time_seconds/3600:.1f} hours)")
        
        def time_callback(from_index, to_index):
            """Returns travel time between locations in seconds."""
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            from_location_id = location_list[from_node]
            to_location_id = location_list[to_node]
            
            # Get travel time using instance's duration method or estimate from distance
            try:
                travel_time = instance.get_duration(from_location_id, to_location_id)
            except (AttributeError, NotImplementedError):
                # Fallback: approximate from distance using 50 km/h speed
                distance = instance.get_distance(from_location_id, to_location_id)
                distance_km = distance * 111  # Approximate conversion
                travel_time = (distance_km / 50) * 3600  # seconds at 50 km/h
            
            # Add service time (10 minutes at pickup/dropoff)
            service_time = 0
            if to_location_id.startswith("pickup") or to_location_id.startswith("dropoff"):
                service_time = 600  # 10 minutes in seconds
            
            return int(travel_time + service_time)
        
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        
        # Add time dimension
        routing.AddDimension(
            time_callback_index,
            300,  # Allow wait time (5 minutes)
            max_time_seconds,  # Max time per vehicle
            False,  # Don't force start cumul to zero (allow flexible depot departure)
            'Time'
        )
        time_dimension = routing.GetDimensionOrDie('Time')
        
        # Add time window constraints for all locations
        time_windows_added = 0
        for loc_id, location in instance.locations.items():
            if loc_id in location_to_index and hasattr(location, 'time_window_start') and location.time_window_start is not None:
                loc_idx = location_to_index[loc_id]
                node_index = manager.NodeToIndex(loc_idx)
                
                start_min = int(location.time_window_start * 60)  # Convert to seconds
                end_max = int(location.time_window_end * 60)  # Convert to seconds
                
                # Fix potential issues: ensure end_max > start_min
                if end_max <= start_min:
                    print(f"⚠️ Fixing invalid time window for {loc_id}: [{start_min/60}, {end_max/60}]")
                    end_max = start_min + 3600  # Add 60 min buffer
                
                time_dimension.CumulVar(node_index).SetRange(start_min, end_max)
                time_windows_added += 1
                
                if time_windows_added <= 5 or time_windows_added % 10 == 0:
                    print(f"Added time window for {loc_id}: [{start_min/60:.1f}, {end_max/60:.1f}] minutes")
        
        print(f"Added {time_windows_added} time windows")
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(30)  # 30 seconds for debugging
        
        # Solve the problem
        print("\nSolving with OR-Tools (debug mode)...")
        solution = routing.SolveWithParameters(search_parameters)
        
        # Output solution
        if solution:
            print("\n✅ Solution found!")
            
            # Extract solution
            routes = {}
            total_distance = 0
            total_time = 0
            vehicle_ids = sorted(instance.vehicles.keys())
            
            for vehicle_idx in range(num_vehicles):
                vehicle_id = vehicle_ids[vehicle_idx]
                route = [depot_id]  # Start at depot
                
                index = routing.Start(vehicle_idx)
                route_distance = 0
                route_time = 0
                
                # Track time at each stop
                times_at_stops = []
                capacities_at_stops = []
                
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    
                    if node_index != 0:  # Skip depot nodes in middle
                        location_id = location_list[node_index]
                        route.append(location_id)
                        
                        # Get time and capacity at this stop
                        time_var = time_dimension.CumulVar(index)
                        capacity_var = capacity_dimension.CumulVar(index)
                        
                        times_at_stops.append((location_id, 
                                             solution.Min(time_var)/60, 
                                             solution.Max(time_var)/60))
                        capacities_at_stops.append((location_id,
                                                  solution.Min(capacity_var),
                                                  solution.Max(capacity_var)))
                    
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_idx)
                    
                    # Get time and distance for this leg
                    if previous_index != index:
                        route_time += solution.Value(time_dimension.CumulVar(index)) - solution.Value(time_dimension.CumulVar(previous_index))
                
                # Add depot at end
                route.append(depot_id)
                
                # Only include routes that actually visit locations
                if len(route) > 2:  # More than just start/end depot
                    routes[vehicle_id] = route
                    total_distance += route_distance
                    total_time += route_time
                    
                    print(f"\n🚗 Vehicle {vehicle_id} route:")
                    print(f"  Stops: {len(route) - 2}")
                    print(f"  Distance: {route_distance}")
                    print(f"  Time: {route_time/60:.1f} minutes")
                    print(f"  Route: {' -> '.join(route)}")
                    
                    print(f"\n  Time at each stop (minutes):")
                    for loc_id, min_time, max_time in times_at_stops:
                        is_pickup = loc_id.startswith("pickup")
                        is_dropoff = loc_id.startswith("dropoff")
                        type_str = "P" if is_pickup else "D" if is_dropoff else " "
                        
                        # Check if this location has a time window
                        has_tw = False
                        tw_str = ""
                        if loc_id in instance.locations:
                            loc = instance.locations[loc_id]
                            if hasattr(loc, 'time_window_start') and loc.time_window_start is not None:
                                has_tw = True
                                tw_start = loc.time_window_start
                                tw_end = loc.time_window_end
                                tw_str = f" (TW: [{tw_start}, {tw_end}])"
                        
                        print(f"    {type_str} {loc_id}: {min_time:.1f} - {max_time:.1f}{tw_str}")
                    
                    print(f"\n  Capacity at each stop:")
                    for loc_id, min_cap, max_cap in capacities_at_stops:
                        print(f"    {loc_id}: {min_cap} - {max_cap}")
            
            # Print overall solution metrics
            print(f"\n📊 Overall solution:")
            print(f"  Total distance: {total_distance}")
            print(f"  Total time: {total_time/60:.1f} minutes")
            print(f"  Vehicles used: {len(routes)}/{num_vehicles}")
            print(f"  Runtime: {(time.time() - start_time) * 1000:.1f}ms")
            
        else:
            print("\n❌ No solution found!")
            print("Analyzing potential reasons for infeasibility:")
            
            # Check for time window conflicts
            print("\n⏰ Checking time window feasibility:")
            tw_locations = []
            for loc_id, loc in instance.locations.items():
                if hasattr(loc, 'time_window_start') and loc.time_window_start is not None:
                    tw_locations.append((loc_id, loc.time_window_start, loc.time_window_end))
            
            # Sort by start time to check for conflicts
            tw_locations.sort(key=lambda x: x[1])
            print(f"  Time windows (sorted by start time):")
            for loc_id, start, end in tw_locations[:10]:  # Show first 10
                print(f"    {loc_id}: [{start}, {end}] (duration: {end-start})")
            
            if len(tw_locations) > 10:
                print(f"    ... and {len(tw_locations)-10} more time windows")
            
            # Check pickup-delivery time window conflicts
            print("\n🔄 Checking pickup-delivery time window conflicts:")
            for request in instance.ride_requests:
                pickup_id = request.pickup_location
                dropoff_id = request.dropoff_location
                
                pickup_tw = (None, None)
                dropoff_tw = (None, None)
                
                if pickup_id in instance.locations:
                    pickup_loc = instance.locations[pickup_id]
                    if hasattr(pickup_loc, 'time_window_start') and pickup_loc.time_window_start is not None:
                        pickup_tw = (pickup_loc.time_window_start, pickup_loc.time_window_end)
                
                if dropoff_id in instance.locations:
                    dropoff_loc = instance.locations[dropoff_id]
                    if hasattr(dropoff_loc, 'time_window_start') and dropoff_loc.time_window_start is not None:
                        dropoff_tw = (dropoff_loc.time_window_start, dropoff_loc.time_window_end)
                
                if pickup_tw[0] is not None and dropoff_tw[0] is not None:
                    # Check if pickup must happen after dropoff due to time windows
                    if pickup_tw[0] >= dropoff_tw[1]:
                        print(f"  ❌ CONFLICT in request {request.id}: Pickup {pickup_id} window [{pickup_tw[0]}, {pickup_tw[1]}] "
                              f"starts after dropoff {dropoff_id} window [{dropoff_tw[0]}, {dropoff_tw[1]}] ends")
                    elif pickup_tw[0] > dropoff_tw[0]:
                        # This is not necessarily a conflict, but might be tight
                        travel_time = instance.get_duration(pickup_id, dropoff_id) / 60  # convert to minutes
                        buffer = dropoff_tw[1] - pickup_tw[0]
                        
                        if buffer < travel_time:
                            print(f"  ❌ CONFLICT in request {request.id}: Not enough time between pickup {pickup_id} "
                                  f"and dropoff {dropoff_id}, need {travel_time:.1f} min but have {buffer:.1f} min")
                        elif buffer < travel_time * 1.5:
                            print(f"  ⚠️ TIGHT timing in request {request.id}: Only {buffer:.1f} minutes between pickup "
                                  f"{pickup_id} and dropoff {dropoff_id}, travel time is {travel_time:.1f} min")
                    
                    # Check if latest pickup + travel time exceeds earliest dropoff
                    if pickup_tw[1] is not None and dropoff_tw[0] is not None:
                        # Calculate minimum travel time between pickup and dropoff
                        try:
                            travel_time = instance.get_duration(pickup_id, dropoff_id) / 60  # convert to minutes
                            if pickup_tw[1] + travel_time > dropoff_tw[0]:
                                print(f"  ⚠️ Potential timing issue: Latest pickup ({pickup_tw[1]}) + travel time ({travel_time:.1f} min) "
                                      f"exceeds earliest dropoff ({dropoff_tw[0]})")
                        except:
                            pass
            
            # Check vehicle time constraints
            vehicle_time = getattr(next(iter(instance.vehicles.values())), "max_time", None)
            if vehicle_time:
                max_tw_end = max(loc.time_window_end for loc in instance.locations.values() 
                                if hasattr(loc, 'time_window_end') and loc.time_window_end is not None)
                
                print(f"\n🕒 Vehicle time constraint check:")
                print(f"  Vehicle max time: {vehicle_time} minutes ({vehicle_time/60:.1f} hours)")
                print(f"  Latest time window end: {max_tw_end} minutes ({max_tw_end/60:.1f} hours)")
                
                if max_tw_end > vehicle_time:
                    print(f"  ❌ CONFLICT: Latest time window ({max_tw_end}) exceeds vehicle time limit ({vehicle_time})")
    
    except Exception as e:
        print(f"Error in debug OR-Tools solver: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point."""
    print("=" * 70)
    print("DEBUGGING TIME WINDOWS AND FRACTIONAL CARGO IN OR-TOOLS")
    print("=" * 70)
    
    # Run main test
    test_moda_small_with_debug()

if __name__ == "__main__":
    main()
