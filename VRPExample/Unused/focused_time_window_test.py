#!/usr/bin/env python3
"""
Focused test to identify the exact time window issue.
"""

import os
import sys

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from vrp_scenarios import create_moda_small_scenario
from vrp_data_models import VRPInstance, Location, Vehicle, RideRequest

def create_minimal_time_window_test():
    """Create a minimal VRP scenario with time windows to test OR-Tools integration."""
    instance = VRPInstance("Minimal Time Window Test")
    
    # Add depot (no time window)
    instance.add_location(Location("depot", 0, 0, demand=0))
    
    # Add pickup location with time window
    pickup = Location("pickup_1", 10, 10, demand=0, 
                     time_window_start=100, time_window_end=200, service_time=15)
    instance.add_location(pickup)
    
    # Add dropoff location with time window
    dropoff = Location("dropoff_1", 20, 20, demand=0,
                      time_window_start=250, time_window_end=350, service_time=10)
    instance.add_location(dropoff)
    
    # Add vehicle
    instance.add_vehicle(Vehicle("vehicle_1", capacity=1000, depot_id="depot"))
    
    # Add ride request
    instance.add_ride_request(RideRequest("request_1", "pickup_1", "dropoff_1", passengers=100))
    
    # Calculate distance matrix
    instance.calculate_distance_matrix(distance_method="euclidean")
    
    return instance

def test_minimal_scenario_with_ortools(scenario: VRPInstance):
    """Test minimal scenario with OR-Tools to isolate time window issues."""
    print(f"\n🧪 Testing {scenario.name}...")
    
    try:
        from ortools.constraint_solver import routing_enums_pb2
        from ortools.constraint_solver import pywrapcp
        
        # Create routing model with single depot
        num_locations = len(scenario.locations)
        num_vehicles = len(scenario.vehicles)
        depot_index = 0  # Single depot at index 0
        
        print(f"   Creating model: {num_locations} locations, {num_vehicles} vehicles, depot at {depot_index}")
        
        manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_index)
        routing = pywrapcp.RoutingModel(manager)
        
        # Distance callback using the distance matrix
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            if hasattr(scenario, 'distance_matrix') and scenario.distance_matrix is not None:
                return int(scenario.distance_matrix[from_node][to_node])
            else:
                # Fallback calculation
                locations = list(scenario.locations.values())
                from_loc = locations[from_node]
                to_loc = locations[to_node]
                dx = from_loc.x - to_loc.x
                dy = from_loc.y - to_loc.y
                return int((dx*dx + dy*dy)**0.5)
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        print("   ✅ Basic model and distance callback created")
        
        # Test WITHOUT time windows first
        print("   🔧 Testing without time windows...")
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.time_limit.FromSeconds(10)
        
        assignment = routing.SolveWithParameters(search_parameters)
        if assignment:
            print("   ✅ Solved WITHOUT time windows")
        else:
            print("   ❌ Failed even WITHOUT time windows - basic infeasibility")
            return False
        
        # Now add time windows
        print("   🔧 Adding time windows...")
        routing.AddDimension(
            transit_callback_index,  # Use distance as time proxy
            300,  # Allow 300 units waiting time
            1440, # Maximum route duration (24 hours in minutes)
            False, # Don't force start cumul to zero
            'Time'
        )
        time_dimension = routing.GetDimensionOrDie('Time')
        
        # Add time window constraints for locations
        location_list = list(scenario.locations.values())
        for location_idx, location in enumerate(location_list):
            if hasattr(location, 'time_window_start') and location.time_window_start is not None:
                print(f"   Adding time window for {location.id}: [{location.time_window_start}, {location.time_window_end}]")
                index = manager.NodeToIndex(location_idx)
                time_dimension.CumulVar(index).SetRange(
                    int(location.time_window_start),
                    int(location.time_window_end)
                )
        
        print("   🔧 Testing WITH time windows...")
        assignment = routing.SolveWithParameters(search_parameters)
        
        if assignment:
            print("   ✅ Solved WITH time windows")
            
            # Print solution details
            total_distance = routing.GetCost(assignment)
            print(f"   Total distance: {total_distance}")
            
            # Print route
            for vehicle_id in range(routing.vehicles()):
                index = routing.Start(vehicle_id)
                route_distance = 0
                route_time = 0
                route = []
                
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    location = location_list[node_index]
                    
                    time_var = time_dimension.CumulVar(index)
                    time_value = assignment.Value(time_var)
                    
                    route.append(f"{location.id}@{time_value}")
                    
                    if not routing.IsEnd(assignment.Value(routing.NextVar(index))):
                        route_distance += routing.GetArcCostForVehicle(
                            index, assignment.Value(routing.NextVar(index)), vehicle_id)
                    
                    index = assignment.Value(routing.NextVar(index))
                
                if len(route) > 1:  # Vehicle was used
                    print(f"   Vehicle {vehicle_id}: {' -> '.join(route)} (distance: {route_distance})")
            
            return True
        else:
            print("   ❌ Failed WITH time windows - time window constraints too restrictive")
            
            # Try to analyze why it failed
            print("   🔍 Analyzing time window feasibility...")
            for location_idx, location in enumerate(location_list):
                if hasattr(location, 'time_window_start') and location.time_window_start is not None:
                    print(f"     {location.id}: window [{location.time_window_start}, {location.time_window_end}] (size: {location.time_window_end - location.time_window_start})")
            
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_moda_time_windows():
    """Analyze the specific time window configuration in MODA scenarios."""
    print("\n🔍 Analyzing MODA Time Window Configuration...")
    
    scenario = create_moda_small_scenario()
    
    # Find pickup-dropoff pairs and their time constraints
    print("\n📋 Pickup-Dropoff Time Window Analysis:")
    
    if scenario.ride_requests:
        for req in scenario.ride_requests:
            pickup_loc = scenario.locations.get(req.pickup_location)
            dropoff_loc = scenario.locations.get(req.dropoff_location)
            
            if pickup_loc and dropoff_loc:
                pickup_tw = (getattr(pickup_loc, 'time_window_start', None), 
                           getattr(pickup_loc, 'time_window_end', None))
                dropoff_tw = (getattr(dropoff_loc, 'time_window_start', None),
                            getattr(dropoff_loc, 'time_window_end', None))
                
                print(f"   {req.id}:")
                print(f"     Pickup:  {pickup_tw[0]:4} - {pickup_tw[1]:4} (size: {pickup_tw[1] - pickup_tw[0]:3})")
                print(f"     Dropoff: {dropoff_tw[0]:4} - {dropoff_tw[1]:4} (size: {dropoff_tw[1] - dropoff_tw[0]:3})")
                
                # Calculate minimum required time between pickup and dropoff
                if scenario.distance_matrix:
                    pickup_idx = list(scenario.locations.keys()).index(req.pickup_location)
                    dropoff_idx = list(scenario.locations.keys()).index(req.dropoff_location)
                    travel_distance = scenario.distance_matrix[pickup_idx][dropoff_idx]
                    min_travel_time = travel_distance  # Assuming 1 unit distance = 1 unit time
                    service_time = getattr(pickup_loc, 'service_time', 0)
                    total_min_time = min_travel_time + service_time
                    
                    gap = dropoff_tw[0] - pickup_tw[1]
                    print(f"     Travel distance: {travel_distance:.1f}, Min time needed: {total_min_time:.1f}")
                    print(f"     Gap between windows: {gap} (feasible: {gap >= total_min_time})")
                    
                    if gap < total_min_time:
                        print(f"     ⚠️ WARNING: Insufficient time gap!")
                
                print()

def main():
    print("Focused Time Window Issue Diagnosis")
    print("=" * 50)
    
    # Test 1: Minimal scenario
    print("\n🔧 Test 1: Minimal Time Window Scenario")
    minimal_scenario = create_minimal_time_window_test()
    success = test_minimal_scenario_with_ortools(minimal_scenario)
    
    if success:
        print("✅ Minimal time window scenario works - issue is in MODA complexity")
    else:
        print("❌ Even minimal time window scenario fails - fundamental OR-Tools issue")
    
    # Test 2: Analyze MODA time windows
    print("\n🔧 Test 2: MODA Time Window Analysis")
    analyze_moda_time_windows()

if __name__ == "__main__":
    main()
