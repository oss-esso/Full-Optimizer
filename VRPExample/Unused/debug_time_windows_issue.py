#!/usr/bin/env python3
"""
Simple script to debug time window issues in MODA scenarios.
"""

import os
import sys
import json

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from vrp_scenarios import create_moda_small_scenario
from vrp_data_models import VRPInstance

def analyze_time_windows(scenario: VRPInstance):
    """Analyze time window implementation in the given scenario."""
    print(f"\n=== Time Window Analysis for {scenario.name} ===")
    
    time_windows = []
    problematic_pairs = []
    
    # Collect all time window info
    for loc_id, location in scenario.locations.items():
        if hasattr(location, 'time_window_start') and location.time_window_start is not None:
            start = location.time_window_start
            end = location.time_window_end if hasattr(location, 'time_window_end') else None
            window_size = end - start if end is not None else None
            
            time_windows.append({
                'location_id': loc_id,
                'start': start,
                'end': end,
                'window_size': window_size
            })
            
            if window_size is not None and window_size <= 0:
                print(f"🔴 ERROR: Location {loc_id} has invalid time window: [{start}, {end}] (size: {window_size})")
    
    # Check pickup-dropoff pairs
    if scenario.ride_requests:
        for req in scenario.ride_requests:
            pickup_loc = scenario.locations.get(req.pickup_location)
            dropoff_loc = scenario.locations.get(req.dropoff_location)
            
            if pickup_loc and dropoff_loc:
                pickup_tw_start = getattr(pickup_loc, 'time_window_start', None)
                pickup_tw_end = getattr(pickup_loc, 'time_window_end', None)
                dropoff_tw_start = getattr(dropoff_loc, 'time_window_start', None)
                dropoff_tw_end = getattr(dropoff_loc, 'time_window_end', None)
                
                if all(tw is not None for tw in [pickup_tw_start, pickup_tw_end, dropoff_tw_start, dropoff_tw_end]):
                    # Check if pickup window ends after dropoff window starts
                    if pickup_tw_end > dropoff_tw_start:
                        problem = {
                            'request_id': req.id,
                            'pickup_window': f"[{pickup_tw_start}, {pickup_tw_end}]",
                            'dropoff_window': f"[{dropoff_tw_start}, {dropoff_tw_end}]",
                            'gap': dropoff_tw_start - pickup_tw_end
                        }
                        problematic_pairs.append(problem)
                        print(f"🔴 ERROR: Request {req.id} - Pickup ends at {pickup_tw_end}, dropoff starts at {dropoff_tw_start} (gap: {problem['gap']})")
                    
                    # Check for very tight windows (less than 60 minutes total)
                    total_available_time = dropoff_tw_end - pickup_tw_start
                    if total_available_time < 60:
                        print(f"⚠️ WARNING: Request {req.id} has very tight time constraint: total {total_available_time} minutes")
    
    # Summary statistics
    if time_windows:
        starts = [tw['start'] for tw in time_windows]
        ends = [tw['end'] for tw in time_windows]
        sizes = [tw['window_size'] for tw in time_windows if tw['window_size'] is not None]
        
        print(f"\n📊 Time Window Statistics:")
        print(f"   Total locations with time windows: {len(time_windows)}")
        print(f"   Start times: {min(starts)} - {max(starts)} (span: {max(starts) - min(starts)} min)")
        print(f"   End times: {min(ends)} - {max(ends)} (span: {max(ends) - min(ends)} min)")
        print(f"   Window sizes: {min(sizes)} - {max(sizes)} (avg: {sum(sizes)/len(sizes):.1f} min)")
        print(f"   Invalid windows: {len([s for s in sizes if s <= 0])}")
        print(f"   Problematic pickup-dropoff pairs: {len(problematic_pairs)}")
        
        return {
            'total_time_windows': len(time_windows),
            'invalid_windows': len([s for s in sizes if s <= 0]),
            'problematic_pairs': len(problematic_pairs),
            'min_window_size': min(sizes) if sizes else None,
            'max_window_size': max(sizes) if sizes else None
        }
    else:
        print(f"   No time windows found")
        return {'total_time_windows': 0}

def test_basic_ortools_compatibility():
    """Test basic OR-Tools compatibility without time windows."""
    print("\n=== OR-Tools Compatibility Test ===")
    
    try:
        from ortools.constraint_solver import routing_enums_pb2
        from ortools.constraint_solver import pywrapcp
        print("✅ OR-Tools imported successfully")
        
        # Test simple model creation
        manager = pywrapcp.RoutingIndexManager(5, 2, 0)  # 5 locations, 2 vehicles, depot at 0
        routing = pywrapcp.RoutingModel(manager)
        print("✅ Basic routing model created successfully")
        
        return True
    except Exception as e:
        print(f"❌ OR-Tools error: {e}")
        return False

def main():
    print("Debugging Time Window Issues in MODA Scenarios")
    print("=" * 60)
    
    # Test OR-Tools first
    ortools_ok = test_basic_ortools_compatibility()
    
    if not ortools_ok:
        print("❌ Cannot proceed - OR-Tools not working properly")
        return
    
    try:
        # Create MODA small scenario
        print("\n🔧 Creating MODA_small scenario...")
        scenario = create_moda_small_scenario()
        
        # Analyze time windows
        analysis = analyze_time_windows(scenario)
        
        # Test with simplified scenario (no time windows)
        print("\n🔧 Testing scenario without time windows...")
        simplified_scenario = create_simplified_scenario(scenario)
        test_scenario_solvability(simplified_scenario, "Simplified (no time windows)")
        
        # Test with original scenario
        print("\n🔧 Testing original scenario with time windows...")
        test_scenario_solvability(scenario, "Original MODA_small")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def create_simplified_scenario(original_scenario: VRPInstance) -> VRPInstance:
    """Create a simplified version without time windows."""
    from vrp_data_models import VRPInstance, Location, Vehicle, RideRequest
    
    simplified = VRPInstance(f"{original_scenario.name} - Simplified")
    
    # Copy locations without time windows
    for loc_id, location in original_scenario.locations.items():
        new_loc = Location(loc_id, location.x, location.y, location.demand)
        if hasattr(location, 'address'):
            new_loc.address = location.address
        if hasattr(location, 'lat'):
            new_loc.lat = location.lat
        if hasattr(location, 'lon'):
            new_loc.lon = location.lon
        simplified.add_location(new_loc)
    
    # Copy vehicles
    for vehicle_id, vehicle in original_scenario.vehicles.items():
        new_vehicle = Vehicle(vehicle_id, vehicle.capacity, vehicle.depot_id)
        if hasattr(vehicle, 'max_time'):
            new_vehicle.max_time = vehicle.max_time
        simplified.add_vehicle(new_vehicle)
    
    # Copy ride requests
    if original_scenario.ride_requests:
        for request in original_scenario.ride_requests:
            new_request = RideRequest(request.id, request.pickup_location, request.dropoff_location, request.passengers)
            simplified.add_ride_request(new_request)
      # Copy distance matrix if available
    if hasattr(original_scenario, 'distance_matrix') and original_scenario.distance_matrix is not None:
        simplified.distance_matrix = original_scenario.distance_matrix
    
    return simplified

def test_scenario_solvability(scenario: VRPInstance, scenario_name: str):
    """Test if a scenario is solvable using OR-Tools with minimal configuration."""
    print(f"\n🧪 Testing {scenario_name}...")
    
    try:
        from ortools.constraint_solver import routing_enums_pb2
        from ortools.constraint_solver import pywrapcp
        
        # Create routing model
        num_locations = len(scenario.locations)
        num_vehicles = len(scenario.vehicles)
        depot_indices = [0]  # Simplified: use first location as depot
        
        print(f"   Locations: {num_locations}, Vehicles: {num_vehicles}")
        
        manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_indices)
        routing = pywrapcp.RoutingModel(manager)
        
        # Simple distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return abs(from_node - to_node) * 100  # Simple distance
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add time windows if they exist
        time_windows_added = False
        if any(hasattr(loc, 'time_window_start') and loc.time_window_start is not None 
               for loc in scenario.locations.values()):
            
            routing.AddDimension(
                transit_callback_index,
                30,    # Allow waiting time
                1440,  # Maximum time per vehicle
                False, # Don't force start cumul to zero
                'Time'
            )
            time_dimension = routing.GetDimensionOrDie('Time')
            
            # Add time window constraints
            for location_idx, (loc_id, location) in enumerate(scenario.locations.items()):
                if hasattr(location, 'time_window_start') and location.time_window_start is not None:
                    index = manager.NodeToIndex(location_idx)
                    time_dimension.CumulVar(index).SetRange(
                        int(location.time_window_start),
                        int(location.time_window_end)
                    )
            time_windows_added = True
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.time_limit.FromSeconds(30)  # Short time limit for testing
        
        # Solve
        assignment = routing.SolveWithParameters(search_parameters)
        
        if assignment:
            total_distance = routing.GetCost(assignment)
            print(f"   ✅ SOLVED - Distance: {total_distance}, Time windows: {time_windows_added}")
            return True
        else:
            print(f"   ❌ NOT SOLVED - Time windows: {time_windows_added}")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    main()
