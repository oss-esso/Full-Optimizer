#!/usr/bin/env python3
"""
Test script to verify trucking company time window logic.
Demonstrates that drivers can work 10-hour shifts starting at any time,
and time windows represent when locations are available (not driver constraints).
"""

import os
import sys
import random

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from vrp_data_models import VRPInstance, Location, Vehicle, RideRequest
from vrp_optimizer_fixed import VRPQuantumOptimizer
from vrp_data_models import VRPObjective

def create_test_trucking_scenario() -> VRPInstance:
    """
    Create a test scenario that demonstrates proper trucking company logic:
    - Time windows = when locations are available for service (can exceed 600 min)
    - Vehicle max_time = 600 min = driver can work max 10 hours per shift
    - Driver can start at any time and work 10 hours from that start time
    """
    random.seed(42)  # For reproducibility
    
    instance = VRPInstance("Trucking Company Test Scenario")
    instance.is_realistic = True
    
    print("Creating trucking company test scenario...")
    print("Demonstrating: Driver 10-hour shifts can start at any time")
    
    # Add depot (available most of the day)
    depot_start = 300  # 5 AM (300 minutes from midnight)
    depot_end = 1200   # 8 PM (1200 minutes from midnight) = 15-hour availability
    depot = Location("depot", 0, 0, demand=0, 
                    time_window_start=depot_start, time_window_end=depot_end, service_time=10)
    instance.add_location(depot)
    print(f"Depot available: {depot_start} to {depot_end} minutes ({depot_start//60}:00 to {depot_end//60}:00)")
    
    # Add vehicle with 10-hour limit
    vehicle = Vehicle("truck_1", capacity=10, depot_id="depot", max_time=600)  # 10 hours max
    instance.add_vehicle(vehicle)
    print(f"Vehicle max driving time: 600 minutes (10 hours)")
    
    # Add pickup location (available afternoon/evening)
    pickup_start = 600  # 10 AM (600 minutes from midnight)
    pickup_end = 1020   # 5 PM (1020 minutes from midnight) = 7-hour window
    pickup = Location("pickup_1", 10, 10, demand=0,
                     time_window_start=pickup_start, time_window_end=pickup_end, service_time=15)
    instance.add_location(pickup)
    print(f"Pickup available: {pickup_start} to {pickup_end} minutes ({pickup_start//60}:00 to {pickup_end//60}:00)")
    
    # Add dropoff location (available late afternoon/evening)
    dropoff_start = 720  # 12 PM (720 minutes from midnight)
    dropoff_end = 1140   # 7 PM (1140 minutes from midnight) = 7-hour window
    dropoff = Location("dropoff_1", 20, 20, demand=0,
                      time_window_start=dropoff_start, time_window_end=dropoff_end, service_time=10)
    instance.add_location(dropoff)
    print(f"Dropoff available: {dropoff_start} to {dropoff_end} minutes ({dropoff_start//60}:00 to {dropoff_end//60}:00)")
    
    # Add ride request
    request = RideRequest("transport_1", "pickup_1", "dropoff_1", passengers=5)
    instance.add_ride_request(request)
    
    # Calculate distance matrix
    instance.calculate_distance_matrix(distance_method="manhattan")
    
    print("\nScenario Summary:")
    print("- Depot: 5 AM to 8 PM (15-hour availability)")
    print("- Pickup: 10 AM to 5 PM (7-hour availability)")  
    print("- Dropoff: 12 PM to 7 PM (7-hour availability)")
    print("- Driver limit: 10 hours maximum")
    print("- Dropoff window ends at 1140 min (19:00), which exceeds 600 min")
    print("- But driver can start at 9:00 (540 min) and work until 19:00 (1140 min) = 10 hours")
    
    return instance

def test_scenario_feasibility():
    """Test if the trucking scenario is feasible with OR-Tools."""
    scenario = create_test_trucking_scenario()
    
    print("\n" + "="*60)
    print("TESTING SCENARIO FEASIBILITY")
    print("="*60)
    
    # Test with OR-Tools
    optimizer = VRPQuantumOptimizer(scenario, VRPObjective.MINIMIZE_DISTANCE)
    result = optimizer.optimize_with_ortools()
    
    is_solved = result.status == "optimal" and result.objective_value > 0
    
    print(f"\nOR-Tools Result:")
    print(f"Status: {result.status}")
    print(f"Solved: {'✅ YES' if is_solved else '❌ NO'}")
    print(f"Objective: {result.objective_value}")
    print(f"Runtime: {result.runtime:.2f}ms")
    
    if result.routes:
        print(f"\nRoute found:")
        for vehicle_id, route in result.routes.items():
            if len(route) > 2:  # More than just start and end depot
                print(f"  {vehicle_id}: {' -> '.join(route)}")
    
    # Verify time window logic
    print("\n" + "="*60)
    print("TIME WINDOW ANALYSIS")
    print("="*60)
    
    for loc_id, location in scenario.locations.items():
        if hasattr(location, 'time_window_start') and location.time_window_start is not None:
            start = location.time_window_start
            end = location.time_window_end
            duration = end - start
            exceeds_600 = end > 600
            
            start_time = f"{start//60:02d}:{start%60:02d}"
            end_time = f"{end//60:02d}:{end%60:02d}"
            
            print(f"{loc_id}:")
            print(f"  Window: {start} to {end} minutes ({start_time} to {end_time})")
            print(f"  Duration: {duration} minutes ({duration/60:.1f} hours)")
            print(f"  Exceeds 600 min: {'YES' if exceeds_600 else 'NO'}")
            print()
    
    print("Key Insight:")
    print("- Time windows can exceed 600 minutes because they represent location availability")
    print("- Vehicle max_time=600 means driver works max 10 hours, but can start anytime")
    print("- A driver starting at 9:00 AM can work until 7:00 PM (10 hours total)")
    print("- This allows serving locations with time windows ending after minute 600")
    
    return is_solved

if __name__ == "__main__":
    success = test_scenario_feasibility()
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    if success:
        print("✅ TRUCKING LOGIC WORKS!")
        print("Time windows can exceed 600 minutes while respecting 10-hour driver shifts.")
    else:
        print("❌ ISSUE FOUND")
        print("Need to investigate why the scenario is not solvable.")
