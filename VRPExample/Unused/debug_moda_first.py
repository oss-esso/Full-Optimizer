#!/usr/bin/env python3
"""
Debug MODA_first scenario to understand why it's showing as infeasible.
"""

import os
import sys

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

import vrp_scenarios
from vrp_optimizer_fixed import VRPQuantumOptimizer
from vrp_data_models import VRPObjective

def analyze_moda_first_issue():
    """Debug the MODA_first scenario issue."""
    print("="*60)
    print("DEBUGGING MODA_FIRST SCENARIO")
    print("="*60)
    
    # Load the scenario
    scenarios = vrp_scenarios.get_all_scenarios()
    moda_first = scenarios.get('MODA_first')
    
    if not moda_first:
        print("❌ Could not load MODA_first scenario")
        return
    
    print(f"✅ Loaded MODA_first scenario:")
    print(f"  - {len(moda_first.locations)} locations")
    print(f"  - {len(moda_first.vehicles)} vehicles")
    print(f"  - {len(moda_first.ride_requests)} ride requests")
    
    # Analyze vehicles
    print(f"\n📊 VEHICLE ANALYSIS:")
    sample_vehicle = next(iter(moda_first.vehicles.values()))
    print(f"  - Capacity: {sample_vehicle.capacity}")
    print(f"  - Max time: {sample_vehicle.max_time} minutes")
    print(f"  - Depot: {sample_vehicle.depot_id}")
    
    # Analyze time windows
    print(f"\n📊 TIME WINDOW ANALYSIS:")
    time_windows = []
    for loc_id, location in moda_first.locations.items():
        if hasattr(location, 'time_window_start') and location.time_window_start is not None:
            time_windows.append({
                'id': loc_id,
                'start': location.time_window_start,
                'end': location.time_window_end,
                'duration': location.time_window_end - location.time_window_start
            })
    
    if time_windows:
        starts = [tw['start'] for tw in time_windows]
        ends = [tw['end'] for tw in time_windows]
        durations = [tw['duration'] for tw in time_windows]
        
        print(f"  - Count: {len(time_windows)}")
        print(f"  - Start times: {min(starts)} to {max(starts)} min")
        print(f"  - End times: {min(ends)} to {max(ends)} min") 
        print(f"  - Durations: {min(durations)} to {max(durations)} min")
        
        # Check for problematic time windows
        max_vehicle_time = sample_vehicle.max_time if sample_vehicle.max_time else float('inf')
        
        print(f"\n🔍 CONSTRAINT VALIDATION:")
        print(f"  - Vehicle max time: {max_vehicle_time} minutes")
        print(f"  - Time windows ending after {max_vehicle_time}: {sum(1 for tw in time_windows if tw['end'] > max_vehicle_time)}")
        
        # This is the KEY INSIGHT - time windows can end after 600 min!
        late_windows = [tw for tw in time_windows if tw['end'] > max_vehicle_time]
        if late_windows:
            print(f"  - Latest time window ends at: {max(tw['end'] for tw in late_windows)} min")
            print(f"  - ✅ This is VALID for trucking company logic!")
            print(f"    (Driver can start later to serve late customers within 10-hour limit)")
    
    # Test with OR-Tools directly
    print(f"\n🧪 OR-TOOLS DIRECT TEST:")
    try:
        optimizer = VRPQuantumOptimizer(moda_first, VRPObjective.MINIMIZE_DISTANCE)
        result = optimizer.optimize_with_ortools()
        
        print(f"  - Status: {result.status}")
        print(f"  - Objective: {result.objective_value}")
        print(f"  - Runtime: {result.runtime:.2f}ms")
        print(f"  - Vehicles used: {len([r for r in result.routes.values() if len(r) > 2]) if result.routes else 0}")
        
        if result.status != "optimal":
            print(f"\n🔴 FAILURE ANALYSIS:")
            print(f"  - The scenario is indeed failing, but NOT due to time window logic")
            print(f"  - Need to investigate other constraints (capacity, demand, etc.)")
            
            # Check demand vs capacity
            total_demand = sum(req.passengers for req in moda_first.ride_requests)
            total_capacity = sum(vehicle.capacity for vehicle in moda_first.vehicles.values())
            print(f"  - Total demand: {total_demand}")
            print(f"  - Total capacity: {total_capacity}")
            print(f"  - Demand/capacity ratio: {total_demand/total_capacity:.2f}")
            
            if total_demand > total_capacity:
                print(f"  - 🚨 ISSUE FOUND: Demand exceeds capacity!")
        else:
            print(f"  - ✅ Actually solvable - comparison script may have issues")
            
    except Exception as e:
        print(f"  - ❌ Error during optimization: {e}")

if __name__ == "__main__":
    analyze_moda_first_issue()
