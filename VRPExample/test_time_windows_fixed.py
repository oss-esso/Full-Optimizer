#!/usr/bin/env python3
"""
Test script to verify both MODA scenarios now have proper time windows and are solvable.
"""

import sys
import os

# Add path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

import vrp_scenarios
from vrp_optimizer_fixed import VRPQuantumOptimizer
from vrp_data_models import VRPObjective

def test_scenario_time_windows(scenario_name, scenario):
    """Test time window properties of a scenario."""
    print(f"\n=== Testing {scenario_name} ===")
    print(f"Locations: {len(scenario.locations)}")
    print(f"Vehicles: {len(scenario.vehicles)}")
    print(f"Ride requests: {len(scenario.ride_requests)}")
    
    # Check time windows
    locations_with_tw = 0
    invalid_tw = 0
    tw_exceeding_vehicle_limit = 0
    
    # Get vehicle time limit
    vehicle_time_limit = None
    if scenario.vehicles:
        first_vehicle = next(iter(scenario.vehicles.values()))
        vehicle_time_limit = getattr(first_vehicle, 'max_time', None)
    
    print(f"Vehicle time limit: {vehicle_time_limit} minutes")
    
    for loc_id, location in scenario.locations.items():
        if hasattr(location, 'time_window_start') and location.time_window_start is not None:
            locations_with_tw += 1
            start = location.time_window_start
            end = location.time_window_end if hasattr(location, 'time_window_end') else None
            
            if end is not None:
                duration = end - start
                if duration <= 0:
                    print(f"  ⚠️ Invalid time window at {loc_id}: [{start}, {end}] (duration: {duration})")
                    invalid_tw += 1
                
                if vehicle_time_limit and end > vehicle_time_limit:
                    print(f"  ⚠️ Time window end ({end}) exceeds vehicle limit ({vehicle_time_limit}) at {loc_id}")
                    tw_exceeding_vehicle_limit += 1
    
    print(f"Locations with time windows: {locations_with_tw}/{len(scenario.locations)}")
    print(f"Invalid time windows (negative/zero duration): {invalid_tw}")
    print(f"Time windows exceeding vehicle limit: {tw_exceeding_vehicle_limit}")
    
    # Test solvability
    print(f"Testing solvability...")
    try:
        optimizer = VRPQuantumOptimizer(scenario, VRPObjective.MINIMIZE_DISTANCE)
        result = optimizer.optimize_with_ortools()
        
        is_solved = result.status == "optimal" and result.objective_value > 0
        vehicles_used = len([r for r in result.routes.values() if len(r) > 2]) if result.routes else 0
        
        print(f"Result: {'✅ SOLVED' if is_solved else '❌ NOT SOLVED'}")
        print(f"Status: {result.status}")
        print(f"Vehicles used: {vehicles_used}")
        print(f"Objective value: {result.objective_value}")
        print(f"Runtime: {result.runtime:.2f}ms")
        
        return is_solved, invalid_tw == 0, tw_exceeding_vehicle_limit == 0
        
    except Exception as e:
        print(f"❌ ERROR during optimization: {e}")
        return False, invalid_tw == 0, tw_exceeding_vehicle_limit == 0

def main():
    """Main test function."""
    print("Testing MODA scenarios with updated time windows")
    print("=" * 60)
    
    # Load scenarios
    scenarios = vrp_scenarios.get_all_scenarios()
    print(f"Available scenarios: {list(scenarios.keys())}")
    
    # Test both MODA scenarios
    results = {}
    
    for scenario_name in ['MODA_small', 'MODA_first']:
        if scenario_name in scenarios:
            scenario = scenarios[scenario_name]
            solved, valid_tw, within_limits = test_scenario_time_windows(scenario_name, scenario)
            results[scenario_name] = {
                'solved': solved,
                'valid_time_windows': valid_tw,
                'within_vehicle_limits': within_limits
            }
        else:
            print(f"\n⚠️ {scenario_name} not found in scenarios")
            results[scenario_name] = {
                'solved': False,
                'valid_time_windows': False,
                'within_vehicle_limits': False
            }
    
    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    
    all_working = True
    for scenario_name, result in results.items():
        status = "✅ WORKING" if all(result.values()) else "❌ ISSUES FOUND"
        print(f"{scenario_name}: {status}")
        
        if not result['solved']:
            print(f"  - Not solvable")
            all_working = False
        if not result['valid_time_windows']:
            print(f"  - Invalid time windows found")
            all_working = False
        if not result['within_vehicle_limits']:
            print(f"  - Time windows exceed vehicle limits")
            all_working = False
    
    if all_working:
        print(f"\n🎉 SUCCESS: All MODA scenarios are working properly!")
        print(f"   - Time windows are valid (positive duration)")
        print(f"   - Time windows respect vehicle time limits (10 hours)")
        print(f"   - All scenarios are solvable with OR-Tools")
    else:
        print(f"\n❌ ISSUES: Some scenarios still have problems")

if __name__ == "__main__":
    main()
