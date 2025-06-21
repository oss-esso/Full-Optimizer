#!/usr/bin/env python3
"""
Test to verify depot time windows are fixed.
"""

from vrp_scenarios import create_moda_small_scenario

def test_depot_time_windows():
    """Test that depot time windows are now 24/7."""
    print("Testing Depot Time Windows Fix")
    print("=" * 40)
    
    scenario = create_moda_small_scenario()
    
    print(f"Scenario: {scenario.name}")
    print(f"Total locations: {len(scenario.locations)}")
    
    depot_count = 0
    for loc_id, location in scenario.locations.items():
        if 'depot' in loc_id:
            depot_count += 1
            start = getattr(location, 'time_window_start', None)
            end = getattr(location, 'time_window_end', None)
            service_time = getattr(location, 'service_time', 0)
            
            print(f"\n🏭 {loc_id}:")
            print(f"   Time window: [{start} - {end}] minutes")
            print(f"   Window size: {end - start if start is not None and end is not None else 'N/A'} minutes")
            print(f"   Service time: {service_time} minutes")
            
            if start == 0 and end == 1440:
                print(f"   ✅ FIXED: 24/7 availability")
            else:
                print(f"   ❌ ISSUE: Limited hours ({start/60:.1f}h - {end/60:.1f}h)")
    
    print(f"\nTotal depots found: {depot_count}")
    
    # Test with working optimizer
    print("\n" + "=" * 40)
    print("Testing with Working Optimizer")
    
    from vrp_optimizer_working import VRPOptimizerWorking
    optimizer = VRPOptimizerWorking()
    result = optimizer.solve(scenario, time_limit_seconds=30)
    
    print(f"✅ Solver result: {result['success']}")
    if result['success']:
        print(f"   Objective: {result['objective_value']}")
        print(f"   Vehicles used: {result['vehicles_used']}")
        print(f"   📈 PROBLEM IS NOW SOLVABLE!")
    else:
        print(f"   Error: {result.get('error', 'Unknown')}")

if __name__ == "__main__":
    test_depot_time_windows()
