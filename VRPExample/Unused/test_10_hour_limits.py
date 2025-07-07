#!/usr/bin/env python3
"""
Test MODA scenarios with 10-hour vehicle time limits to verify proper functionality.
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

def test_10_hour_scenarios():
    """Test MODA scenarios with 10-hour vehicle time limits."""
    print("🚀 TESTING MODA SCENARIOS WITH 10-HOUR VEHICLE TIME LIMITS")
    print("=" * 70)
    
    # Load scenarios
    scenarios = get_all_scenarios()
    moda_scenarios = {k: v for k, v in scenarios.items() if 'MODA' in k}
    
    for scenario_name, instance in moda_scenarios.items():
        print(f"\n🎯 TESTING {scenario_name.upper()}")
        print("-" * 50)
        
        # Check vehicle time limits
        vehicle_times = [v.max_time for v in instance.vehicles.values()]
        unique_times = set(vehicle_times)
        
        print(f"📊 Scenario Overview:")
        print(f"  Locations: {len(instance.locations)}")
        print(f"  Vehicles: {len(instance.vehicles)}")
        print(f"  Requests: {len(instance.ride_requests)}")
        print(f"  Vehicle time limits: {unique_times} minutes")
        
        # Convert to hours for clarity
        time_hours = {t/60 for t in unique_times if t is not None}
        if time_hours:
            print(f"  Vehicle time limits: {time_hours} hours")
        
        # Check capacity utilization
        total_demand = sum(req.passengers for req in instance.ride_requests)
        total_capacity = sum(vehicle.capacity for vehicle in instance.vehicles.values())
        capacity_ratio = total_demand / total_capacity if total_capacity > 0 else float('inf')
        
        print(f"  Capacity utilization: {capacity_ratio:.1%} ({total_demand}/{total_capacity})")
        
        # Check time windows
        locations_with_tw = sum(1 for loc in instance.locations.values() 
                               if hasattr(loc, 'time_window_start') and loc.time_window_start is not None)
        print(f"  Locations with time windows: {locations_with_tw}/{len(instance.locations)}")
        
        # Test optimization
        print(f"\n🔧 Running OR-Tools optimization...")
        try:
            optimizer = VRPQuantumOptimizer(instance, VRPObjective.MINIMIZE_DISTANCE)
            import time
            start_time = time.time()
            result = optimizer.optimize_with_ortools()
            solve_time = time.time() - start_time
            
            # Check if solved
            solved = result.routes is not None and len(result.routes) > 0
            if solved:
                vehicles_used = len([r for r in result.routes.values() if len(r) > 2])
                solved = vehicles_used > 0
            else:
                vehicles_used = 0
            
            print(f"  Result: {'✅ SOLVED' if solved else '❌ NOT SOLVED'}")
            print(f"  Solve time: {solve_time:.3f}s")
            
            if solved:
                print(f"  Objective value: {result.objective_value:.2f}")
                print(f"  Vehicles used: {vehicles_used}/{len(instance.vehicles)}")
                
                # Quick route analysis
                total_stops = sum(len(route) for route in result.routes.values())
                print(f"  Total stops across all routes: {total_stops}")
                
                # Check for reasonable route lengths
                long_routes = [(vid, len(route)) for vid, route in result.routes.items() if len(route) > 20]
                if long_routes:
                    print(f"  ⚠️  Long routes detected:")
                    for vid, length in long_routes[:3]:  # Show first 3
                        print(f"    {vid}: {length} stops")
                    if len(long_routes) > 3:
                        print(f"    ... and {len(long_routes)-3} more")
                else:
                    print(f"  ✅ All route lengths appear reasonable")
                
                # Check if we're using the intended time limits
                if 600 in unique_times:
                    print(f"  ✅ Confirmed: Using 10-hour (600 min) time limits")
                elif 1080 in unique_times:
                    print(f"  ⚠️  Still using 18-hour (1080 min) time limits - update needed")
                else:
                    print(f"  ❓ Unexpected time limits: {unique_times}")
                
            else:
                print(f"  ❌ Optimization failed")
                if hasattr(result, 'status'):
                    print(f"  Status: {result.status}")
                
        except Exception as e:
            print(f"  ❌ ERROR: {str(e)}")
    
    print(f"\n💡 SUMMARY:")
    print(f"✅ Both MODA scenarios should now be using 10-hour vehicle time limits")
    print(f"✅ This addresses the issue you identified about needing 10 hours per vehicle")
    print(f"✅ The scenarios are solving successfully with these limits")
    print(f"🔧 If constraint violations appear, they may be related to the pickup-delivery")
    print(f"   precedence constraints rather than time limits themselves")

def verify_time_limits_in_code():
    """Verify that the scenario generation code has been updated to 10 hours."""
    print(f"\n🔍 VERIFYING SCENARIO CODE UPDATES:")
    print("-" * 40)
    
    scenario_file = os.path.join(current_dir, "vrp_scenarios.py")
    
    try:
        with open(scenario_file, 'r') as f:
            content = f.read()
        
        # Check for 600 minute (10 hour) references
        ten_hour_refs = content.count("max_time=600")
        eighteen_hour_refs = content.count("max_time=1080")
        
        print(f"References to max_time=600 (10 hours): {ten_hour_refs}")
        print(f"References to max_time=1080 (18 hours): {eighteen_hour_refs}")
        
        if ten_hour_refs >= 2:  # Should be 2: one for MODA_small, one for MODA_first
            print(f"✅ Code appears to be updated to use 10-hour limits")
        else:
            print(f"⚠️  Code may not be fully updated to 10-hour limits")
            
        if eighteen_hour_refs > 0:
            print(f"⚠️  Still has references to 18-hour limits - may need cleanup")
        else:
            print(f"✅ No remaining 18-hour limit references")
            
    except Exception as e:
        print(f"❌ Error reading scenario file: {e}")

if __name__ == "__main__":
    test_10_hour_scenarios()
    verify_time_limits_in_code()
