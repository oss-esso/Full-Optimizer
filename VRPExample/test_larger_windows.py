#!/usr/bin/env python3

import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vrp_scenarios import create_moda_first_scenario
from create_moda_large_windows import create_moda_first_larger_windows
from vrp_optimizer_rolling_window import VRPOptimizerRollingWindow
from vrp_data_models import VRPObjective

def compare_time_windows():
    """Compare time windows between original and larger window versions."""
    
    print("=" * 80)
    print("COMPARING TIME WINDOWS: Original vs Larger Windows")
    print("=" * 80)
    
    # Test original MODA_first
    print("\n--- ORIGINAL MODA_first Time Windows ---")
    original = create_moda_first_scenario()
    
    original_spans = []
    for location_id, location in original.locations.items():
        if (hasattr(location, 'time_window_start') and hasattr(location, 'time_window_end') and
            location.time_window_start is not None and location.time_window_end is not None):
            span = location.time_window_end - location.time_window_start
            original_spans.append(span)
    
    if original_spans:
        print(f"Time window spans:")
        print(f"  - Average: {sum(original_spans)/len(original_spans):.1f} minutes ({sum(original_spans)/len(original_spans)/60:.1f} hours)")
        print(f"  - Minimum: {min(original_spans)} minutes ({min(original_spans)/60:.1f} hours)")
        print(f"  - Maximum: {max(original_spans)} minutes ({max(original_spans)/60:.1f} hours)")
        
        # Calculate total time span across all locations
        all_starts = [getattr(loc, 'time_window_start', 0) for loc in original.locations.values() 
                     if hasattr(loc, 'time_window_start') and getattr(loc, 'time_window_start', None) is not None]
        all_ends = [getattr(loc, 'time_window_end', 0) for loc in original.locations.values() 
                   if hasattr(loc, 'time_window_end') and getattr(loc, 'time_window_end', None) is not None]
        
        if all_starts and all_ends:
            earliest = min(all_starts)
            latest = max(all_ends)
            total_span = latest - earliest
            print(f"  - Total time span: {total_span} minutes ({total_span/60:.1f} hours)")
            print(f"  - From {earliest} to {latest} minutes")
      # Test larger window MODA_first
    print("\n--- LARGER WINDOWS MODA_first Time Windows ---")
    larger = create_moda_first_larger_windows()
    
    larger_spans = []
    for location_id, location in larger.locations.items():
        if (hasattr(location, 'time_window_start') and hasattr(location, 'time_window_end') and
            location.time_window_start is not None and location.time_window_end is not None):
            span = location.time_window_end - location.time_window_start
            larger_spans.append(span)
    
    if larger_spans:
        print(f"Time window spans:")
        print(f"  - Average: {sum(larger_spans)/len(larger_spans):.1f} minutes ({sum(larger_spans)/len(larger_spans)/60:.1f} hours)")
        print(f"  - Minimum: {min(larger_spans)} minutes ({min(larger_spans)/60:.1f} hours)")
        print(f"  - Maximum: {max(larger_spans)} minutes ({max(larger_spans)/60:.1f} hours)")
          # Calculate total time span across all locations
        all_starts = [getattr(loc, 'time_window_start', 0) for loc in larger.locations.values() 
                     if hasattr(loc, 'time_window_start') and getattr(loc, 'time_window_start', None) is not None]
        all_ends = [getattr(loc, 'time_window_end', 0) for loc in larger.locations.values() 
                   if hasattr(loc, 'time_window_end') and getattr(loc, 'time_window_end', None) is not None]
        all_ends = [getattr(loc, 'time_window_end', 0) for loc in larger.locations.values() if hasattr(loc, 'time_window_end')]
        
        if all_starts and all_ends:
            earliest = min(all_starts)
            latest = max(all_ends)
            total_span = latest - earliest
            print(f"  - Total time span: {total_span} minutes ({total_span/60:.1f} hours)")
            print(f"  - From {earliest} to {latest} minutes")
    
    return original, larger

def test_larger_windows_optimization():
    """Test if larger time windows make MODA_first feasible."""
    
    print("\n" + "=" * 80)
    print("TESTING OPTIMIZATION WITH LARGER TIME WINDOWS")
    print("=" * 80)
    
    # Test original scenario
    print("\n--- Testing Original MODA_first ---")
    original = create_moda_first_scenario()
    
    optimizer_orig = VRPOptimizerRollingWindow(original, VRPObjective.MINIMIZE_DISTANCE)
    result_orig = optimizer_orig.optimize_with_rolling_window()
    
    print(f"Original result: {result_orig.status}")
    if result_orig.status == "optimal":
        print(f"  - Vehicles used: {result_orig.metrics.get('vehicles_used', 0)}")
        print(f"  - Total distance: {result_orig.metrics.get('total_distance', 0):.2f}")
        print(f"  - Runtime: {result_orig.runtime:.2f} ms")
    else:
        print(f"  - Failed: {result_orig.metrics}")
    
    # Test larger windows scenario
    print("\n--- Testing Larger Windows MODA_first ---")
    larger = create_moda_first_larger_windows()
    
    optimizer_larger = VRPOptimizerRollingWindow(larger, VRPObjective.MINIMIZE_DISTANCE)
    result_larger = optimizer_larger.optimize_with_rolling_window()
    
    print(f"Larger windows result: {result_larger.status}")
    if result_larger.status == "optimal":
        print(f"  - Vehicles used: {result_larger.metrics.get('vehicles_used', 0)}")
        print(f"  - Total distance: {result_larger.metrics.get('total_distance', 0):.2f}")
        print(f"  - Runtime: {result_larger.runtime:.2f} ms")
        
        # Analyze routes
        active_routes = {k: v for k, v in result_larger.routes.items() if len(v) > 2}
        print(f"  - Active routes: {len(active_routes)}")
        
        for vehicle_id, route in list(active_routes.items())[:5]:  # Show first 5 routes
            route_length = len(route) - 2  # Exclude start/end depot
            print(f"    - {vehicle_id}: {route_length} stops")
    else:
        print(f"  - Failed: {result_larger.metrics}")
    
    # Summary
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Original MODA_first: {result_orig.status}")
    print(f"Larger windows MODA_first: {result_larger.status}")
    
    if result_orig.status != "optimal" and result_larger.status == "optimal":
        print("✅ SUCCESS: Larger time windows made the scenario feasible!")
    elif result_orig.status == "optimal" and result_larger.status == "optimal":
        print("✅ Both scenarios are feasible - larger windows may provide better solutions")
    elif result_orig.status != "optimal" and result_larger.status != "optimal":
        print("❌ Even with larger windows, the scenario remains infeasible")
        print("   This suggests the issue is not time window constraints but other factors")
    else:
        print("🤔 Unexpected result - need further investigation")

if __name__ == "__main__":
    print("Comparing MODA scenarios with different time window sizes...")
    
    # Compare time windows
    original, larger = compare_time_windows()
    
    # Test optimization
    test_larger_windows_optimization()
