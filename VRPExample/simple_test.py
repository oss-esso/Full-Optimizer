#!/usr/bin/env python3
"""
Simple test to verify the enhanced VRP optimizers work correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vrp_scenarios import create_moda_small_scenario, create_moda_first_scenario
from vrp_optimizer_rolling_window import VRPOptimizerRollingWindow
from vrp_data_models import VRPObjective

def main():
    print("Testing enhanced VRP optimizers...")
    
    # Test MODA_small first (should be feasible)
    print("\n1. Testing MODA_small scenario...")
    try:
        instance = create_moda_small_scenario()
        print(f"Created scenario with {len(instance.location_ids)} locations, {len(instance.vehicles)} vehicles, {len(instance.ride_requests)} requests")
        
        optimizer = VRPOptimizerRollingWindow(instance, VRPObjective.MINIMIZE_DISTANCE)
        
        # Test with different time limits
        for time_limit in [60, 300, 900]:  # 1min, 5min, 15min
            print(f"\nTesting with {time_limit}s time limit...")
            result = optimizer.optimize_with_rolling_window(
                timeout_duration=time_limit + 30,
                time_limit_seconds=time_limit
            )
            
            print(f"  Status: {result.status}")
            print(f"  Objective: {result.objective_value}")
            print(f"  Feasible: {result.status == 'optimal' and result.objective_value < float('inf')}")
            
            if result.status == 'optimal' and result.objective_value < float('inf'):
                print(f"  ✅ FOUND SOLUTION with {time_limit}s!")
                break
                
    except Exception as e:
        print(f"Error testing MODA_small: {e}")
    
    # Test MODA_first (challenging)
    print("\n2. Testing MODA_first scenario...")
    try:
        instance = create_moda_first_scenario()
        print(f"Created scenario with {len(instance.location_ids)} locations, {len(instance.vehicles)} vehicles, {len(instance.ride_requests)} requests")
        
        optimizer = VRPOptimizerRollingWindow(instance, VRPObjective.MINIMIZE_DISTANCE)
        
        # Test with longer time limit
        time_limit = 1800  # 30 minutes
        print(f"\nTesting with {time_limit}s time limit...")
        result = optimizer.optimize_with_rolling_window(
            timeout_duration=time_limit + 30,
            time_limit_seconds=time_limit
        )
        
        print(f"  Status: {result.status}")
        print(f"  Objective: {result.objective_value}")
        print(f"  Feasible: {result.status == 'optimal' and result.objective_value < float('inf')}")
        
        if result.status == 'optimal' and result.objective_value < float('inf'):
            print(f"  ✅ FOUND SOLUTION for MODA_first!")
        else:
            print(f"  ❌ Could not solve MODA_first with {time_limit}s")
            
    except Exception as e:
        print(f"Error testing MODA_first: {e}")
    
    print("\nSimple test completed!")

if __name__ == "__main__":
    main()
