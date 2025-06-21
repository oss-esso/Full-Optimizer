#!/usr/bin/env python3
"""
Test to understand why old scenarios fail with rolling window optimizer.
"""

import sys
sys.path.append('.')

from vrp_scenarios import VRPScenarioGenerator, create_moda_small_scenario
from vrp_optimizer_fixed import VRPQuantumOptimizer  
from vrp_optimizer_rolling_window import VRPOptimizerRollingWindow

def test_scenario_structure():
    """Test the structure of different scenario types."""
    print("=" * 80)
    print("TESTING SCENARIO STRUCTURES")
    print("=" * 80)
    
    # Test old delivery scenario
    gen = VRPScenarioGenerator()
    old_scenario = gen.create_small_delivery_scenario()
    
    print("\n1. OLD DELIVERY SCENARIO:")
    print(f"   Locations: {len(old_scenario.locations)}")
    print(f"   Vehicles: {len(old_scenario.vehicles)}")
    print(f"   Ride requests: {len(old_scenario.ride_requests)}")
    
    # Check if locations have demand
    first_loc = list(old_scenario.locations.values())[0]
    print(f"   First location: {first_loc.id}")
    print(f"   Has demand attribute: {hasattr(first_loc, 'demand')}")
    if hasattr(first_loc, 'demand'):
        print(f"   Demand: {first_loc.demand}")
    
    # Test MODA scenario
    moda_scenario = create_moda_small_scenario()
    
    print("\n2. MODA SCENARIO:")
    print(f"   Locations: {len(moda_scenario.locations)}")
    print(f"   Vehicles: {len(moda_scenario.vehicles)}")
    print(f"   Ride requests: {len(moda_scenario.ride_requests)}")
    
    if moda_scenario.ride_requests:
        first_request = moda_scenario.ride_requests[0]
        print(f"   First request: {first_request.id}")
        print(f"   Pickup: {first_request.pickup_location}")
        print(f"   Dropoff: {first_request.dropoff_location}")

def test_old_optimizer():
    """Test old scenarios with the original optimizer."""
    print("\n" + "=" * 80)
    print("TESTING OLD SCENARIOS WITH ORIGINAL OPTIMIZER")
    print("=" * 80)
    
    gen = VRPScenarioGenerator()
    small_scenario = gen.create_small_delivery_scenario()
    
    print(f"\nTesting: {small_scenario.name}")
    print(f"Locations: {len(small_scenario.locations)}, Vehicles: {len(small_scenario.vehicles)}")
    
    try:
        optimizer = VRPQuantumOptimizer(small_scenario)
        result = optimizer.optimize()
        
        print(f"✅ Result: {result.status}")
        print(f"   Objective: {result.objective_value}")
        print(f"   Routes: {len(result.routes) if result.routes else 0}")
        
        if result.routes:
            for vehicle_id, route in list(result.routes.items())[:2]:  # Show first 2 routes
                print(f"   {vehicle_id}: {len(route)} stops")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_rolling_window_on_old():
    """Test old scenarios with rolling window optimizer."""
    print("\n" + "=" * 80)
    print("TESTING OLD SCENARIOS WITH ROLLING WINDOW OPTIMIZER")
    print("=" * 80)
    
    gen = VRPScenarioGenerator()
    small_scenario = gen.create_small_delivery_scenario()
    
    print(f"\nTesting: {small_scenario.name}")
    print(f"Locations: {len(small_scenario.locations)}, Vehicles: {len(small_scenario.vehicles)}")
    
    try:
        optimizer = VRPOptimizerRollingWindow(small_scenario)
        result = optimizer.optimize_with_rolling_window(timeout_duration=60, time_limit_seconds=30)
        
        print(f"✅ Result: {result.status}")
        print(f"   Objective: {result.objective_value}")
        print(f"   Routes: {len(result.routes) if result.routes else 0}")
        
        if result.routes:
            for vehicle_id, route in list(result.routes.items())[:2]:  # Show first 2 routes
                print(f"   {vehicle_id}: {len(route)} stops")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_scenario_structure()
    test_old_optimizer()
    test_rolling_window_on_old()
