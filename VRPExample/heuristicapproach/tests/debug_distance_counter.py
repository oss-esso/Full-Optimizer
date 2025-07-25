#!/usr/bin/env python3
"""
Debug script to test distance calculation counter.
This script will help us understand why the distance counter shows 0.
"""

import sys
import os

print("🔍 DEBUG: Distance Calculation Counter Test")
print("="*60)

try:
    # Add algo directory to path
    current_dir = os.path.dirname(__file__)
    algo_dir = os.path.join(current_dir, '..', 'algo')
    sys.path.insert(0, algo_dir)
    
    print(f"Current dir: {current_dir}")
    print(f"Algo dir: {algo_dir}")
    print(f"Algo dir exists: {os.path.exists(algo_dir)}")
except Exception as e:
    print(f"Setup error: {e}")
    sys.exit(1)

# Test 1: Direct function import and test
print("\n1️⃣ Testing direct import and function calls:")
try:
    from second_level import haversine_distance, reset_distance_calculation_count, get_distance_calculation_count
    
    reset_distance_calculation_count()
    print(f"   ✅ Counter after reset: {get_distance_calculation_count()}")
    
    # Test haversine function
    distance = haversine_distance(45.0, 9.0, 46.0, 10.0)
    print(f"   ✅ Haversine distance: {distance:.2f} km")
    print(f"   ✅ Counter after haversine call: {get_distance_calculation_count()}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Route distance calculation
print("\n2️⃣ Testing route distance calculation:")
try:
    from epdt_data_structures import Route
    
    # Create mock vehicle
    class MockVehicle:
        def __init__(self):
            self.id = 'test_vehicle'
            self.weight_capacity = 1000
            self.volume_capacity = 20
    
    # Test route distance calculation
    reset_distance_calculation_count()
    route = Route(vehicle=MockVehicle())
    
    print(f"   ✅ Counter before route calc: {get_distance_calculation_count()}")
    
    # Test the internal distance calculation method
    distance = route._calculate_distance_between_points(45.0, 9.0, 46.0, 10.0)
    print(f"   ✅ Route distance: {distance:.2f} km")
    print(f"   ✅ Counter after route calc: {get_distance_calculation_count()}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check if algorithm imports are different
print("\n3️⃣ Testing algorithm import path:")
try:
    # Test the same import path as the main algorithm
    reset_distance_calculation_count()
    
    # Import the way the main test script does it
    from algo.second_level import get_distance_calculation_count as algo_get_count
    from algo.second_level import haversine_distance as algo_haversine
    from algo.second_level import reset_distance_calculation_count as algo_reset
    
    print(f"   ⚠️  Different import path test:")
    print(f"   ✅ Counter (algo import): {algo_get_count()}")
    
    # Call haversine via algo import
    distance = algo_haversine(45.0, 9.0, 46.0, 10.0)
    print(f"   ✅ Algo haversine distance: {distance:.2f} km")
    print(f"   ✅ Counter (algo import): {algo_get_count()}")
    
    # Check if counters are the same
    direct_count = get_distance_calculation_count()
    algo_count = algo_get_count()
    
    if direct_count == algo_count:
        print(f"   ✅ Counters match: {direct_count}")
    else:
        print(f"   ❌ Counter mismatch! Direct: {direct_count}, Algo: {algo_count}")
        print(f"   💡 This explains why the main test shows 0!")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Simulate the full scenario run
print("\n4️⃣ Testing full scenario simulation:")
try:
    # Add src directory for scenario loading
    src_dir = os.path.join(current_dir, '..', 'src')
    sys.path.insert(0, src_dir)
    
    from moda_scenarios import create_furgoni_scenario
    from data_adapter import convert_instance_to_epdt_input
    from first_level import l1_heuristic
    
    print("   🏗️  Loading furgoni scenario...")
    instance = create_furgoni_scenario()
    
    print("   🔄 Converting to EPDT format...")
    orders, vehicles = convert_instance_to_epdt_input(instance)
    
    print(f"   📊 Created {len(orders)} orders and {len(vehicles)} vehicles")
    
    # Reset counter and run a minimal version
    reset_distance_calculation_count()
    print(f"   ✅ Counter before algorithm: {get_distance_calculation_count()}")
    
    # Configure minimal parameters
    params = {
        'tabu_tenure': 5,
        'M1': 1,  # Just 1 iteration for quick test
        'M2': 1,  # Just 1 iteration for quick test
        'exploration_strategy': 'vnd',
        'enable_advanced_neighborhoods': False,
        'enable_granular_search': False,
        'enable_parallelization': False,
        'local_search_strategy': 'first_improvement',
        'initialization_method': 'best_insertion',
        'debug_assignment': False
    }
    
    print("   🚀 Running minimal algorithm...")
    solution = l1_heuristic(orders, vehicles, params)
    
    final_count = get_distance_calculation_count()
    print(f"   ✅ Counter after algorithm: {final_count}")
    
    # Calculate total distance from solution
    total_distance = 0
    for route in solution.routes.values():
        if route and route.tasks:
            route_distance = route.get_total_distance()
            total_distance += route_distance
    
    print(f"   📏 Total solution distance: {total_distance:.2f} km")
    print(f"   📊 Assignment rate: {len(solution.get_assigned_orders())}/{len(orders)} orders")
    
    if final_count > 0:
        print(f"   ✅ SUCCESS: Counter working! {final_count} calculations made")
    else:
        print(f"   ❌ PROBLEM: Counter still 0 despite distance calculations")
        
        # Check if maybe routes are empty or failed
        active_routes = sum(1 for route in solution.routes.values() if route and route.tasks)
        print(f"   🔍 Active routes: {active_routes}/{len(solution.routes)}")
        
        if active_routes == 0:
            print(f"   💡 No active routes - this explains why no distance calculations!")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("🏁 Debug test complete!")
