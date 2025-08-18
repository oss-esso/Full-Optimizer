#!/usr/bin/env python3
"""
Test 1: Core Component Validation (_create_base_route)

Goal: Confirm that the most fundamental route-creation function works as expected 
and that the feasibility check correctly validates a simple, valid route.
"""

import sys
import os

# Add the algo directory to the Python path
algo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'algo')
sys.path.insert(0, algo_path)

try:
    # Import required classes and functions
    from epdt_data_structures import Route, Vehicle, Task, TaskType
    from first_level import _create_base_route
    from second_level import is_feasible
    
    print("🧪 Test 1: Core Component Validation (_create_base_route)")
    print("=" * 60)
    
    # Create a dummy Vehicle object
    print("📋 Creating dummy vehicle...")
    vehicle = Vehicle(
        id="TEST_VEHICLE_001",
        depot_id="DEPOT-ASTI",
        weight_capacity=1000.0,
        volume_capacity=100.0,
        pallet_capacity=10,
        cost_per_km=1.0,
        cost_per_hour=50.0
    )
    print(f"✅ Created vehicle: {vehicle.id}")
    
    # Call _create_base_route
    print("\n🏗️ Creating base route using _create_base_route...")
    route = _create_base_route(vehicle)
    print(f"✅ Base route created with {len(route.tasks)} tasks")
    
    # Assert that len(route.tasks) is 2
    print(f"\n🔍 Validating route structure...")
    assert len(route.tasks) == 2, f"Expected 2 tasks, got {len(route.tasks)}"
    print(f"✅ Route has exactly 2 tasks")
    
    # Assert that route.tasks[0].is_depot_start() is True
    assert route.tasks[0].is_depot_start(), f"First task is not depot start: {route.tasks[0]}"
    print(f"✅ First task is depot start: {route.tasks[0].task_type}")
    
    # Assert that route.tasks[-1].is_depot_return() is True
    assert route.tasks[-1].is_depot_return(), f"Last task is not depot return: {route.tasks[-1]}"
    print(f"✅ Last task is depot return: {route.tasks[-1].task_type}")
    
    # Call is_feasible with return_reason=True
    print(f"\n⚖️ Checking route feasibility...")
    try:
        feasible, reason = is_feasible(route, return_reason=True)
        print(f"✅ Feasibility check completed: feasible={feasible}, reason='{reason}'")
        
        # Assert that feasible is True
        assert feasible, f"Route should be feasible but got: {reason}"
        print(f"✅ Route is correctly marked as feasible")
        
    except Exception as e:
        print(f"⚠️ Feasibility check failed with exception: {e}")
        print("Trying feasibility check without return_reason parameter...")
        try:
            feasible = is_feasible(route)
            print(f"✅ Simplified feasibility check: feasible={feasible}")
            assert feasible, f"Route should be feasible"
        except Exception as e2:
            print(f"❌ Both feasibility checks failed: {e2}")
            raise
    
    print(f"\n🎉 TEST 1 PASSED: All assertions successful!")
    print("=" * 60)
    print("✅ _create_base_route function works correctly")
    print("✅ Route structure is valid (depot start -> depot return)")
    print("✅ Feasibility check validates the route as feasible")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required modules are available")
    sys.exit(1)
except AssertionError as e:
    print(f"❌ Assertion failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
