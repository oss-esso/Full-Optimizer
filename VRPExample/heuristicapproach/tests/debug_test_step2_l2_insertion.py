#!/usr/bin/env python3
"""
Test 2: L2 Heuristic Validation (l2_heuristic)

Goal: Verify that the L2 heuristic (intra-route optimization) correctly inserts 
an order into a base route while preserving the depot task structure.
"""

import sys
import os

# Add the algo directory to the Python path
algo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'algo')
sys.path.insert(0, algo_path)

try:
    # Import required classes and functions
    from epdt_data_structures import Route, Vehicle, Task, TaskType, Order
    from first_level import _create_base_route
    from second_level import l2_heuristic, is_feasible
    
    print("🧪 Test 2: L2 Heuristic Validation (l2_heuristic)")
    print("=" * 60)
    
    # Create a dummy Vehicle object
    print("📋 Creating dummy vehicle...")
    vehicle = Vehicle(
        id="TEST_VEHICLE_002",
        depot_id="DEPOT-ASTI",
        weight_capacity=1000.0,
        volume_capacity=100.0,
        pallet_capacity=10,
        cost_per_km=1.0,
        cost_per_hour=50.0
    )
    print(f"✅ Created vehicle: {vehicle.id}")
    
    # Create a simple Order
    print("📦 Creating test order...")
    pickup_task = Task(
        id="PICKUP_001",
        location_id="CUSTOMER_A",
        task_type=TaskType.PICKUP,
        order_id="ORDER_001",
        lat=44.9200,
        lon=8.2200,
        service_time=15.0,
        demand=100.0,
        volume=10.0,
        pallets=2,
        earliest_time=480,  # 8:00 AM
        latest_time=1020    # 5:00 PM
    )
    
    delivery_task = Task(
        id="DELIVERY_001",
        location_id="CUSTOMER_B",
        task_type=TaskType.DELIVERY,
        order_id="ORDER_001",
        lat=44.9500,
        lon=8.2500,
        service_time=10.0,
        demand=100.0,
        volume=10.0,
        pallets=2,
        earliest_time=480,  # 8:00 AM
        latest_time=1020    # 5:00 PM
    )
    
    order = Order(
        id="ORDER_001",
        pickup_tasks=[pickup_task],
        delivery_tasks=[delivery_task]
    )
    print(f"✅ Created order: {order.id} with {len(order.pickup_tasks)} pickup and {len(order.delivery_tasks)} delivery tasks")
    
    # Create a valid base_route using _create_base_route
    print("🏗️ Creating base route...")
    base_route = _create_base_route(vehicle)
    print(f"✅ Base route created with {len(base_route.tasks)} tasks")
    
    # Verify base route structure
    assert len(base_route.tasks) == 2, f"Base route should have 2 tasks, got {len(base_route.tasks)}"
    assert base_route.tasks[0].is_depot_start(), "First task should be depot start"
    assert base_route.tasks[-1].is_depot_return(), "Last task should be depot return"
    print("✅ Base route structure verified")
    
    # Call l2_heuristic
    print("🎯 Calling l2_heuristic to insert order into route...")
    try:
        new_route = l2_heuristic(base_route, order)
        print(f"✅ L2 heuristic completed")
        
        # Check that new_route is not None
        assert new_route is not None, "L2 heuristic should return a route, not None"
        print(f"✅ L2 heuristic returned a valid route with {len(new_route.tasks)} tasks")
        
        # Assert that new_route.tasks[0].is_depot_start() is True
        assert new_route.tasks[0].is_depot_start(), f"First task should be depot start, got: {new_route.tasks[0].task_type}"
        print(f"✅ First task is depot start: {new_route.tasks[0].task_type}")
        
        # Assert that new_route.tasks[-1].is_depot_return() is True
        assert new_route.tasks[-1].is_depot_return(), f"Last task should be depot return, got: {new_route.tasks[-1].task_type}"
        print(f"✅ Last task is depot return: {new_route.tasks[-1].task_type}")
        
        # Verify the route has more than just depot tasks (should contain the order tasks)
        assert len(new_route.tasks) > 2, f"Route should have more than just depot tasks, got {len(new_route.tasks)} tasks"
        print(f"✅ Route contains order tasks (total tasks: {len(new_route.tasks)})")
        
        # Call is_feasible and assert it returns True
        print("⚖️ Checking route feasibility...")
        feasible, reason = is_feasible(new_route, return_reason=True)
        print(f"✅ Feasibility check completed: feasible={feasible}, reason='{reason}'")
        
        assert feasible, f"Route should be feasible but got: {reason}"
        print(f"✅ Route is correctly marked as feasible")
        
        # Print route details for debugging
        print("\\n📋 Final route structure:")
        for i, task in enumerate(new_route.tasks):
            print(f"  {i+1}. {task.task_type.name} - {task.id} (Order: {task.order_id})")
        
        print(f"\\n🎉 TEST 2 PASSED: All assertions successful!")
        print("=" * 60)
        print("✅ L2 heuristic successfully inserts order into route")
        print("✅ Depot task structure is preserved")
        print("✅ Final route is feasible")
        
    except Exception as e:
        print(f"❌ L2 heuristic failed: {e}")
        import traceback
        traceback.print_exc()
        raise
        
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
