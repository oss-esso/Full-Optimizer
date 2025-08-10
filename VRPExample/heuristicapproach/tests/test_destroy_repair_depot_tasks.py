#!/usr/bin/env python3
"""
Test to verify destroy and repair operations maintain depot task structure.
This addresses Issue 8 of TODO_SOLVE_PROD_ISSUES.md.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'algo'))

from algo.first_level import _create_base_route
from algo.destroy_and_repair import _insert_order_into_vehicle, _destroy_orders_from_vehicle
from algo.epdt_data_structures import Vehicle, Order, Task, TaskType, Solution

def test_destroy_repair_depot_tasks():
    """Test that destroy and repair operations preserve depot task structure."""
    
    print("🔧 Testing destroy and repair depot task preservation...")
    
    # Create test vehicle
    vehicle = Vehicle(
        id='TEST_V001',
        depot_id='DEPOT-ASTI', 
        weight_capacity=1000,
        volume_capacity=50,
        pallet_capacity=10
    )
    
    # Create solution
    solution = Solution()
    
    # Test 1: Insert order into new vehicle (should create route with depot structure)
    print("\\n   Test 1: Insert order into new vehicle")
    
    # Create test order
    pickup_task = Task(
        id='test_pickup_001',
        location_id='LOC_001',
        task_type=TaskType.PICKUP,
        order_id='ORDER_001',
        lat=44.9100,
        lon=8.2100,
        service_time=10.0,
        demand=100.0,
        volume=5.0
    )
    
    delivery_task = Task(
        id='test_delivery_001', 
        location_id='LOC_002',
        task_type=TaskType.DELIVERY,
        order_id='ORDER_001',
        lat=44.9200,
        lon=8.2200,
        service_time=10.0,
        demand=-100.0,
        volume=-5.0
    )
    
    test_order = Order(
        id='ORDER_001',
        pickup_tasks=[pickup_task],
        delivery_tasks=[delivery_task]
    )
    
    # Insert order (should create new route if none exists)
    success = _insert_order_into_vehicle(test_order, vehicle, solution, debug=True)
    
    if success:
        print("   ✅ Order insertion successful")
        route = solution.routes.get(vehicle.id)
        if route:
            print(f"   Route now has {len(route.tasks)} tasks:")
            for i, task in enumerate(route.tasks):
                print(f"     [{i}] {task.task_type} - {task.id}")
            
            # Verify depot structure
            if len(route.tasks) >= 2:
                first_task_type = str(route.tasks[0].task_type)
                last_task_type = str(route.tasks[-1].task_type)
                
                if 'DEPOT_START' in first_task_type:
                    print("   ✅ First task is DEPOT_START")
                else:
                    print(f"   ❌ First task is not DEPOT_START: {first_task_type}")
                    return False
                    
                if 'DEPOT_RETURN' in last_task_type:
                    print("   ✅ Last task is DEPOT_RETURN")
                else:
                    print(f"   ❌ Last task is not DEPOT_RETURN: {last_task_type}")
                    return False
                    
                # Verify customer tasks are between depot tasks
                customer_tasks = route.tasks[1:-1]
                if len(customer_tasks) == 2:  # pickup + delivery
                    print("   ✅ Customer tasks correctly inserted between depot tasks")
                else:
                    print(f"   ❌ Expected 2 customer tasks, found {len(customer_tasks)}")
                    return False
            else:
                print(f"   ❌ Route has only {len(route.tasks)} tasks, expected at least 2")
                return False
        else:
            print("   ❌ No route created")
            return False
    else:
        print("   ❌ Order insertion failed")
        return False
    
    # Test 2: Add another order to the same vehicle
    print("\\n   Test 2: Add second order to existing route")
    
    pickup_task2 = Task(
        id='test_pickup_002',
        location_id='LOC_003',
        task_type=TaskType.PICKUP,
        order_id='ORDER_002',
        lat=44.9300,
        lon=8.2300,
        service_time=15.0,
        demand=50.0,
        volume=3.0
    )
    
    delivery_task2 = Task(
        id='test_delivery_002', 
        location_id='LOC_004',
        task_type=TaskType.DELIVERY,
        order_id='ORDER_002',
        lat=44.9400,
        lon=8.2400,
        service_time=15.0,
        demand=-50.0,
        volume=-3.0
    )
    
    test_order2 = Order(
        id='ORDER_002',
        pickup_tasks=[pickup_task2],
        delivery_tasks=[delivery_task2]
    )
    
    success2 = _insert_order_into_vehicle(test_order2, vehicle, solution, debug=True)
    
    if success2:
        print("   ✅ Second order insertion successful")
        route = solution.routes.get(vehicle.id)
        print(f"   Route now has {len(route.tasks)} tasks:")
        for i, task in enumerate(route.tasks):
            print(f"     [{i}] {task.task_type} - {task.id}")
        
        # Verify depot structure is maintained
        first_task_type = str(route.tasks[0].task_type)
        last_task_type = str(route.tasks[-1].task_type)
        if ('DEPOT_START' in first_task_type and 'DEPOT_RETURN' in last_task_type):
            print("   ✅ Depot structure maintained after second insertion")
        else:
            print(f"   ❌ Depot structure corrupted: first={first_task_type}, last={last_task_type}")
            return False
    else:
        print("   ❌ Second order insertion failed")
        return False
    
    # Test 3: Test destroy operation (remove orders while preserving depot)
    print("\\n   Test 3: Test destroy operation")
    
    removed_orders = _destroy_orders_from_vehicle(vehicle, solution, test_order, debug=True)
    
    print(f"   Removed {len(removed_orders)} orders")
    route = solution.routes.get(vehicle.id)
    if route:
        print(f"   Route after destroy has {len(route.tasks)} tasks:")
        for i, task in enumerate(route.tasks):
            print(f"     [{i}] {task.task_type} - {task.id}")
        
        # Verify depot tasks are preserved
        depot_start_count = sum(1 for task in route.tasks if 'DEPOT_START' in str(task.task_type))
        depot_return_count = sum(1 for task in route.tasks if 'DEPOT_RETURN' in str(task.task_type))
        
        if depot_start_count == 1 and depot_return_count == 1:
            print("   ✅ Depot tasks preserved during destroy operation")
        else:
            print(f"   ❌ Depot tasks corrupted: {depot_start_count} DEPOT_START, {depot_return_count} DEPOT_RETURN")
            return False
            
        # If route has depot structure, verify it's correct
        if len(route.tasks) >= 2:
            first_task_type = str(route.tasks[0].task_type)
            last_task_type = str(route.tasks[-1].task_type)
            if ('DEPOT_START' in first_task_type and 'DEPOT_RETURN' in last_task_type):
                print("   ✅ Depot structure correct after destroy operation")
            else:
                print(f"   ❌ Depot structure incorrect: first={first_task_type}, last={last_task_type}")
                return False
    
    return True

if __name__ == "__main__":
    success = test_destroy_repair_depot_tasks()
    if success:
        print("\\n🎉 Destroy and repair depot task test PASSED!")
    else:
        print("\\n💥 Destroy and repair depot task test FAILED!")
    
    exit(0 if success else 1)
