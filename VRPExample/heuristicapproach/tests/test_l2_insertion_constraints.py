#!/usr/bin/env python3
"""
Test to verify L2 insertion logic correctly constrains insertion between depot tasks.
This addresses Step 4 of REFACTOR_ROUTE_CREATION.md and Issue 6 of TODO_SOLVE_PROD_ISSUES.md.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'algo'))

from algo.first_level import _create_base_route
from algo.second_level import _generate_initial_task_sequence
from algo.epdt_data_structures import Vehicle, Order, Task, TaskType

def test_l2_insertion_constraints():
    """Test that L2 insertion respects depot task boundaries."""
    
    print("🔧 Testing L2 insertion constraints...")
    
    # Create a test vehicle with proper depot structure
    vehicle = Vehicle(
        id='TEST_V001',
        depot_id='DEPOT-ASTI', 
        weight_capacity=1000,
        volume_capacity=50
    )
    
    # Create base route with depot structure
    base_route = _create_base_route(vehicle)
    print(f"   Base route has {len(base_route.tasks)} tasks:")
    for i, task in enumerate(base_route.tasks):
        print(f"   [{i}] {task.task_type} - {task.id}")
    
    # Create a simple test order with pickup and delivery
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
    
    print(f"   Test order: {test_order.id} with {len(test_order.pickup_tasks)} pickups, {len(test_order.delivery_tasks)} deliveries")
    
    # Call L2 insertion
    result_routes = _generate_initial_task_sequence(base_route, test_order, debug_assignment=True)
    
    if not result_routes:
        print("❌ L2 insertion failed - no routes returned")
        return False
    
    # Verify the result maintains depot structure
    final_route = result_routes[0]
    print(f"   Final route has {len(final_route.tasks)} tasks:")
    for i, task in enumerate(final_route.tasks):
        print(f"   [{i}] {task.task_type} - {task.id}")
    
    # Critical tests
    success = True
    
    # Test 1: First task must be DEPOT_START
    if not final_route.tasks[0].task_type == TaskType.DEPOT_START:
        print("❌ First task is not DEPOT_START")
        success = False
    else:
        print("✅ First task is DEPOT_START")
    
    # Test 2: Last task must be DEPOT_RETURN  
    if not final_route.tasks[-1].task_type == TaskType.DEPOT_RETURN:
        print("❌ Last task is not DEPOT_RETURN")
        success = False
    else:
        print("✅ Last task is DEPOT_RETURN")
    
    # Test 3: All customer tasks must be between depot tasks
    customer_tasks = final_route.tasks[1:-1]  # Exclude first and last
    depot_tasks_in_middle = [task for task in customer_tasks if task.task_type in [TaskType.DEPOT_START, TaskType.DEPOT_RETURN]]
    if depot_tasks_in_middle:
        print(f"❌ Found depot tasks in middle: {[task.task_type for task in depot_tasks_in_middle]}")
        success = False
    else:
        print("✅ No depot tasks found in middle positions")
    
    # Test 4: Must have both pickup and delivery tasks
    pickup_tasks = [task for task in customer_tasks if task.task_type == TaskType.PICKUP]
    delivery_tasks = [task for task in customer_tasks if task.task_type == TaskType.DELIVERY]
    
    if len(pickup_tasks) != 1:
        print(f"❌ Expected 1 pickup task, found {len(pickup_tasks)}")
        success = False
    else:
        print("✅ Found exactly 1 pickup task")
        
    if len(delivery_tasks) != 1:
        print(f"❌ Expected 1 delivery task, found {len(delivery_tasks)}")
        success = False
    else:
        print("✅ Found exactly 1 delivery task")
    
    # Test 5: Precedence constraint - pickup must come before delivery for same order
    if pickup_tasks and delivery_tasks:
        pickup_pos = None
        delivery_pos = None
        for i, task in enumerate(final_route.tasks):
            if task.task_type == TaskType.PICKUP and task.order_id == 'ORDER_001':
                pickup_pos = i
            elif task.task_type == TaskType.DELIVERY and task.order_id == 'ORDER_001':
                delivery_pos = i
        
        if pickup_pos is not None and delivery_pos is not None:
            if pickup_pos < delivery_pos:
                print("✅ Pickup occurs before delivery (precedence respected)")
            else:
                print(f"❌ Precedence violation: pickup at {pickup_pos}, delivery at {delivery_pos}")
                success = False
        else:
            print("❌ Could not find pickup or delivery positions")
            success = False
    
    return success

if __name__ == "__main__":
    success = test_l2_insertion_constraints()
    if success:
        print("🎉 L2 insertion constraint test PASSED!")
    else:
        print("💥 L2 insertion constraint test FAILED!")
    
    exit(0 if success else 1)
