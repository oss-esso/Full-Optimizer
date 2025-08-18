#!/usr/bin/env python3
"""
Debug script to analyze why L2 heuristic is failing to generate initial routes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from epdt_data_structures import Route, Order, Task, TaskType, Vehicle
    from second_level import _generate_initial_task_sequence, is_feasible
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def debug_single_order():
    """Debug why a single order fails to generate initial routes."""
    
    # Create a simple vehicle (furgone_1)
    vehicle = Vehicle(
        id="furgone_1",
        depot_id="depot",
        weight_capacity=3500,
        volume_capacity=20.0,
        vehicle_type="standard"
    )
    
    # Create empty route
    route = Route(vehicle=vehicle)
    
    # Create a simple depot order (depot_request_1)
    order = Order(
        id="depot_request_1",
        pickup_location="depot_bay_1",
        delivery_location="badia_polesine_ro",
        weight=750,
        volume=3.8,
        priority=1,
        mandatory=True
    )
    
    # Add pickup and delivery tasks to the order
    pickup_task = Task(
        id="pickup_depot_request_1",
        order_id="depot_request_1",
        location="depot_bay_1",
        task_type=TaskType.PICKUP,
        demand=750,
        volume=3.8
    )
    
    delivery_task = Task(
        id="delivery_depot_request_1", 
        order_id="depot_request_1",
        location="badia_polesine_ro",
        task_type=TaskType.DELIVERY,
        demand=-750,
        volume=-3.8
    )
    
    order.pickup_tasks = [pickup_task]
    order.delivery_tasks = [delivery_task]
    
    print(f"🔍 Testing order: {order.id}")
    print(f"   Weight: {order.weight}kg, Volume: {order.volume}m³")
    print(f"   Vehicle capacity: {vehicle.weight_capacity}kg, {vehicle.volume_capacity}m³")
    print(f"   Empty route feasible: {route.is_feasible()}")
    
    # Test basic task insertion
    print(f"\n📋 Testing basic task insertion:")
    
    # Test inserting just the pickup
    test_route = route.copy()
    test_route.insert_task(0, pickup_task)
    print(f"   Route with pickup only: {len(test_route.tasks)} tasks, feasible: {test_route.is_feasible()}")
    
    # Test inserting pickup + delivery
    test_route2 = route.copy()
    test_route2.insert_task(0, pickup_task)
    test_route2.insert_task(1, delivery_task)
    print(f"   Route with pickup+delivery: {len(test_route2.tasks)} tasks, feasible: {test_route2.is_feasible()}")
    
    # Debug the feasibility check for the basic route
    if not test_route2.is_feasible():
        print(f"\n⚠️  Feasibility check failed! Debugging...")
        
        # Check each constraint manually
        print(f"   📏 Weight constraint: {pickup_task.demand + delivery_task.demand}kg <= {vehicle.weight_capacity}kg: {pickup_task.demand + delivery_task.demand <= vehicle.weight_capacity}")
        print(f"   📦 Volume constraint: {pickup_task.volume + delivery_task.volume}m³ <= {vehicle.volume_capacity}m³: {pickup_task.volume + delivery_task.volume <= vehicle.volume_capacity}")
        
        # Let's check what is_feasible is actually checking
        sorted_tasks = test_route2.get_sorted_tasks()
        print(f"   📋 Task sequence: {[f'{t.task_type.name}@{t.location}' for t in sorted_tasks]}")
        
        # Manual capacity simulation
        load_w = 0
        load_v = 0
        for i, task in enumerate(sorted_tasks):
            load_w += task.demand
            load_v += task.volume
            print(f"   Step {i+1}: {task.task_type.name} {task.demand}kg, {task.volume}m³ -> Load: {load_w}kg, {load_v}m³")
            if load_w > vehicle.weight_capacity:
                print(f"      ❌ Weight limit exceeded!")
            if load_v > vehicle.volume_capacity:
                print(f"      ❌ Volume limit exceeded!")
    
    # Now test the L2 heuristic
    print(f"\n🧪 Testing L2 initial route generation:")
    initial_routes = _generate_initial_task_sequence(route, order, debug_assignment=True)
    print(f"   Generated {len(initial_routes)} initial routes")
    
    return len(initial_routes) > 0

if __name__ == "__main__":
    success = debug_single_order()
    print(f"\n🎯 Result: {'SUCCESS' if success else 'FAILED'}")
