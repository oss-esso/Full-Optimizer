"""
Route Validation Tests for EPDT Algorithm

This module tests the route validation logic, specifically the pickup-before-delivery
precedence constraint enforcement as described in TODO2.md item #17.
"""

import sys
import os

# Add the algo directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'algo'))

from epdt_data_structures import Route, Vehicle, Task, TaskType, Order
from second_level import is_feasible


def create_test_vehicle():
    """Create a test vehicle for route validation."""
    return Vehicle(
        id="test_vehicle",
        depot_id="test_depot",
        weight_capacity=10000,
        volume_capacity=50.0,
        pallet_capacity=10,
        lifo_required=False
    )


def create_test_task(task_id: str, task_type: TaskType, order_id: str):
    """Create a test task for route validation."""
    return Task(
        id=task_id,
        task_type=task_type,
        order_id=order_id,
        location_id=f"location_{task_id}",
        lat=0.0,
        lon=0.0,
        service_time=30.0,
        demand=100 if task_type == TaskType.PICKUP else -100,
        volume=1.0 if task_type == TaskType.PICKUP else -1.0
    )


def test_pickup_before_delivery_precedence():
    """
    Test the pickup-before-delivery precedence constraint.
    
    This test validates that:
    1. Routes with interleaved pickups and deliveries are rejected (return False)
    2. Routes with all pickups before all deliveries are accepted (return True)
    """
    print("🧪 Testing pickup-before-delivery precedence constraint...")
    
    # Create test vehicle
    vehicle = create_test_vehicle()
    
    # Test Case 1: Invalid route with interleaved tasks (Pickup A -> Pickup B -> Delivery A -> Pickup C)
    print("\n📋 Test Case 1: Invalid interleaved route")
    
    invalid_route = Route(vehicle=vehicle, driver=None)
    
    # Add tasks in interleaved order
    pickup_a = create_test_task("pickup_a", TaskType.PICKUP, "order_a")
    pickup_b = create_test_task("pickup_b", TaskType.PICKUP, "order_b") 
    delivery_a = create_test_task("delivery_a", TaskType.DELIVERY, "order_a")
    pickup_c = create_test_task("pickup_c", TaskType.PICKUP, "order_c")
    
    invalid_route.tasks = [pickup_a, pickup_b, delivery_a, pickup_c]
    
    # This should return False due to pickup_c coming after delivery_a
    result_invalid = is_feasible(invalid_route, debug_feasibility=True)
    
    print(f"   ❌ Invalid route feasibility: {result_invalid} (expected: False)")
    assert not result_invalid, "Route with interleaved pickups/deliveries should be infeasible"
    
    # Test Case 2: Valid route with pickup-first sequencing (Pickup A -> Pickup B -> Pickup C -> Delivery A)
    print("\n📋 Test Case 2: Valid pickup-first route")
    
    valid_route = Route(vehicle=vehicle, driver=None)
    
    # Add tasks in pickup-first order
    delivery_b = create_test_task("delivery_b", TaskType.DELIVERY, "order_b")
    delivery_c = create_test_task("delivery_c", TaskType.DELIVERY, "order_c")
    
    valid_route.tasks = [pickup_a, pickup_b, pickup_c, delivery_a, delivery_b, delivery_c]
    
    # This should return True as all pickups come before all deliveries
    result_valid = is_feasible(valid_route, debug_feasibility=True)
    
    print(f"   ✅ Valid route feasibility: {result_valid} (expected: True)")
    assert result_valid, "Route with pickup-first sequencing should be feasible"
    
    # Test Case 3: Edge case - Only pickups (should be valid)
    print("\n📋 Test Case 3: Only pickups")
    
    pickup_only_route = Route(vehicle=vehicle, driver=None)
    pickup_only_route.tasks = [pickup_a, pickup_b, pickup_c]
    
    result_pickup_only = is_feasible(pickup_only_route, debug_feasibility=True)
    print(f"   ✅ Pickup-only route feasibility: {result_pickup_only} (expected: True)")
    assert result_pickup_only, "Route with only pickups should be feasible"
    
    # Test Case 4: Edge case - Only deliveries (should be valid)
    print("\n📋 Test Case 4: Only deliveries")
    
    delivery_only_route = Route(vehicle=vehicle, driver=None)
    delivery_only_route.tasks = [delivery_a, delivery_b, delivery_c]
    
    result_delivery_only = is_feasible(delivery_only_route, debug_feasibility=True)
    print(f"   ✅ Delivery-only route feasibility: {result_delivery_only} (expected: True)")
    assert result_delivery_only, "Route with only deliveries should be feasible"
    
    # Test Case 5: Empty route (should be valid)
    print("\n📋 Test Case 5: Empty route")
    
    empty_route = Route(vehicle=vehicle, driver=None)
    empty_route.tasks = []
    
    result_empty = is_feasible(empty_route, debug_feasibility=True)
    print(f"   ✅ Empty route feasibility: {result_empty} (expected: True)")
    assert result_empty, "Empty route should be feasible"
    
    print("\n🎉 All pickup-before-delivery precedence tests passed!")


if __name__ == "__main__":
    test_pickup_before_delivery_precedence()
