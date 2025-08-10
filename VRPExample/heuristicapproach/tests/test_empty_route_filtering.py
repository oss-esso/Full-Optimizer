#!/usr/bin/env python3
"""
Test to verify empty depot-only routes are filtered from final solution.
This addresses Issue 7 of TODO_SOLVE_PROD_ISSUES.md.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'algo'))

from algo.first_level import _create_base_route, _validate_and_filter_solution
from algo.epdt_data_structures import Vehicle, Solution

def test_empty_route_filtering():
    """Test that routes with only depot tasks are filtered out."""
    
    print("🧹 Testing empty route filtering...")
    
    # Create test vehicles with proper constraints
    vehicles = []
    for i in range(5):
        vehicle = Vehicle(
            id=f'V{i:03d}',
            depot_id='DEPOT-ASTI', 
            weight_capacity=1000,
            volume_capacity=50,
            pallet_capacity=10  # Add pallet capacity to avoid None comparison error
        )
        vehicles.append(vehicle)
    
    # Create solution with mix of empty and depot-only routes
    solution = Solution()
    
    # Vehicle 0: Route with only depot tasks (should be filtered)
    depot_only_route = _create_base_route(vehicles[0])
    solution.add_route(vehicles[0].id, depot_only_route)
    print(f"   Added depot-only route for {vehicles[0].id}: {len(depot_only_route.tasks)} tasks")
    
    # Vehicle 1: Another route with only depot tasks (should be filtered)
    depot_only_route2 = _create_base_route(vehicles[1])
    solution.add_route(vehicles[1].id, depot_only_route2)
    print(f"   Added depot-only route for {vehicles[1].id}: {len(depot_only_route2.tasks)} tasks")
    
    # Vehicle 2: Route with customer tasks (should be kept)
    # We'll create a route with depot tasks + customer task to simulate a real route
    customer_route = _create_base_route(vehicles[2])
    # Simulate adding a customer task (we'll just add a dummy task with customer order_id)
    from algo.epdt_data_structures import Task, TaskType
    customer_task = Task(
        id='customer_task_001',
        location_id='CUSTOMER_LOC',
        task_type=TaskType.PICKUP,
        order_id='CUSTOMER_ORDER_001', 
        lat=44.9100,
        lon=8.2100,
        service_time=10.0,
        demand=100.0,
        volume=5.0
    )
    # Insert between depot tasks
    customer_route.tasks.insert(1, customer_task)
    solution.add_route(vehicles[2].id, customer_route)
    print(f"   Added customer route for {vehicles[2].id}: {len(customer_route.tasks)} tasks")
    
    # Vehicle 3: Empty route (no tasks) - should be filtered
    from algo.epdt_data_structures import Route
    empty_route = Route(vehicle=vehicles[3])
    solution.add_route(vehicles[3].id, empty_route)
    print(f"   Added empty route for {vehicles[3].id}: {len(empty_route.tasks)} tasks")
    
    # Vehicle 4: Another depot-only route
    depot_only_route3 = _create_base_route(vehicles[4])
    solution.add_route(vehicles[4].id, depot_only_route3)
    print(f"   Added depot-only route for {vehicles[4].id}: {len(depot_only_route3.tasks)} tasks")
    
    print(f"\\n   Initial solution has {len(solution.routes)} routes")
    
    # Apply filtering
    filtered_solution = _validate_and_filter_solution(solution)
    
    print(f"   Filtered solution has {len(filtered_solution.routes)} routes\\n")
    
    # Verify results
    success = True
    
    # Should only have the customer route remaining
    if len(filtered_solution.routes) != 1:
        print(f"❌ Expected 1 route after filtering, got {len(filtered_solution.routes)}")
        success = False
    else:
        print("✅ Correct number of routes after filtering")
    
    # The remaining route should be the customer route
    if vehicles[2].id not in filtered_solution.routes:
        print(f"❌ Customer route for {vehicles[2].id} was incorrectly filtered out")
        success = False
    else:
        remaining_route = filtered_solution.routes[vehicles[2].id]
        if len(remaining_route.tasks) != 3:  # DEPOT_START + customer + DEPOT_RETURN
            print(f"❌ Customer route has {len(remaining_route.tasks)} tasks, expected 3")
            success = False
        else:
            print("✅ Customer route correctly preserved")
    
    # Verify depot-only routes were filtered
    depot_only_vehicles = [vehicles[0].id, vehicles[1].id, vehicles[4].id]
    for vehicle_id in depot_only_vehicles:
        if vehicle_id in filtered_solution.routes:
            print(f"❌ Depot-only route for {vehicle_id} was NOT filtered out")
            success = False
    
    if success:
        print("✅ All depot-only routes correctly filtered")
        
    # Verify empty route was filtered
    if vehicles[3].id in filtered_solution.routes:
        print(f"❌ Empty route for {vehicles[3].id} was NOT filtered out")
        success = False
    else:
        print("✅ Empty route correctly filtered")
    
    return success

if __name__ == "__main__":
    success = test_empty_route_filtering()
    if success:
        print("🎉 Empty route filtering test PASSED!")
    else:
        print("💥 Empty route filtering test FAILED!")
    
    exit(0 if success else 1)
