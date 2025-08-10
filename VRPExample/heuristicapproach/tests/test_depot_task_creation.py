#!/usr/bin/env python3
"""
Test depot task creation and finalization.

Issue 5: Ensure Depot Tasks are Added to All Final Routes
This test verifies that all routes in the final solution properly 
start and end with depot tasks.
"""

import sys
import os

# Add path for imports
test_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(test_dir, '..')
algo_dir = os.path.join(heuristic_root, 'algo')
sys.path.insert(0, heuristic_root)
sys.path.insert(0, algo_dir)

print("=== Testing Depot Task Creation and Finalization ===")

try:
    from first_level import l1_heuristic, _add_depot_tasks_to_route
    from second_level import is_feasible
    from epdt_data_structures import Route, Task, Vehicle, TaskType, Solution, Order
    
    print("Successfully imported required modules")
    
    # Create test data
    vehicle1 = Vehicle(
        id="TEST_VEHICLE_1", 
        depot_id="DEPOT1",
        weight_capacity=1000, 
        volume_capacity=1000,
        pallet_capacity=10
    )
    
    vehicle2 = Vehicle(
        id="TEST_VEHICLE_2", 
        depot_id="DEPOT1",
        weight_capacity=1000, 
        volume_capacity=1000,
        pallet_capacity=10
    )
    
    vehicles = [vehicle1, vehicle2]
    
    # Create test orders
    orders = []
    for i in range(3):
        pickup_task = Task(
            id=f"pickup_{i}",
            location_id=f"location_pickup_{i}",
            task_type=TaskType.PICKUP,
            order_id=f"order_{i}",
            lat=45.0 + (i * 0.01),
            lon=7.0 + (i * 0.01),
            service_time=30,
            earliest_time=0,
            latest_time=1440,
            demand=100
        )
        
        delivery_task = Task(
            id=f"delivery_{i}",
            location_id=f"location_delivery_{i}",
            task_type=TaskType.DELIVERY,
            order_id=f"order_{i}",
            lat=45.1 + (i * 0.01),
            lon=7.1 + (i * 0.01),
            service_time=30,
            earliest_time=0,
            latest_time=1440,
            demand=-100
        )
        
        order = Order(
            id=f"order_{i}",
            pickup_tasks=[pickup_task],
            delivery_tasks=[delivery_task]
        )
        orders.append(order)
    
    print(f"Created {len(orders)} test orders and {len(vehicles)} vehicles")
    
    print(f"\n=== Test 1: Manual Depot Task Addition ===")
    
    # Create a route without depot tasks
    route_without_depot = Route(vehicle=vehicle1)
    route_without_depot.tasks = [
        orders[0].pickup_tasks[0],
        orders[0].delivery_tasks[0]
    ]
    
    print(f"  Route before depot addition: {len(route_without_depot.tasks)} tasks")
    for task in route_without_depot.tasks:
        print(f"    - {task.task_type.value}: {task.id}")
    
    # Check feasibility before depot addition
    feasible_before = is_feasible(route_without_depot, debug_feasibility=False, return_reason=True)
    print(f"  Feasible before depot addition: {feasible_before}")
    
    # Add depot tasks
    _add_depot_tasks_to_route(route_without_depot)
    
    print(f"  Route after depot addition: {len(route_without_depot.tasks)} tasks")
    for task in route_without_depot.tasks:
        print(f"    - {task.task_type.value}: {task.id}")
    
    # Check feasibility after depot addition
    feasible_after = is_feasible(route_without_depot, debug_feasibility=False, return_reason=True)
    print(f"  Feasible after depot addition: {feasible_after}")
    
    # Verify depot structure
    has_depot_start = route_without_depot.tasks[0].is_depot_start()
    has_depot_return = route_without_depot.tasks[-1].is_depot_return()
    
    if has_depot_start and has_depot_return and feasible_after[0]:
        print("  ✅ Manual depot task addition working correctly")
    else:
        print("  ❌ Manual depot task addition has issues")
        print(f"    - Depot start: {has_depot_start}")
        print(f"    - Depot return: {has_depot_return}")
        print(f"    - Feasible: {feasible_after}")
    
    print(f"\n=== Test 2: Full L1 Heuristic with Depot Finalization ===")
    
    # Run the full L1 heuristic
    params = {
        'M1': 5,  # Short run for testing
        'M2': 5,
        'tabu_tenure': 3,
        'initialization_method': 'cluster_aware'
    }
    
    solution = l1_heuristic(orders, vehicles, params)
    
    print(f"  L1 heuristic completed")
    print(f"  Final solution has {len(solution.routes)} routes")
    
    # Verify all routes have proper depot structure
    all_routes_valid = True
    
    for vehicle_id, route in solution.routes.items():
        if not route.tasks:
            print(f"    - Route {vehicle_id}: EMPTY")
            continue
            
        print(f"    - Route {vehicle_id}: {len(route.tasks)} tasks")
        
        # Check depot structure
        first_is_depot_start = route.tasks[0].is_depot_start()
        last_is_depot_return = route.tasks[-1].is_depot_return()
        route_feasible = is_feasible(route, debug_feasibility=False)
        
        print(f"      - First task depot start: {first_is_depot_start}")
        print(f"      - Last task depot return: {last_is_depot_return}")
        print(f"      - Route feasible: {route_feasible}")
        
        if not (first_is_depot_start and last_is_depot_return and route_feasible):
            all_routes_valid = False
            print(f"      ❌ Route {vehicle_id} has depot structure issues")
    
    if all_routes_valid:
        print("  ✅ All routes have proper depot structure")
    else:
        print("  ❌ Some routes have depot structure issues")
    
    print(f"\n=== Test 3: Edge Cases ===")
    
    # Test empty route
    empty_route = Route(vehicle=vehicle2)
    empty_route.tasks = []
    
    print(f"  Testing empty route...")
    _add_depot_tasks_to_route(empty_route)
    
    if len(empty_route.tasks) == 2:
        print(f"  ✅ Empty route correctly populated with depot tasks")
    else:
        print(f"  ❌ Empty route depot addition failed: {len(empty_route.tasks)} tasks")
    
    # Test route that already has depot tasks
    route_with_depot = Route(vehicle=vehicle1)
    depot_start = Task.create_depot_start_task(
        vehicle_id=vehicle1.id,
        depot_location_id="DEPOT1",
        depot_lat=45.0,
        depot_lon=7.0
    )
    depot_return = Task.create_depot_return_task(
        vehicle_id=vehicle1.id,
        depot_location_id="DEPOT1",
        depot_lat=45.0,
        depot_lon=7.0
    )
    
    route_with_depot.tasks = [depot_start, orders[1].pickup_tasks[0], orders[1].delivery_tasks[0], depot_return]
    original_task_count = len(route_with_depot.tasks)
    
    print(f"  Testing route with existing depot tasks ({original_task_count} tasks)...")
    _add_depot_tasks_to_route(route_with_depot)
    
    if len(route_with_depot.tasks) == original_task_count + 2:
        print(f"  ⚠️  Depot tasks added to route that already had them (now {len(route_with_depot.tasks)} tasks)")
        print(f"    - This suggests the function doesn't check for existing depot tasks")
    else:
        print(f"  ✅ Route with existing depot tasks unchanged")
    
    print(f"\n=== Test 4: Robustness Check ===")
    
    # This tests if the current logic can handle edge cases safely
    
    # Test with None tasks
    none_route = Route(vehicle=vehicle1)
    none_route.tasks = None
    
    try:
        _add_depot_tasks_to_route(none_route)
        print(f"  ⚠️  Function handled None tasks without error")
    except Exception as e:
        print(f"  ❌ Function failed with None tasks: {e}")
    
    print(f"\n=== Summary ===")
    
    print(f"  Manual depot addition: {'✅ Working' if feasible_after[0] else '❌ Failed'}")
    print(f"  L1 heuristic depot finalization: {'✅ Working' if all_routes_valid else '❌ Failed'}")
    print(f"  Edge case handling: {'⚠️ Needs improvement' if len(route_with_depot.tasks) > original_task_count else '✅ Good'}")
    
    if feasible_after[0] and all_routes_valid:
        print(f"\n🎉 DEPOT TASK CREATION IS WORKING!")
        print(f"✅ Routes can be manually fixed with depot tasks")
        print(f"✅ L1 heuristic produces routes with proper depot structure")
    else:
        print(f"\n⚠️  DEPOT TASK CREATION NEEDS ATTENTION")
        if not feasible_after[0]:
            print(f"❌ Manual depot addition not working properly")
        if not all_routes_valid:
            print(f"❌ L1 heuristic not ensuring depot structure")

except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()
