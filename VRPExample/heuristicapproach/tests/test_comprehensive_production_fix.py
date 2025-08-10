#!/usr/bin/env python3
"""
Comprehensive test to validate all production issue fixes.
"""

import sys
import os

# Add path for imports
test_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(test_dir, '..')
algo_dir = os.path.join(heuristic_root, 'algo')
sys.path.insert(0, heuristic_root)
sys.path.insert(0, algo_dir)

print("=== Comprehensive Production Issues Fix Validation ===")

try:
    from second_level import is_feasible
    from first_level import _validate_and_filter_solution
    from epdt_data_structures import Route, Task, Vehicle, TaskType, Solution
    
    print("Successfully imported required modules")
    
    # Create test vehicles
    vehicle1 = Vehicle(
        id="VEHICLE_1", 
        depot_id="DEPOT1",
        weight_capacity=1000, 
        volume_capacity=1000,
        pallet_capacity=10
    )
    
    vehicle2 = Vehicle(
        id="VEHICLE_2", 
        depot_id="DEPOT1",
        weight_capacity=1000, 
        volume_capacity=1000,
        pallet_capacity=10
    )
    
    print(f"\n=== Test 1: HoS Violations are Prevented ===")
    
    # Create depot tasks
    depot_start1 = Task(
        id="depot_start_1",
        location_id="DEPOT1", 
        task_type=TaskType.DEPOT_START,
        order_id="depot_start_order_1",
        lat=45.0,
        lon=7.0,
        service_time=0,
        earliest_time=0,
        latest_time=1440,
        demand=0
    )
    
    depot_return1 = Task(
        id="depot_return_1",
        location_id="DEPOT1", 
        task_type=TaskType.DEPOT_RETURN,
        order_id="depot_return_order_1",
        lat=45.0,
        lon=7.0,
        service_time=0,
        earliest_time=0,
        latest_time=1440,
        demand=0
    )
    
    # Create route with HoS violation
    hos_violation_task = Task(
        id="hos_task",
        location_id="location1",
        task_type=TaskType.PICKUP,
        order_id="order_hos",
        lat=45.1,
        lon=7.1,
        service_time=900,  # 15 hours - should violate work limits
        earliest_time=0,
        latest_time=1440,
        demand=10
    )
    
    route1 = Route(vehicle=vehicle1)
    route1.tasks = [depot_start1, hos_violation_task, depot_return1]
    
    result1 = is_feasible(route1, debug_feasibility=False, return_reason=True)
    print(f"  HoS violation route: {result1}")
    
    if result1[0] == False and "HoS" in result1[1]:
        print("  ✅ HoS violations correctly prevented")
    else:
        print("  ❌ HoS violations not being prevented")
    
    print(f"\n=== Test 2: Empty Routes are Filtered ===")
    
    # Create depot-only route
    depot_start2 = Task(
        id="depot_start_2",
        location_id="DEPOT1", 
        task_type=TaskType.DEPOT_START,
        order_id="depot_start_order_2",
        lat=45.0,
        lon=7.0,
        service_time=0,
        earliest_time=0,
        latest_time=1440,
        demand=0
    )
    
    depot_return2 = Task(
        id="depot_return_2",
        location_id="DEPOT1", 
        task_type=TaskType.DEPOT_RETURN,
        order_id="depot_return_order_2",
        lat=45.0,
        lon=7.0,
        service_time=0,
        earliest_time=0,
        latest_time=1440,
        demand=0
    )
    
    route2 = Route(vehicle=vehicle2)
    route2.tasks = [depot_start2, depot_return2]  # Only depot tasks
    
    # Test solution filtering
    test_solution = Solution()
    test_solution.routes = {
        "VEHICLE_1": route1,  # HoS violation route
        "VEHICLE_2": route2   # Empty route
    }
    
    print(f"  Original solution has {len(test_solution.routes)} routes")
    
    filtered_solution = _validate_and_filter_solution(test_solution)
    
    print(f"  Filtered solution has {len(filtered_solution.routes)} routes")
    
    if len(filtered_solution.routes) == 0:
        print("  ✅ All problematic routes correctly filtered out")
    else:
        print("  ⚠️  Some routes remained after filtering:")
        for vehicle_id, route in filtered_solution.routes.items():
            print(f"    - {vehicle_id}: {len(route.tasks)} tasks")
    
    print(f"\n=== Test 3: Valid Routes are Preserved ===")
    
    # Create a valid route
    depot_start3 = Task(
        id="depot_start_3",
        location_id="DEPOT1", 
        task_type=TaskType.DEPOT_START,
        order_id="depot_start_order_3",
        lat=45.0,
        lon=7.0,
        service_time=0,
        earliest_time=0,
        latest_time=1440,
        demand=0
    )
    
    depot_return3 = Task(
        id="depot_return_3",
        location_id="DEPOT1", 
        task_type=TaskType.DEPOT_RETURN,
        order_id="depot_return_order_3",
        lat=45.0,
        lon=7.0,
        service_time=0,
        earliest_time=0,
        latest_time=1440,
        demand=0
    )
    
    valid_task = Task(
        id="valid_task",
        location_id="location3",
        task_type=TaskType.PICKUP,
        order_id="order_valid",
        lat=45.05,  # Close to depot
        lon=7.05,
        service_time=30,    # Reasonable service time
        earliest_time=0,
        latest_time=1440,   # Flexible time window
        demand=10
    )
    
    vehicle3 = Vehicle(
        id="VEHICLE_3", 
        depot_id="DEPOT1",
        weight_capacity=1000, 
        volume_capacity=1000,
        pallet_capacity=10
    )
    
    route3 = Route(vehicle=vehicle3)
    route3.tasks = [depot_start3, valid_task, depot_return3]
    
    valid_solution = Solution()
    valid_solution.routes = {
        "VEHICLE_3": route3   # Valid route
    }
    
    print(f"  Testing valid route with {len(route3.tasks)} tasks")
    
    filtered_valid = _validate_and_filter_solution(valid_solution)
    
    if len(filtered_valid.routes) == 1:
        print("  ✅ Valid routes correctly preserved")
    else:
        print("  ❌ Valid routes incorrectly filtered out")
    
    print(f"\n=== Test 4: Depot Structure Validation ===")
    
    # Create route without proper depot structure
    non_depot_task1 = Task(
        id="non_depot_1",
        location_id="location4",
        task_type=TaskType.PICKUP,
        order_id="order_bad",
        lat=45.1,
        lon=7.1,
        service_time=30,
        earliest_time=0,
        latest_time=1440,
        demand=10
    )
    
    non_depot_task2 = Task(
        id="non_depot_2",
        location_id="location5",
        task_type=TaskType.DELIVERY,
        order_id="order_bad",
        lat=45.2,
        lon=7.2,
        service_time=30,
        earliest_time=0,
        latest_time=1440,
        demand=-10
    )
    
    route4 = Route(vehicle=vehicle1)
    route4.tasks = [non_depot_task1, non_depot_task2]  # No depot tasks
    
    result4 = is_feasible(route4, debug_feasibility=False, return_reason=True)
    print(f"  Route without depot structure: {result4}")
    
    if result4[0] == False and "depot" in result4[1].lower():
        print("  ✅ Depot structure violations correctly detected")
    else:
        print("  ❌ Depot structure violations not being detected")
    
    print(f"\n=== Final Summary ===")
    
    hos_working = result1[0] == False and "HoS" in result1[1]
    depot_working = result4[0] == False and "depot" in result4[1].lower()
    filtering_working = len(filtered_solution.routes) == 0
    valid_preserved = len(filtered_valid.routes) == 1
    
    if hos_working and depot_working and filtering_working and valid_preserved:
        print("🎉 ALL PRODUCTION ISSUES HAVE BEEN RESOLVED!")
        print("✅ HoS violations are prevented")
        print("✅ Empty routes are filtered")
        print("✅ Valid routes are preserved")
        print("✅ Depot structure is validated")
        print("")
        print("The system should now produce clean results without violations.")
    else:
        print("⚠️  Some issues remain:")
        if not hos_working:
            print("  - HoS violations not being prevented")
        if not depot_working:
            print("  - Depot structure not being validated")
        if not filtering_working:
            print("  - Route filtering not working")
        if not valid_preserved:
            print("  - Valid routes being incorrectly removed")
    
    print(f"\n=== Comprehensive Validation Complete ===")

except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()
