#!/usr/bin/env python3
"""
Test for Issue 2: Fix Time Window Violations

This test creates a scenario with tight time windows that are impossible to meet
given the travel times, and verifies that such routes are marked as infeasible.
"""

import sys
import os

# Add path for imports
test_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(test_dir, '..')
algo_dir = os.path.join(heuristic_root, 'algo')
sys.path.insert(0, heuristic_root)
sys.path.insert(0, algo_dir)

print("=== Time Window Violations Test ===")

try:
    from second_level import is_feasible
    from epdt_data_structures import Route, Task, Vehicle, TaskType
    
    print("Successfully imported required modules")
    
    # Create a test vehicle
    vehicle = Vehicle(
        id="TEST_VEHICLE", 
        depot_id="DEPOT1",
        weight_capacity=1000, 
        volume_capacity=1000,
        pallet_capacity=10
    )
    
    print(f"\n=== Scenario 1: Impossible Time Window Sequence ===")
    
    # Create depot tasks
    depot_start = Task(
        id="depot_start",
        location_id="DEPOT1", 
        task_type=TaskType.DEPOT_START,
        order_id="depot_start_order",
        lat=45.0,
        lon=7.0,
        service_time=0,
        earliest_time=0,
        latest_time=1440,
        demand=0
    )
    
    depot_return = Task(
        id="depot_return",
        location_id="DEPOT1", 
        task_type=TaskType.DEPOT_RETURN,
        order_id="depot_return_order",
        lat=45.0,
        lon=7.0,
        service_time=0,
        earliest_time=0,
        latest_time=1440,
        demand=0
    )
    
    # Create a sequence that should violate time windows
    task1 = Task(
        id="task1",
        location_id="location1",
        task_type=TaskType.PICKUP,
        order_id="order1",
        lat=45.5,  # Far from depot
        lon=7.5,
        service_time=60,     # 1 hour service
        earliest_time=180,   # Can start at minute 180 (3 hours)
        latest_time=240,     # Must start by minute 240 (4 hours)
        demand=10
    )
    
    task2 = Task(
        id="task2",
        location_id="location2",
        task_type=TaskType.DELIVERY,
        order_id="order1",
        lat=44.0,  # Very far from task1
        lon=6.0,
        service_time=30,     # 30 minutes service
        earliest_time=250,   # Can start at minute 250
        latest_time=280,     # Must start by minute 280 (IMPOSSIBLE given travel time)
        demand=-10
    )
    
    route = Route(vehicle=vehicle)
    route.tasks = [depot_start, task1, task2, depot_return]
    
    print(f"Testing route with impossible time window sequence:")
    print(f"  Task 1: service at minute 180-240, service_time=60min")
    print(f"  Task 2: service at minute 250-280, but far from Task 1")
    print(f"  Expected: Route should be infeasible due to time window violation")
    
    result = is_feasible(route, debug_feasibility=True, return_reason=True)
    print(f"  Result: {result}")
    
    if result[0] == False:
        print("  ✅ SUCCESS: Time window violation correctly detected")
    else:
        print("  ❌ FAILURE: Time window violation not detected")
    
    print(f"\n=== Scenario 2: Very Tight But Possible Time Windows ===")
    
    # Create a sequence that should just barely work
    task3 = Task(
        id="task3",
        location_id="location3",
        task_type=TaskType.PICKUP,
        order_id="order2",
        lat=45.1,  # Close to depot
        lon=7.1,
        service_time=30,     # 30 minutes service
        earliest_time=60,    # Can start at minute 60
        latest_time=120,     # Must start by minute 120
        demand=10
    )
    
    task4 = Task(
        id="task4",
        location_id="location4",
        task_type=TaskType.DELIVERY,
        order_id="order2",
        lat=45.2,  # Close to task3
        lon=7.2,
        service_time=20,     # 20 minutes service
        earliest_time=200,   # Can start at minute 200 (reasonable gap)
        latest_time=400,     # Must start by minute 400 (generous window)
        demand=-10
    )
    
    route2 = Route(vehicle=vehicle)
    route2.tasks = [depot_start, task3, task4, depot_return]
    
    print(f"Testing route with tight but feasible time windows:")
    print(f"  Task 3: service at minute 60-120, service_time=30min")
    print(f"  Task 4: service at minute 200-400, close to Task 3")
    print(f"  Expected: Route should be feasible")
    
    result2 = is_feasible(route2, debug_feasibility=True, return_reason=True)
    print(f"  Result: {result2}")
    
    if result2[0] == True:
        print("  ✅ SUCCESS: Feasible route correctly accepted")
    else:
        print("  ❌ FAILURE: Feasible route incorrectly rejected")
        print(f"    Reason: {result2[1]}")
    
    print(f"\n=== Scenario 3: Late Start Violation ===")
    
    # Create a task that starts too late
    late_task = Task(
        id="late_task",
        location_id="location5",
        task_type=TaskType.PICKUP,
        order_id="order3",
        lat=46.0,  # Very far, requiring long travel
        lon=8.0,
        service_time=30,
        earliest_time=50,    # Can start at minute 50
        latest_time=60,      # Must start by minute 60 (impossible given distance)
        demand=10
    )
    
    route3 = Route(vehicle=vehicle)
    route3.tasks = [depot_start, late_task, depot_return]
    
    print(f"Testing route with late start violation:")
    print(f"  Late task: far from depot, must start by minute 60")
    print(f"  Expected: Route should be infeasible due to late arrival")
    
    result3 = is_feasible(route3, debug_feasibility=True, return_reason=True)
    print(f"  Result: {result3}")
    
    if result3[0] == False:
        print("  ✅ SUCCESS: Late arrival correctly detected")
    else:
        print("  ❌ FAILURE: Late arrival not detected")
    
    print(f"\n=== Summary ===")
    
    scenario1_correct = result[0] == False    # Should be infeasible
    scenario2_correct = result2[0] == True    # Should be feasible
    scenario3_correct = result3[0] == False   # Should be infeasible
    
    if scenario1_correct and scenario2_correct and scenario3_correct:
        print("🎉 ALL TIME WINDOW TESTS PASSED!")
        print("✅ Time window violations are being properly detected and prevented")
    else:
        print("⚠️  Some time window tests failed:")
        if not scenario1_correct:
            print("  - Impossible time window sequence not detected")
        if not scenario2_correct:
            print("  - Feasible tight windows incorrectly rejected")  
        if not scenario3_correct:
            print("  - Late arrival not detected")
    
    print(f"\n=== Time Window Violations Test Complete ===")

except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()
