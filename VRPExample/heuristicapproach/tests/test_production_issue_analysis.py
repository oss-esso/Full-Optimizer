#!/usr/bin/env python3
"""
Test to analyze the current production issues with HoS and time window violations.
"""

import sys
import os

# Add path for imports
test_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(test_dir, '..')
algo_dir = os.path.join(heuristic_root, 'algo')
sys.path.insert(0, heuristic_root)
sys.path.insert(0, algo_dir)

print("=== Production Issue Analysis Test ===")

try:
    from second_level import is_feasible, _check_hos_multiday
    from epdt_data_structures import Route, Task, Vehicle, TaskType, DriverState
    
    print("Successfully imported required modules")
    
    # Create a test vehicle
    vehicle = Vehicle(
        id="TEST_VEHICLE", 
        depot_id="DEPOT1",
        weight_capacity=1000, 
        volume_capacity=1000,
        pallet_capacity=10
    )
    
    print(f"\n=== Issue 1: Testing HoS Integration ===")
    
    # Create a route that should violate HoS
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
    
    # Long journey task
    long_task = Task(
        id="long_task",
        location_id="far_location",
        task_type=TaskType.PICKUP,
        order_id="order1",
        lat=47.0,  # Far location
        lon=9.0,
        service_time=600,  # 10 hours service time (should violate work limits)
        earliest_time=0,
        latest_time=1440,
        demand=10
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
    
    route = Route(vehicle=vehicle)
    route.tasks = [depot_start, long_task, depot_return]
    
    print(f"Testing route with excessive service time (10 hours):")
    print(f"  - Depot start: {depot_start.id}")
    print(f"  - Long task: {long_task.id} (service_time={long_task.service_time} minutes)")
    print(f"  - Depot return: {depot_return.id}")
    
    # Test is_feasible function
    feasible_result = is_feasible(route, debug_feasibility=True, return_reason=True)
    print(f"is_feasible result: {feasible_result}")
    
    # Test direct HoS checking
    driver_state = DriverState()
    sorted_tasks = route.tasks
    hos_result = _check_hos_multiday(route, driver_state, sorted_tasks)
    print(f"Direct HoS check result: {hos_result}")
    
    print(f"\n=== Issue 2: Testing Time Window Violations ===")
    
    # Create a route with time window violations
    early_task = Task(
        id="early_task",
        location_id="location1",
        task_type=TaskType.PICKUP,
        order_id="order2",
        lat=45.1,
        lon=7.1,
        service_time=30,
        earliest_time=60,    # Can start at minute 60
        latest_time=90,      # Must start by minute 90 (tight window)
        demand=10
    )
    
    late_task = Task(
        id="late_task",
        location_id="location2",
        task_type=TaskType.DELIVERY,
        order_id="order2",
        lat=45.2,
        lon=7.2,
        service_time=30,
        earliest_time=200,   # Can start at minute 200
        latest_time=220,     # Must start by minute 220 (impossible given travel time)
        demand=-10
    )
    
    route2 = Route(vehicle=vehicle)
    route2.tasks = [depot_start, early_task, late_task, depot_return]
    
    print(f"Testing route with tight time windows:")
    print(f"  - Early task: window=[{early_task.earliest_time}, {early_task.latest_time}]")
    print(f"  - Late task: window=[{late_task.earliest_time}, {late_task.latest_time}]")
    
    feasible_result2 = is_feasible(route2, debug_feasibility=True, return_reason=True)
    print(f"Time window feasibility result: {feasible_result2}")
    
    print(f"\n=== Issue 3: Testing Depot Structure Validation ===")
    
    # Create a route without proper depot structure
    route3 = Route(vehicle=vehicle)
    route3.tasks = [early_task, late_task]  # Missing depot tasks
    
    print(f"Testing route without depot structure:")
    print(f"  - Tasks: {[task.id for task in route3.tasks]}")
    
    feasible_result3 = is_feasible(route3, debug_feasibility=True, return_reason=True)
    print(f"Depot structure validation result: {feasible_result3}")
    
    print(f"\n=== Analysis Complete ===")

except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()
