#!/usr/bin/env python3
"""
Test to validate the fixed HoS and time window enforcement.
"""

import sys
import os

# Add path for imports
test_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(test_dir, '..')
algo_dir = os.path.join(heuristic_root, 'algo')
sys.path.insert(0, heuristic_root)
sys.path.insert(0, algo_dir)

print("=== Fixed HoS and Time Window Validation Test ===")

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
    
    print(f"\n=== Test 1: Verifying HoS Enforcement (No Bypass) ===")
    
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
    
    # Create a route that should violate HoS (excessive service time)
    violation_task = Task(
        id="violation_task",
        location_id="far_location",
        task_type=TaskType.PICKUP,
        order_id="order_violation",
        lat=47.0,  # Far location requiring travel
        lon=9.0,
        service_time=800,  # 13+ hours service time (should violate work limits)
        earliest_time=0,
        latest_time=1440,
        demand=10
    )
    
    route1 = Route(vehicle=vehicle)
    route1.tasks = [depot_start, violation_task, depot_return]
    
    print(f"Testing route with excessive service time (13+ hours):")
    print(f"  Service time: {violation_task.service_time} minutes")
    
    # This should now be rejected (no longer bypassed during initialization)
    result1 = is_feasible(route1, debug_feasibility=True, return_reason=True)
    print(f"  Result: {result1}")
    
    if result1[0] == False:
        print("  ✅ SUCCESS: HoS violations are now properly detected and rejected")
    else:
        print("  ❌ ISSUE: HoS violations are still being allowed")
    
    print(f"\n=== Test 2: Verifying Time Window Enforcement ===")
    
    # Create tasks with impossible time windows
    early_task = Task(
        id="early_task",
        location_id="location1",
        task_type=TaskType.PICKUP,
        order_id="order_tw",
        lat=45.1,
        lon=7.1,
        service_time=30,
        earliest_time=60,     # Can start at minute 60
        latest_time=90,       # Must start by minute 90
        demand=10
    )
    
    impossible_task = Task(
        id="impossible_task",
        location_id="location2",
        task_type=TaskType.DELIVERY,
        order_id="order_tw",
        lat=45.5,  # Far location requiring significant travel time
        lon=7.5,
        service_time=30,
        earliest_time=100,    # Can start at minute 100
        latest_time=110,      # Must start by minute 110 (impossible given travel + service)
        demand=-10
    )
    
    route2 = Route(vehicle=vehicle)
    route2.tasks = [depot_start, early_task, impossible_task, depot_return]
    
    print(f"Testing route with impossible time window sequence:")
    print(f"  Early task: window=[{early_task.earliest_time}, {early_task.latest_time}]")
    print(f"  Impossible task: window=[{impossible_task.earliest_time}, {impossible_task.latest_time}]")
    
    result2 = is_feasible(route2, debug_feasibility=True, return_reason=True)
    print(f"  Result: {result2}")
    
    if result2[0] == False:
        print("  ✅ SUCCESS: Time window violations are properly detected and rejected")
    else:
        print("  ❌ ISSUE: Time window violations are still being allowed")
    
    print(f"\n=== Test 3: Verifying Valid Route Acceptance ===")
    
    # Create a valid route that should pass all checks
    valid_task = Task(
        id="valid_task",
        location_id="nearby_location",
        task_type=TaskType.PICKUP,
        order_id="order_valid",
        lat=45.05,  # Close location
        lon=7.05,
        service_time=30,     # Reasonable service time
        earliest_time=0,
        latest_time=1440,    # Flexible time window
        demand=10
    )
    
    route3 = Route(vehicle=vehicle)
    route3.tasks = [depot_start, valid_task, depot_return]
    
    print(f"Testing valid route:")
    print(f"  Service time: {valid_task.service_time} minutes")
    print(f"  Time window: [{valid_task.earliest_time}, {valid_task.latest_time}]")
    
    result3 = is_feasible(route3, debug_feasibility=True, return_reason=True)
    print(f"  Result: {result3}")
    
    if result3[0] == True:
        print("  ✅ SUCCESS: Valid routes are still being accepted")
    else:
        print("  ❌ ISSUE: Valid routes are being incorrectly rejected")
        print(f"    Reason: {result3[1]}")
    
    print(f"\n=== Summary ===")
    
    hos_working = result1[0] == False
    tw_working = result2[0] == False
    valid_working = result3[0] == True
    
    if hos_working and tw_working and valid_working:
        print("🎉 ALL TESTS PASSED: HoS and time window enforcement is working correctly!")
        print("✅ The production issues should now be resolved")
    else:
        print("⚠️  Some tests failed:")
        if not hos_working:
            print("  - HoS enforcement not working")
        if not tw_working:
            print("  - Time window enforcement not working")  
        if not valid_working:
            print("  - Valid route acceptance not working")
    
    print(f"\n=== Fixed Validation Test Complete ===")

except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()
