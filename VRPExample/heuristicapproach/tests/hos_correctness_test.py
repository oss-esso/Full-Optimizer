"""
Test to verify the corrected HoS implementation with proper break logic.
"""

import sys
import os

# Add path for imports
test_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(test_dir, '..')
algo_dir = os.path.join(heuristic_root, 'algo')
sys.path.insert(0, heuristic_root)
sys.path.insert(0, algo_dir)

print("=== HoS Implementation Correctness Test ===")

try:
    from second_level import _check_hos_multiday
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
    
    # Create route with tasks that would require breaks
    route = Route(vehicle=vehicle)
    
    # Test Case 1: Journey requiring one break during travel
    print(f"\n=== Test 1: Long Journey Requiring Break ===")
    
    # Task 1: Starting point
    task1 = Task(
        id="task1",
        location_id="loc1", 
        task_type=TaskType.PICKUP,
        order_id="order1",
        lat=45.0,
        lon=7.0,
        service_time=30,  # 30 minutes service
        earliest_time=0,
        latest_time=1440,  # 24 hours
        demand=10
    )
    
    # Task 2: 5-hour drive away (longer than 4.5 hour limit)
    # Simulate by placing it far enough to require 5 hours of travel
    task2 = Task(
        id="task2",
        location_id="loc2",
        task_type=TaskType.DELIVERY, 
        order_id="order1",
        lat=40.0,  # Much further south to simulate long distance
        lon=7.0,
        service_time=20,  # 20 minutes service
        earliest_time=0,
        latest_time=1440,
        demand=-10
    )
    
    route.tasks = [task1, task2]
    sorted_tasks = route.tasks
    
    # Start with fresh driver state
    driver_state = DriverState()
    
    # Test the corrected HoS function
    print(f"Testing route with {len(sorted_tasks)} tasks...")
    print(f"Initial driver state - drive_since_break: {driver_state.drive_since_break}, work_today: {driver_state.work_today}")
    
    result = _check_hos_multiday(route, driver_state, sorted_tasks)
    
    print(f"HoS check result: {result}")
    print(f"Final driver state - drive_since_break: {driver_state.drive_since_break}, work_today: {driver_state.work_today}")
    
    # Test Case 2: Multiple short journeys within limits
    print(f"\n=== Test 2: Multiple Short Journeys ===")
    
    # Reset driver state
    driver_state = DriverState()
    
    # Create multiple nearby tasks
    short_tasks = []
    for i in range(5):
        task = Task(
            id=f"task{i}",
            location_id=f"loc{i}",
            task_type=TaskType.PICKUP if i % 2 == 0 else TaskType.DELIVERY,
            order_id=f"order{i}",
            lat=45.0 + (i * 0.01),  # Very close together
            lon=7.0 + (i * 0.01),
            service_time=30,
            earliest_time=0,
            latest_time=1440,
            demand=10 if i % 2 == 0 else -10
        )
        short_tasks.append(task)
    
    route.tasks = short_tasks
    sorted_tasks = route.tasks
    
    print(f"Testing route with {len(sorted_tasks)} short tasks...")
    result = _check_hos_multiday(route, driver_state, sorted_tasks)
    
    print(f"HoS check result: {result}")
    print(f"Final driver state - drive_since_break: {driver_state.drive_since_break}, work_today: {driver_state.work_today}")
    
    # Test Case 3: Route that would exceed daily limits
    print(f"\n=== Test 3: Route Exceeding Daily Limits ===")
    
    # Reset driver state but pre-load with 8 hours of driving already done
    driver_state = DriverState()
    driver_state.drive_today = 8 * 60  # 8 hours already driven
    driver_state.work_today = 10 * 60  # 10 hours already worked
    
    # Try to add more work that would exceed 9-hour driving or 13-hour work limit
    excessive_task = Task(
        id="excessive_task",
        location_id="far_location",
        task_type=TaskType.DELIVERY,
        order_id="excessive_order",
        lat=40.0,  # Far away to require long travel
        lon=10.0,
        service_time=60,  # 1 hour service
        earliest_time=0,
        latest_time=1440,
        demand=-10
    )
    
    route.tasks = [task1, excessive_task]  # Long journey after already working 10 hours
    sorted_tasks = route.tasks
    
    print(f"Testing route that should exceed daily limits...")
    print(f"Pre-loaded driver state - drive_today: {driver_state.drive_today/60:.1f}h, work_today: {driver_state.work_today/60:.1f}h")
    
    result = _check_hos_multiday(route, driver_state, sorted_tasks)
    
    print(f"HoS check result (should be False): {result}")
    
    print(f"\n=== HoS Implementation Correctness Test Complete ===")
    
    if result == False:
        print("✅ SUCCESS: HoS correctly identified violation for excessive daily limits")
    else:
        print("❌ ISSUE: HoS should have rejected route with excessive daily limits")

except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()
