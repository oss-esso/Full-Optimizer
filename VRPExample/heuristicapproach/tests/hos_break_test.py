"""
Detailed test to verify break logic in HoS implementation.
"""

import sys
import os

# Add path for imports
test_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(test_dir, '..')
algo_dir = os.path.join(heuristic_root, 'algo')
sys.path.insert(0, heuristic_root)
sys.path.insert(0, algo_dir)

print("=== HoS Break Logic Test ===")

try:
    from second_level import _check_hos_multiday
    from epdt_data_structures import Route, Task, Vehicle, TaskType, DriverState
    
    # Create test vehicle
    vehicle = Vehicle(
        id="TEST_VEHICLE", 
        depot_id="DEPOT1",
        weight_capacity=1000, 
        volume_capacity=1000,
        pallet_capacity=10
    )
    
    # Test: Simulated 6-hour journey that should require a break
    print(f"\n=== Testing 6-Hour Journey (Should Require Break) ===")
    
    # Create a driver state with some initial driving
    driver_state = DriverState()
    
    # Manually set up a scenario that will definitely trigger the break logic
    # Drive for 3 hours first
    driver_state.drive_since_break = 3 * 60  # 3 hours (180 minutes)
    driver_state.drive_today = 3 * 60
    driver_state.work_today = 3 * 60
    
    print(f"Initial state: drive_since_break={driver_state.drive_since_break/60:.1f}h")
    print(f"MAX_DRIVE_WITHOUT_BREAK={driver_state.MAX_DRIVE_WITHOUT_BREAK/60:.1f}h")
    
    # Create tasks with a very long journey between them
    task1 = Task(
        id="start",
        location_id="start_loc", 
        task_type=TaskType.PICKUP,
        order_id="test_order",
        lat=45.0,
        lon=7.0,
        service_time=10,  # 10 minutes service
        earliest_time=0,
        latest_time=1440,
        demand=10
    )
    
    # Create a task very far away to force a long journey
    task2 = Task(
        id="end",
        location_id="end_loc",
        task_type=TaskType.DELIVERY, 
        order_id="test_order",
        lat=35.0,  # Much further south (about 1000+ km)
        lon=7.0,
        service_time=10,
        earliest_time=0,
        latest_time=1440,
        demand=-10
    )
    
    route = Route(vehicle=vehicle)
    route.tasks = [task1, task2]
    
    print(f"Testing route with long journey...")
    print(f"Distance: from ({task1.lat}, {task1.lon}) to ({task2.lat}, {task2.lon})")
    
    result = _check_hos_multiday(route, driver_state, route.tasks)
    
    print(f"Result: {result}")
    print(f"Final state: drive_since_break={driver_state.drive_since_break/60:.1f}h, work_today={driver_state.work_today/60:.1f}h")
    
    # Check if break was taken (drive_since_break should be less than initial + full journey)
    if driver_state.drive_since_break < 4.5 * 60:  # Less than 4.5 hours
        print("✅ SUCCESS: Break logic worked - drive_since_break reset during journey")
    else:
        print("⚠️  WARNING: Break logic may not have triggered properly")
    
    # Test exceeding the 4.5 hour limit
    print(f"\n=== Testing Immediate Violation (Already at 4.5h limit) ===")
    
    driver_state_limit = DriverState()
    driver_state_limit.drive_since_break = 4.5 * 60  # Exactly at limit
    driver_state_limit.drive_today = 4.5 * 60
    driver_state_limit.work_today = 4.5 * 60
    
    print(f"Starting exactly at limit: drive_since_break={driver_state_limit.drive_since_break/60:.1f}h")
    
    result_limit = _check_hos_multiday(route, driver_state_limit, route.tasks)
    
    print(f"Result: {result_limit}")
    print(f"Final state: drive_since_break={driver_state_limit.drive_since_break/60:.1f}h, work_today={driver_state_limit.work_today/60:.1f}h")
    
    if driver_state_limit.drive_since_break < 4.0 * 60:  # Should have reset after break
        print("✅ SUCCESS: Break taken when starting at limit")
    else:
        print("⚠️  Check: May need to verify break timing")

except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()
