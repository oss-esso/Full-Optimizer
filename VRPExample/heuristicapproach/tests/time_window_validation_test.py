"""
Simple test to verify time window violation fixes.
"""

import sys
import os

# Add path for imports
test_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(test_dir, '..')
algo_dir = os.path.join(heuristic_root, 'algo')
sys.path.insert(0, heuristic_root)
sys.path.insert(0, algo_dir)

print("=== Time Window Validation Test ===")

try:
    from second_level import is_feasible
    from epdt_data_structures import Route, Task, Vehicle, TaskType
    
    print("Successfully imported required modules")
    
    # Create a simple vehicle
    vehicle = Vehicle(
        id="TEST_VEHICLE", 
        depot_id="DEPOT1",
        weight_capacity=1000, 
        volume_capacity=1000,
        pallet_capacity=10  # Add pallet capacity to avoid None comparison error
    )
    
    # Create test tasks with time windows - proper depot structure
    depot_start = Task(
        id="depot_start",
        location_id="DEPOT1", 
        task_type=TaskType.DEPOT_START,
        order_id="depot_start_order",
        lat=45.0,
        lon=7.0,
        service_time=0,  # No service time at depot
        earliest_time=0,
        latest_time=1440,  # 24 hours
        demand=0
    )
    
    task1 = Task(
        id="task1",
        location_id="loc1", 
        task_type=TaskType.PICKUP,
        order_id="order1",
        lat=45.0,
        lon=7.0,
        service_time=30,  # 30 minutes service
        earliest_time=100,  # Can start at minute 100
        latest_time=200,    # Must start by minute 200
        demand=10
    )
    
    task2 = Task(
        id="task2",
        location_id="loc2",
        task_type=TaskType.DELIVERY, 
        order_id="order1",
        lat=45.1,
        lon=7.1,
        service_time=20,  # 20 minutes service
        earliest_time=250,  # Can start at minute 250
        latest_time=300,    # Must start by minute 300
        demand=-10
    )
    
    depot_return = Task(
        id="depot_return",
        location_id="DEPOT1", 
        task_type=TaskType.DEPOT_RETURN,
        order_id="depot_return_order",
        lat=45.0,
        lon=7.0,
        service_time=0,  # No service time at depot
        earliest_time=0,
        latest_time=1440,  # 24 hours
        demand=0
    )
    
    # Create route
    route = Route(vehicle=vehicle)
    route.tasks = [depot_start, task1, task2, depot_return]
    
    print(f"\nCreated test route with {len(route.tasks)} tasks:")
    print(f"Depot Start: service_time={depot_start.service_time}")
    print(f"Task 1: service_time={task1.service_time}, window=[{task1.earliest_time}, {task1.latest_time}]")
    print(f"Task 2: service_time={task2.service_time}, window=[{task2.earliest_time}, {task2.latest_time}]")
    print(f"Depot Return: service_time={depot_return.service_time}")
    
    # Test 1: Normal feasible route (should pass)
    print(f"\n=== Test 1: Normal Route (should be feasible) ===")
    feasible = is_feasible(route, debug_feasibility=True, return_reason=True)
    print(f"Result: {feasible}")
    
    # Test 2: Tight time window (potential violation)
    print(f"\n=== Test 2: Tight Time Window ===")
    # Make task2's latest time very tight
    task2.latest_time = 150  # Very tight - might be impossible to reach in time
    
    feasible = is_feasible(route, debug_feasibility=True, return_reason=True)
    print(f"Result: {feasible}")
    
    # Test 3: Impossible time window
    print(f"\n=== Test 3: Impossible Time Window ===")
    task2.latest_time = 120  # Impossible - earlier than task1 completion
    
    feasible = is_feasible(route, debug_feasibility=True, return_reason=True)
    print(f"Result: {feasible}")
    
    print(f"\n=== Time Window Validation Test Complete ===")

except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()
