#!/usr/bin/env python3

import sys
import os

# Add path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, parent_dir)

try:
    from algo.second_level import is_feasible
    print("Successfully imported is_feasible function")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

try:
    from algo.epdt_data_structures import Route, Task, Vehicle, TaskType
    print("Successfully imported data structures")
except ImportError as e:
    print(f"Data structures import error: {e}")
    sys.exit(1)

# Create a simple test route
def create_simple_route():
    """Create a simple route with one pickup and one delivery"""
    
    # Create a basic vehicle
    vehicle = Vehicle(
        id="test_vehicle",
        depot_id="depot",
        weight_capacity=1000.0,
        volume_capacity=10.0,
        lifo_required=False
    )
    
    # Create simple tasks
    pickup_task = Task(
        id="pickup_1",
        location_id="loc1", 
        task_type=TaskType.PICKUP,
        order_id="order1",
        lat=45.0,
        lon=9.0,
        service_time=10.0,
        demand=100.0,
        volume=1.0,
        earliest_time=0,
        latest_time=1440  # 24 hours
    )
    
    delivery_task = Task(
        id="delivery_1",
        location_id="loc2",
        task_type=TaskType.DELIVERY, 
        order_id="order1",
        lat=45.1,
        lon=9.1,
        service_time=10.0,
        demand=-100.0,  # negative for delivery
        volume=-1.0,    # negative for delivery
        earliest_time=0,
        latest_time=1440  # 24 hours
    )
    
    # Create route
    route = Route(vehicle=vehicle, tasks=[pickup_task, delivery_task])
    
    return route

def main():
    print("Testing simple route feasibility...")
    
    # Create a simple route
    route = create_simple_route()
    print(f"Created route with {len(route.tasks)} tasks")
    
    # Test feasibility
    try:
        feasible = is_feasible(route)
        print(f"Route feasibility: {feasible}")
        
        if not feasible:
            print("Route is marked as infeasible. Let's investigate why...")
            
    except Exception as e:
        print(f"Error during feasibility check: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
