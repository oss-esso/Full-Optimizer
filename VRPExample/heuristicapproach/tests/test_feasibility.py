#!/usr/bin/env python3
"""
Simple feasibility test to debug HoS validation issues
"""

import sys
import os

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
algo_dir = os.path.join(current_dir, '..', 'algo')
sys.path.insert(0, algo_dir)

from epdt_data_structures import Vehicle, Task, TaskType, Order, Route
from second_level import is_feasible, DriverState

def test_simple_route():
    """Test a very simple route for feasibility"""
    print("🧪 Testing simple 2-task route feasibility...")
    
    # Create a simple vehicle
    vehicle = Vehicle(
        id='test_vehicle', 
        depot_id='depot', 
        weight_capacity=3500.0, 
        volume_capacity=20.0,
        vehicle_type='standard'
    )
    
    # Create a route
    route = Route(vehicle=vehicle)
    
    # Create simple tasks
    pickup_task = Task(
        id='p1', 
        location_id='depot', 
        task_type=TaskType.PICKUP, 
        order_id='order1', 
        lat=0.0,
        lon=0.0,
        service_time=10.0,
        demand=500.0, 
        volume=2.5
    )
    
    delivery_task = Task(
        id='d1', 
        location_id='nearby', 
        task_type=TaskType.DELIVERY, 
        order_id='order1', 
        lat=1.0,
        lon=1.0,
        service_time=10.0,
        demand=-500.0, 
        volume=-2.5
    )
    
    # Create order with tasks
    order1 = Order(
        id='order1',
        pickup_tasks=[pickup_task],
        delivery_tasks=[delivery_task]
    )
    
    route.tasks = [pickup_task, delivery_task]
    
    # Test feasibility
    try:
        result = is_feasible(route)
        print(f"✅ Simple route feasibility: {result}")
        return result
    except Exception as e:
        print(f"❌ Error testing feasibility: {e}")
        return False

def test_empty_route():
    """Test an empty route"""
    print("🧪 Testing empty route feasibility...")
    
    vehicle = Vehicle(
        id='empty_test', 
        depot_id='depot', 
        weight_capacity=3500.0, 
        volume_capacity=20.0,
        vehicle_type='standard'
    )
    
    route = Route(vehicle=vehicle)
    route.tasks = []
    
    try:
        result = is_feasible(route)
        print(f"✅ Empty route feasibility: {result}")
        return result
    except Exception as e:
        print(f"❌ Error testing empty route: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Debugging Route Feasibility Issues")
    print("=" * 50)
    
    test_empty_route()
    test_simple_route()
    
    print("\n🔍 Analysis:")
    print("If simple routes are failing feasibility, the HoS validation may be too strict.")
    print("If they pass, the issue is likely with the complex routes in the main test.")
