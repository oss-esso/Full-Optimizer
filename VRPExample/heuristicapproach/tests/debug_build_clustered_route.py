#!/usr/bin/env python3
"""
Quick debug test to check what's happening in build_clustered_route
"""

import sys
import os

# Add the algo directory to the Python path
algo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'algo')
sys.path.insert(0, algo_path)

try:
    from epdt_data_structures import Route, Vehicle, Task, TaskType, Order
    from first_level import _create_base_route, build_clustered_route
    
    print("🔍 Debug Test: Checking build_clustered_route behavior")
    print("=" * 60)
    
    # Create vehicle
    vehicle = Vehicle(
        id="DEBUG_VEHICLE",
        depot_id="DEPOT-ASTI", 
        weight_capacity=1000.0,
        volume_capacity=100.0,
        pallet_capacity=10,
        cost_per_km=1.0,
        cost_per_hour=50.0
    )
    
    # Create base route
    print("📋 Creating base route...")
    base_route = _create_base_route(vehicle)
    print(f"✅ Base route has {len(base_route.tasks)} tasks")
    print("   Tasks:")
    for i, task in enumerate(base_route.tasks):
        print(f"     {i+1}. {task.task_type.name} - {task.id}")
    
    # Create simple order
    pickup = Task(
        id="PICKUP_DEBUG", location_id="CUSTOMER_A", task_type=TaskType.PICKUP,
        order_id="ORDER_DEBUG", lat=44.9200, lon=8.2200, service_time=15.0,
        demand=100.0, volume=10.0, pallets=2, earliest_time=480, latest_time=1020
    )
    delivery = Task(
        id="DELIVERY_DEBUG", location_id="CUSTOMER_B", task_type=TaskType.DELIVERY,
        order_id="ORDER_DEBUG", lat=44.9500, lon=8.2500, service_time=10.0,
        demand=100.0, volume=10.0, pallets=2, earliest_time=480, latest_time=1020
    )
    order = Order(id="ORDER_DEBUG", pickup_tasks=[pickup], delivery_tasks=[delivery])
    
    print(f"\\n📦 Created order: {order.id}")
    
    # Test build_clustered_route
    print(f"\\n🎯 Calling build_clustered_route...")
    result_route = build_clustered_route(base_route, [order], debug_assignment=True)
    
    if result_route:
        print(f"✅ build_clustered_route returned route with {len(result_route.tasks)} tasks")
        print("   Final route tasks:")
        for i, task in enumerate(result_route.tasks):
            print(f"     {i+1}. {task.task_type.name} - {task.id}")
        
        # Check depot structure
        if result_route.tasks:
            print(f"\\n🔍 Route structure analysis:")
            print(f"   First task is depot start: {result_route.tasks[0].is_depot_start()}")
            print(f"   Last task is depot return: {result_route.tasks[-1].is_depot_return()}")
            
            if not result_route.tasks[0].is_depot_start():
                print(f"   ❌ PROBLEM: First task is {result_route.tasks[0].task_type.name}, not DEPOT_START")
            if not result_route.tasks[-1].is_depot_return():
                print(f"   ❌ PROBLEM: Last task is {result_route.tasks[-1].task_type.name}, not DEPOT_RETURN")
    else:
        print("❌ build_clustered_route returned None")
    
    print("\\n" + "=" * 60)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
