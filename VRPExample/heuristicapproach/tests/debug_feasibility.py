#!/usr/bin/env python3
"""
Debug script to investigate why simple routes are failing feasibility checks
"""

import sys
import os

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
algo_dir = os.path.join(current_dir, '..', 'algo')
sys.path.insert(0, algo_dir)

from epdt_data_structures import Vehicle, Task, TaskType, Order, Route
from second_level import is_feasible, DriverState, _simulate_hos_advanced, _sort_tasks_chronologically

def debug_route_feasibility():
    """Debug the specific route that's failing feasibility"""
    print("🔍 Debugging Route Feasibility for furgone_6 scenario")
    print("=" * 60)
    
    # Recreate the exact scenario from the test output
    vehicle = Vehicle(
        id='furgone_6',
        depot_id='depot',
        weight_capacity=3500.0,
        volume_capacity=20.0,
        vehicle_type='standard'
    )
    
    # Create the pickup task (cazzano_bg)
    pickup_task = Task(
        id='pickup_cazzano_bg',
        location_id='cazzano_bg',
        task_type=TaskType.PICKUP,
        order_id='pickup_request_26',
        lat=45.7311,  # Approximate coordinates for Cazzano Sant'Andrea, BG
        lon=9.7347,
        service_time=15.0,  # 15 minutes as shown in output
        demand=1200.0,
        volume=6.0
    )
    
    # Create the delivery task (pickup_bay_6)
    delivery_task = Task(
        id='delivery_pickup_bay_6',
        location_id='pickup_bay_6',
        task_type=TaskType.DELIVERY,
        order_id='pickup_request_26',
        lat=45.4773,  # Approximate depot coordinates
        lon=9.1815,
        service_time=10.0,
        demand=-1200.0,
        volume=-6.0
    )
    
    # Create the route
    route = Route(vehicle=vehicle)
    route.tasks = [pickup_task, delivery_task]
    
    print(f"📋 Route Details:")
    print(f"   Vehicle: {vehicle.id} (capacity: {vehicle.weight_capacity}kg, {vehicle.volume_capacity}m³)")
    print(f"   Tasks: {len(route.tasks)}")
    print(f"   Total load: {sum(t.demand for t in route.tasks if t.is_pickup())}kg")
    print(f"   Total volume: {sum(t.volume for t in route.tasks if t.is_pickup())}m³")
    
    # Test each constraint individually
    print(f"\n🔍 Testing Individual Constraints:")
    
    # 1. Test basic capacity constraints
    print(f"1️⃣ Capacity Constraints:")
    total_weight = sum(t.demand for t in route.tasks if t.is_pickup())
    total_volume = sum(t.volume for t in route.tasks if t.is_pickup())
    weight_ok = total_weight <= vehicle.weight_capacity
    volume_ok = total_volume <= vehicle.volume_capacity
    print(f"   Weight: {total_weight}kg / {vehicle.weight_capacity}kg = {'✅ OK' if weight_ok else '❌ FAIL'}")
    print(f"   Volume: {total_volume}m³ / {vehicle.volume_capacity}m³ = {'✅ OK' if volume_ok else '❌ FAIL'}")
    
    # 2. Test precedence constraints
    print(f"2️⃣ Precedence Constraints:")
    pickup_index = next((i for i, t in enumerate(route.tasks) if t.is_pickup()), -1)
    delivery_index = next((i for i, t in enumerate(route.tasks) if t.is_delivery()), -1)
    precedence_ok = pickup_index < delivery_index
    print(f"   Pickup before delivery: {'✅ OK' if precedence_ok else '❌ FAIL'}")
    print(f"   Pickup index: {pickup_index}, Delivery index: {delivery_index}")
    
    # 3. Test LIFO constraints (if applicable)
    print(f"3️⃣ LIFO Constraints:")
    lifo_required = getattr(vehicle, 'lifo_required', False)
    print(f"   LIFO required: {lifo_required}")
    if lifo_required:
        # Test LIFO logic
        lifo_stack = []
        lifo_ok = True
        for task in route.tasks:
            if task.is_pickup():
                lifo_stack.append(task.order_id)
            elif task.is_delivery():
                if not lifo_stack or lifo_stack[-1] != task.order_id:
                    lifo_ok = False
                    break
                lifo_stack.pop()
        print(f"   LIFO compliance: {'✅ OK' if lifo_ok else '❌ FAIL'}")
    else:
        print(f"   LIFO compliance: ✅ N/A (not required)")
    
    # 4. Test HoS constraints in detail
    print(f"4️⃣ Hours of Service (HoS) Constraints:")
    try:
        driver_state = DriverState()
        sorted_tasks = _sort_tasks_chronologically(route.tasks)
        
        print(f"   Driver state initialized: ✅")
        print(f"   Tasks sorted: {len(sorted_tasks)} tasks")
        
        # Run HoS simulation with detailed output
        hos_feasible, total_time = _simulate_hos_advanced(route, driver_state, sorted_tasks)
        
        estimated_days = max(1, int(total_time / (24 * 60)) + (1 if total_time % (24 * 60) > 0 else 0))
        
        print(f"   HoS simulation result: {'✅ FEASIBLE' if hos_feasible else '❌ INFEASIBLE'}")
        print(f"   Total simulation time: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
        print(f"   Estimated days: {estimated_days}")
        
        # Check against standard limits
        daily_limit = 9 * 60  # 9 hours in minutes
        weekly_limit = 56 * 60  # 56 hours in minutes
        
        print(f"   vs Daily limit (9h): {total_time:.1f} / {daily_limit} = {'✅ OK' if total_time <= daily_limit else '❌ EXCEEDED'}")
        print(f"   vs Weekly limit (56h): {total_time:.1f} / {weekly_limit} = {'✅ OK' if total_time <= weekly_limit else '❌ EXCEEDED'}")
        
        if not hos_feasible:
            print(f"   🚨 HoS VIOLATION DETECTED - This is likely the cause!")
            
    except Exception as e:
        print(f"   ❌ Error in HoS simulation: {e}")
        hos_feasible = False
    
    # 5. Run full feasibility check
    print(f"\n5️⃣ Full Feasibility Check:")
    try:
        full_feasible = is_feasible(route)
        print(f"   Result: {'✅ FEASIBLE' if full_feasible else '❌ INFEASIBLE'}")
    except Exception as e:
        print(f"   ❌ Error in feasibility check: {e}")
        full_feasible = False
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Basic constraints: {'✅ PASS' if weight_ok and volume_ok and precedence_ok else '❌ FAIL'}")
    print(f"   HoS constraints: {'✅ PASS' if hos_feasible else '❌ FAIL'}")
    print(f"   Overall feasibility: {'✅ FEASIBLE' if full_feasible else '❌ INFEASIBLE'}")
    
    if not full_feasible:
        print(f"\n🔧 Recommendation:")
        if not hos_feasible:
            print(f"   The route fails due to Hours of Service regulations.")
            print(f"   Even a {total_time/60:.1f}h route may require mandatory breaks or rest periods.")
            print(f"   Consider investigating HoS simulation parameters or break requirements.")
        else:
            print(f"   The route fails for reasons other than HoS.")
            print(f"   Check other constraints like time windows or vehicle capabilities.")

if __name__ == "__main__":
    debug_route_feasibility()
