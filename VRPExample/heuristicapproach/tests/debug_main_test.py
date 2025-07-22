#!/usr/bin/env python3
"""
Targeted debug to examine the exact Route objects from the main test
"""

import sys
import os

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
src_dir = os.path.join(heuristic_root, 'src')
algo_dir = os.path.join(heuristic_root, 'algo')

sys.path.insert(0, heuristic_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, algo_dir)

# Import the same functions as the main test
from moda_scenarios import create_furgoni_scenario
from data_adapter import convert_instance_to_epdt_input
from second_level import is_feasible

def debug_main_test_routes():
    """Debug the exact routes created in the main test"""
    print("🔍 Debugging Main Test Route Objects")
    print("=" * 50)
    
    # Recreate the exact scenario from main test
    print("1️⃣ Loading scenario...")
    vrp_instance = create_furgoni_scenario()
    
    print("2️⃣ Converting to EPDT format...")
    orders, vehicles, depot = convert_instance_to_epdt_input(vrp_instance)
    
    print("3️⃣ Finding furgone_6 and pickup_request_26...")
    
    # Find furgone_6 vehicle
    furgone_6 = None
    for vehicle in vehicles:
        if vehicle.id == 'furgone_6':
            furgone_6 = vehicle
            break
    
    if not furgone_6:
        print("❌ furgone_6 not found")
        return
    
    # Find pickup_request_26 order
    pickup_request_26 = None
    for order in orders:
        if order.id == 'pickup_request_26':
            pickup_request_26 = order
            break
    
    if not pickup_request_26:
        print("❌ pickup_request_26 not found")
        return
    
    print(f"✅ Found furgone_6: {furgone_6.id}")
    print(f"✅ Found pickup_request_26: {pickup_request_26.id}")
    print(f"   Pickup tasks: {len(pickup_request_26.pickup_tasks)}")
    print(f"   Delivery tasks: {len(pickup_request_26.delivery_tasks)}")
    
    # Create the route exactly as the main test does
    from epdt_data_structures import Route
    
    route = Route(vehicle=furgone_6)
    
    # Add tasks in the same order as main test (pickup first, then delivery)
    all_tasks = pickup_request_26.pickup_tasks + pickup_request_26.delivery_tasks
    route.tasks = all_tasks
    
    print(f"\n4️⃣ Route Details:")
    print(f"   Vehicle: {route.vehicle.id}")
    print(f"   Tasks: {len(route.tasks)}")
    for i, task in enumerate(route.tasks):
        task_type = "📦" if task.is_pickup() else "🏪"
        print(f"   {i+1}. {task_type} {task.location_id} ({task.order_id}) - {task.demand}kg, {task.volume}m³")
    
    print(f"\n5️⃣ Testing feasibility with both methods:")
    
    # Method 1: Direct function call (like my debug script)
    direct_feasible = is_feasible(route)
    print(f"   Direct function call: {'✅ FEASIBLE' if direct_feasible else '❌ INFEASIBLE'}")
    
    # Method 2: Route method call (like main test)
    try:
        method_feasible = route.is_feasible()
        print(f"   Route method call: {'✅ FEASIBLE' if method_feasible else '❌ INFEASIBLE'}")
    except Exception as e:
        print(f"   Route method call: ❌ ERROR - {e}")
    
    # Method 3: Check cache state
    print(f"   Route cache state: {getattr(route, '_is_feasible_cached', 'None')}")
    
    # Method 4: Detail comparison
    if direct_feasible != method_feasible:
        print(f"\n🚨 DISCREPANCY DETECTED!")
        print(f"   Direct call: {direct_feasible}")
        print(f"   Route method: {method_feasible}")
        print(f"   This explains why the main test shows different results!")
        
        # Clear cache and test again
        route._is_feasible_cached = None
        try:
            method_feasible_2 = route.is_feasible()
            print(f"   Route method (cache cleared): {'✅ FEASIBLE' if method_feasible_2 else '❌ INFEASIBLE'}")
        except Exception as e:
            print(f"   Route method (cache cleared): ❌ ERROR - {e}")

if __name__ == "__main__":
    debug_main_test_routes()
