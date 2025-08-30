#!/usr/bin/env python3

"""
Targeted investigation for Order 6 assignment failure.
This test focuses specifically on understanding why Order 6 gets rejected
during L2 heuristic insertion despite having valid time windows and capacity.
"""

import sys
import os

# Add the parent directory to Python path for imports
test_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(test_dir, '..')
sys.path.insert(0, heuristic_root)

def main():
    print("=== ORDER 6 INVESTIGATION TEST ===")
    print("Loading scenario and creating Order 6 route...")
    
    # Import required modules
    try:
        from src.create_scenario_from_excel import create_scenario_from_excel
    except ImportError:
        # Try alternative import path
        sys.path.insert(0, os.path.join(heuristic_root, 'src'))
        from create_scenario_from_excel import create_scenario_from_excel
    
    from algo.second_level import l2_heuristic
    from algo.epdt_data_structures import Route
    
    # Load scenario
    excel_file = os.path.join(heuristic_root, 'src', 'furgoni_con_prova.xlsx')
    scenario = create_scenario_from_excel(excel_file)
    
    print(f"Loaded scenario: {len(scenario.orders)} orders, {len(scenario.vehicles)} vehicles")
    
    # Find Order 6
    order_6 = None
    for order in scenario.orders:
        if str(order.id) == '6':
            order_6 = order
            break
    
    if not order_6:
        print("ERROR: Could not find Order 6")
        return
    
    print(f"\nOrder 6 details:")
    print(f"  ID: {order_6.id}")
    print(f"  Pickups: {len(order_6.get_pickups())}")
    print(f"  Deliveries: {len(order_6.get_deliveries())}")
    
    for i, pickup in enumerate(order_6.get_pickups()):
        print(f"    Pickup {i+1}: lat={getattr(pickup, 'lat', 'missing')}, lon={getattr(pickup, 'lon', 'missing')}")
        print(f"               earliest={getattr(pickup, 'earliest_time', 'missing')}, latest={getattr(pickup, 'latest_time', 'missing')}")
    
    for i, delivery in enumerate(order_6.get_deliveries()):
        print(f"    Delivery {i+1}: lat={getattr(delivery, 'lat', 'missing')}, lon={getattr(delivery, 'lon', 'missing')}")
        print(f"                  earliest={getattr(delivery, 'earliest_time', 'missing')}, latest={getattr(delivery, 'latest_time', 'missing')}")
    
    # Find a suitable vehicle (LOW_TEMP with enough capacity)
    suitable_vehicle = None
    for vehicle in scenario.vehicles:
        if (hasattr(vehicle, 'has_low_temp') and vehicle.has_low_temp and 
            getattr(vehicle, 'pallet_capacity', 0) >= 2):
            suitable_vehicle = vehicle
            break
    
    if not suitable_vehicle:
        print("ERROR: Could not find suitable vehicle")
        return
    
    print(f"\nUsing vehicle: {suitable_vehicle.id}")
    print(f"  Weight capacity: {getattr(suitable_vehicle, 'weight_capacity', 'missing')}")
    print(f"  Volume capacity: {getattr(suitable_vehicle, 'volume_capacity', 'missing')}")
    print(f"  Pallet capacity: {getattr(suitable_vehicle, 'pallet_capacity', 'missing')}")
    print(f"  Has LOW_TEMP: {getattr(suitable_vehicle, 'has_low_temp', 'missing')}")
    
    # Create an empty route for this vehicle
    empty_route = Route(suitable_vehicle, None)
    
    # Add depot start and return tasks
    from algo.epdt_data_structures import Task, TaskType
    
    depot_start = Task(
        id=f"DEPOT_START_{suitable_vehicle.id}",
        task_type=TaskType.DEPOT_START,
        order_id=None,
        lat=getattr(suitable_vehicle, 'depot_lat', 44.9009),
        lon=getattr(suitable_vehicle, 'depot_lon', 8.2057),
        demand=0.0,
        volume=0.0,
        pallets=0.0,
        earliest_time=None,
        latest_time=None
    )
    
    depot_return = Task(
        id=f"DEPOT_RETURN_{suitable_vehicle.id}",
        task_type=TaskType.DEPOT_RETURN,
        order_id=None,
        lat=getattr(suitable_vehicle, 'depot_lat', 44.9009),
        lon=getattr(suitable_vehicle, 'depot_lon', 8.2057),
        demand=0.0,
        volume=0.0,
        pallets=0.0,
        earliest_time=None,
        latest_time=None
    )
    
    empty_route.tasks = [depot_start, depot_return]
    
    print(f"\nCreated empty route with {len(empty_route.tasks)} tasks")
    
    # Try to insert Order 6 using L2 heuristic
    print(f"\n=== ATTEMPTING L2 HEURISTIC INSERTION ===")
    result_route = l2_heuristic(empty_route, order_6, debug_assignment=True, enhanced_diagnostics=True)
    
    if result_route:
        print(f"\nSUCCESS: Order 6 was assigned!")
        print(f"Final route has {len(result_route.tasks)} tasks")
    else:
        print(f"\nFAILURE: Order 6 was not assigned")
        
        print(f"\n=== MANUAL FEASIBILITY CHECK ===")
        # Try manual feasibility check on the empty route
        from algo.second_level import is_feasible
        
        print(f"Testing feasibility of empty route...")
        empty_feasible = is_feasible(empty_route, debug_feasibility=True, return_reason=False)
        print(f"Empty route feasible: {empty_feasible}")
        
        if empty_feasible:
            print(f"\nEmpty route is feasible, so the issue is with Order 6 insertion...")
        else:
            print(f"\nEmpty route itself is infeasible! This indicates a deeper issue.")

if __name__ == "__main__":
    main()
