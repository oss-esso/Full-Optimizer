#!/usr/bin/env python3
"""
Simple Order 7 debug test to see why it's being rejected.
"""

import sys
import os

# Add the src directory to the path  
src_path = os.path.join(os.path.dirname(__file__), 'src')
algo_path = os.path.join(os.path.dirname(__file__), 'algo')
utils_path = os.path.join(os.path.dirname(__file__), 'utils')

sys.path.insert(0, src_path)
sys.path.insert(0, algo_path) 
sys.path.insert(0, utils_path)

def test_order_7_assignment():
    print("=== ORDER 7 ASSIGNMENT DEBUG TEST ===")
    
    try:
        from create_scenario_from_excel import create_scenario_from_excel
        from l1_heuristic import l1_heuristic
        
        # Load scenario
        excel_file = os.path.join(os.path.dirname(__file__), '..', 'src', 'furgoni_con_prova.xlsx')
        print(f"Loading scenario from: {excel_file}")
        
        scenario = create_scenario_from_excel(excel_file)
        print(f"Loaded: {len(scenario.orders)} orders, {len(scenario.vehicles)} vehicles")
        
        # Find Order 7
        order_7 = None
        for order in scenario.orders:
            if hasattr(order, 'id') and str(order.id) == '7':
                order_7 = order
                break
        
        if not order_7:
            print("ERROR: Order 7 not found!")
            return
            
        print(f"\nOrder 7 found with {len(order_7.tasks)} tasks")
        
        # Find compatible vehicles (those with HANGERS)
        hangers_vehicles = []
        for vehicle in scenario.vehicles:
            vehicle_caps = []
            if hasattr(vehicle, 'capabilities') and vehicle.capabilities:
                for cap in vehicle.capabilities:
                    cap_name = cap.name if hasattr(cap, 'name') else str(cap)
                    vehicle_caps.append(cap_name)
                    
            if 'HANGERS' in vehicle_caps:
                hangers_vehicles.append(vehicle)
                
        print(f"Found {len(hangers_vehicles)} vehicles with HANGERS capability")
        for vehicle in hangers_vehicles[:5]:  # Show first 5
            print(f"  - {vehicle.id}: {getattr(vehicle, 'weight_capacity', 'N/A')}kg, {getattr(vehicle, 'volume_capacity', 'N/A')}m³, {getattr(vehicle, 'pallet_capacity', 'N/A')}pal")
        
        # Try to assign Order 7 to GA637VG specifically
        ga637vg = None
        for vehicle in scenario.vehicles:
            if hasattr(vehicle, 'id') and vehicle.id == 'GA637VG':
                ga637vg = vehicle
                break
                
        if ga637vg:
            print(f"\nTesting Order 7 assignment to GA637VG...")
            print(f"GA637VG capacity: {getattr(ga637vg, 'weight_capacity', 'N/A')}kg, {getattr(ga637vg, 'volume_capacity', 'N/A')}m³, {getattr(ga637vg, 'pallet_capacity', 'N/A')}pal")
            
            # Create a basic route with just depot tasks for GA637VG
            from epdt_data_structures import Route
            from epdt_data_structures import Task, TaskType
            
            # Create depot start and end tasks
            depot_start = Task(
                id="DEPOT_START",
                location_id="DEPOT",
                task_type=TaskType.PICKUP,
                order_id="DEPOT",
                lat=45.4642,
                lon=9.1900,
                service_time=0,
                earliest_time=0,
                latest_time=1440,
                demand=0,
                volume=0,
                pallets=0
            )
            
            depot_end = Task(
                id="DEPOT_END", 
                location_id="DEPOT",
                task_type=TaskType.DELIVERY,
                order_id="DEPOT",
                lat=45.4642,
                lon=9.1900,
                service_time=0,
                earliest_time=0,
                latest_time=1440,
                demand=0,
                volume=0,
                pallets=0
            )
            
            # Create route
            route = Route(vehicle=ga637vg, tasks=[depot_start, depot_end])
            
            # Try L2 assignment
            from second_level import find_best_sequence_for_complex_order
            
            print(f"\nCalling find_best_sequence_for_complex_order with debug=True...")
            result = find_best_sequence_for_complex_order(route, order_7, debug=True)
            
            if result:
                print(f"\n*** SUCCESS *** Order 7 can be assigned to GA637VG!")
            else:
                print(f"\n*** FAILURE *** Order 7 cannot be assigned to GA637VG")
        else:
            print("ERROR: GA637VG vehicle not found!")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_order_7_assignment()
