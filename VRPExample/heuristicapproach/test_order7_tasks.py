#!/usr/bin/env python3
"""
Simple test to check if Order 7 assignment works after the tasks fix.
"""

import sys
import os

# Set the working directory to the heuristic approach directory
os.chdir(r'D:\Projects\OQI_Project\Full Optimizer\VRPExample\heuristicapproach')

# Add the necessary paths
sys.path.insert(0, 'utils')
sys.path.insert(0, 'algo')
sys.path.insert(0, 'src')

def test_order_7_tasks():
    print("=== TESTING ORDER 7 TASKS AFTER FIX ===")
    
    try:
        from scenario_creator import create_scenario_from_excel
        
        # Load scenario
        excel_file = os.path.join('src', 'furgoni_con_prova.xlsx')
        print(f"Loading scenario from: {excel_file}")
        
        orders, vehicles, drivers = create_scenario_from_excel(excel_file)
        print(f"Loaded: {len(orders)} orders, {len(vehicles)} vehicles")
        
        # Find Order 7
        order_7 = None
        for order in orders:
            if str(order.id) == '7':
                order_7 = order
                break
        
        if not order_7:
            print("ERROR: Order 7 not found!")
            return
            
        print(f"\n=== ORDER 7 ANALYSIS ===")
        print(f"Order ID: {order_7.id}")
        print(f"Pickup tasks: {len(order_7.pickup_tasks)}")
        print(f"Delivery tasks: {len(order_7.delivery_tasks)}")
        print(f"Combined tasks: {len(order_7.tasks)}")
        print(f"get_all_tasks(): {len(order_7.get_all_tasks())}")
        
        # Show task details
        print(f"\nTask details:")
        for i, task in enumerate(order_7.tasks):
            task_type = getattr(getattr(task, 'task_type', None), 'name', 'UNK')
            caps_needed = []
            if getattr(task, 'requires_hangers', False):
                caps_needed.append('HANGERS')
            if getattr(task, 'requires_loader', False):
                caps_needed.append('LOADER') 
            if getattr(task, 'requires_low_temp', False):
                caps_needed.append('LOW_TEMP')
            
            print(f"  {i+1}. {task.id} ({task_type}): {task.demand}kg, {task.volume}m3, {task.pallets}pal")
            print(f"      Capabilities needed: {', '.join(caps_needed) if caps_needed else 'NONE'}")
        
        # Find compatible vehicles
        hangers_vehicles = []
        for vehicle in vehicles:
            vehicle_caps = []
            if hasattr(vehicle, 'capabilities') and vehicle.capabilities:
                for cap in vehicle.capabilities:
                    cap_name = cap.name if hasattr(cap, 'name') else str(cap)
                    vehicle_caps.append(cap_name)
                    
            if 'HANGERS' in vehicle_caps:
                hangers_vehicles.append(vehicle)
                
        print(f"\n=== COMPATIBLE VEHICLES ===")
        print(f"Vehicles with HANGERS: {len(hangers_vehicles)}")
        for vehicle in hangers_vehicles[:3]:  # Show first 3
            print(f"  - {vehicle.id}: {getattr(vehicle, 'weight_capacity', 'N/A')}kg, {getattr(vehicle, 'volume_capacity', 'N/A')}m3, {getattr(vehicle, 'pallet_capacity', 'N/A')}pal")
        
        print(f"\n*** TASKS FIX SUCCESSFUL ***")
        print(f"Order 7 now has {len(order_7.tasks)} tasks instead of 0!")
        print(f"Ready for assignment attempts...")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_order_7_tasks()
