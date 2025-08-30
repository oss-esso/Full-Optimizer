#!/usr/bin/env python3
"""
Focused debugging test for Orders 5 and 6 to identify why these simple orders are failing.
"""

import os
import sys
import time
from pathlib import Path

# Add necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
src_dir = os.path.join(heuristic_root, 'src')
algo_dir = os.path.join(heuristic_root, 'algo')

sys.path.insert(0, heuristic_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, algo_dir)

from create_scenario_from_excel import create_scenario_from_excel
from first_level import l1_heuristic

def main():
    print("=== DEBUG ORDERS 5 & 6 TEST ===")
    
    # Load the scenario
    excel_path = os.path.join(heuristic_root, 'src', 'furgoni_con_prova.xlsx')
    print(f"Loading scenario from: {excel_path}")
    
    scenario = create_scenario_from_excel(excel_path)
    orders = scenario['orders']
    vehicles = scenario['vehicles']
    
    print(f"Loaded {len(orders)} orders and {len(vehicles)} vehicles")
    
    # Find Orders 5 and 6
    target_orders = []
    for order in orders:
        if str(getattr(order, 'id', 'unknown')) in ['5', '6']:
            target_orders.append(order)
    
    print(f"\nFound {len(target_orders)} target orders")
    
    # Analyze Orders 5 and 6 in detail
    for order in target_orders:
        order_id = str(getattr(order, 'id', 'unknown'))
        print(f"\n=== DETAILED ANALYSIS OF ORDER {order_id} ===")
        
        # Get tasks
        pickup_tasks = getattr(order, 'pickup_tasks', [])
        delivery_tasks = getattr(order, 'delivery_tasks', [])
        
        print(f"  Configuration: {len(pickup_tasks)} pickups + {len(delivery_tasks)} deliveries")
        
        # Analyze each task
        total_weight = 0.0
        total_volume = 0.0 
        total_pallets = 0.0
        
        print(f"  PICKUP TASKS:")
        for i, task in enumerate(pickup_tasks):
            weight = getattr(task, 'demand', 0.0)
            volume = getattr(task, 'volume', 0.0)
            pallets = getattr(task, 'pallets', 0.0)
            service_time = getattr(task, 'service_time', 0.0)
            earliest = getattr(task, 'earliest_time', None)
            latest = getattr(task, 'latest_time', None)
            
            total_weight += weight
            total_volume += volume
            total_pallets += pallets
            
            print(f"    P{i+1} ({getattr(task, 'id', 'no-id')}): {weight:.1f}kg, {volume:.1f}m³, {pallets:.0f}pal, service: {service_time:.1f}min")
            print(f"        Time window: {earliest} -> {latest}")
        
        print(f"  DELIVERY TASKS:")
        for i, task in enumerate(delivery_tasks):
            weight = getattr(task, 'demand', 0.0)
            volume = getattr(task, 'volume', 0.0)
            pallets = getattr(task, 'pallets', 0.0)
            service_time = getattr(task, 'service_time', 0.0)
            earliest = getattr(task, 'earliest_time', None)
            latest = getattr(task, 'latest_time', None)
            
            total_weight += weight
            total_volume += volume
            total_pallets += pallets
            
            print(f"    D{i+1} ({getattr(task, 'id', 'no-id')}): {weight:.1f}kg, {volume:.1f}m³, {pallets:.0f}pal, service: {service_time:.1f}min")
            print(f"        Time window: {earliest} -> {latest}")
        
        print(f"  TOTAL DEMAND: {total_weight:.1f}kg, {total_volume:.1f}m³, {total_pallets:.0f}pal")
        
        # Check if any tasks have zero service time
        zero_service_tasks = []
        for task in pickup_tasks + delivery_tasks:
            if getattr(task, 'service_time', 5.0) == 0.0:
                zero_service_tasks.append(getattr(task, 'id', 'no-id'))
        
        if zero_service_tasks:
            print(f"  WARNING: Tasks with zero service time: {zero_service_tasks}")
        
        # Check vehicle compatibility
        print(f"\n  VEHICLE COMPATIBILITY CHECK:")
        compatible_vehicles = 0
        for vehicle in vehicles:
            weight_cap = getattr(vehicle, 'weight_capacity', float('inf'))
            volume_cap = getattr(vehicle, 'volume_capacity', float('inf'))
            pallet_cap = getattr(vehicle, 'pallet_capacity', float('inf'))
            
            weight_ok = abs(total_weight) <= weight_cap
            volume_ok = abs(total_volume) <= volume_cap
            pallets_ok = abs(total_pallets) <= pallet_cap
            
            if weight_ok and volume_ok and pallets_ok:
                compatible_vehicles += 1
            else:
                print(f"    {vehicle.id}: INCOMPATIBLE - Weight:{weight_ok}, Volume:{volume_ok}, Pallets:{pallets_ok}")
                print(f"      Vehicle: {weight_cap:.0f}kg, {volume_cap:.1f}m³, {pallet_cap:.0f}pal")
                print(f"      Required: {abs(total_weight):.1f}kg, {abs(total_volume):.1f}m³, {abs(total_pallets):.0f}pal")
        
        print(f"  Compatible vehicles: {compatible_vehicles}/{len(vehicles)}")
        
        # Check time window conflicts
        print(f"\n  TIME WINDOW ANALYSIS:")
        all_tasks = pickup_tasks + delivery_tasks
        earliest_pickup = None
        latest_delivery = None
        
        for task in pickup_tasks:
            earliest = getattr(task, 'earliest_time', None)
            if earliest is not None:
                if earliest_pickup is None or earliest < earliest_pickup:
                    earliest_pickup = earliest
        
        for task in delivery_tasks:
            latest = getattr(task, 'latest_time', None)
            if latest is not None:
                if latest_delivery is None or latest > latest_delivery:
                    latest_delivery = latest
        
        if earliest_pickup is not None and latest_delivery is not None:
            time_span = latest_delivery - earliest_pickup
            print(f"    Time span: {earliest_pickup} -> {latest_delivery} ({time_span:.0f} minutes)")
            
            if time_span < 60:
                print(f"    WARNING: Very tight time window ({time_span:.0f} minutes)")
        
    # Now try to run L1 heuristic with detailed debugging for these orders
    print(f"\n=== TESTING L1 HEURISTIC WITH DETAILED DEBUGGING ===")
    
    # Configure parameters
    params = {
        'M1': 2000,
        'M2': 500,
        'enhanced_diagnostics': True
    }
    
    print(f"Running L1 heuristic with detailed debugging enabled...")
    start_time = time.time()
    
    try:
        solution = l1_heuristic(orders, vehicles, params)
        runtime = time.time() - start_time
        
        print(f"L1 heuristic completed in {runtime:.2f} seconds")
        
        # Check results for target orders
        print(f"\n=== FINAL ASSIGNMENT RESULTS ===")
        
        for order_id in ['5', '6']:
            assigned = False
            assigned_vehicle = None
            
            for vehicle_id, route in solution.routes.items():
                if route and hasattr(route, 'tasks'):
                    for task in route.tasks:
                        if hasattr(task, 'order_id') and str(task.order_id) == order_id:
                            assigned = True
                            assigned_vehicle = vehicle_id
                            break
                    if assigned:
                        break
            
            if assigned:
                print(f"  Order {order_id}: ASSIGNED to {assigned_vehicle}")
            else:
                print(f"  Order {order_id}: UNASSIGNED")
    
    except Exception as e:
        print(f"L1 heuristic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
