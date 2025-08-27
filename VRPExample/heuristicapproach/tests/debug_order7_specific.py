#!/usr/bin/env python3
"""
Specific debugging test for Order 7 assignment issues.

This focused test investigates why Order 7 fails to assign to seemingly compatible vehicles
FF235DM and XA819VA during the destroy-and-repair phase.
"""

import os
import sys

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
src_dir = os.path.join(heuristic_root, 'src')
algo_dir = os.path.join(heuristic_root, 'algo')

sys.path.insert(0, heuristic_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, algo_dir)

from scenario_creator import create_scenario_from_excel
from algo.second_level import l2_heuristic
from algo.route import Route
from typing import List

def test_order7_assignment():
    """Test Order 7 assignment to specific vehicles."""
    print("="*60)
    print("ORDER 7 SPECIFIC ASSIGNMENT TEST")
    print("="*60)
    
    # Load scenario
    excel_path = os.path.join(heuristic_root, 'src', 'furgoni3_2.xlsx')
    orders, vehicles, drivers = create_scenario_from_excel(excel_path)
    
    # Find Order 7
    order_7 = None
    for order in orders:
        if str(order.id) == "7":
            order_7 = order
            break
    
    if not order_7:
        print("ERROR: Order 7 not found!")
        return
    
    print(f"Found Order 7:")
    print(f"  Requirements: {order_7.get_total_demand():.0f}kg, {order_7.get_total_volume():.1f}m³, {order_7.get_total_pallets()}pal")
    print(f"  Tasks: {len(order_7.get_all_tasks())}")
    
    # Find target vehicles
    ff235dm = None
    xa819va = None
    xa356kw = None
    
    for vehicle in vehicles:
        if vehicle.id == "FF235DM":
            ff235dm = vehicle
        elif vehicle.id == "XA819VA":
            xa819va = vehicle
        elif vehicle.id == "XA356KW":
            xa356kw = vehicle
    
    target_vehicles = [
        ("FF235DM", ff235dm),
        ("XA819VA", xa819va),
        ("XA356KW", xa356kw)
    ]
    
    print(f"\nTesting Order 7 assignment to target vehicles:")
    print("-"*60)
    
    for name, vehicle in target_vehicles:
        if not vehicle:
            print(f"{name}: NOT FOUND")
            continue
            
        print(f"\n{name}: Testing assignment...")
        print(f"  Capacity: {vehicle.weight_capacity:.0f}kg, {vehicle.volume_capacity:.1f}m³, {vehicle.pallet_capacity}pal")
        print(f"  Capabilities: loader={getattr(vehicle, 'has_loader', False)}, "
              f"low_temp={getattr(vehicle, 'has_low_temp', False)}, "
              f"hangers={getattr(vehicle, 'has_hangers', False)}")
        
        # Create empty route for this vehicle
        empty_route = Route(vehicle)
        
        # Attempt L2 heuristic insertion
        print(f"  Attempting L2 insertion...")
        try:
            result_route = l2_heuristic(
                empty_route, 
                order_7, 
                debug_assignment=True,
                enhanced_diagnostics=True,
                show_diagnostics=True
            )
            
            if result_route:
                print(f"  ✅ SUCCESS: Order 7 assigned to {name}")
                print(f"     Route has {len(result_route.tasks)} tasks")
            else:
                print(f"  ❌ FAILED: Order 7 could not assign to {name}")
                
        except Exception as e:
            print(f"  ❌ ERROR: Exception during assignment: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"\n" + "="*60)
    print("ORDER 7 ASSIGNMENT TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_order7_assignment()
