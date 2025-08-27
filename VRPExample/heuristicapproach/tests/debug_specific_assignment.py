#!/usr/bin/env python3
"""
Focused debug script to understand why Order 38 fails to assign to FF235DM (44,000kg idle vehicle)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.create_scenario_from_excel import create_scenario_from_excel

def debug_specific_assignment():
    """Debug why a specific order fails to assign to a specific vehicle"""
    
    # Load the scenario
    print("🔍 Loading scenario...")
    excel_path = "../src/furgoni3_2.xlsx"
    orders, vehicles, drivers = create_scenario_from_excel(excel_path)
    
    # Find Order 38 and Vehicle FF235DM
    order_38 = None
    vehicle_ff235dm = None
    
    for order in orders:
        if str(order.id) == "38":
            order_38 = order
            break
    
    for vehicle in vehicles:
        if vehicle.id == "FF235DM":
            vehicle_ff235dm = vehicle
            break
    
    if not order_38:
        print("❌ Could not find Order 38")
        return
    
    if not vehicle_ff235dm:
        print("❌ Could not find Vehicle FF235DM")
        return
    
    print(f"✅ Found Order 38: {order_38.get_total_demand():.0f}kg, {order_38.get_total_volume():.1f}m³, {order_38.get_total_pallets()} pallets")
    print(f"✅ Found Vehicle FF235DM: {vehicle_ff235dm.weight_capacity:.0f}kg, {vehicle_ff235dm.volume_capacity:.1f}m³, {getattr(vehicle_ff235dm, 'pallet_capacity', 'unknown')} pallets")
    
    # Create empty route for the vehicle
    from algo.first_level import _create_base_route
    base_route = _create_base_route(vehicle_ff235dm)
    
    print(f"✅ Created base route with {len(base_route.tasks)} tasks")
    
    # Try L2 heuristic with full debugging
    print("\n🔬 ATTEMPTING L2 HEURISTIC WITH FULL DEBUGGING:")
    print("=" * 60)
    
    from algo.second_level import l2_heuristic
    result_route = l2_heuristic(
        route=base_route, 
        order=order_38, 
        debug_assignment=True, 
        enhanced_diagnostics=True,
        sequencing_strategy='clustered'
    )
    
    print("=" * 60)
    if result_route:
        print(f"✅ SUCCESS: L2 heuristic returned a valid route")
        print(f"   Final route has {len(result_route.tasks)} tasks")
        print(f"   Route is feasible: {result_route.is_feasible()}")
    else:
        print(f"❌ FAILURE: L2 heuristic returned None")
        print(f"   This explains why Order 38 cannot be assigned to FF235DM")
        
        # Try to understand the root cause
        print(f"\n🔍 ROOT CAUSE ANALYSIS:")
        print(f"   Order 38 demand: {order_38.get_total_demand():.0f}kg")
        print(f"   Vehicle FF235DM capacity: {vehicle_ff235dm.weight_capacity:.0f}kg")
        print(f"   Capacity utilization would be: {(order_38.get_total_demand()/vehicle_ff235dm.weight_capacity)*100:.1f}%")
        print(f"   This should be EASILY feasible!")

if __name__ == "__main__":
    debug_specific_assignment()
