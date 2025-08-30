#!/usr/bin/env python3
"""
Debug script to understand why Order 7 cannot be assigned to GA637VG.

This script will:
1. Load the scenario
2. Find Order 7 and GA637VG 
3. Test the specific assignment
4. Report detailed feasibility check results
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    print("=== ORDER 7 FEASIBILITY DEBUG ===")
    
    try:
        # Import required modules
        from create_scenario_from_excel import create_scenario_from_excel
        from algo.second_level import find_best_sequence_for_complex_order
        from algo.constraints import check_vehicle_capabilities_match_order
        
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
            
        print(f"\n=== ORDER 7 DETAILS ===")
        print(f"Order ID: {order_7.id}")
        print(f"Tasks: {len(order_7.tasks)}")
        
        # Check if order has capability requirements
        if hasattr(order_7, 'required_capabilities'):
            print(f"Required capabilities: {order_7.required_capabilities}")
        else:
            print("No required_capabilities attribute found")
            
        # Check individual tasks for capabilities
        for i, task in enumerate(order_7.tasks):
            print(f"  Task {i+1}: {task.id}")
            if hasattr(task, 'required_capabilities'):
                print(f"    Required capabilities: {task.required_capabilities}")
            else:
                print(f"    No required_capabilities attribute")
            if hasattr(task, 'capabilities'):
                print(f"    Capabilities: {task.capabilities}")
            else:
                print(f"    No capabilities attribute")
        
        # Find GA637VG vehicle
        ga637vg = None
        for vehicle in scenario.vehicles:
            if hasattr(vehicle, 'id') and vehicle.id == 'GA637VG':
                ga637vg = vehicle
                break
                
        if not ga637vg:
            print("ERROR: GA637VG vehicle not found!")
            return
            
        print(f"\n=== GA637VG DETAILS ===")
        print(f"Vehicle ID: {ga637vg.id}")
        print(f"Capacity: {ga637vg.max_weight}kg, {ga637vg.max_volume}m3, {ga637vg.max_pallets}pal")
        
        if hasattr(ga637vg, 'capabilities'):
            print(f"Capabilities: {ga637vg.capabilities}")
        else:
            print("No capabilities attribute found")
            
        # Test capability matching
        print(f"\n=== CAPABILITY CHECK ===")
        try:
            capability_match = check_vehicle_capabilities_match_order(ga637vg, order_7)
            print(f"Capability match result: {capability_match}")
        except Exception as e:
            print(f"Capability check failed: {e}")
            
        # Test sequence generation
        print(f"\n=== SEQUENCE GENERATION TEST ===")
        try:
            result = find_best_sequence_for_complex_order(order_7, ga637vg)
            if result:
                print(f"Sequence generation SUCCESS!")
                print(f"Best sequence: {len(result.get('sequence', []))} tasks")
                print(f"Peak pallets: {result.get('peak_pallets', 'unknown')}")
            else:
                print(f"Sequence generation FAILED - returned None")
        except Exception as e:
            print(f"Sequence generation ERROR: {e}")
            import traceback
            traceback.print_exc()
            
        # Test direct capacity check
        print(f"\n=== DIRECT CAPACITY CHECK ===")
        order_weight = sum(abs(task.demand) for task in order_7.tasks if hasattr(task, 'demand'))
        order_volume = sum(abs(task.volume) for task in order_7.tasks if hasattr(task, 'volume'))
        order_pallets = sum(abs(task.pallets) for task in order_7.tasks if hasattr(task, 'pallets'))
        
        print(f"Order 7 total: {order_weight}kg, {order_volume}m3, {order_pallets}pal")
        print(f"GA637VG capacity: {ga637vg.max_weight}kg, {ga637vg.max_volume}m3, {ga637vg.max_pallets}pal")
        
        weight_ok = order_weight <= ga637vg.max_weight
        volume_ok = order_volume <= ga637vg.max_volume  
        pallets_ok = order_pallets <= ga637vg.max_pallets
        
        print(f"Weight check: {weight_ok} ({order_weight} <= {ga637vg.max_weight})")
        print(f"Volume check: {volume_ok} ({order_volume} <= {ga637vg.max_volume})")
        print(f"Pallets check: {pallets_ok} ({order_pallets} <= {ga637vg.max_pallets})")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
