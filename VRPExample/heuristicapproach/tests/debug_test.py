#!/usr/bin/env python3
"""
Minimal debug test to identify the issue
"""

import sys
import os

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
src_dir = os.path.join(heuristic_root, 'src')
algo_dir = os.path.join(heuristic_root, 'algo')
utils_dir = os.path.join(heuristic_root, 'utils')

sys.path.insert(0, heuristic_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, algo_dir)
sys.path.insert(0, utils_dir)

print("=== DEBUG TEST ===")
print(f"Current dir: {current_dir}")
print(f"Heuristic root: {heuristic_root}")

try:
    print("Testing scenario_creator import...")
    from scenario_creator import create_scenario_from_excel
    print("✅ scenario_creator imported successfully")
    
    print("Testing scenario loading...")
    excel_file = os.path.join(src_dir, "furgoni.xlsx")
    print(f"Loading from: {excel_file}")
    
    if os.path.exists(excel_file):
        print("✅ Excel file exists")
        orders, vehicles = create_scenario_from_excel(excel_file)
        print(f"✅ Loaded {len(orders)} orders, {len(vehicles)} vehicles")
        
        # Check the problematic orders
        for order in orders:
            if order.id in ['ORDER_GFM_IMBOTTITURE_SNC_31', 'ORDER_CURTI_SRL_10']:
                print(f"Found problematic order: {order.id}")
                tasks = order.get_all_tasks()
                total_weight = sum(abs(task.demand) for task in tasks) / 2
                total_volume = sum(abs(task.volume) for task in tasks) / 2
                print(f"  Requirements: {total_weight}kg, {total_volume}m³")
                
        # Check vehicle capacities - first check what attributes exist
        if vehicles:
            sample_vehicle = vehicles[0]
            print(f"Sample vehicle attributes: {dir(sample_vehicle)}")
            print(f"Sample vehicle: {sample_vehicle}")
            
            # Try different possible attribute names
            weight_attr = None
            volume_attr = None
            pallet_attr = None
            
            for attr in ['max_weight', 'max_weight_kg', 'weight_capacity', 'capacity_weight']:
                if hasattr(sample_vehicle, attr):
                    weight_attr = attr
                    break
                    
            for attr in ['max_volume', 'max_volume_m3', 'volume_capacity', 'capacity_volume']:
                if hasattr(sample_vehicle, attr):
                    volume_attr = attr
                    break
                    
            for attr in ['max_pallets', 'max_pallet', 'pallet_capacity', 'capacity_pallets']:
                if hasattr(sample_vehicle, attr):
                    pallet_attr = attr
                    break
            
            print(f"Weight attribute: {weight_attr}")
            print(f"Volume attribute: {volume_attr}")
            print(f"Pallet attribute: {pallet_attr}")
            
            if weight_attr and volume_attr:
                weights = [getattr(v, weight_attr) for v in vehicles]
                volumes = [getattr(v, volume_attr) for v in vehicles]
                print(f"Vehicle capacities: {min(weights)}-{max(weights)}kg, {min(volumes)}-{max(volumes)}m³")
                
                # Check if any vehicle can handle the orders
                suitable_for_order1 = sum(1 for v in vehicles if getattr(v, weight_attr) >= 1000 and getattr(v, volume_attr) >= 21.79)
                suitable_for_order2 = sum(1 for v in vehicles if getattr(v, weight_attr) >= 25100 and getattr(v, volume_attr) >= 31.68)
                print(f"Vehicles that can handle ORDER_GFM_IMBOTTITURE_SNC_31: {suitable_for_order1}/{len(vehicles)}")
                print(f"Vehicles that can handle ORDER_CURTI_SRL_10: {suitable_for_order2}/{len(vehicles)}")
            else:
                print("❌ Could not determine vehicle capacity attributes")
        
    else:
        print("❌ Excel file not found")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("=== DEBUG TEST COMPLETED ===")
