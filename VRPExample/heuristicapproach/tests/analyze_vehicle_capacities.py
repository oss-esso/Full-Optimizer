#!/usr/bin/env python3
"""
Quick script to analyze vehicle capacities from the Excel file
"""

import sys
import os

# Add the parent directory to Python path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils.scenario_creator import create_scenario_from_excel

def main():
    excel_file = os.path.join(parent_dir, "src", "furgoni.xlsx")
    
    print("=== VEHICLE CAPACITY ANALYSIS ===")
    print(f"Loading from: {excel_file}")
    
    try:
        orders, vehicles = create_scenario_from_excel(excel_file)
        print(f"✅ Loaded {len(orders)} orders and {len(vehicles)} vehicles")
        
        # Group vehicles by pallet capacity
        pallet_groups = {}
        weight_groups = {}
        volume_groups = {}
        
        for vehicle in vehicles:
            pallet_cap = vehicle.pallet_capacity
            weight_cap = vehicle.weight_capacity
            volume_cap = vehicle.volume_capacity
            
            if pallet_cap not in pallet_groups:
                pallet_groups[pallet_cap] = []
            pallet_groups[pallet_cap].append(vehicle.id)
            
            # Round weight and volume for grouping
            weight_key = round(weight_cap, 0)
            if weight_key not in weight_groups:
                weight_groups[weight_key] = []
            weight_groups[weight_key].append(vehicle.id)
            
            volume_key = round(volume_cap, 1)
            if volume_key not in volume_groups:
                volume_groups[volume_key] = []
            volume_groups[volume_key].append(vehicle.id)
        
        print(f"\n🚛 PALLET CAPACITY DISTRIBUTION:")
        for pallet_cap in sorted(pallet_groups.keys()):
            vehicles_list = pallet_groups[pallet_cap]
            print(f"   • {pallet_cap} pallets: {len(vehicles_list)} vehicles")
            if len(vehicles_list) <= 5:
                print(f"     - Vehicles: {', '.join(vehicles_list)}")
            else:
                print(f"     - Sample vehicles: {', '.join(vehicles_list[:3])}...")
        
        print(f"\n⚖️  WEIGHT CAPACITY DISTRIBUTION:")
        for weight_cap in sorted(weight_groups.keys()):
            vehicles_list = weight_groups[weight_cap]
            print(f"   • {weight_cap} kg: {len(vehicles_list)} vehicles")
            if len(vehicles_list) <= 5:
                print(f"     - Vehicles: {', '.join(vehicles_list)}")
        
        print(f"\n📦 VOLUME CAPACITY DISTRIBUTION:")
        for volume_cap in sorted(volume_groups.keys()):
            vehicles_list = volume_groups[volume_cap]
            print(f"   • {volume_cap} m³: {len(vehicles_list)} vehicles")
            if len(vehicles_list) <= 5:
                print(f"     - Vehicles: {', '.join(vehicles_list)}")
        
        # Find vehicles that can handle 33+ pallets
        high_pallet_vehicles = [v for v in vehicles if v.pallet_capacity >= 33]
        
        print(f"\n🎯 VEHICLES THAT CAN HANDLE 33+ PALLETS:")
        if high_pallet_vehicles:
            for vehicle in high_pallet_vehicles:
                print(f"   • {vehicle.id}: {vehicle.pallet_capacity} pallets, {vehicle.weight_capacity} kg, {vehicle.volume_capacity:.1f} m³")
        else:
            print("   ❌ No vehicles found with 33+ pallet capacity")
            max_pallets = max(v.pallet_capacity for v in vehicles)
            print(f"   📊 Maximum pallet capacity available: {max_pallets} pallets")
        
        # Check for large orders that might need special handling
        print(f"\n📋 ANALYZING PROBLEMATIC ORDERS:")
        large_orders = []
        for order in orders:
            total_pallets = 0
            total_weight = 0.0
            total_volume = 0.0
            
            for task in order.get_all_tasks():
                total_pallets += abs(task.pallets)
                total_weight += abs(task.demand)
                total_volume += abs(task.volume)
            
            # Calculate actual requirements (divide by 2 for depot bay pairs)
            actual_pallets = total_pallets / 2
            actual_weight = total_weight / 2
            actual_volume = total_volume / 2
            
            if actual_pallets > 10 or actual_weight > 10000:  # Large orders
                large_orders.append((order.id, actual_pallets, actual_weight, actual_volume))
        
        large_orders.sort(key=lambda x: x[1], reverse=True)  # Sort by pallets
        
        print(f"   Found {len(large_orders)} large orders:")
        for order_id, pallets, weight, volume in large_orders[:10]:
            print(f"   • {order_id}: {pallets:.0f} pallets, {weight:.0f} kg, {volume:.1f} m³")
            
            # Check if any vehicle can handle this order
            compatible_vehicles = [
                v for v in vehicles 
                if v.pallet_capacity >= pallets and 
                   v.weight_capacity >= weight and 
                   v.volume_capacity >= volume
            ]
            
            if compatible_vehicles:
                print(f"     ✅ Can be handled by {len(compatible_vehicles)} vehicles (e.g., {compatible_vehicles[0].id})")
            else:
                print(f"     ❌ Cannot be handled by any vehicle!")
                # Find the limiting constraint
                pallet_ok = len([v for v in vehicles if v.pallet_capacity >= pallets])
                weight_ok = len([v for v in vehicles if v.weight_capacity >= weight])
                volume_ok = len([v for v in vehicles if v.volume_capacity >= volume])
                
                print(f"        - Pallet constraint: {pallet_ok} vehicles can handle {pallets:.0f} pallets")
                print(f"        - Weight constraint: {weight_ok} vehicles can handle {weight:.0f} kg")
                print(f"        - Volume constraint: {volume_ok} vehicles can handle {volume:.1f} m³")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
