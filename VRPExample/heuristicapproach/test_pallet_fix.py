#!/usr/bin/env python3
"""
Quick test to verify the pallet constraint fix
"""

import sys
import os

# Add paths
sys.path.append('.')
sys.path.append('./algo')
sys.path.append('./src')

print("🔧 Testing Pallet Constraint Fix for ORDER_CURTI_SRL_10")
print("=" * 60)

try:
    from utils.scenario_creator import create_scenario_from_excel
    
    # Test loading scenario
    excel_file = 'src/furgoni.xlsx'
    if os.path.exists(excel_file):
        print("📂 Loading scenario from Excel...")
        orders, vehicles, drivers = create_scenario_from_excel(excel_file)
        print(f"✅ Loaded {len(orders)} orders and {len(vehicles)} vehicles")
        
        # Find ORDER_CURTI_SRL_10 or similar
        test_order = None
        for order in orders:
            if 'CURTI' in order.id:
                test_order = order
                break
        
        if test_order:
            print(f"\n🎯 Analyzing order: {test_order.id}")
            all_tasks = test_order.get_all_tasks()
            total_pallets = sum(task.pallets for task in all_tasks)
            
            pickup_tasks = [t for t in all_tasks if t.is_pickup()]
            delivery_tasks = [t for t in all_tasks if t.is_delivery()]
            
            print(f"   📦 Pickup tasks: {len(pickup_tasks)}")
            print(f"   🏪 Delivery tasks: {len(delivery_tasks)}")
            print(f"   📊 Total pallet demand: {total_pallets} (should be 0 for depot bay pairs)")
            
            print(f"\n📋 Task Details:")
            for task in all_tasks:
                sign = "+" if task.pallets >= 0 else ""
                print(f"   • {task.id}: {task.task_type.value}, pallets={sign}{task.pallets}")
            
            # Check peak pallet requirement
            max_pallets_needed = max(abs(task.pallets) for task in all_tasks)
            print(f"\n🚛 Max pallets needed at any point: {max_pallets_needed}")
            
            # Check vehicle capacities
            large_vehicles = [v for v in vehicles if v.pallet_capacity >= max_pallets_needed]
            print(f"   Compatible vehicles: {len(large_vehicles)}/{len(vehicles)}")
            
            if large_vehicles:
                example_vehicle = large_vehicles[0]
                print(f"   Example: {example_vehicle.id} has {example_vehicle.pallet_capacity} pallet capacity")
                
                # Test simulation
                print(f"\n🧪 Simulating pallet load for vehicle {example_vehicle.id}:")
                load_pallets = 0
                
                for i, task in enumerate(all_tasks):
                    if task.is_pickup():
                        load_pallets += task.pallets  # Add pallets (positive)
                        print(f"   Step {i+1}: After pickup {task.id}: load = {load_pallets} pallets")
                    elif task.is_delivery():
                        load_pallets += task.pallets  # Add pallets (negative for delivery)
                        print(f"   Step {i+1}: After delivery {task.id}: load = {load_pallets} pallets")
                    
                    # Check constraint
                    exceeds = load_pallets > example_vehicle.pallet_capacity
                    status = "❌ EXCEEDS" if exceeds else "✅ OK"
                    print(f"              Constraint check: {load_pallets} ≤ {example_vehicle.pallet_capacity} → {status}")
                    
                    if exceeds:
                        print(f"   ⚠️  VIOLATION: Route would exceed pallet capacity!")
                        break
                
                print(f"\n📊 Final Result:")
                if load_pallets == 0:
                    print(f"   ✅ Perfect: Final load is 0 pallets (pickup/delivery pair balanced)")
                else:
                    print(f"   ⚠️  Warning: Final load is {load_pallets} pallets (should be 0)")
            else:
                print("   ❌ No compatible vehicles found!")
                
        else:
            print("❌ ORDER_CURTI_SRL_10 not found in scenario")
    else:
        print(f"❌ Excel file not found: {excel_file}")
        
except Exception as e:
    print(f"❌ Error during test: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🏁 Test completed")
