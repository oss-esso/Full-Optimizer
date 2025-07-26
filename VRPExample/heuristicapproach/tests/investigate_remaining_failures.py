"""
Remaining Assignment Failures Investigation

After fixing the pallet capacity constraint bug, we still have 28 orders failing.
This script investigates what's causing these remaining failures.
"""

import sys
import os

# Add necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
src_dir = os.path.join(heuristic_root, 'src')
algo_dir = os.path.join(heuristic_root, 'algo')
utils_dir = os.path.join(heuristic_root, 'utils')

sys.path.insert(0, heuristic_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, algo_dir)
sys.path.insert(0, utils_dir)

from scenario_creator import create_scenario_from_excel
from second_level import l2_heuristic, is_feasible
from epdt_data_structures import Route

print("=== REMAINING ASSIGNMENT FAILURES INVESTIGATION ===")

def investigate_remaining_failures():
    """Test the remaining 28 failing orders with detailed feasibility debugging."""
    excel_file = os.path.join(src_dir, 'furgoni.xlsx')
    
    if not os.path.exists(excel_file):
        print(f"❌ Excel file not found: {excel_file}")
        return
    
    print(f"📂 Loading scenario from: {excel_file}")
    
    try:
        orders, vehicles = create_scenario_from_excel(excel_file)
        print(f"✅ Loaded {len(orders)} orders and {len(vehicles)} vehicles")
        
        # List of orders that were still failing after the pallet fix
        failing_order_ids = [
            "ORDER_DIPHARMA_FRANCIS_SRL_39",
            "ORDER_GEODIS_18", 
            "ORDER_COLUMBIA__SERRAVALLE_0",
            "ORDER_AGRIACRO_TECH_SRL_19",
            "ORDER_FIAT_V._I_SPA_STAB_B_21"
        ]
        
        print(f"\n🔍 INVESTIGATING 5 SAMPLE FAILING ORDERS:")
        
        for order_id in failing_order_ids:
            order = None
            for o in orders:
                if o.id == order_id:
                    order = o
                    break
            
            if not order:
                print(f"\n❌ Order {order_id} not found")
                continue
            
            print(f"\n{'='*60}")
            print(f"🎯 TESTING ORDER: {order.id}")
            print(f"   • Pickup tasks: {len(order.pickup_tasks)}")
            print(f"   • Delivery tasks: {len(order.delivery_tasks)}")
            
            # Calculate total order requirements
            total_weight = 0.0
            total_volume = 0.0
            total_pallets = 0
            
            all_tasks = order.pickup_tasks + order.delivery_tasks
            for task in all_tasks:
                total_weight += task.demand
                total_volume += task.volume
                total_pallets += task.pallets
            
            print(f"   • Total weight: {total_weight:.1f} kg")
            print(f"   • Total volume: {total_volume:.2f} m³")
            print(f"   • Total pallets: {total_pallets}")
            
            # Test with a few different vehicles
            test_vehicles = []
            for vehicle in vehicles[:10]:  # Test first 10 vehicles
                # Basic capacity check
                can_handle_volume = vehicle.volume_capacity >= abs(total_volume)
                can_handle_pallets = vehicle.pallet_capacity >= abs(total_pallets) if vehicle.pallet_capacity else True
                can_handle_weight = vehicle.weight_capacity >= abs(total_weight)  # Weight is soft but check anyway
                
                if can_handle_volume and can_handle_pallets:
                    test_vehicles.append(vehicle)
            
            print(f"   • Vehicles with sufficient capacity: {len(test_vehicles)}")
            
            if not test_vehicles:
                print(f"   ❌ NO VEHICLES WITH SUFFICIENT CAPACITY!")
                continue
            
            # Test L2 heuristic with compatible vehicles
            success_count = 0
            
            for i, vehicle in enumerate(test_vehicles[:3]):  # Test first 3 compatible vehicles
                print(f"\n   🧪 TESTING with Vehicle {vehicle.id}:")
                print(f"      Volume: {vehicle.volume_capacity:.2f} m³ (req: {abs(total_volume):.2f})")
                print(f"      Pallets: {vehicle.pallet_capacity} (req: {abs(total_pallets)})")
                print(f"      Weight: {vehicle.weight_capacity:.1f} kg (req: {abs(total_weight):.1f})")
                
                # Create empty route
                empty_route = Route(vehicle=vehicle, tasks=[])
                
                # Test L2 heuristic with debug
                print(f"      🔍 DETAILED L2 FEASIBILITY TEST:")
                result = l2_heuristic(empty_route, order, debug_assignment=True)
                
                if result:
                    print(f"      ✅ L2 SUCCESS!")
                    success_count += 1
                    
                    # Test final route feasibility
                    print(f"      🔍 FINAL ROUTE FEASIBILITY TEST:")
                    feasible = is_feasible(result, debug_feasibility=True)
                    print(f"      Final feasibility: {'✅' if feasible else '❌'}")
                    break
                else:
                    print(f"      ❌ L2 FAILED!")
            
            if success_count == 0:
                print(f"   💥 ALL VEHICLES FAILED - DEEPER INVESTIGATION NEEDED")
            else:
                print(f"   ✅ SUCCESS with {success_count}/{len(test_vehicles)} tested vehicles")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    investigate_remaining_failures()
