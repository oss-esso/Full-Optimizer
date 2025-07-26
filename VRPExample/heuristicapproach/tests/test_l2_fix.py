"""
Direct L2 Heuristic Test

Test the fixed L2 heuristic directly with a problematic delivery-only order.
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
from second_level import l2_heuristic
from epdt_data_structures import Route

print("=== DIRECT L2 HEURISTIC TEST ===")

def test_l2_fix():
    excel_file = os.path.join(src_dir, 'furgoni.xlsx')
    
    if not os.path.exists(excel_file):
        print(f"❌ Excel file not found: {excel_file}")
        return
    
    print(f"📂 Loading scenario from: {excel_file}")
    
    try:
        orders, vehicles = create_scenario_from_excel(excel_file)
        print(f"✅ Loaded {len(orders)} orders and {len(vehicles)} vehicles")
        
        # Find a problematic delivery-only order
        target_order = None
        for order in orders:
            if order.id == "ORDER_COLUMBIA__SERRAVALLE_0":
                target_order = order
                break
        
        if not target_order:
            print("❌ Target order not found")
            return
        
        print(f"\n🎯 Testing order: {target_order.id}")
        print(f"   Pickups: {len(target_order.pickup_tasks)}")
        print(f"   Deliveries: {len(target_order.delivery_tasks)}")
        
        # Find a compatible vehicle
        test_vehicle = None
        for vehicle in vehicles:
            if vehicle.id == "FX192HX":
                test_vehicle = vehicle
                break
        
        if not test_vehicle:
            print("❌ Test vehicle not found")
            return
        
        print(f"\n🚛 Testing vehicle: {test_vehicle.id}")
        print(f"   Volume capacity: {test_vehicle.volume_capacity:.2f} m³")
        print(f"   Pallet capacity: {test_vehicle.pallet_capacity}")
        print(f"   Weight capacity: {test_vehicle.weight_capacity:.1f} kg")
        
        # Create empty route
        empty_route = Route(vehicle=test_vehicle, tasks=[])
        
        print(f"\n🧪 CALLING L2 HEURISTIC (with debug)...")
        print("-" * 50)
        
        # Call L2 heuristic with debug enabled
        result = l2_heuristic(empty_route, target_order, debug_assignment=True)
        
        print("-" * 50)
        
        if result:
            print(f"✅ L2 SUCCESS!")
            print(f"   Result route has {len(result.tasks)} tasks")
            print(f"   Route feasible: {result.is_feasible()}")
        else:
            print(f"❌ L2 FAILED!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_l2_fix()
