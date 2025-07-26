"""
Delivery-Only Order Feasibility Investigation

This script investigates why delivery-only orders fail on smaller vehicles.
The hypothesis: delivery-only orders require the vehicle to start with cargo loaded,
but smaller vehicles might have constraints preventing this initial loading state.
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
from second_level import is_feasible
from epdt_data_structures import Route

print("=== DELIVERY-ONLY ORDER FEASIBILITY INVESTIGATION ===")

def debug_delivery_only_feasibility():
    """Debug why delivery-only orders fail feasibility checks on smaller vehicles."""
    excel_file = os.path.join(src_dir, 'furgoni.xlsx')
    
    if not os.path.exists(excel_file):
        print(f"❌ Excel file not found: {excel_file}")
        return
    
    print(f"📂 Loading scenario from: {excel_file}")
    
    try:
        orders, vehicles = create_scenario_from_excel(excel_file)
        print(f"✅ Loaded {len(orders)} orders and {len(vehicles)} vehicles")
        
        # Find a delivery-only order that fails on small vehicles
        test_order = None
        for order in orders:
            if order.id == "ORDER_COLUMBIA__SERRAVALLE_0":
                test_order = order
                break
        
        if not test_order:
            print("❌ Test order not found")
            return
        
        print(f"\n🎯 ANALYZING ORDER: {test_order.id}")
        print(f"   • Pickup tasks: {len(test_order.pickup_tasks)}")
        print(f"   • Delivery tasks: {len(test_order.delivery_tasks)}")
        
        if test_order.delivery_tasks:
            delivery_task = test_order.delivery_tasks[0]
            print(f"   • Delivery task demands:")
            print(f"     - Weight: {delivery_task.demand} kg (negative = removing cargo)")
            print(f"     - Volume: {delivery_task.volume} m³")
            print(f"     - Pallets: {delivery_task.pallets}")
        
        # Test with small vs large vehicle
        small_vehicle = None
        large_vehicle = None
        
        for vehicle in vehicles:
            if vehicle.id == "FX192HX":  # Small vehicle that fails
                small_vehicle = vehicle
            elif vehicle.id == "FX194HX":  # Large vehicle that succeeds
                large_vehicle = vehicle
        
        if not small_vehicle or not large_vehicle:
            print("❌ Test vehicles not found")
            return
        
        print(f"\n🔍 COMPARING SMALL VS LARGE VEHICLE:")
        print(f"   Small Vehicle ({small_vehicle.id}):")
        print(f"     - Volume capacity: {small_vehicle.volume_capacity:.2f} m³")
        print(f"     - Pallet capacity: {small_vehicle.pallet_capacity}")
        print(f"     - Weight capacity: {small_vehicle.weight_capacity:.1f} kg")
        
        print(f"   Large Vehicle ({large_vehicle.id}):")
        print(f"     - Volume capacity: {large_vehicle.volume_capacity:.2f} m³")
        print(f"     - Pallet capacity: {large_vehicle.pallet_capacity}")
        print(f"     - Weight capacity: {large_vehicle.weight_capacity:.1f} kg")
        
        # Test feasibility with both vehicles
        print(f"\n🧪 TESTING DELIVERY-ONLY FEASIBILITY:")
        
        for vehicle_name, vehicle in [("Small", small_vehicle), ("Large", large_vehicle)]:
            print(f"\n   {vehicle_name} Vehicle ({vehicle.id}):")
            
            # Create route with just the delivery task
            route = Route(vehicle=vehicle, tasks=[])
            route.tasks = test_order.delivery_tasks.copy()
            
            print(f"   📊 Route Analysis:")
            print(f"     - Tasks in route: {len(route.tasks)}")
            print(f"     - Task type: {route.tasks[0].task_type.value}")
            print(f"     - Task demands: {route.tasks[0].demand} kg, {route.tasks[0].volume} m³, {route.tasks[0].pallets} pallets")
            
            # Manual capacity calculation simulation
            print(f"   🔍 Manual Capacity Simulation:")
            print(f"     Starting state: empty vehicle")
            print(f"     - Initial load: 0 kg, 0 m³, 0 pallets")
            
            # Simulate what happens when we process the delivery
            delivery = route.tasks[0]
            
            # For delivery: we remove cargo (negative demand values)
            final_weight = 0 + delivery.demand  # delivery.demand is negative
            final_volume = 0 + delivery.volume  # delivery.volume is negative  
            final_pallets = 0 - delivery.pallets  # delivery.pallets is positive, we subtract
            
            print(f"     After delivery:")
            print(f"     - Final load: {final_weight} kg, {final_volume} m³, {final_pallets} pallets")
            
            # Check if final loads are valid
            weight_violation = final_weight < 0
            volume_violation = final_volume < 0
            pallet_violation = final_pallets < 0
            
            print(f"     Constraint violations:")
            print(f"     - Negative weight: {'❌' if weight_violation else '✅'} ({final_weight} kg)")
            print(f"     - Negative volume: {'❌' if volume_violation else '✅'} ({final_volume} m³)")
            print(f"     - Negative pallets: {'❌' if pallet_violation else '✅'} ({final_pallets} pallets)")
            
            # Test actual feasibility function
            print(f"   🔍 Actual Feasibility Check:")
            feasible = is_feasible(route, debug_feasibility=True)
            print(f"     Result: {'✅ FEASIBLE' if feasible else '❌ INFEASIBLE'}")
            
            # If infeasible, this reveals the issue!
            if not feasible:
                print(f"     🚨 ISSUE IDENTIFIED: Vehicle cannot deliver cargo it doesn't have!")
                print(f"     💡 SOLUTION NEEDED: Pre-load vehicle with required cargo for delivery-only orders")
        
        print(f"\n📊 CONCLUSION:")
        print(f"   The issue is that delivery-only orders try to remove cargo from empty vehicles.")
        print(f"   This creates negative cargo states which violate feasibility constraints.")
        print(f"   Solution: Either pre-load vehicles or handle delivery-only orders as special cases.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def investigate_initial_cargo_loading():
    """Investigate if vehicles can be pre-loaded with cargo for delivery-only orders."""
    print(f"\n🔍 INVESTIGATING INITIAL CARGO LOADING SOLUTION:")
    
    try:
        excel_file = os.path.join(src_dir, 'furgoni.xlsx')
        orders, vehicles = create_scenario_from_excel(excel_file)
        
        # Find the problematic delivery-only order
        test_order = None
        for order in orders:
            if order.id == "ORDER_COLUMBIA__SERRAVALLE_0":
                test_order = order
                break
        
        small_vehicle = None
        for vehicle in vehicles:
            if vehicle.id == "FX192HX":
                small_vehicle = vehicle
                break
        
        if not test_order or not small_vehicle:
            print("❌ Test objects not found")
            return
        
        delivery_task = test_order.delivery_tasks[0]
        
        print(f"\n💡 TESTING PRE-LOADING SOLUTION:")
        print(f"   Order: {test_order.id}")
        print(f"   Vehicle: {small_vehicle.id}")
        print(f"   Delivery demand: {delivery_task.demand} kg, {delivery_task.volume} m³, {delivery_task.pallets} pallets")
        
        # Test if pre-loading the vehicle would work
        required_initial_weight = abs(delivery_task.demand)
        required_initial_volume = abs(delivery_task.volume)
        required_initial_pallets = delivery_task.pallets
        
        print(f"\n   Required initial cargo:")
        print(f"   - Weight: {required_initial_weight} kg")
        print(f"   - Volume: {required_initial_volume} m³")
        print(f"   - Pallets: {required_initial_pallets}")
        
        # Check if vehicle can handle this initial cargo
        can_preload_weight = required_initial_weight <= small_vehicle.weight_capacity
        can_preload_volume = required_initial_volume <= small_vehicle.volume_capacity
        can_preload_pallets = required_initial_pallets <= (small_vehicle.pallet_capacity or float('inf'))
        
        print(f"\n   Vehicle capacity check:")
        print(f"   - Can pre-load weight: {'✅' if can_preload_weight else '❌'}")
        print(f"   - Can pre-load volume: {'✅' if can_preload_volume else '❌'}")
        print(f"   - Can pre-load pallets: {'✅' if can_preload_pallets else '❌'}")
        
        if can_preload_weight and can_preload_volume and can_preload_pallets:
            print(f"\n   ✅ SOLUTION FEASIBLE: Vehicle can be pre-loaded with required cargo!")
            print(f"   💡 Implementation: Modify feasibility logic to allow pre-loaded state for delivery-only orders")
        else:
            print(f"\n   ❌ SOLUTION NOT FEASIBLE: Vehicle cannot handle required initial cargo")
        
    except Exception as e:
        print(f"❌ Error in pre-loading investigation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_delivery_only_feasibility()
    investigate_initial_cargo_loading()
