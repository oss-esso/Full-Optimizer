"""
Depot Bay Operations Investigation

This script investigates if the delivery-only order failures are caused by missing
depot bay operations (depot pickups for delivery-only orders).

The hypothesis: delivery-only orders fail on smaller vehicles because they need
initial cargo loading at depot, but we're not creating depot pickup tasks.
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
from epdt_data_structures import Route, Task, TaskType

print("=== DEPOT BAY OPERATIONS INVESTIGATION ===")

def analyze_depot_bay_requirements():
    """Analyze if orders need depot bay operations and test with manual depot tasks."""
    excel_file = os.path.join(src_dir, 'furgoni.xlsx')
    
    if not os.path.exists(excel_file):
        print(f"❌ Excel file not found: {excel_file}")
        return
    
    print(f"📂 Loading scenario from: {excel_file}")
    
    try:
        orders, vehicles = create_scenario_from_excel(excel_file)
        print(f"✅ Loaded {len(orders)} orders and {len(vehicles)} vehicles")
        
        # Focus on failing delivery-only orders
        failing_orders = [
            "ORDER_COLUMBIA__SERRAVALLE_0",  # Delivery-only: -649.0 kg, -7.31 m³, 6 pallets
            "ORDER_FIAT_V._I_SPA_STAB_B_21"  # Delivery-only: -598.0 kg, -1.66 m³, 1 pallet
        ]
        
        # Test vehicles: small one that fails, large one that works
        test_vehicles = []
        for vehicle in vehicles:
            if vehicle.id in ["FX192HX", "FX194HX"]:  # Small vs Large
                test_vehicles.append(vehicle)
        
        print(f"\n🔍 TESTING DEPOT BAY HYPOTHESIS:")
        print(f"   Test vehicles: {[v.id for v in test_vehicles]}")
        
        for order_id in failing_orders:
            order = None
            for o in orders:
                if o.id == order_id:
                    order = o
                    break
            
            if not order:
                print(f"\n❌ Order {order_id} not found")
                continue
            
            print(f"\n{'='*80}")
            print(f"🎯 ANALYZING ORDER: {order.id}")
            
            # Check if this is delivery-only
            delivery_task = order.delivery_tasks[0] if order.delivery_tasks else None
            if not delivery_task:
                print(f"   ❌ No delivery task found")
                continue
            
            print(f"   📦 Delivery Task:")
            print(f"      Weight: {delivery_task.demand:.1f} kg (negative = delivery)")
            print(f"      Volume: {delivery_task.volume:.2f} m³")
            print(f"      Pallets: {delivery_task.pallets}")
            
            # Test 1: Try original order (without depot pickup)
            print(f"\n   🧪 TEST 1: Original Order (No Depot Pickup)")
            
            for vehicle in test_vehicles:
                print(f"\n      Vehicle {vehicle.id}:")
                empty_route = Route(vehicle=vehicle, tasks=[])
                
                result = l2_heuristic(empty_route, order, debug_assignment=False)
                success = result is not None
                print(f"         L2 Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
                
                if success:
                    feasible = is_feasible(result, debug_feasibility=False)
                    print(f"         Feasibility: {'✅ FEASIBLE' if feasible else '❌ INFEASIBLE'}")
            
            # Test 2: Create depot pickup task and test combined order
            print(f"\n   🧪 TEST 2: With Manual Depot Pickup Task")
            
            # Create a depot pickup task to load the cargo initially
            depot_pickup = Task(
                id=f"DEPOT_PICKUP_{order.id}",
                location_id="DEPOT_MAIN",
                task_type=TaskType.PICKUP,
                order_id=order.id,
                lat=45.4064,  # Padova depot coordinates
                lon=11.8768,
                service_time=10.0,  # 10 minutes to load at depot
                demand=abs(delivery_task.demand),  # Positive for pickup
                volume=abs(delivery_task.volume),   # Positive for pickup  
                pallets=delivery_task.pallets,      # Same pallets count
                earliest_time=None,
                latest_time=None,
                soft_time_window=True
            )
            
            # Create modified order with depot pickup
            from epdt_data_structures import Order
            modified_order = Order(
                id=f"{order.id}_WITH_DEPOT",
                pickup_tasks=[depot_pickup],
                delivery_tasks=order.delivery_tasks.copy()
            )
            
            print(f"      📦 Added Depot Pickup:")
            print(f"         Weight: {depot_pickup.demand:.1f} kg (positive = pickup)")
            print(f"         Volume: {depot_pickup.volume:.2f} m³")
            print(f"         Pallets: {depot_pickup.pallets}")
            
            for vehicle in test_vehicles:
                print(f"\n      Vehicle {vehicle.id}:")
                empty_route = Route(vehicle=vehicle, tasks=[])
                
                result = l2_heuristic(empty_route, modified_order, debug_assignment=False)
                success = result is not None
                print(f"         L2 Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
                
                if success:
                    feasible = is_feasible(result, debug_feasibility=False)
                    print(f"         Feasibility: {'✅ FEASIBLE' if feasible else '❌ INFEASIBLE'}")
                    
                    # Show route structure
                    print(f"         Route Tasks: {len(result.tasks)}")
                    for i, task in enumerate(result.tasks):
                        task_location = "DEPOT" if "DEPOT" in task.location_id else "CUSTOMER"
                        print(f"           {i+1}. {task.task_type.value} at {task_location} ({task.demand:.1f}kg)")
            
            # Test 3: Test feasibility constraints in detail
            print(f"\n   🔍 TEST 3: Detailed Feasibility Analysis")
            
            small_vehicle = test_vehicles[0]  # FX192HX
            print(f"\n      Small Vehicle ({small_vehicle.id}) Constraint Analysis:")
            print(f"         Capacity - Weight: {small_vehicle.weight_capacity:.1f} kg")
            print(f"         Capacity - Volume: {small_vehicle.volume_capacity:.2f} m³")
            print(f"         Capacity - Pallets: {small_vehicle.pallet_capacity}")
            
            # Simulate route with depot pickup + delivery
            print(f"\n         Simulated Route Progression:")
            load_w = 0.0
            load_v = 0.0
            load_p = 0
            
            # Step 1: Depot pickup
            load_w += abs(delivery_task.demand)
            load_v += abs(delivery_task.volume)
            load_p += delivery_task.pallets
            
            print(f"         After Depot Pickup:")
            print(f"            Load - Weight: {load_w:.1f} kg ({'✅' if load_w <= small_vehicle.weight_capacity else '❌ EXCEEDS'})")
            print(f"            Load - Volume: {load_v:.2f} m³ ({'✅' if load_v <= small_vehicle.volume_capacity else '❌ EXCEEDS'})")
            print(f"            Load - Pallets: {load_p} ({'✅' if load_p <= small_vehicle.pallet_capacity else '❌ EXCEEDS'})")
            
            # Step 2: Customer delivery
            load_w += delivery_task.demand  # Negative, reduces load
            load_v += delivery_task.volume  # Negative, reduces load
            load_p -= delivery_task.pallets  # Remove pallets
            
            print(f"         After Customer Delivery:")
            print(f"            Load - Weight: {load_w:.1f} kg")
            print(f"            Load - Volume: {load_v:.2f} m³")
            print(f"            Load - Pallets: {load_p}")
            
            constraint_violations = []
            if abs(delivery_task.volume) > small_vehicle.volume_capacity:
                constraint_violations.append(f"Volume constraint: {abs(delivery_task.volume):.2f} > {small_vehicle.volume_capacity:.2f}")
            if delivery_task.pallets > small_vehicle.pallet_capacity:
                constraint_violations.append(f"Pallet constraint: {delivery_task.pallets} > {small_vehicle.pallet_capacity}")
            
            if constraint_violations:
                print(f"         ❌ CONSTRAINT VIOLATIONS:")
                for violation in constraint_violations:
                    print(f"            • {violation}")
            else:
                print(f"         ✅ NO CONSTRAINT VIOLATIONS")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_depot_bay_requirements()
