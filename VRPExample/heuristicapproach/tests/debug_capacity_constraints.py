"""
Capacity Constraints Debugging Script

This script investigates why orders are failing assignment by examining:
1. Volume constraints (hard constraint)
2. Pallet count constraints (hard constraint) 
3. Weight constraints (soft constraint with penalty)

Focus: Understanding the hard constraint violations preventing order assignment.
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

print("=== CAPACITY CONSTRAINTS DEBUGGING ===")

def analyze_order_vehicle_compatibility(orders, vehicles):
    """
    Analyze which orders can be assigned to which vehicles based on hard constraints.
    """
    print(f"\n🔍 ANALYZING ORDER-VEHICLE COMPATIBILITY")
    print(f"   Orders: {len(orders)}")
    print(f"   Vehicles: {len(vehicles)}")
    
    # Get some problematic orders from previous run
    problematic_order_ids = [
        "ORDER_DIPHARMA_FRANCIS_SRL_39",
        "ORDER_GEODIS_18", 
        "ORDER_COLUMBIA__SERRAVALLE_0",
        "ORDER_AGRIACRO_TECH_SRL_19",
        "ORDER_FIAT_V._I_SPA_STAB_B_21"
    ]
    
    problematic_orders = []
    for order in orders:
        if order.id in problematic_order_ids:
            problematic_orders.append(order)
    
    print(f"\n📦 ANALYZING {len(problematic_orders)} PROBLEMATIC ORDERS:")
    
    for order in problematic_orders[:3]:  # Analyze first 3
        print(f"\n--- ORDER: {order.id} ---")
        
        # Calculate order requirements
        total_weight = 0.0
        total_volume = 0.0
        total_pallets = 0
        
        all_tasks = order.pickup_tasks + order.delivery_tasks
        for task in all_tasks:
            total_weight += getattr(task, 'weight_demand', 0.0)
            total_volume += getattr(task, 'volume_demand', 0.0)
            total_pallets += getattr(task, 'pallet_demand', 0)
        
        print(f"   Order Requirements:")
        print(f"   • Weight: {total_weight:.1f} kg")
        print(f"   • Volume: {total_volume:.2f} m³")
        print(f"   • Pallets: {total_pallets}")
        print(f"   • Task count: {len(all_tasks)}")
        
        # Test against different vehicle types
        compatible_vehicles = []
        
        for vehicle in vehicles[:10]:  # Test first 10 vehicles
            can_handle_volume = vehicle.volume_capacity >= abs(total_volume)
            can_handle_pallets = vehicle.pallet_capacity >= abs(total_pallets)
            can_handle_weight_soft = True  # Weight is soft constraint
            
            # Check hard constraints only
            is_compatible = can_handle_volume and can_handle_pallets
            
            if is_compatible:
                compatible_vehicles.append(vehicle)
            
            print(f"   Vehicle {vehicle.id}:")
            print(f"     - Volume: {vehicle.volume_capacity:.2f} m³ (req: {abs(total_volume):.2f}) {'✅' if can_handle_volume else '❌'}")
            print(f"     - Pallets: {vehicle.pallet_capacity} (req: {abs(total_pallets)}) {'✅' if can_handle_pallets else '❌'}")
            print(f"     - Weight: {vehicle.weight_capacity:.1f} kg (req: {abs(total_weight):.1f}) ⚠️ SOFT")
            print(f"     - Compatible: {'✅' if is_compatible else '❌'}")
        
        print(f"   📊 RESULT: {len(compatible_vehicles)} compatible vehicles found")
        
        if len(compatible_vehicles) > 0:
            print(f"   ✅ Order should be assignable to: {[v.id for v in compatible_vehicles[:3]]}")
            
            # Test L2 heuristic with first compatible vehicle
            print(f"\n   🧪 TESTING L2 HEURISTIC with {compatible_vehicles[0].id}:")
            test_route = Route(vehicle=compatible_vehicles[0], tasks=[])
            try:
                result = l2_heuristic(test_route, order, {})
                if result:
                    print(f"      ✅ L2 heuristic SUCCESS")
                else:
                    print(f"      ❌ L2 heuristic FAILED - examining why...")
                    # Try to get more details
                    analyze_l2_failure(test_route, order, compatible_vehicles[0])
            except Exception as e:
                print(f"      💥 L2 heuristic ERROR: {e}")
        else:
            print(f"   ❌ NO COMPATIBLE VEHICLES - Hard constraint violations")


def analyze_l2_failure(route, order, vehicle):
    """
    Deeper analysis of why L2 heuristic fails.
    """
    print(f"      🔍 L2 FAILURE ANALYSIS:")
    
    # Check individual task constraints
    all_tasks = order.pickup_tasks + order.delivery_tasks
    
    for i, task in enumerate(all_tasks):
        print(f"        Task {i+1} ({task.task_type.value}):")
        print(f"          - Weight: {getattr(task, 'weight_demand', 0.0):.1f} kg")
        print(f"          - Volume: {getattr(task, 'volume_demand', 0.0):.2f} m³")
        print(f"          - Pallets: {getattr(task, 'pallet_demand', 0)}")
        
        # Check if individual task exceeds vehicle capacity
        exceeds_volume = abs(getattr(task, 'volume_demand', 0.0)) > vehicle.volume_capacity
        exceeds_pallets = abs(getattr(task, 'pallet_demand', 0)) > vehicle.pallet_capacity
        
        if exceeds_volume:
            print(f"          ❌ VOLUME VIOLATION: {abs(getattr(task, 'volume_demand', 0.0)):.2f} > {vehicle.volume_capacity:.2f}")
        if exceeds_pallets:
            print(f"          ❌ PALLET VIOLATION: {abs(getattr(task, 'pallet_demand', 0))} > {vehicle.pallet_capacity}")


def check_vehicle_capacity_distribution(vehicles):
    """
    Analyze the distribution of vehicle capacities to understand coverage.
    """
    print(f"\n📊 VEHICLE CAPACITY DISTRIBUTION:")
    
    vehicles_sorted_by_volume = sorted(vehicles, key=lambda v: v.volume_capacity)
    vehicles_sorted_by_pallets = sorted(vehicles, key=lambda v: v.pallet_capacity)
    vehicles_sorted_by_weight = sorted(vehicles, key=lambda v: v.weight_capacity)
    
    print(f"\n   Volume capacity range:")
    print(f"   • Min: {vehicles_sorted_by_volume[0].volume_capacity:.2f} m³ ({vehicles_sorted_by_volume[0].id})")
    print(f"   • Max: {vehicles_sorted_by_volume[-1].volume_capacity:.2f} m³ ({vehicles_sorted_by_volume[-1].id})")
    print(f"   • Median: {vehicles_sorted_by_volume[len(vehicles)//2].volume_capacity:.2f} m³")
    
    print(f"\n   Pallet capacity range:")
    print(f"   • Min: {vehicles_sorted_by_pallets[0].pallet_capacity} pallets ({vehicles_sorted_by_pallets[0].id})")
    print(f"   • Max: {vehicles_sorted_by_pallets[-1].pallet_capacity} pallets ({vehicles_sorted_by_pallets[-1].id})")
    print(f"   • Median: {vehicles_sorted_by_pallets[len(vehicles)//2].pallet_capacity} pallets")
    
    print(f"\n   Weight capacity range:")
    print(f"   • Min: {vehicles_sorted_by_weight[0].weight_capacity:.1f} kg ({vehicles_sorted_by_weight[0].id})")
    print(f"   • Max: {vehicles_sorted_by_weight[-1].weight_capacity:.1f} kg ({vehicles_sorted_by_weight[-1].id})")
    print(f"   • Median: {vehicles_sorted_by_weight[len(vehicles)//2].weight_capacity:.1f} kg")


def main():
    excel_file = os.path.join(src_dir, 'furgoni.xlsx')
    
    if not os.path.exists(excel_file):
        print(f"❌ Excel file not found: {excel_file}")
        return
    
    print(f"📂 Loading scenario from: {excel_file}")
    
    try:
        orders, vehicles = create_scenario_from_excel(excel_file)
        print(f"✅ Loaded {len(orders)} orders and {len(vehicles)} vehicles")
        
        # Analyze vehicle capacity distribution
        check_vehicle_capacity_distribution(vehicles)
        
        # Analyze order-vehicle compatibility
        analyze_order_vehicle_compatibility(orders, vehicles)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
