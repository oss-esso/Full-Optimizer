"""
Debug Script for L1 Heuristic Order Assignment Issues

This script investigates why 30 out of 53 orders are failing to be assigned
during the cluster_aware_initializer phase of the l1_heuristic.

The script will:
1. Load the same scenario that's failing
2. Examine the properties of failing orders 
3. Check vehicle capacities and constraints
4. Test L2 feasibility checks manually for specific failing orders
5. Identify the root cause of assignment failures
"""

import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

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

# Import required modules
try:
    from scenario_creator import create_scenario_from_excel
    from first_level import l1_heuristic
    from second_level import l2_heuristic
    from epdt_data_structures import Route
    from route_provider import set_testing_mode
    set_testing_mode(use_haversine=True)
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def analyze_order_properties(orders):
    """Analyze the properties of all orders to understand what might be problematic."""
    print("📋 ORDER ANALYSIS:")
    print("=" * 60)
    
    total_weight = 0
    total_volume = 0
    weight_distribution = {}
    volume_distribution = {}
    
    for order in orders:
        # Get order demands
        weight = order.get_total_demand()
        volume = order.get_total_volume()
        
        total_weight += weight
        total_volume += volume
        
        # Categorize weight
        if weight <= 100:
            weight_cat = "Light (≤100kg)"
        elif weight <= 500:
            weight_cat = "Medium (101-500kg)"
        elif weight <= 1000:
            weight_cat = "Heavy (501-1000kg)"
        else:
            weight_cat = "Very Heavy (>1000kg)"
        
        weight_distribution[weight_cat] = weight_distribution.get(weight_cat, 0) + 1
        
        # Categorize volume
        if volume <= 1.0:
            volume_cat = "Small (≤1m³)"
        elif volume <= 5.0:
            volume_cat = "Medium (1-5m³)"
        elif volume <= 10.0:
            volume_cat = "Large (5-10m³)"
        else:
            volume_cat = "Very Large (>10m³)"
        
        volume_distribution[volume_cat] = volume_distribution.get(volume_cat, 0) + 1
    
    print(f"Total Orders: {len(orders)}")
    print(f"Total Weight Demand: {total_weight:.1f} kg")
    print(f"Total Volume Demand: {total_volume:.1f} m³")
    print(f"Average Weight per Order: {total_weight/len(orders):.1f} kg")
    print(f"Average Volume per Order: {total_volume/len(orders):.1f} m³")
    
    print("\nWeight Distribution:")
    for category, count in weight_distribution.items():
        print(f"  {category}: {count} orders")
    
    print("\nVolume Distribution:")
    for category, count in volume_distribution.items():
        print(f"  {category}: {count} orders")


def analyze_vehicle_properties(vehicles):
    """Analyze vehicle properties to understand capacity constraints."""
    print("\n🚛 VEHICLE ANALYSIS:")
    print("=" * 60)
    
    total_weight_capacity = 0
    total_volume_capacity = 0
    capacity_distribution = {}
    
    for vehicle in vehicles:
        weight_cap = vehicle.weight_capacity
        volume_cap = vehicle.volume_capacity
        
        total_weight_capacity += weight_cap
        total_volume_capacity += volume_cap
        
        # Categorize capacity
        if weight_cap <= 3500:
            cap_cat = "Light (≤3.5t)"
        elif weight_cap <= 7500:
            cap_cat = "Medium (3.5-7.5t)"
        elif weight_cap <= 15000:
            cap_cat = "Heavy (7.5-15t)"
        else:
            cap_cat = "Very Heavy (>15t)"
        
        capacity_distribution[cap_cat] = capacity_distribution.get(cap_cat, 0) + 1
    
    print(f"Total Vehicles: {len(vehicles)}")
    print(f"Total Weight Capacity: {total_weight_capacity:.1f} kg")
    print(f"Total Volume Capacity: {total_volume_capacity:.1f} m³")
    print(f"Average Weight Capacity: {total_weight_capacity/len(vehicles):.1f} kg")
    print(f"Average Volume Capacity: {total_volume_capacity/len(vehicles):.1f} m³")
    
    print("\nCapacity Distribution:")
    for category, count in capacity_distribution.items():
        print(f"  {category}: {count} vehicles")


def test_basic_feasibility(failing_orders, vehicles):
    """Test basic capacity feasibility for failing orders."""
    print("\n🔍 BASIC FEASIBILITY ANALYSIS:")
    print("=" * 60)
    
    print("Testing first 5 failing orders against all vehicles...")
    
    test_orders = failing_orders[:5]  # Test first 5 failing orders
    
    for i, order in enumerate(test_orders):
        print(f"\n📦 Order {i+1}: {order.id}")
        print(f"   Weight: {order.get_total_demand():.1f} kg")
        print(f"   Volume: {order.get_total_volume():.1f} m³")
        print(f"   Pickup tasks: {len(order.get_pickups())}")
        print(f"   Delivery tasks: {len(order.get_deliveries())}")
        
        # Examine the tasks in detail
        pickups = order.get_pickups()
        deliveries = order.get_deliveries()
        
        if pickups:
            for j, pickup in enumerate(pickups):
                print(f"     Pickup {j+1}: {pickup.location_id if hasattr(pickup, 'location_id') else 'Unknown'}")
        
        if deliveries:
            for j, delivery in enumerate(deliveries):
                print(f"     Delivery {j+1}: {delivery.location_id if hasattr(delivery, 'location_id') else 'Unknown'}")
                if hasattr(delivery, 'demand'):
                    print(f"       Demand: {delivery.demand} kg")
                if hasattr(delivery, 'volume'):
                    print(f"       Volume: {delivery.volume} m³")
        
        # Check if this is a delivery-only order (problematic pattern)
        if len(pickups) == 0 and len(deliveries) > 0:
            print(f"   🚨 DELIVERY-ONLY ORDER detected! This might be the issue!")
        
        compatible_vehicles = 0
        
        for vehicle in vehicles:
            weight_ok = order.get_total_demand() <= vehicle.weight_capacity
            volume_ok = order.get_total_volume() <= vehicle.volume_capacity
            
            if weight_ok and volume_ok:
                compatible_vehicles += 1
        
        print(f"   Compatible vehicles (basic capacity): {compatible_vehicles}/{len(vehicles)}")
        
        if compatible_vehicles == 0:
            print(f"   ❌ NO VEHICLES CAN HANDLE THIS ORDER!")
            # Find the smallest vehicle that could handle the weight
            vehicles_by_weight = sorted(vehicles, key=lambda v: v.weight_capacity, reverse=True)
            vehicles_by_volume = sorted(vehicles, key=lambda v: v.volume_capacity, reverse=True)
            
            print(f"   Largest vehicle weight capacity: {vehicles_by_weight[0].weight_capacity:.1f} kg")
            print(f"   Largest vehicle volume capacity: {vehicles_by_volume[0].volume_capacity:.1f} m³")


def test_l2_insertion_debugging(failing_orders, vehicles):
    """Test L2 insertion with debugging for specific failing orders."""
    print("\n🧪 L2 INSERTION DEBUGGING:")
    print("=" * 60)
    
    # Test the first failing order with the first few vehicles
    if failing_orders:
        test_order = failing_orders[0]
        print(f"\nTesting order: {test_order.id}")
        
        test_vehicles = vehicles[:5]  # Test with first 5 vehicles
        
        for vehicle in test_vehicles:
            print(f"\n🚛 Testing vehicle: {vehicle.id}")
            print(f"   Weight capacity: {vehicle.weight_capacity:.1f} kg")
            print(f"   Volume capacity: {vehicle.volume_capacity:.1f} m³")
            
            # Create empty route
            empty_route = Route(vehicle=vehicle)
            
            # Try L2 insertion with debugging
            try:
                result_route = l2_heuristic(empty_route, test_order, True)  # Use positional argument
                if result_route is not None:
                    print(f"   ✅ L2 insertion SUCCEEDED!")
                    print(f"   Route tasks: {len(result_route.tasks)}")
                else:
                    print(f"   ❌ L2 insertion FAILED!")
            except Exception as e:
                print(f"   💥 L2 insertion CRASHED: {e}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()[:200]}...")


def main():
    """Main debugging function."""
    print("🐛 DEBUGGING L1 HEURISTIC ORDER ASSIGNMENT FAILURES")
    print("=" * 80)
    
    # Load the same scenario that's failing
    excel_file = os.path.join(src_dir, 'furgoni.xlsx')
    
    if not os.path.exists(excel_file):
        print(f"❌ Excel file not found: {excel_file}")
        return
    
    print(f"📂 Loading scenario from: {excel_file}")
    
    try:
        orders, vehicles = create_scenario_from_excel(excel_file)
        print(f"✅ Loaded {len(orders)} orders and {len(vehicles)} vehicles")
    except Exception as e:
        print(f"❌ Error loading scenario: {e}")
        return
    
    # Analyze order and vehicle properties
    analyze_order_properties(orders)
    analyze_vehicle_properties(vehicles)
    
    # Define the failing order IDs (from the comprehensive test output)
    failing_order_ids = [
        "ORDER_DIPHARMA_FRANCIS_SRL_39",
        "ORDER_GEODIS_18", 
        "ORDER_COLUMBIA__SERRAVALLE_0",
        "ORDER_AGRIACRO_TECH_SRL_19",
        "ORDER_FIAT_V._I_SPA_STAB_B_21",
        "ORDER_A.DI_DEDDA_P/C__ENER_2"
    ]
    
    # Find the failing order objects
    failing_orders = [order for order in orders if order.id in failing_order_ids]
    
    print(f"\n🎯 Found {len(failing_orders)} failing orders for detailed analysis")
    
    # Test basic feasibility
    test_basic_feasibility(failing_orders, vehicles)
    
    # Test L2 insertion with debugging
    test_l2_insertion_debugging(failing_orders, vehicles)
    
    print("\n" + "=" * 80)
    print("🏁 DEBUGGING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
