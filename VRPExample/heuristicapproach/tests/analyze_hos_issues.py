"""
Diagnostic script to analyze HoS violations and initialization issues.

This script will help us understand:
1. Which specific routes are causing HoS violations
2. What characteristics make these routes problematic
3. Whether the issue is in initialization or optimization
"""

import sys
import os

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
sys.path.insert(0, heuristic_root)
sys.path.insert(0, os.path.join(heuristic_root, 'src'))
sys.path.insert(0, os.path.join(heuristic_root, 'algo'))
sys.path.insert(0, os.path.join(heuristic_root, 'utils'))

def analyze_unassigned_orders():
    """Analyze the characteristics of unassigned orders."""
    print("=== Analyzing Unassigned Orders ===")
    
    # Based on the test output, these orders consistently fail assignment:
    problematic_orders = [14, 17, 18, 19, 34, 36, 42, 48, 51]
    
    try:
        from scenario_creator import create_scenario_from_excel
        
        # Load the scenario
        excel_path = os.path.join(heuristic_root, 'src', 'furgoni3_2.xlsx')
        print(f"Loading scenario from: {excel_path}")
        
        orders, vehicles, drivers = create_scenario_from_excel(excel_path)
        print(f"Loaded: {len(orders)} orders, {len(vehicles)} vehicles")
        
        # Analyze problematic orders
        print(f"\nAnalyzing {len(problematic_orders)} problematic orders...")
        
        for order_id in problematic_orders:
            if order_id <= len(orders):
                order = orders[order_id - 1]  # Assuming 0-based indexing
                print(f"\nOrder {order_id}:")
                print(f"  Weight: {getattr(order, 'weight', 'N/A')} kg")
                print(f"  Volume: {getattr(order, 'volume', 'N/A')} m³")
                print(f"  Pallets: {getattr(order, 'pallets', 'N/A')}")
                
                # Check if order exceeds any vehicle capacity
                max_weight = max(v.weight_capacity for v in vehicles)
                max_volume = max(v.volume_capacity for v in vehicles)
                max_pallets = max(v.pallet_capacity for v in vehicles)
                
                print(f"  Max vehicle capacities: {max_weight}kg, {max_volume}m³, {max_pallets} pallets")
                
                weight_ok = getattr(order, 'weight', 0) <= max_weight
                volume_ok = getattr(order, 'volume', 0) <= max_volume  
                pallets_ok = getattr(order, 'pallets', 0) <= max_pallets
                
                print(f"  Capacity check: Weight {weight_ok}, Volume {volume_ok}, Pallets {pallets_ok}")
                
                if not (weight_ok and volume_ok and pallets_ok):
                    print(f"  *** ORDER {order_id} EXCEEDS VEHICLE CAPACITIES ***")
        
        # Analyze vehicle distribution
        print(f"\n=== Vehicle Capacity Analysis ===")
        vehicle_types = {}
        for vehicle in vehicles:
            v_type = f"{vehicle.weight_capacity:.0f}kg_{vehicle.volume_capacity:.0f}m³_{vehicle.pallet_capacity}pal"
            if v_type not in vehicle_types:
                vehicle_types[v_type] = 0
            vehicle_types[v_type] += 1
            
        print("Vehicle type distribution:")
        for v_type, count in sorted(vehicle_types.items()):
            print(f"  {v_type}: {count} vehicles")
            
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

def analyze_hos_violations():
    """Analyze patterns in HoS violations."""
    print("\n=== HoS Violation Analysis ===")
    
    # Based on test output, these vehicles had HoS violations:
    hos_violating_vehicles = [
        "FX192HX", "GA621VG", "GA646VG", "GA649VG", "GC079LX", 
        "GC086LX", "GE030FZ", "GP507LA", "XA819VA", "XA346KW", "XA345KW"
    ]
    
    print(f"Found {len(hos_violating_vehicles)} vehicles with HoS violations:")
    for vehicle_id in hos_violating_vehicles:
        print(f"  - {vehicle_id}")
    
    # The issue might be that these routes have too many orders or long distances
    print("\nThis suggests the problem might be:")
    print("1. Routes are too long (exceed driving time limits)")
    print("2. Too many orders assigned to single vehicles")
    print("3. Poor geographical clustering (inefficient routes)")
    print("4. Insufficient rest time calculation")

if __name__ == "__main__":
    analyze_unassigned_orders()
    analyze_hos_violations()
    
    print("\n=== Recommendations ===")
    print("1. Try cluster-aware initialization (already testing)")
    print("2. Reduce maximum orders per route")
    print("3. Implement geographical pre-clustering")
    print("4. Review HoS parameters for conservativeness")
    print("5. Consider order splitting for oversized orders")
