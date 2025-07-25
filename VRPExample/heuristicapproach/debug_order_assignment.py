#!/usr/bin/env python3
"""
Debug script to analyze why specific orders are not being assigned
"""

import sys
import os
import time

# Add necessary paths for imports (same as test runner)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Since this script is in the heuristic root (not in tests/), the paths are different
heuristic_root = current_dir  # We're already in the heuristic root
src_dir = os.path.join(heuristic_root, 'src')
algo_dir = os.path.join(heuristic_root, 'algo')

sys.path.insert(0, heuristic_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, algo_dir)

# Debug path information
print(f"Debug: Current working directory: {os.getcwd()}")
print(f"Debug: Script directory: {current_dir}")
print(f"Debug: Heuristic root: {heuristic_root}")
print(f"Debug: Source directory: {src_dir}")
print(f"Debug: Algorithm directory: {algo_dir}")
print(f"Debug: moda_scenarios.py exists: {os.path.exists(os.path.join(src_dir, 'moda_scenarios.py'))}")

def debug_unassigned_orders():
    """Debug why specific orders are failing assignment"""
    
    print("🔍 Debugging Unassigned Order Assignment")
    print("="*60)
    
    # Import required modules - use exact pattern from working test script
    try:
        from moda_scenarios import create_furgoni_scenario
        print("✅ Imported moda_scenarios")
        
        from data_adapter import convert_instance_to_epdt_input
        print("✅ Imported data_adapter")
        
        from epdt_data_structures import Route
        print("✅ Imported epdt_data_structures")
        
        from second_level import is_feasible, l2_heuristic
        print("✅ Imported second_level")
        
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Available modules in path:")
        for path in sys.path[:5]:  # Show first 5 paths
            if os.path.exists(path):
                print(f"  {path}: {os.listdir(path)[:5]}")  # Show first 5 files
        return False
    
    # Create scenario
    print("\n1️⃣ Creating scenario...")
    vrp_instance = create_furgoni_scenario()
    orders, vehicles = convert_instance_to_epdt_input(vrp_instance)
    print(f"✅ Scenario created: {len(orders)} orders, {len(vehicles)} vehicles")
    
    # Find a problematic order from our analysis
    problem_order = None
    for order in orders:
        if order.id == 'depot_request_6':  # This was one that should fit but wasn't assigned
            problem_order = order
            break
    
    if not problem_order:
        print("❌ Could not find depot_request_6")
        return False
    
    print(f"\n2️⃣ Analyzing problem order: {problem_order.id}")
    print(f"   Weight: {problem_order.get_total_demand():.0f} kg")
    print(f"   Volume: {problem_order.get_total_volume():.1f} m³")
    
    # Find a suitable vehicle (camion_16_t had capacity)
    test_vehicle = None
    for vehicle in vehicles:
        if vehicle.id == 'camion_16_t':
            test_vehicle = vehicle
            break
    
    if not test_vehicle:
        print("❌ Could not find camion_16_t")
        return False
    
    print(f"\n3️⃣ Testing vehicle: {test_vehicle.id}")
    print(f"   Weight capacity: {test_vehicle.weight_capacity:.0f} kg")
    print(f"   Volume capacity: {test_vehicle.volume_capacity:.1f} m³")
    
    # Create a minimal route with just this order
    print(f"\n4️⃣ Testing order insertion...")
    empty_route = Route(vehicle=test_vehicle)
    empty_route.tasks = []
    
    print(f"   Empty route feasible: {is_feasible(empty_route)}")
    
    # Try to insert the order using L2 heuristic
    print(f"\n5️⃣ Testing L2 insertion...")
    try:
        start_time = time.time()
        result_route = l2_heuristic(empty_route, problem_order)
        end_time = time.time()
        
        if result_route:
            print(f"✅ L2 insertion successful in {end_time - start_time:.2f}s")
            print(f"   Result route tasks: {len(result_route.tasks)}")
            print(f"   Result route feasible: {is_feasible(result_route)}")
        else:
            print(f"❌ L2 insertion failed after {end_time - start_time:.2f}s")
            print(f"   This explains why the order was not assigned!")
            
            # Test basic feasibility with manual task creation
            print(f"\n6️⃣ Testing manual route creation...")
            manual_route = Route(vehicle=test_vehicle)
            manual_route.tasks = problem_order.pickup_tasks + problem_order.delivery_tasks
            
            print(f"   Manual route tasks: {len(manual_route.tasks)}")
            manual_feasible = is_feasible(manual_route)
            print(f"   Manual route feasible: {manual_feasible}")
            
            if not manual_feasible:
                print(f"   💡 The order itself violates constraints!")
            else:
                print(f"   💡 The order is feasible but L2 heuristic can't find insertion!")
                
    except Exception as e:
        print(f"❌ Error in L2 insertion: {e}")
        import traceback
        traceback.print_exc()
    
    return True

if __name__ == "__main__":
    debug_unassigned_orders()
