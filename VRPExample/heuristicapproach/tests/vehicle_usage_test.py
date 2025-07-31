"""
Simple test to verify vehicle usage fixes - should now use more vehicles when penalty is low.
"""

import sys
import os

# Add path for imports
test_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(test_dir, '..')
algo_dir = os.path.join(heuristic_root, 'algo')
utils_dir = os.path.join(heuristic_root, 'utils')
sys.path.insert(0, heuristic_root)
sys.path.insert(0, algo_dir)
sys.path.insert(0, utils_dir)

print("=== Vehicle Usage Test ===")

try:
    # Test the fix by running a quick scenario with different vehicle penalty values
    from scenario_creator import create_scenario_from_excel
    from first_level import l1_heuristic
    
    print("Successfully imported required modules")
    
    # Load a small scenario
    excel_path = os.path.join(heuristic_root, "src", "furgoni2.xlsx")
    
    print(f"Loading scenario from: {excel_path}")
    epdt_orders, epdt_vehicles, drivers = create_scenario_from_excel(excel_path)
    
    print(f"Loaded {len(epdt_orders)} orders and {len(epdt_vehicles)} vehicles")
    
    # Test 1: High vehicle penalty (should use fewer vehicles)
    params_high_penalty = {
        'tabu_tenure': 10,
        'M1': 3,  # Very short test
        'M2': 10,
        'vehicle_penalty_per_vehicle': 1000.0,  # High penalty - prefer fewer vehicles
        'unassigned_order_base_penalty': 10000.0,
        'debug_regret': False,
        'debug_destroy_repair': False
    }
    
    print(f"\n=== Test 1: High Vehicle Penalty (1000.0) ===")
    solution_high = l1_heuristic(epdt_orders, epdt_vehicles, params_high_penalty)
    
    vehicles_used_high = len([route for route in solution_high.routes if route and hasattr(route, 'tasks') and route.tasks])
    print(f"Vehicles used with HIGH penalty: {vehicles_used_high}/{len(epdt_vehicles)}")
    
    # Test 2: Low vehicle penalty (should use more vehicles)
    params_low_penalty = {
        'tabu_tenure': 10,
        'M1': 3,  # Very short test
        'M2': 10,
        'vehicle_penalty_per_vehicle': 0.0,  # Zero penalty - vehicles are free!
        'unassigned_order_base_penalty': 10000.0,
        'debug_regret': False,
        'debug_destroy_repair': False
    }
    
    print(f"\n=== Test 2: Low Vehicle Penalty (0.0) ===")
    solution_low = l1_heuristic(epdt_orders, epdt_vehicles, params_low_penalty)
    
    vehicles_used_low = len([route for route in solution_low.routes if route and hasattr(route, 'tasks') and route.tasks])
    print(f"Vehicles used with LOW penalty: {vehicles_used_low}/{len(epdt_vehicles)}")
    
    # Analysis
    print(f"\n=== Analysis ===")
    print(f"High penalty: {vehicles_used_high} vehicles")
    print(f"Low penalty: {vehicles_used_low} vehicles")
    
    if vehicles_used_low > vehicles_used_high:
        print("✅ SUCCESS: Low penalty uses MORE vehicles (fix working!)")
    elif vehicles_used_low == vehicles_used_high:
        print("⚠️  NEUTRAL: Same number of vehicles used (may need longer test)")
    else:
        print("❌ UNEXPECTED: Low penalty uses FEWER vehicles (investigate)")
    
    print(f"\n=== Vehicle Usage Test Complete ===")

except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()
