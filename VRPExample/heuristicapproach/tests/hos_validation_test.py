"""
Simple test to verify HoS violation fixes.
"""

import sys
import os

# Add path for imports
test_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(test_dir, '..')
sys.path.insert(0, heuristic_root)

print("=== HoS Validation Test ===")
print(f"Test directory: {test_dir}")
print(f"Heuristic root: {heuristic_root}")

try:
    # Import required modules
    from algo.first_level import l1_heuristic
    from utils.scenario_creator import create_scenario_from_excel
    
    print("Successfully imported required modules")
    
    # Find the furgoni2.xlsx file
    src_dir = os.path.join(heuristic_root, '..', '..', '..', 'src')
    excel_file = os.path.join(src_dir, 'furgoni2.xlsx')
    
    if not os.path.exists(excel_file):
        print(f"Excel file not found at: {excel_file}")
        print("Looking for alternative Excel files...")
        # Try to find any Excel file in tests
        for f in os.listdir(test_dir):
            if f.endswith('.xlsx'):
                excel_file = os.path.join(test_dir, f)
                print(f"Using Excel file: {excel_file}")
                break
    
    # Create test scenario
    orders, vehicles, drivers = create_scenario_from_excel(excel_file)
    print(f"Created scenario: {len(orders)} orders, {len(vehicles)} vehicles, {len(drivers)} drivers")
    
    # Create parameters dict
    params = {
        'max_iterations': 50,  # Reduced for faster testing
        'tabu_tenure': 10,
        'stagnation_limit': 20,
        'debug_feasibility': False,
        'enable_destroy_and_repair': False,
        'M1': 1000,  # Distance penalty
        'M2': 5000,  # Hard constraint penalty
        'initialization_method': 'cluster_aware'
    }
    
    print("Running L1 heuristic with HoS validation fix...")
    
    # Run the heuristic
    solution = l1_heuristic(orders, vehicles, params)
    
    print(f"\n=== SOLUTION ANALYSIS ===")
    print(f"Total routes: {len(solution.routes)}")
    
    # Analyze each route for HoS compliance
    hos_compliant_routes = 0
    hos_violation_routes = 0
    
    for i, route in enumerate(solution.routes):
        print(f"\nRoute {i+1}: {route}")
        
        # Check if route is actually a Route object
        if hasattr(route, 'tasks') and route.tasks:
            print(f"  Tasks: {len(route.tasks)}")
            
            # Try to import the feasibility check
            try:
                from algo.second_level import is_feasible
                feasible = is_feasible(route, debug_feasibility=False)
                if feasible:
                    print(f"  HoS Status: COMPLIANT")
                    hos_compliant_routes += 1
                else:
                    print(f"  HoS Status: VIOLATION")
                    hos_violation_routes += 1
            except Exception as e:
                print(f"  HoS Status: ERROR checking feasibility - {e}")
        else:
            print(f"  Type: {type(route)} (not a Route object with tasks)")
            print(f"  This suggests the solution structure may be different than expected")
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"HoS Compliant Routes: {hos_compliant_routes}")
    print(f"HoS Violation Routes: {hos_violation_routes}")
    
    if hos_violation_routes == 0:
        print("SUCCESS: No HoS violations detected in final solution!")
    else:
        print(f"ISSUE: {hos_violation_routes} routes still have HoS violations")
    
    print("\n=== HoS Validation Test Complete ===")

except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()
