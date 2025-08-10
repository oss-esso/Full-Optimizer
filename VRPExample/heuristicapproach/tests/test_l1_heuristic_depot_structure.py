#!/usr/bin/env python3
"""
Quick test to verify l1_heuristic produces routes with depot structure.
This investigates the root cause of comprehensive test failures.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'algo'))

def test_l1_heuristic_depot_structure():
    """Test that l1_heuristic produces routes with proper depot structure."""
    
    print("🔍 Testing l1_heuristic depot structure...")
    
    try:
        # Import required modules
        from utils.scenario_creator import create_scenario_from_excel
        from algo.first_level import l1_heuristic
        
        # Load a small scenario
        excel_file = os.path.join(os.path.dirname(__file__), '..', 'src', 'furgoni2.xlsx')
        print(f"   Loading scenario from: {excel_file}")
        
        orders, vehicles, drivers = create_scenario_from_excel(excel_file)
        orders = orders[:5]  # Take only first 5 orders for quick test
        vehicles = vehicles[:10]  # Take only first 10 vehicles
        
        print(f"   Testing with {len(orders)} orders and {len(vehicles)} vehicles")
        
        # Create parameters matching comprehensive test (with regret_k initialization)
        params = {
            'tabu_tenure': 50,
            'M1': 30,
            'M2': 200,
            'vehicle_penalty_per_vehicle': 0.0,
            'unassigned_order_base_penalty': 50000.0,
            'time_window_violation_penalty': 500000.0,
            'capacity_violation_penalty': 100.0,
            'distance_violation_penalty': 50.0,
            'Lo': 1000.0,
            'wk_ID': 60.0,
            'wk_IE': 60.0,
            'wk_IF': 40.0,
            'wk_IH': 30.0,
            'wk_IJ': 10.0,
            'M': 5000.0,
            'P_task': 500000.0,
            'initialization_method': 'regret_k',  # Use same as comprehensive test
            'regret_k_value': 3,
            'debug_regret': True
        }
        
        # Run l1_heuristic
        solution = l1_heuristic(orders, vehicles, params)
        
        print(f"   L1 heuristic returned solution with {len(solution.routes)} routes")
        
        # Check each route for depot structure
        depot_structure_issues = 0
        valid_routes = 0
        
        for vehicle_id, route in solution.routes.items():
            if route.tasks:
                first_task = route.tasks[0]
                last_task = route.tasks[-1]
                
                first_is_depot_start = 'DEPOT_START' in str(first_task.task_type)
                last_is_depot_return = 'DEPOT_RETURN' in str(last_task.task_type)
                
                print(f"   Route {vehicle_id}: {len(route.tasks)} tasks")
                print(f"     First: {first_task.task_type} ({first_task.id})")
                print(f"     Last: {last_task.task_type} ({last_task.id})")
                
                if first_is_depot_start and last_is_depot_return:
                    print("     ✅ Proper depot structure")
                    valid_routes += 1
                else:
                    print("     ❌ Missing depot structure")
                    depot_structure_issues += 1
            else:
                print(f"   Route {vehicle_id}: EMPTY")
        
        print(f"\\n   Summary:")
        print(f"     Valid routes: {valid_routes}")
        print(f"     Invalid routes: {depot_structure_issues}")
        
        if depot_structure_issues == 0:
            print("   🎉 All routes have proper depot structure!")
            return True
        else:
            print(f"   💥 {depot_structure_issues} routes missing depot structure!")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_l1_heuristic_depot_structure()
    if success:
        print("🎉 L1 heuristic depot structure test PASSED!")
    else:
        print("💥 L1 heuristic depot structure test FAILED!")
    
    exit(0 if success else 1)
