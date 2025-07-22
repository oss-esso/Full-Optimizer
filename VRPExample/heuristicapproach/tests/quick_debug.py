#!/usr/bin/env python3
"""
Simple debug script to test one route from the mock solution
"""
import sys
import os

# Add path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, parent_dir)

# Temporarily monkey patch the is_feasible function to add debug output
def monkey_patch_feasibility_function():
    from algo import second_level
    
    original_simulate_hos = second_level._simulate_hos_advanced
    
    def debug_simulate_hos_advanced(route, driver_state, sorted_tasks):
        print(f"DEBUG HoS: Starting simulation with {len(sorted_tasks)} tasks")
        try:
            result = original_simulate_hos(route, driver_state, sorted_tasks)
            print(f"DEBUG HoS: Result = {result}")
            return result
        except Exception as e:
            print(f"DEBUG HoS: Exception occurred: {e}")
            import traceback
            traceback.print_exc()
            return False, 0
    
    # Replace the function
    second_level._simulate_hos_advanced = debug_simulate_hos_advanced
    
    original_is_feasible = second_level.is_feasible
    
    def debug_is_feasible(route):
        print(f"\nDEBUG: Testing route with {len(route.tasks)} tasks")
        
        try:
            # Call original function but catch where it fails
            result = original_is_feasible(route)
            print(f"DEBUG: is_feasible returned {result}")
            return result
        except Exception as e:
            print(f"DEBUG: is_feasible threw exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    second_level.is_feasible = debug_is_feasible

def main():
    monkey_patch_feasibility_function()
    
    try:
        # Import after patching
        from algo.second_level import is_feasible
        from run_scenario_test import _create_mock_solution
        from src.data_adapter import convert_instance_to_epdt_input
        from src.moda_scenarios import create_furgoni_scenario
        
        print("Creating test scenario...")
        vrp_instance = create_furgoni_scenario()
        orders, vehicles, params = convert_instance_to_epdt_input(vrp_instance)
        solution = _create_mock_solution(orders, vehicles)
        
        # Test first route
        for vehicle_id, route in solution.routes.items():
            if route.tasks:
                print(f"\nTesting {vehicle_id} with {len(route.tasks)} tasks")
                result = is_feasible(route)
                print(f"Final result: {result}")
                break
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
