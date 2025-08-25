"""
Debug script to verify HoS validation is working correctly.

This will help us understand why routes with HoS violations are being accepted
during the optimization process.
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

def test_hos_validation():
    """Test HoS validation on a simple route."""
    print("=== HoS Validation Debug Test ===")
    
    try:
        # Import scenario creation
        from scenario_creator import create_scenario_from_excel
        
        # Load the scenario
        excel_path = os.path.join(heuristic_root, 'src', 'furgoni3_2.xlsx')
        print(f"Loading scenario from: {excel_path}")
        
        orders, vehicles, drivers = create_scenario_from_excel(excel_path)
        print(f"Loaded: {len(orders)} orders, {len(vehicles)} vehicles")
        
        # Import L2 heuristic and feasibility functions
        from second_level import l2_heuristic, is_feasible
        from hos_simulation import build_compliant_timeline
        
        # Take first order and try to assign it to first vehicle
        test_order = orders[0]
        test_vehicle = vehicles[0]
        
        print(f"Testing order {test_order.id} on vehicle {test_vehicle.id}")
        
        # Try to create a route
        test_route = l2_heuristic(test_order, test_vehicle, debug_insertion=True)
        
        if test_route:
            print(f"Route created with {len(test_route.tasks)} tasks")
            
            # Test feasibility check
            print("\nTesting is_feasible function...")
            feasible = is_feasible(test_route, debug_feasibility=True, return_reason=True)
            print(f"Feasibility result: {feasible}")
            
            # Test HoS directly
            print("\nTesting HoS simulation directly...")
            try:
                timeline, rest_costs = build_compliant_timeline(test_route)
                print(f"HoS timeline created with {len(timeline)} events, rest cost: {rest_costs}")
            except Exception as e:
                print(f"HoS simulation failed: {e}")
                import traceback
                traceback.print_exc()
                
        else:
            print("Failed to create route")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hos_validation()
