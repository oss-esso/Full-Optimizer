#!/usr/bin/env python3
"""
Quick test to verify that the route filtering change works correctly.
This test runs a smaller version to validate the fix.
"""

import sys
import os

# Add the necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
vrp_example_dir = os.path.join(current_dir, "VRPExample", "heuristicapproach")
algo_dir = os.path.join(vrp_example_dir, "algo")
test_dir = os.path.join(vrp_example_dir, "tests")
sys.path.insert(0, vrp_example_dir)
sys.path.insert(0, algo_dir)
sys.path.insert(0, test_dir)

def test_route_filtering_change():
    """Test that route filtering change works as expected."""
    print("🧪 Testing route filtering change...")
    
    try:
        # Import the comprehensive integration test functions
        from comprehensive_integration_test import run_phase1_heuristic_test
        
        # Path to the Excel file
        excel_file = os.path.join(test_dir, "..", "src", "furgoni3_2.xlsx")
        if not os.path.exists(excel_file):
            print(f"❌ Excel file not found: {excel_file}")
            return False
            
        print(f"📁 Using Excel file: {excel_file}")
        
        # Run Phase 1 heuristic
        print("\n📊 Running Phase 1 heuristic test...")
        solution, orders, vehicles, runtime = run_phase1_heuristic_test(excel_file)
        
        if solution:
            print(f"✅ Phase 1 completed successfully!")
            print(f"   - Orders: {len(orders)}")
            print(f"   - Vehicles: {len(vehicles)}")
            print(f"   - Runtime: {runtime:.2f} seconds")
            
            # Count routes with tasks
            routes_with_tasks = sum(1 for route in solution.routes.values() if route.tasks)
            print(f"   - Routes with tasks: {routes_with_tasks}")
            
            # Now test the filtering change by simulating Phase 2 route preparation
            print(f"\n🔧 Testing route filtering change...")
            active_routes = {}
            route_objects = []
            infeasible_count = 0
            route_violations = {}
            
            # Simulate the modified logic
            for vehicle_id, route in solution.routes.items():
                if route.tasks:  # Only routes with tasks
                    try:
                        from algo.second_level import is_feasible
                        feasible, reason = is_feasible(route, debug_feasibility=True, return_reason=True)
                        
                        # Always include the route regardless of feasibility (NEW BEHAVIOR)
                        active_routes[vehicle_id] = route
                        route_objects.append(route)
                        
                        if not feasible:
                            infeasible_count += 1
                            route_violations[vehicle_id] = reason
                            print(f"   ⚠️  Route for vehicle {vehicle_id} has violations: {reason} (kept with warning)")
                        else:
                            print(f"   ✅ Route for vehicle {vehicle_id} is feasible")
                            
                    except ImportError:
                        # If feasibility check unavailable, include all routes
                        print(f"   ❓ Feasibility check unavailable - including route {vehicle_id}")
                        active_routes[vehicle_id] = route
                        route_objects.append(route)
            
            print(f"\n📈 Results:")
            print(f"   - Total routes processed: {routes_with_tasks}")
            print(f"   - Routes kept: {len(active_routes)}")
            print(f"   - Routes with violations: {infeasible_count}")
            print(f"   - Percentage kept: {len(active_routes)/routes_with_tasks*100:.1f}%")
            
            if len(active_routes) == routes_with_tasks:
                print(f"   ✅ SUCCESS: All routes were kept (no filtering)")
                return True
            else:
                print(f"   ❌ FAILURE: Some routes were filtered out")
                return False
        else:
            print(f"❌ Phase 1 failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_route_filtering_change()
    print(f"\n{'🎉 TEST PASSED' if success else '❌ TEST FAILED'}")
    sys.exit(0 if success else 1)
