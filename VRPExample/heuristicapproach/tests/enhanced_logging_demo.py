#!/usr/bin/env python3
"""
Focused test to demonstrate enhanced logging with forced assignment failures.
"""

import sys
import os
from pathlib import Path

# Add paths for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / ".." / "utils"))
sys.path.insert(0, str(current_dir / ".." / "algo"))

print("=== Enhanced Logging Demo with Assignment Failures ===")

try:
    from scenario_creator import create_scenario_from_excel
    from first_level import l1_heuristic
    
    def test_enhanced_logging_demo():
        """Create a scenario that will cause assignment failures to demonstrate enhanced logging."""
        
        # Use existing sample Excel file
        excel_file = "../sample_new_format.xlsx"
        if not Path(excel_file).exists():
            print(f"❌ Excel file not found: {excel_file}")
            return False
        
        print(f"📊 Loading scenario from {excel_file}")
        
        # Create scenario
        orders, vehicles, drivers = create_scenario_from_excel(excel_file)
        
        print(f"✅ Scenario loaded:")
        print(f"   - Orders: {len(orders)}")
        print(f"   - Vehicles: {len(vehicles)}")
        print(f"   - Drivers: {len(drivers)}")
        
        # Deliberately constrain the scenario to force assignment failures
        print(f"\n🎯 Constraining scenario to demonstrate enhanced logging...")
        
        # Reduce vehicle capacity drastically to force failures
        for vehicle in vehicles:
            vehicle.weight_capacity = 50.0  # Very low capacity
            vehicle.volume_capacity = 0.1   # Very low volume
        
        # Set up parameters with enhanced logging enabled
        params = {
            'initialization_method': 'regret_k',
            'regret_k_value': 3,
            'enhanced_logging': True,  # ENABLE ENHANCED LOGGING
            'tabu_tenure': 10,
            'M1': 5,
            'M2': 10,
            'exploration_strategy': 'vnd',
            'enable_advanced_neighborhoods': False,
            'enable_granular_search': False,
            'enable_parallelization': False,
            'local_search_strategy': 'first_improvement',
            'vehicle_penalty_per_vehicle': 0.0,
            'unassigned_order_base_penalty': 50000.0,
            'time_window_violation_penalty': 500000.0,
            'capacity_violation_penalty': 100.0,
            'distance_violation_penalty': 50.0
        }
        
        print(f"\n🔍 Running L1 heuristic with severely constrained vehicles...")
        print(f"   (Vehicle capacities reduced to 50kg/0.1m³ to force failures)")
        
        try:
            solution = l1_heuristic(orders, vehicles, params)
            
            # Analyze the solution
            if solution and solution.routes:
                active_routes = len([r for r in solution.routes.values() if r.tasks])
                print(f"\n📈 Results with constrained vehicles:")
                print(f"   - Active routes: {active_routes}")
                print(f"   - Total vehicles: {len(vehicles)}")
                
                # Check for orders in routes
                assigned_order_ids = set()
                for vehicle_id, route in solution.routes.items():
                    if route.tasks:
                        for task in route.tasks:
                            if hasattr(task, 'order_id') and task.order_id:
                                assigned_order_ids.add(task.order_id)
                
                all_order_ids = {order.id for order in orders}
                unassigned_orders = all_order_ids - assigned_order_ids
                
                print(f"   - Assigned orders: {len(assigned_order_ids)}")
                print(f"   - Unassigned orders: {len(unassigned_orders)}")
                
                if unassigned_orders:
                    print(f"\n⚠️  Unassigned orders (with enhanced diagnostics above):")
                    for order_id in sorted(unassigned_orders):
                        print(f"   - {order_id}")
                else:
                    print(f"\n✅ All orders assigned despite constraints!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during L1 heuristic: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Run the demo
    success = test_enhanced_logging_demo()
    
    if success:
        print(f"\n🎉 Enhanced logging demonstration completed!")
        print(f"   Check the detailed diagnostic output above for assignment failures.")
    else:
        print(f"\n❌ Enhanced logging demonstration failed!")
        
except Exception as e:
    print(f"Error during test: {e}")
    import traceback
    traceback.print_exc()
