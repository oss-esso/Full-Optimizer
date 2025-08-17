#!/usr/bin/env python3
"""
Test the enhanced assignment logging functionality for unassigned orders.
"""

import sys
import os
from pathlib import Path

# Add paths for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "utils"))
sys.path.insert(0, str(current_dir / "algo"))

print("=== Enhanced Assignment Logging Test ===")

try:
    # Import required modules
    from scenario_creator import create_scenario_from_excel
    from first_level import l1_heuristic
    
    def test_enhanced_logging():
        """Test the enhanced logging functionality."""
        
        # Use existing sample Excel file
        excel_file = "sample_new_format.xlsx"
        if not Path(excel_file).exists():
            print(f"❌ Excel file not found: {excel_file}")
            return False
        
        print(f"📊 Testing enhanced logging with {excel_file}")
        
        # Create scenario
        orders, vehicles, drivers = create_scenario_from_excel(excel_file)
        
        print(f"✅ Scenario loaded:")
        print(f"   - Orders: {len(orders)}")
        print(f"   - Vehicles: {len(vehicles)}")
        print(f"   - Drivers: {len(drivers)}")
        
        # Run L1 heuristic with enhanced logging enabled
        print("\n🔍 Running L1 heuristic with enhanced logging...")
        
        # Set up parameters with enhanced logging enabled
        params = {
            'initialization_method': 'regret_k',
            'regret_k_value': 3,
            'enhanced_logging': True,  # Enable our enhanced logging
            'tabu_tenure': 10,  # Required for L1 heuristic
            'M1': 5,  # Required for L1 heuristic
            'M2': 10,  # Required for L1 heuristic
            'exploration_strategy': 'vnd',
            'enable_advanced_neighborhoods': False,
            'enable_granular_search': False,
            'enable_parallelization': False,
            'local_search_strategy': 'first_improvement',
            'max_iterations': 100,  # Maximum iterations for tabu search
            'vehicle_penalty_per_vehicle': 0.0,
            'unassigned_order_base_penalty': 50000.0,
            'time_window_violation_penalty': 500000.0,
            'capacity_violation_penalty': 100.0,
            'distance_violation_penalty': 50.0
        }
        
        try:
            solution = l1_heuristic(orders, vehicles, params)
            
            # Analyze the solution
            assigned_orders = []
            unassigned_orders = []
            active_routes = 0
            
            if solution and solution.routes:
                active_routes = len([r for r in solution.routes.values() if r.tasks])
                
                # Check which orders were assigned
                for vehicle_id, route in solution.routes.items():
                    if route.tasks:
                        for task in route.tasks:
                            if hasattr(task, 'order_id') and task.order_id:
                                assigned_orders.append(task.order_id)
                
                # Find unassigned orders
                all_order_ids = [order.id for order in orders]
                unassigned_orders = [order_id for order_id in all_order_ids if order_id not in assigned_orders]
            
            print(f"\n📈 Assignment Results:")
            print(f"   - Assigned orders: {len(set(assigned_orders))}")
            print(f"   - Active routes: {active_routes}")
            print(f"   - Unassigned orders: {len(unassigned_orders)}")
            
            if unassigned_orders:
                print(f"\n⚠️  Unassigned orders with enhanced diagnostics:")
                for order_id in unassigned_orders:
                    print(f"   - Order {order_id}")
                print("\n(Check console output above for detailed diagnostic information)")
            else:
                print(f"\n✅ All orders were successfully assigned!")
                
            return True
            
        except Exception as e:
            print(f"❌ Error during L1 heuristic: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Run the test
    success = test_enhanced_logging()
    
    if success:
        print(f"\n✅ Enhanced logging test completed successfully!")
    else:
        print(f"\n❌ Enhanced logging test failed!")
        
except Exception as e:
    print(f"Error during test: {e}")
    import traceback
    traceback.print_exc()
