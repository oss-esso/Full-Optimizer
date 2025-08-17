"""
Test script to verify enhanced assignment failure logging works for any unassigned order
"""
import sys
import os

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
src_dir = os.path.join(heuristic_root, 'src')
algo_dir = os.path.join(heuristic_root, 'algo')

sys.path.insert(0, heuristic_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, algo_dir)

print("=== Enhanced Assignment Logging Test ===")

try:
    # Import required modules
    try:
        from src.scenario_creator import create_scenario_from_excel
    except ImportError:
        from scenario_creator import create_scenario_from_excel
        
    try:
        from algo.first_level import l1_heuristic
    except ImportError:
        from first_level import l1_heuristic
    
    # Load the scenario
    excel_path = os.path.join(heuristic_root, 'src', 'furgoni2.xlsx')
    
    print(f"Loading scenario from: {excel_path}")
    scenario = create_scenario_from_excel(excel_path)
    
    print(f"Loaded {len(scenario.orders)} orders and {len(scenario.vehicles)} vehicles")
    
    # Configure parameters to enable enhanced logging
    params = {
        'initialization_method': 'regret_k',
        'regret_k_value': 3,
        'debug_regret': True,  # Enable regret debugging 
        'enable_enhanced_validation': True,
        'enable_force_assignment': False,  # Disable to see natural assignment failures
        'enable_destroy_and_repair': False,  # Disable to see natural assignment failures
    }
    
    print("Running l1_heuristic with enhanced assignment failure logging enabled...")
    print("This will show detailed diagnostics for any orders that cannot be assigned")
    print("-" * 80)
    
    # Run optimization
    solution = l1_heuristic(scenario.orders, scenario.vehicles, params)
    
    print("-" * 80)
    print(f"Optimization completed.")
    print(f"Solution routes: {len(solution.routes)}")
    
    # Count assigned vs unassigned orders
    assigned_orders = set()
    for route in solution.routes.values():
        for task in route.tasks:
            if hasattr(task, 'order') and task.order and hasattr(task.order, 'id'):
                assigned_orders.add(task.order.id)
    
    total_orders = len(scenario.orders)
    assigned_count = len(assigned_orders)
    unassigned_count = total_orders - assigned_count
    
    print(f"Assignment Summary:")
    print(f"  Total orders: {total_orders}")
    print(f"  Assigned orders: {assigned_count} ({assigned_count/total_orders*100:.1f}%)")
    print(f"  Unassigned orders: {unassigned_count}")
    
    if hasattr(solution, 'unassigned_orders') and solution.unassigned_orders:
        print(f"  Unassigned order IDs: {list(solution.unassigned_orders)}")

except Exception as e:
    print(f"Error during test: {e}")
    import traceback
    traceback.print_exc()
