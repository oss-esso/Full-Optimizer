"""
Test Large Order Assignment with Advanced Strategies

This test specifically validates the implementation of advanced order insertion strategies
for large/constrained orders, including:
- Regret-k Insertion Heuristic
- Destroy and Repair Operator

The test loads a scenario with known difficult orders and validates that the new
heuristics can successfully assign them where the basic initialization fails.
"""

import os
import sys
import time

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
src_dir = os.path.join(heuristic_root, 'src')
algo_dir = os.path.join(heuristic_root, 'algo')

sys.path.insert(0, heuristic_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, algo_dir)

def test_regret_k_vs_cluster_aware():
    """Test regret-k initialization vs cluster-aware initialization for large orders."""
    print("="*80)
    print("🧪 TESTING REGRET-K VS CLUSTER-AWARE INITIALIZATION")
    print("="*80)
    
    try:
        from scenario_creator import create_scenario_from_excel
        from first_level import l1_heuristic
        
        # Load scenario with problematic large orders
        excel_file = os.path.join(heuristic_root, 'src', 'furgoni.xlsx')
        orders, vehicles, drivers = create_scenario_from_excel(excel_file)
        
        print(f"📦 Loaded scenario: {len(orders)} orders, {len(vehicles)} vehicles, {len(drivers)} drivers")
        
        # Test 1: Cluster-aware initialization
        print("\n🏗️  Testing cluster-aware initialization...")
        params_cluster = {
            'tabu_tenure': 20,
            'M1': 10,
            'M2': 50,
            'initialization_method': 'cluster_aware',
            'enable_destroy_and_repair': False,
            'enable_force_assignment': False,
            'debug_assignment': False
        }
        
        start_time = time.time()
        solution_cluster = l1_heuristic(orders, vehicles, params_cluster)
        cluster_time = time.time() - start_time
        
        cluster_assigned = count_assigned_orders(solution_cluster, orders)
        print(f"✅ Cluster-aware result: {cluster_assigned}/{len(orders)} orders assigned ({cluster_assigned/len(orders)*100:.1f}%) in {cluster_time:.2f}s")
        
        # Test 2: Regret-k initialization
        print("\n🧠 Testing regret-k initialization...")
        params_regret = {
            'tabu_tenure': 20,
            'M1': 10,
            'M2': 50,
            'initialization_method': 'regret_k',
            'regret_k_value': 3,
            'enable_destroy_and_repair': False,
            'enable_force_assignment': False,
            'debug_regret': True,
            'debug_assignment': False
        }
        
        start_time = time.time()
        solution_regret = l1_heuristic(orders, vehicles, params_regret)
        regret_time = time.time() - start_time
        
        regret_assigned = count_assigned_orders(solution_regret, orders)
        print(f"✅ Regret-k result: {regret_assigned}/{len(orders)} orders assigned ({regret_assigned/len(orders)*100:.1f}%) in {regret_time:.2f}s")
        
        # Test 3: Regret-k + Destroy and Repair
        print("\n🔧 Testing regret-k + destroy and repair...")
        params_advanced = {
            'tabu_tenure': 20,
            'M1': 10,
            'M2': 50,
            'initialization_method': 'regret_k',
            'regret_k_value': 3,
            'enable_destroy_and_repair': True,
            'max_destroy_attempts': 3,
            'enable_force_assignment': False,
            'debug_regret': False,
            'debug_destroy_repair': True,
            'debug_assignment': False
        }
        
        start_time = time.time()
        solution_advanced = l1_heuristic(orders, vehicles, params_advanced)
        advanced_time = time.time() - start_time
        
        advanced_assigned = count_assigned_orders(solution_advanced, orders)
        print(f"✅ Advanced result: {advanced_assigned}/{len(orders)} orders assigned ({advanced_assigned/len(orders)*100:.1f}%) in {advanced_time:.2f}s")
        
        # Summary comparison
        print("\n📊 COMPARISON SUMMARY:")
        print("="*60)
        print(f"Cluster-aware:           {cluster_assigned:2d}/{len(orders)} ({cluster_assigned/len(orders)*100:5.1f}%) - {cluster_time:6.2f}s")
        print(f"Regret-k:                {regret_assigned:2d}/{len(orders)} ({regret_assigned/len(orders)*100:5.1f}%) - {regret_time:6.2f}s")
        print(f"Regret-k + Destroy&Repair: {advanced_assigned:2d}/{len(orders)} ({advanced_assigned/len(orders)*100:5.1f}%) - {advanced_time:6.2f}s")
        
        # Determine the winner
        improvements = []
        if regret_assigned > cluster_assigned:
            improvements.append(f"Regret-k improved by {regret_assigned - cluster_assigned} orders")
        if advanced_assigned > regret_assigned:
            improvements.append(f"Destroy&Repair improved by {advanced_assigned - regret_assigned} more orders")
        if advanced_assigned > cluster_assigned:
            improvements.append(f"Total improvement: {advanced_assigned - cluster_assigned} orders")
        
        if improvements:
            print("\n🎯 IMPROVEMENTS DETECTED:")
            for improvement in improvements:
                print(f"  ✅ {improvement}")
        else:
            print("\n⚠️  No significant improvements detected")
        
        # Analyze unassigned orders
        analyze_unassigned_orders(solution_advanced, orders)
        
        return advanced_assigned == len(orders)  # Return True if 100% assignment achieved
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def count_assigned_orders(solution, orders):
    """Count how many orders are assigned in the solution."""
    assigned_order_ids = set()
    
    if hasattr(solution, 'routes') and solution.routes:
        for route in solution.routes.values():
            if route and hasattr(route, 'tasks') and route.tasks:
                for task in route.tasks:
                    if hasattr(task, 'order_id') and task.order_id:
                        order_id = task.order_id
                        if not ('depot' in str(order_id).lower()):
                            assigned_order_ids.add(order_id)
    
    return len(assigned_order_ids)


def analyze_unassigned_orders(solution, orders):
    """Analyze which orders remain unassigned and why."""
    assigned_order_ids = set()
    
    if hasattr(solution, 'routes') and solution.routes:
        for route in solution.routes.values():
            if route and hasattr(route, 'tasks') and route.tasks:
                for task in route.tasks:
                    if hasattr(task, 'order_id') and task.order_id:
                        order_id = task.order_id
                        if not ('depot' in str(order_id).lower()):
                            assigned_order_ids.add(order_id)
    
    unassigned_orders = [order for order in orders if order.id not in assigned_order_ids]
    
    if unassigned_orders:
        print(f"\n❌ UNASSIGNED ORDERS ANALYSIS ({len(unassigned_orders)} orders):")
        print("-" * 60)
        
        for order in unassigned_orders:
            try:
                weight = order.get_total_demand()
                volume = order.get_total_volume()
                pallets = getattr(order, 'total_pallets', 0)
                print(f"  • {order.id}: {weight:.1f}kg, {volume:.2f}m³, {pallets:.0f} pallets")
            except Exception as e:
                print(f"  • {order.id}: [Error analyzing: {e}]")
    else:
        print("\n✅ ALL ORDERS SUCCESSFULLY ASSIGNED!")


def test_specific_difficult_orders():
    """Test specific known difficult orders from the scenario."""
    print("\n" + "="*80)
    print("🎯 TESTING SPECIFIC DIFFICULT ORDERS")
    print("="*80)
    
    try:
        from scenario_creator import create_scenario_from_excel
        from first_level import regret_k_initializer
        
        # Load scenario
        excel_file = os.path.join(heuristic_root, 'src', 'furgoni.xlsx')
        orders, vehicles, drivers = create_scenario_from_excel(excel_file)
        
        # Identify the most difficult orders
        difficult_orders = []
        for order in orders:
            try:
                weight = order.get_total_demand()
                if weight > 20000:  # Orders over 20 tons
                    difficult_orders.append((order, weight))
            except:
                continue
        
        difficult_orders.sort(key=lambda x: x[1], reverse=True)
        
        print(f"🏋️  Found {len(difficult_orders)} heavy orders (>20 tons):")
        for order, weight in difficult_orders[:5]:  # Show top 5
            print(f"  • {order.id}: {weight:.1f}kg")
        
        if difficult_orders:
            # Test regret-k specifically on these orders
            params = {
                'regret_k_value': 2,
                'debug_regret': True
            }
            
            print(f"\n🧠 Testing regret-k initialization on {len(difficult_orders)} difficult orders...")
            solution = regret_k_initializer(orders, vehicles, params)
            
            # Check assignment success for difficult orders
            assigned_difficult = 0
            for order, weight in difficult_orders:
                if is_order_assigned(solution, order):
                    assigned_difficult += 1
                    print(f"  ✅ {order.id} ({weight:.1f}kg) - ASSIGNED")
                else:
                    print(f"  ❌ {order.id} ({weight:.1f}kg) - UNASSIGNED")
            
            success_rate = assigned_difficult / len(difficult_orders) * 100
            print(f"\n📊 Difficult order assignment: {assigned_difficult}/{len(difficult_orders)} ({success_rate:.1f}%)")
            
            return success_rate > 50  # Success if >50% of difficult orders assigned
        else:
            print("⚠️  No heavy orders found in scenario")
            return True
            
    except Exception as e:
        print(f"❌ Specific order test failed: {e}")
        return False


def is_order_assigned(solution, order):
    """Check if a specific order is assigned in the solution."""
    if not hasattr(solution, 'routes') or not solution.routes:
        return False
    
    for route in solution.routes.values():
        if route and hasattr(route, 'tasks') and route.tasks:
            for task in route.tasks:
                if hasattr(task, 'order_id') and task.order_id == order.id:
                    return True
    return False


def main():
    """Run all large order assignment tests."""
    print("🧪 LARGE ORDER ASSIGNMENT TEST SUITE")
    print("="*80)
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Regret-k vs Cluster-aware comparison
    print("Test 1: Algorithm Comparison")
    if test_regret_k_vs_cluster_aware():
        tests_passed += 1
        print("✅ Test 1 PASSED")
    else:
        print("❌ Test 1 FAILED")
    
    # Test 2: Specific difficult orders
    print("\nTest 2: Difficult Order Handling")
    if test_specific_difficult_orders():
        tests_passed += 1
        print("✅ Test 2 PASSED")
    else:
        print("❌ Test 2 FAILED")
    
    # Final summary
    print("\n" + "="*80)
    print(f"🎯 TEST SUMMARY: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED - Advanced order insertion strategies working!")
    else:
        print("⚠️  SOME TESTS FAILED - Advanced strategies may need refinement")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
