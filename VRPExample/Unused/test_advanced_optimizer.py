#!/usr/bin/env python3
"""
Test the Advanced Initial Solution Optimizer integration
"""
import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from advanced_initial_solution_optimizer import AdvancedInitialSolutionOptimizer
from vrp_scenarios import get_all_scenarios
from vrp_data_models import VRPObjective

def test_advanced_optimizer():
    """Test the advanced optimizer with different methods."""
    print("Testing Advanced Initial Solution Optimizer")
    print("=" * 60)
    
    # Get scenarios
    scenarios = get_all_scenarios()
    
    # Test on MODA_small for quick testing
    test_scenario = "MODA_small"
    
    if test_scenario not in scenarios:
        print(f"Scenario {test_scenario} not found")
        return
    
    instance = scenarios[test_scenario]
    print(f"Testing scenario: {test_scenario}")
    print(f"  Locations: {len(instance.locations)}")
    print(f"  Vehicles: {len(instance.vehicles)}")
    print(f"  Ride requests: {len(instance.ride_requests) if instance.ride_requests else 0}")
    
    # Initialize optimizer
    optimizer = AdvancedInitialSolutionOptimizer(instance, VRPObjective.MINIMIZE_DISTANCE)
    
    if not optimizer.generator:
        print("  No customers found, skipping...")
        return
    
    print(f"  Converted to: {len(optimizer.customers)} customers, {len(optimizer.vehicles)} vehicles")
    
    # Test individual methods
    methods = [
        ("Nearest Neighbor", lambda: optimizer.optimize_nearest_neighbor(0.1)),
        ("Savings Algorithm", lambda: optimizer.optimize_savings_algorithm(0.12)),
        ("Firefly Algorithm", lambda: optimizer.optimize_firefly_algorithm(15, 0.12)),
        ("Greedy Insertion", lambda: optimizer.optimize_greedy_insertion(True, 0.1)),
    ]
    
    results = {}
    
    print(f"\nTesting individual methods:")
    print("-" * 40)
    
    for method_name, method_func in methods:
        try:
            print(f"  {method_name}...")
            result = method_func()
            
            results[method_name] = result
            
            if result.status == "optimal":
                print(f"    ✅ Distance: {result.objective_value:.3f}")
                print(f"    ✅ Runtime: {result.runtime:.1f}ms")
                print(f"    ✅ Vehicles used: {result.metrics.get('vehicles_used', 0)}")
                print(f"    ✅ Customers served: {result.metrics.get('customers_served', 0)}")
                print(f"    ✅ Capacity utilization: {result.metrics.get('avg_capacity_utilization', 0):.1%}")
                
                # Show first route
                if result.routes:
                    first_vehicle = list(result.routes.keys())[0]
                    first_route = result.routes[first_vehicle]
                    route_preview = " -> ".join(first_route[:4]) + ("..." if len(first_route) > 4 else "")
                    print(f"    ✅ Sample route: {route_preview}")
            else:
                print(f"    ❌ Status: {result.status}")
                if "error" in result.metrics:
                    print(f"    ❌ Error: {result.metrics['error']}")
                    
        except Exception as e:
            print(f"    ❌ Exception: {str(e)}")
            results[method_name] = None
    
    # Compare results
    print(f"\nMethod Comparison:")
    print("-" * 40)
    
    optimal_results = {name: res for name, res in results.items() 
                      if res and res.status == "optimal"}
    
    if optimal_results:
        best_distance = min(res.objective_value for res in optimal_results.values())
        
        for method_name, result in optimal_results.items():
            ratio = result.objective_value / best_distance
            print(f"  {method_name:20s}: {result.objective_value:7.3f} ({ratio:5.2f}x best) "
                  f"- {result.runtime:6.1f}ms")
    
    # Test best-of-multiple approach
    print(f"\nTesting Best-of-Multiple approach:")
    print("-" * 40)
    
    try:
        best_result = optimizer.optimize_best_of_multiple(4)
        
        if best_result.status == "optimal":
            print(f"  ✅ Best distance: {best_result.objective_value:.3f}")
            print(f"  ✅ Best method: {best_result.metrics.get('best_method', 'unknown')}")
            print(f"  ✅ Total runtime: {best_result.runtime:.1f}ms")
            print(f"  ✅ Methods tried: {best_result.metrics.get('total_methods_tried', 0)}")
            
            if "method_comparison" in best_result.metrics:
                print("  ✅ Method comparison:")
                for method, distance in best_result.metrics["method_comparison"].items():
                    print(f"       {method}: {distance:.3f}")
        else:
            print(f"  ❌ Status: {best_result.status}")
            
    except Exception as e:
        print(f"  ❌ Exception: {str(e)}")

def test_integration_with_vrp_main():
    """Test integration potential with main VRP system."""
    print(f"\n" + "=" * 60)
    print("Testing Integration with Main VRP System")
    print("=" * 60)
    
    scenarios = get_all_scenarios()
    
    # Test on different scenario types
    test_scenarios = ["MODA_small", "MODA_first"]
    
    for scenario_name in test_scenarios:
        if scenario_name not in scenarios:
            continue
            
        print(f"\nScenario: {scenario_name}")
        instance = scenarios[scenario_name]
        
        # Initialize advanced optimizer
        advanced_optimizer = AdvancedInitialSolutionOptimizer(instance, VRPObjective.MINIMIZE_DISTANCE)
        
        if not advanced_optimizer.generator:
            print("  No customers found, skipping...")
            continue
        
        # Test firefly algorithm specifically
        try:
            result = advanced_optimizer.optimize_firefly_algorithm(20, 0.12)
            
            if result.status == "optimal":
                print(f"  🔥 Firefly Algorithm Success:")
                print(f"     Distance: {result.objective_value:.2f}")
                print(f"     Vehicles: {result.metrics.get('vehicles_used', 0)}")
                print(f"     Runtime: {result.runtime:.1f}ms")
                print(f"     Routes generated: {len(result.routes)}")
                
                # Compare with a simple baseline
                baseline_result = advanced_optimizer.optimize_nearest_neighbor(0.0)
                if baseline_result.status == "optimal":
                    improvement = (baseline_result.objective_value - result.objective_value) / baseline_result.objective_value
                    print(f"     Improvement over Nearest Neighbor: {improvement:.1%}")
            else:
                print(f"  ❌ Firefly Algorithm failed: {result.status}")
                
        except Exception as e:
            print(f"  ❌ Exception: {str(e)}")

if __name__ == "__main__":
    test_advanced_optimizer()
    test_integration_with_vrp_main()
