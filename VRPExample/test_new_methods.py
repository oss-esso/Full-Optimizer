#!/usr/bin/env python3
"""
Simple test script to verify the new optimization methods work correctly.
"""

from vrp_scenarios import get_all_scenarios
from vrp_optimizer_fixed import VRPQuantumOptimizer
from vrp_data_models import VRPObjective

def test_new_methods():
    """Test the new optimization methods."""
    print("Testing new optimization methods...")
    
    # Get a simple scenario
    scenarios = get_all_scenarios()
    scenario_name = list(scenarios.keys())[0]
    print(f"Testing with scenario: {scenario_name}")
    
    instance = scenarios[scenario_name]
    optimizer = VRPQuantumOptimizer(instance, VRPObjective.MINIMIZE_DISTANCE)
    
    # Test advanced heuristics
    print("\n1. Testing advanced heuristics...")
    try:
        result1 = optimizer.optimize_with_advanced_heuristics()
        distance1 = result1.metrics.get("total_distance", 0)
        runtime1 = result1.runtime
        print(f"   ✓ Advanced heuristics: distance={distance1:.2f}, runtime={runtime1:.2f}ms")
    except Exception as e:
        print(f"   ✗ Advanced heuristics failed: {e}")
    
    # Test 2-opt improvement
    print("\n2. Testing 2-opt improvement...")
    try:
        result2 = optimizer.optimize_with_2opt_improvement()
        distance2 = result2.metrics.get("total_distance", 0)
        runtime2 = result2.runtime
        print(f"   ✓ 2-opt improvement: distance={distance2:.2f}, runtime={runtime2:.2f}ms")
    except Exception as e:
        print(f"   ✗ 2-opt improvement failed: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_new_methods()
