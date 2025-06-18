#!/usr/bin/env python3
"""
Test the integrated VRP pipeline with all 5 methods
"""

import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from vrp_scenarios import get_all_scenarios
from vrp_optimizer_fixed import VRPQuantumOptimizer
from vrp_data_models import VRPObjective

def test_integrated_pipeline():
    """Test the integrated pipeline with all optimization methods."""
    print("Testing integrated VRP pipeline with all 5 methods...")
    
    # Get scenarios
    scenarios = get_all_scenarios()
      # Test with a simple scenario
    test_scenario = 'small_delivery'
    if test_scenario not in scenarios:
        print(f"Test scenario '{test_scenario}' not found. Available scenarios:")
        for name in scenarios.keys():
            print(f"  - {name}")
        return
    
    instance = scenarios[test_scenario]
    print(f"\nTesting with scenario: {test_scenario}")
    print(f"Locations: {len(instance.location_ids)}, Vehicles: {len(instance.vehicles)}")
    
    # Create optimizer
    optimizer = VRPQuantumOptimizer(instance, VRPObjective.MINIMIZE_DISTANCE)
    
    print("\nRunning all optimization methods...")
    
    # Test all methods
    results = {}
    
    try:
        print("1. Quantum-Enhanced Benders...")
        results['quantum'] = optimizer.optimize_with_quantum_benders()
        print(f"   ✓ Distance: {results['quantum'].metrics.get('total_distance', 0):.2f}, Runtime: {results['quantum'].runtime:.2f}ms")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    try:
        print("2. Classical Benders...")
        results['classical'] = optimizer.optimize_with_classical_benders()
        print(f"   ✓ Distance: {results['classical'].metrics.get('total_distance', 0):.2f}, Runtime: {results['classical'].runtime:.2f}ms")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    try:
        print("3. OR-Tools...")
        results['ortools'] = optimizer.optimize_with_ortools()
        print(f"   ✓ Distance: {results['ortools'].metrics.get('total_distance', 0):.2f}, Runtime: {results['ortools'].runtime:.2f}ms")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    try:
        print("4. Advanced Heuristics...")
        results['advanced'] = optimizer.optimize_with_advanced_heuristics()
        print(f"   ✓ Distance: {results['advanced'].metrics.get('total_distance', 0):.2f}, Runtime: {results['advanced'].runtime:.2f}ms")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    try:
        print("5. 2-opt Improvement...")
        results['twopt'] = optimizer.optimize_with_2opt_improvement()
        print(f"   ✓ Distance: {results['twopt'].metrics.get('total_distance', 0):.2f}, Runtime: {results['twopt'].runtime:.2f}ms")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print(f"\nSuccessfully tested {len(results)}/5 methods")
    
    # Find best method
    if results:
        best_method = min(results.items(), key=lambda x: x[1].metrics.get('total_distance', float('inf')))
        print(f"Best method: {best_method[0]} with distance {best_method[1].metrics.get('total_distance', 0):.2f}")
    
    print("\nIntegration test completed!")

if __name__ == "__main__":
    test_integrated_pipeline()
