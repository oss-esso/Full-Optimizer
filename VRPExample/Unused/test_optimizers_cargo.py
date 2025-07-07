#!/usr/bin/env python3
"""
Test the updated VRP optimizers with cargo weight scenarios.
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from vrp_scenarios import get_all_scenarios
from vrp_optimizer_fixed import VRPQuantumOptimizer
from vrp_data_models import VRPObjective

def test_optimizers_with_cargo():
    """Test both optimizers with the updated cargo weight scenarios."""
    print("Testing VRP optimizers with trucking cargo scenarios")
    print("="*60)
    
    # Load scenarios
    scenarios = get_all_scenarios()
    
    # Test MODA_small
    print("\n--- Testing MODA_small with Cargo Weights ---")
    moda_small = scenarios['MODA_small']
    
    print(f"Vehicle capacity: {list(moda_small.vehicles.values())[0].capacity} kg")
    cargo_weights = [req.passengers for req in moda_small.ride_requests[:5]]
    print(f"Sample cargo weights: {cargo_weights} kg")
    
    # Test with OR-Tools
    optimizer = VRPQuantumOptimizer(moda_small, VRPObjective.MINIMIZE_DISTANCE)
    result = optimizer.optimize_with_ortools()
    
    print(f"OR-Tools result: {result.status}")
    print(f"Vehicles used: {result.metrics.get('vehicles_used', 0)}")
    print(f"Total distance: {result.metrics.get('total_distance', 0):.2f}")
    print(f"Runtime: {result.runtime:.2f}ms")
    
    # Test with quantum method
    try:
        quantum_result = optimizer.optimize_with_quantum_benders()
        print(f"Quantum result: {quantum_result.status}")
        print(f"Vehicles used: {quantum_result.metrics.get('vehicles_used', 0)}")
    except Exception as e:
        print(f"Quantum method error: {e}")
    
    # Test with classical method
    try:
        classical_result = optimizer.optimize_with_classical_benders()
        print(f"Classical result: {classical_result.status}")
        print(f"Vehicles used: {classical_result.metrics.get('vehicles_used', 0)}")
    except Exception as e:
        print(f"Classical method error: {e}")
    
    print("\n--- Summary ---")
    if result.status == "optimal":
        print("✅ MODA_small with cargo weights: SUCCESSFUL")
        print(f"   Capacity handling: {list(moda_small.vehicles.values())[0].capacity} kg per truck")
        print(f"   Cargo range: {min(cargo_weights)}-{max(cargo_weights)} kg per shipment")
        print(f"   Solution quality: {result.metrics.get('vehicles_used', 0)} vehicles, {result.metrics.get('total_distance', 0):.2f} distance")
    else:
        print("❌ MODA_small with cargo weights: FAILED")
        print(f"   Status: {result.status}")

if __name__ == "__main__":
    test_optimizers_with_cargo()
