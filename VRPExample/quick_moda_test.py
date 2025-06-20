#!/usr/bin/env python3
"""Quick test for MODA scenarios with fixed quantum solver."""

from vrp_scenarios import get_all_scenarios
from vrp_optimizer import VRPQuantumOptimizer

def test_moda_quantum():
    """Test MODA scenarios with quantum solver."""
    scenarios = get_all_scenarios()
    
    print("=== Quick MODA Quantum Test ===")
    
    # Test just MODA scenarios
    for name, scenario in scenarios.items():
        if name.startswith('MODA'):
            print(f"\nTesting {name}...")
            print(f"  Vehicles: {len(scenario.vehicles)}")
            print(f"  Requests: {len(scenario.ride_requests)}")
            
            # Test quantum solver
            optimizer = VRPQuantumOptimizer(scenario)
            result = optimizer.optimize_with_quantum_benders()
            
            print(f"  Quantum: status={result.status}, objective={result.objective_value:.2f}")
            print(f"  Vehicles used: {result.metrics.get('vehicles_used', 0)}")

if __name__ == "__main__":
    test_moda_quantum()
