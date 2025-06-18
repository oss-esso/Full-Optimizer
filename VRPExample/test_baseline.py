#!/usr/bin/env python3
"""
Quick baseline test of current OR-Tools implementation
"""
import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from vrp_scenarios import get_all_scenarios
from vrp_optimizer_fixed import VRPQuantumOptimizer
from vrp_data_models import VRPObjective

def test_baseline():
    """Test current OR-Tools implementation"""
    print("Current OR-Tools Implementation Baseline Test")
    print("=" * 50)
    
    scenarios = get_all_scenarios()
    moda_small = scenarios['MODA_small']
    
    print(f'MODA_small: {len(moda_small.vehicles)} vehicles, {len(moda_small.ride_requests)} requests')
    
    optimizer = VRPQuantumOptimizer(moda_small, VRPObjective.MINIMIZE_DISTANCE)
    result = optimizer.optimize_with_ortools()
    
    vehicles_used = len([r for r in result.routes.values() if len(r) > 2])
    total_distance = result.metrics.get('total_distance', 0)
    
    print(f'Vehicles used: {vehicles_used}/{len(moda_small.vehicles)}')
    print(f'Distance: {total_distance:.2f}')
    print(f'Runtime: {result.runtime:.0f}ms')
    print(f'Objective: {result.objective_value:.2f}')
    
    print('\nRoute details:')
    for vehicle_id, route in result.routes.items():
        if len(route) > 2:
            requests = (len(route) - 2) // 2
            print(f'  {vehicle_id}: {len(route)} stops ({requests} requests)')

if __name__ == "__main__":
    test_baseline()
