#!/usr/bin/env python3
"""
Quick test to verify time constraints are working properly.
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

def test_time_constraints():
    """Test that time constraints are properly enforced."""
    print("🔬 VERIFYING TIME CONSTRAINT ENFORCEMENT")
    print("=" * 50)
    
    scenarios = get_all_scenarios()
    
    # Test MODA_small first (should be quick)
    instance = scenarios['MODA_small']
    
    print(f"📊 Testing MODA_small:")
    print(f"  Vehicles: {len(instance.vehicles)}")
    print(f"  Vehicle time limits: {[v.max_time for v in instance.vehicles.values()]}")
    
    # Get vehicle time limit
    vehicle_time_limit = list(instance.vehicles.values())[0].max_time  # 600 minutes
    
    try:
        optimizer = VRPQuantumOptimizer(instance, VRPObjective.MINIMIZE_DISTANCE)
        result = optimizer.optimize_with_ortools()
        
        if result.routes:
            print(f"✅ Solution found with {len([r for r in result.routes.values() if len(r) > 2])} vehicles")
            
            # Check if routes respect time constraints
            violations = 0
            for vehicle_id, route in result.routes.items():
                if len(route) > 2:  # More than just depot -> depot
                    # Estimate route time (rough calculation)
                    route_stops = len(route) - 2  # Exclude start and end depot
                    estimated_service_time = route_stops * 10  # Assume 10 min per stop
                    estimated_travel_time = result.objective_value * 2 / len(result.routes)  # Rough estimate
                    total_estimated_time = estimated_service_time + estimated_travel_time
                    
                    print(f"  {vehicle_id}: {route_stops} stops, ~{total_estimated_time:.0f} min estimated")
                    
                    if total_estimated_time > vehicle_time_limit:
                        violations += 1
                        print(f"    ⚠️  Potential time violation!")
            
            if violations == 0:
                print(f"✅ All routes appear to respect the {vehicle_time_limit}-minute time limit")
            else:
                print(f"⚠️  {violations} potential time limit violations detected")
        else:
            print("❌ No solution found")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_time_constraints()
