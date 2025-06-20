#!/usr/bin/env python3
"""
Test basic VRP without any ride requests to see if that's the issue.
"""

import os
import sys

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from vrp_data_models import VRPInstance, Location, Vehicle
from vrp_optimizer_fixed import VRPQuantumOptimizer
from vrp_data_models import VRPObjective

def test_basic_vrp():
    """Test a basic VRP without ride requests."""
    print("Testing basic VRP without ride requests...")
    
    instance = VRPInstance("Basic VRP Test")
    
    # Add depot
    depot = Location("depot", 0, 0, demand=0, 
                    time_window_start=0, time_window_end=1440, service_time=0)
    instance.add_location(depot)
    
    # Add customer with demand
    customer = Location("customer_1", 10, 10, demand=1,
                       time_window_start=0, time_window_end=1440, service_time=30)
    instance.add_location(customer)
    
    # Add vehicle
    vehicle = Vehicle("truck", capacity=10, depot_id="depot", max_time=600)
    instance.add_vehicle(vehicle)
    
    # NO ride requests - this should be standard VRP
    
    instance.calculate_distance_matrix(distance_method="manhattan")
    
    print(f"Instance has ride_requests: {bool(instance.ride_requests)}")
    print(f"Number of ride_requests: {len(instance.ride_requests) if instance.ride_requests else 0}")
    
    # Test with OR-Tools
    optimizer = VRPQuantumOptimizer(instance, VRPObjective.MINIMIZE_DISTANCE)
    result = optimizer.optimize_with_ortools()
    
    print(f"\nResult: {result.status}")
    print(f"Solved: {'✅ YES' if result.status == 'optimal' else '❌ NO'}")
    
    if result.routes:
        print("Routes:")
        for vehicle_id, route in result.routes.items():
            print(f"  {vehicle_id}: {' -> '.join(route)}")
    
    return result.status == "optimal"

if __name__ == "__main__":
    success = test_basic_vrp()
    print(f"\nBasic VRP test: {'✅ PASS' if success else '❌ FAIL'}")
