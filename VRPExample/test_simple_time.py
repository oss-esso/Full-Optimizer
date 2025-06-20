#!/usr/bin/env python3
"""
Ultra-simple test to understand the time constraint interpretation in OR-Tools.
"""

import os
import sys

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from vrp_data_models import VRPInstance, Location, Vehicle, RideRequest
from vrp_optimizer_fixed import VRPQuantumOptimizer
from vrp_data_models import VRPObjective

def test_simple_case():
    """Test a very simple case to understand time constraints."""
    print("Testing ultra-simple scenario...")
    
    instance = VRPInstance("Ultra Simple Test")
    
    # Add depot (available all day)
    depot = Location("depot", 0, 0, demand=0, 
                    time_window_start=0, time_window_end=1440, service_time=0)
    instance.add_location(depot)
    
    # Add customer (available all day) 
    customer = Location("customer", 10, 10, demand=1,
                       time_window_start=0, time_window_end=1440, service_time=30)
    instance.add_location(customer)
    
    # Add vehicle with 10-hour limit
    vehicle = Vehicle("truck", capacity=10, depot_id="depot", max_time=600)
    instance.add_vehicle(vehicle)
    
    instance.calculate_distance_matrix(distance_method="manhattan")
    
    print("Scenario: depot->customer->depot")
    print("All locations available 0-1440 minutes (all day)")
    print("Vehicle max time: 600 minutes (10 hours)")
    print("Travel time: about 20 minutes total")
    print("Service time: 30 minutes")
    print("Total time needed: ~50 minutes (well under 600)")
    
    # Test with OR-Tools
    optimizer = VRPQuantumOptimizer(instance, VRPObjective.MINIMIZE_DISTANCE)
    result = optimizer.optimize_with_ortools()
    
    print(f"\nResult: {result.status}")
    print(f"Solved: {'✅ YES' if result.status == 'optimal' else '❌ NO'}")
    
    return result.status == "optimal"

def test_late_time_windows():
    """Test with time windows that end after 600 minutes."""
    print("\nTesting with late time windows...")
    
    instance = VRPInstance("Late Time Windows Test")
    
    # Add depot (available morning to late evening)
    depot = Location("depot", 0, 0, demand=0, 
                    time_window_start=480, time_window_end=1200, service_time=0)  # 8 AM to 8 PM
    instance.add_location(depot)
    
    # Add customer (available late afternoon to evening)
    customer = Location("customer", 10, 10, demand=1,
                       time_window_start=900, time_window_end=1080, service_time=30)  # 3 PM to 6 PM
    instance.add_location(customer)
    
    # Add vehicle with 10-hour limit
    vehicle = Vehicle("truck", capacity=10, depot_id="depot", max_time=600)
    instance.add_vehicle(vehicle)
    
    instance.calculate_distance_matrix(distance_method="manhattan")
    
    print("Scenario: depot (8 AM-8 PM) -> customer (3 PM-6 PM) -> depot")
    print("Customer window ends at 1080 min (6 PM) > 600 min threshold")
    print("Driver can start at 8 AM (480 min) and work until 6 PM (1080 min) = 10 hours")
    
    # Test with OR-Tools
    optimizer = VRPQuantumOptimizer(instance, VRPObjective.MINIMIZE_DISTANCE)
    result = optimizer.optimize_with_ortools()
    
    print(f"\nResult: {result.status}")
    print(f"Solved: {'✅ YES' if result.status == 'optimal' else '❌ NO'}")
    
    return result.status == "optimal"

def test_start_time_flexibility():
    """Test if driver can start later to accommodate late time windows."""
    print("\nTesting driver start time flexibility...")
    
    instance = VRPInstance("Start Time Flexibility Test")
    
    # Add depot (available all day)
    depot = Location("depot", 0, 0, demand=0, 
                    time_window_start=0, time_window_end=1440, service_time=0)
    instance.add_location(depot)
    
    # Add customer with late time window
    customer = Location("customer", 10, 10, demand=1,
                       time_window_start=1020, time_window_end=1200, service_time=30)  # 5 PM to 8 PM
    instance.add_location(customer)
    
    # Add vehicle with 10-hour limit
    vehicle = Vehicle("truck", capacity=10, depot_id="depot", max_time=600)
    instance.add_vehicle(vehicle)
    
    instance.calculate_distance_matrix(distance_method="manhattan")
    
    print("Scenario: customer available 5 PM-8 PM (1020-1200 min)")
    print("Driver should start around 10 AM to serve customer at 5 PM and finish by 8 PM")
    print("Total shift: 10 AM to 8 PM = 10 hours (exactly max_time=600)")
    
    # Test with OR-Tools
    optimizer = VRPQuantumOptimizer(instance, VRPObjective.MINIMIZE_DISTANCE)
    result = optimizer.optimize_with_ortools()
    
    print(f"\nResult: {result.status}")
    print(f"Solved: {'✅ YES' if result.status == 'optimal' else '❌ NO'}")
    
    if result.routes:
        print("Routes found:")
        for vehicle_id, route in result.routes.items():
            if len(route) > 2:
                print(f"  {vehicle_id}: {' -> '.join(route)}")
    
    return result.status == "optimal"

if __name__ == "__main__":
    print("="*60)
    print("OR-TOOLS TIME CONSTRAINT TESTING")
    print("="*60)
    
    test1 = test_simple_case()
    test2 = test_late_time_windows()
    test3 = test_start_time_flexibility()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Simple case (all day availability): {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Late time windows (ends after 600 min): {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Start time flexibility (late customer): {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if all([test1, test2, test3]):
        print("\n✅ ALL TESTS PASS - Trucking logic works correctly!")
    else:
        print("\n❌ SOME TESTS FAIL - Need to investigate OR-Tools time constraint handling")
