#!/usr/bin/env python3
"""
Test script for dual capacity constraints with a feasible scenario.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import the dual capacity optimizer
import importlib.util
current_dir = os.path.dirname(__file__)
optimizer_path = os.path.join(current_dir, "vrp_optimizer_clean copy.py")
spec = importlib.util.spec_from_file_location("vrp_optimizer_clean_copy", optimizer_path)
vrp_optimizer_clean_copy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vrp_optimizer_clean_copy)

from vrp_data_models import VRPInstance, Location, Vehicle, RideRequest

# Use the correct class name
CleanVRPOptimizer = vrp_optimizer_clean_copy.CleanVRPOptimizer

def create_simple_feasible_scenario() -> VRPInstance:
    """Create a simple feasible scenario for testing dual capacity constraints."""
    instance = VRPInstance("Simple Feasible Test")
    
    # Add depot
    depot = Location("depot", 45.0, 9.0, 0)
    instance.add_location(depot)
    
    # Add 3 pickup locations
    instance.add_location(Location("pickup_1", 45.1, 9.1, 0))
    instance.add_location(Location("pickup_2", 45.2, 9.2, 0))
    instance.add_location(Location("pickup_3", 45.3, 9.3, 0))
    
    # Add 3 delivery locations
    instance.add_location(Location("delivery_1", 45.4, 9.4, 0))
    instance.add_location(Location("delivery_2", 45.5, 9.5, 0))
    instance.add_location(Location("delivery_3", 45.6, 9.6, 0))
    
    # Add 1 vehicle with reasonable capacity
    vehicle = Vehicle("truck_1", capacity=10000, depot_id="depot", volume_capacity=50.0)
    instance.add_vehicle(vehicle)
    
    # Add 3 ride requests with modest demands
    # Request 1: 2000kg, 10m³ (weight-limited)
    request1 = RideRequest("req_1", "pickup_1", "delivery_1", passengers=2000, volume=10.0)
    instance.add_ride_request(request1)
    
    # Request 2: 1000kg, 15m³ (volume-limited)
    request2 = RideRequest("req_2", "pickup_2", "delivery_2", passengers=1000, volume=15.0)
    instance.add_ride_request(request2)
    
    # Request 3: 1500kg, 12m³ (balanced)
    request3 = RideRequest("req_3", "pickup_3", "delivery_3", passengers=1500, volume=12.0)
    instance.add_ride_request(request3)
    
    # Total: 4500kg (45% of 10000kg), 37m³ (74% of 50m³) - should be feasible
    
    print(f"Created simple feasible scenario:")
    print(f"  Vehicle capacity: 10000kg, 50m³")
    print(f"  Total demand: 4500kg (45%), 37m³ (74%)")
    print(f"  Expected to be feasible with both constraints satisfied")
    
    return instance

def test_feasible_dual_capacity():
    """Test dual capacity constraints with a feasible scenario."""
    print("=" * 80)
    print("🧪 Testing Dual Capacity with Feasible Scenario")
    print("=" * 80)
    
    # Create simple feasible scenario
    scenario = create_simple_feasible_scenario()
    
    # Create optimizer
    optimizer = CleanVRPOptimizer(vrp_instance=scenario)
    
    # Solve with capacity constraints
    result = optimizer.solve(constraint_level="capacity", verbose=True)
    
    if result['status'] == 'SUCCESS':
        print(f"\n✅ SUCCESS: Feasible solution found!")
        print(f"   Routes: {len(result['routes'])}")
        
        # Check for limiting constraints
        for route in result['routes']:
            if route['capacity_usage']:
                print(f"\n🚛 {route['vehicle_id']} capacity usage:")
                for stop in route['capacity_usage']:
                    print(f"   {stop['location_id']}: {stop['weight_usage_kg']}kg ({stop['weight_utilization_pct']}%), "
                          f"{stop['volume_usage_m3']}m³ ({stop['volume_utilization_pct']}%) → {stop['limiting_constraint']}")
        
        return True
    else:
        print(f"\n❌ FAILED: {result.get('message', 'Unknown error')}")
        return False

def main():
    """Run feasible dual capacity test."""
    print("🚀 Testing Dual Capacity Constraints with Feasible Scenario")
    
    success = test_feasible_dual_capacity()
    
    print("\n" + "=" * 80)
    if success:
        print("✅ Feasible dual capacity test passed!")
        print("   Dual capacity constraints working correctly for feasible scenarios")
    else:
        print("❌ Feasible dual capacity test failed!")

if __name__ == "__main__":
    main()
