#!/usr/bin/env python3
"""
Test driver regulations and mixed fleet implementation.
"""

import os
import sys

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

import vrp_scenarios
from vrp_optimizer_fixed import VRPQuantumOptimizer
from vrp_data_models import VRPObjective

def test_mixed_fleet_generation():
    """Test that scenarios generate correct mixed fleet."""
    print("=== Testing Mixed Fleet Generation ===")
    
    # Test MODA_small scenario
    print("\nTesting MODA_small scenario:")
    scenarios = vrp_scenarios.get_all_scenarios()
    moda_small = scenarios.get('MODA_small')
    
    if moda_small:
        # Check fleet composition
        standard_trucks = [v for v in moda_small.vehicles.values() if v.vehicle_type == 'standard']
        heavy_trucks = [v for v in moda_small.vehicles.values() if v.vehicle_type == 'heavy']
        
        print(f"  Total vehicles: {len(moda_small.vehicles)}")
        print(f"  Standard trucks (4t): {len(standard_trucks)}")
        print(f"  Heavy trucks (24t): {len(heavy_trucks)}")
        
        # Check capacities
        standard_capacities = [v.capacity for v in standard_trucks]
        heavy_capacities = [v.capacity for v in heavy_trucks]
        
        print(f"  Standard truck capacities: {set(standard_capacities)}")
        print(f"  Heavy truck capacities: {set(heavy_capacities)}")
        
        # Check driver regulations
        print(f"  Heavy truck regulations:")
        for v in heavy_trucks[:2]:  # Show first 2
            print(f"    {v.id}: max_driving={v.max_driving_time}min, break={v.required_break_time}min, total_work={v.max_total_work_time}min")
        
        # Check feasibility
        total_capacity = sum(v.capacity for v in moda_small.vehicles.values())
        total_demand = sum(req.passengers for req in moda_small.ride_requests) if moda_small.ride_requests else 0
        print(f"  Total capacity: {total_capacity} kg")
        print(f"  Total demand: {total_demand} kg")
        print(f"  Feasible: {total_demand <= total_capacity}")
        
    else:
        print("  ❌ MODA_small scenario not found")
    
    # Test MODA_first scenario  
    print("\nTesting MODA_first scenario:")
    moda_first = scenarios.get('MODA_first')
    
    if moda_first:
        # Check fleet composition
        standard_trucks = [v for v in moda_first.vehicles.values() if v.vehicle_type == 'standard']
        heavy_trucks = [v for v in moda_first.vehicles.values() if v.vehicle_type == 'heavy']
        
        print(f"  Total vehicles: {len(moda_first.vehicles)}")
        print(f"  Standard trucks (4t): {len(standard_trucks)}")
        print(f"  Heavy trucks (24t): {len(heavy_trucks)}")
        
        # Check if we achieved the target 40+20 split
        print(f"  Target: 40 standard + 20 heavy = 60 total")
        print(f"  Actual: {len(standard_trucks)} standard + {len(heavy_trucks)} heavy = {len(moda_first.vehicles)} total")
        
        # Check feasibility
        total_capacity = sum(v.capacity for v in moda_first.vehicles.values())
        total_demand = sum(req.passengers for req in moda_first.ride_requests) if moda_first.ride_requests else 0
        print(f"  Total capacity: {total_capacity} kg")
        print(f"  Total demand: {total_demand} kg")
        print(f"  Feasible: {total_demand <= total_capacity}")
        
    else:
        print("  ❌ MODA_first scenario not found")

def test_driver_regulation_enforcement():
    """Test that driver regulations are enforced in optimization."""
    print("\n=== Testing Driver Regulation Enforcement ===")
    
    scenarios = vrp_scenarios.get_all_scenarios()
    moda_small = scenarios.get('MODA_small')
    
    if not moda_small:
        print("❌ MODA_small scenario not available for testing")
        return
    
    print("Testing with MODA_small scenario...")
    
    # Test quantum solver
    print("\nTesting quantum solver:")
    optimizer = VRPQuantumOptimizer(moda_small, VRPObjective.MINIMIZE_DISTANCE)
    result = optimizer.optimize_with_quantum_benders()
    
    print(f"  Status: {result.status}")
    print(f"  Vehicles used: {len([r for r in result.routes.values() if len(r) > 2])}")
    print(f"  Objective value: {result.objective_value}")
    
    # Check routes for heavy trucks
    heavy_vehicle_ids = [v.id for v in moda_small.vehicles.values() if v.vehicle_type == 'heavy']
    
    for vehicle_id in heavy_vehicle_ids:
        if vehicle_id in result.routes:
            route = result.routes[vehicle_id]
            stops = [loc for loc in route if not loc.startswith('depot')]
            print(f"  Heavy truck {vehicle_id}: {len(stops)} stops, route length: {len(route)}")
            
            # Check for break locations (service areas)
            service_stops = [loc for loc in route if loc.startswith('service_area')]
            if service_stops:
                print(f"    Break locations: {service_stops}")
            else:
                print(f"    No break locations detected (route may be short enough)")
    
    # Test OR-Tools solver if available
    print("\nTesting OR-Tools solver:")
    result_ortools = optimizer.optimize_with_ortools()
    
    print(f"  Status: {result_ortools.status}")
    print(f"  Vehicles used: {len([r for r in result_ortools.routes.values() if len(r) > 2])}")
    print(f"  Objective value: {result_ortools.objective_value}")
    
    # Check route lengths (should be constrained to max 8 stops)
    for vehicle_id, route in result_ortools.routes.items():
        if len(route) > 2:  # Vehicle is used
            stops = [loc for loc in route if not loc.startswith('depot')]
            if len(stops) > 8:
                print(f"  ⚠️ Warning: Vehicle {vehicle_id} has {len(stops)} stops (exceeds limit of 8)")
            else:
                print(f"  ✅ Vehicle {vehicle_id}: {len(stops)} stops (within limit)")

def test_break_scenarios():
    """Test specific scenarios that would require breaks."""
    print("\n=== Testing Break Requirement Scenarios ===")
    
    # Create a scenario with long routes that should trigger breaks
    from vrp_data_models import VRPInstance, Location, Vehicle, RideRequest
    import random
    
    instance = VRPInstance("Break Test Scenario")
    
    # Add depot
    instance.add_location(Location("depot_1", 45.4642, 9.1896, demand=0))
    
    # Add pickup/dropoff locations spread far apart to force long routes
    locations = [
        ("pickup_1", 45.0, 8.0),   # Far west
        ("dropoff_1", 46.0, 10.0), # Far east  
        ("pickup_2", 44.5, 9.0),   # South
        ("dropoff_2", 45.8, 9.5),  # North
    ]
    
    for loc_id, lat, lon in locations:
        instance.add_location(Location(loc_id, lon, lat, demand=0))
    
    # Add one heavy truck that should need breaks
    vehicle = Vehicle("heavy_truck_1", capacity=24000, depot_id="depot_1", max_time=540)
    vehicle.vehicle_type = "heavy"
    vehicle.max_driving_time = 270.0  # 4.5 hours
    vehicle.required_break_time = 45.0
    vehicle.max_total_work_time = 540.0
    instance.add_vehicle(vehicle)
    
    # Add requests that create long routes
    instance.add_ride_request(RideRequest("request_1", "pickup_1", "dropoff_1", passengers=5000))
    instance.add_ride_request(RideRequest("request_2", "pickup_2", "dropoff_2", passengers=3000))
    
    # Calculate distance matrix
    instance.calculate_distance_matrix(distance_method="manhattan")
    
    print("Testing break enforcement with long-distance scenario:")
    
    # Test with optimizer
    optimizer = VRPQuantumOptimizer(instance, VRPObjective.MINIMIZE_DISTANCE)
    result = optimizer.optimize_with_quantum_benders()
    
    print(f"  Status: {result.status}")
    if result.routes:
        route = result.routes.get("heavy_truck_1", [])
        print(f"  Heavy truck route: {route}")
        
        # Check for break locations
        service_stops = [loc for loc in route if loc.startswith('service_area') or 'break' in loc.lower()]
        if service_stops:
            print(f"  ✅ Break locations inserted: {service_stops}")
        else:
            print(f"  ℹ️ No breaks needed (route may be short) or service areas not available")

if __name__ == "__main__":
    print("Testing Mixed Fleet and Driver Regulations")
    print("=" * 60)
    
    test_mixed_fleet_generation()
    test_driver_regulation_enforcement()
    test_break_scenarios()
    
    print("\n" + "=" * 60)
    print("Testing completed!")
