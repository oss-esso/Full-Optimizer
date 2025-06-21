#!/usr/bin/env python3
"""
Test Service Areas Integration for Break Planning

This script tests the integration of the Northern Italy service areas database
with the VRP solver for realistic break planning for heavy trucks.
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
from service_areas_db import service_areas_db

def test_service_areas_database():
    """Test the service areas database functionality."""
    print("=== Testing Service Areas Database ===")
    
    # Get database statistics
    stats = service_areas_db.get_service_area_statistics()
    print(f"  Total service areas: {stats['total_areas']}")
    print(f"  With truck parking: {stats['with_truck_parking']}")
    print(f"  With fuel: {stats['with_fuel']}")
    print(f"  With restaurant: {stats['with_restaurant']}")
    
    print("\n  Service areas by highway:")
    for highway, count in stats['by_highway'].items():
        print(f"    {highway}: {count} areas")
    
    # Test finding service areas between major cities
    print("\n  Testing route-based service area finding:")
    
    # Milan to Bologna route
    milan_lat, milan_lon = 45.4642, 9.1896
    bologna_lat, bologna_lon = 44.4949, 11.3426
    
    route_areas = service_areas_db.find_service_areas_near_route(
        milan_lat, milan_lon, bologna_lat, bologna_lon, max_detour_km=15.0
    )
    print(f"    Milan to Bologna: {len(route_areas)} service areas found")
    for area in route_areas[:3]:  # Show first 3
        print(f"      {area.name} ({area.highway})")
    
    # Test finding best break location
    best_break = service_areas_db.find_break_location_between_points(
        milan_lat, milan_lon, bologna_lat, bologna_lon
    )
    if best_break:
        print(f"    Best break location: {best_break.name} ({best_break.highway})")
    else:
        print("    No suitable break location found")

def test_service_area_integration():
    """Test service area integration with VRP scenarios."""
    print("\n=== Testing Service Area Integration ===")
    
    # Load scenarios
    scenarios = vrp_scenarios.get_all_scenarios()
    
    # Test with MODA_small
    moda_small = scenarios.get('MODA_small')
    if moda_small:
        print(f"  MODA_small scenario loaded")
        print(f"    Total locations: {len(moda_small.locations)}")
        
        # Count service areas in the scenario
        service_area_count = sum(1 for loc_id in moda_small.locations.keys() 
                               if loc_id.startswith('SA_'))
        print(f"    Service areas included: {service_area_count}")
        
        # List some service areas
        service_areas = [loc_id for loc_id in moda_small.locations.keys() 
                        if loc_id.startswith('SA_')]
        if service_areas:
            print("    Service areas available:")
            for sa_id in service_areas[:3]:  # Show first 3
                sa_location = moda_small.locations[sa_id]
                print(f"      {sa_id}: {getattr(sa_location, 'address', 'N/A')}")
        else:
            print("    No service areas found in scenario")
    
    # Test with MODA_first
    moda_first = scenarios.get('MODA_first')
    if moda_first:
        print(f"\n  MODA_first scenario loaded")
        print(f"    Total locations: {len(moda_first.locations)}")
        
        # Count service areas in the scenario
        service_area_count = sum(1 for loc_id in moda_first.locations.keys() 
                               if loc_id.startswith('SA_'))
        print(f"    Service areas included: {service_area_count}")

def test_break_enforcement():
    """Test break enforcement with service areas."""
    print("\n=== Testing Break Enforcement with Service Areas ===")
    
    scenarios = vrp_scenarios.get_all_scenarios()
    moda_small = scenarios.get('MODA_small')
    
    if not moda_small:
        print("  MODA_small scenario not available")
        return
    
    print(f"  Testing with MODA_small scenario")
    print(f"    Heavy trucks: {len([v for v in moda_small.vehicles.values() if getattr(v, 'vehicle_type', 'standard') == 'heavy'])}")
    
    # Test quantum solver with break enforcement
    print("\n  Testing quantum solver with break enforcement:")
    optimizer = VRPQuantumOptimizer(moda_small, VRPObjective.MINIMIZE_DISTANCE)
    result = optimizer.optimize_with_quantum_benders()
    
    print(f"    Status: {result.status}")
    print(f"    Vehicles used: {len([r for r in result.routes.values() if len(r) > 2])}")
    
    # Analyze routes for break locations
    break_count = 0
    for vehicle_id, route in result.routes.items():
        if len(route) > 2:  # Vehicle is used
            vehicle = moda_small.vehicles[vehicle_id]
            is_heavy = getattr(vehicle, 'vehicle_type', 'standard') == 'heavy'
            
            if is_heavy:
                # Count service area stops in route
                service_stops = [loc for loc in route if loc.startswith('SA_')]
                print(f"    Heavy truck {vehicle_id}: {len(route)-2} stops, {len(service_stops)} service area stops")
                break_count += len(service_stops)
                
                if service_stops:
                    print(f"      Service area stops: {service_stops}")
    
    print(f"    Total break locations used: {break_count}")

def test_route_optimization_with_breaks():
    """Test route optimization that includes break planning."""
    print("\n=== Testing Route Optimization with Break Planning ===")
    
    # Create a longer distance scenario to trigger break requirements
    from vrp_data_models import VRPInstance, Location, Vehicle, RideRequest
    
    # Create a scenario with long distances that will require breaks
    instance = VRPInstance("Long Distance Break Test")
    
    # Add depot
    instance.add_location(Location("depot_1", 8.2062, 44.9009, demand=0))  # Asti
    
    # Add distant pickup/dropoff pairs
    locations_data = [
        ("pickup_1", 7.6869, 45.0703, "Turin"),      # Turin
        ("dropoff_1", 12.3426, 44.4949, "Bologna"),  # Bologna (long distance)
        ("pickup_2", 9.1896, 45.4642, "Milan"),      # Milan
        ("dropoff_2", 11.8767, 45.4064, "Venice"),   # Venice (long distance)
    ]
    
    for loc_id, lon, lat, city in locations_data:
        location = Location(loc_id, lon, lat, demand=0)
        location.address = f"{city}, Italy"
        location.lat = lat
        location.lon = lon
        instance.add_location(location)
    
    # Add heavy truck that will need breaks
    vehicle = Vehicle("heavy_truck_1", capacity=24000, depot_id="depot_1", max_time=540)
    vehicle.vehicle_type = "heavy"
    vehicle.max_driving_time = 270.0  # 4.5 hours
    vehicle.required_break_time = 45.0  # 45 minutes
    vehicle.max_total_work_time = 540.0  # 9 hours
    instance.add_vehicle(vehicle)
    
    # Add cargo requests for long distances
    instance.add_ride_request(RideRequest("request_1", "pickup_1", "dropoff_1", passengers=5000))
    instance.add_ride_request(RideRequest("request_2", "pickup_2", "dropoff_2", passengers=3000))
    
    # Add service areas to the instance
    try:
        from service_areas_db import service_area_to_location
        
        # Add A1 and A4 service areas for the long routes
        relevant_service_areas = [
            "SA_A1_Fiorenzuola_N", "SA_A1_Secchia_N", 
            "SA_A4_Bergamo_E", "SA_A4_Limenella_E"
        ]
        
        for sa_id in relevant_service_areas:
            if sa_id in service_areas_db.service_areas:
                sa = service_areas_db.service_areas[sa_id]
                instance.locations[sa_id] = service_area_to_location(sa)
        
        print(f"  Added {len(relevant_service_areas)} service areas for long-distance routes")
        
    except Exception as e:
        print(f"  Error adding service areas: {e}")
    
    # Calculate distance matrix
    instance.calculate_distance_matrix(distance_method="euclidean")
    
    # Test optimization
    print(f"  Testing long-distance scenario:")
    print(f"    Locations: {len(instance.locations)}")
    print(f"    Heavy trucks: 1")
    print(f"    Long-distance requests: 2")
    
    optimizer = VRPQuantumOptimizer(instance, VRPObjective.MINIMIZE_DISTANCE)
    result = optimizer.optimize_with_quantum_benders()
    
    print(f"    Status: {result.status}")
    if result.routes:
        for vehicle_id, route in result.routes.items():
            if len(route) > 2:
                service_stops = [loc for loc in route if loc.startswith('SA_')]
                print(f"    Route: {route}")
                print(f"    Service area breaks: {len(service_stops)}")
                if service_stops:
                    print(f"      Break locations: {service_stops}")

def main():
    """Run all service area tests."""
    print("Testing Service Areas Integration for Heavy Truck Break Planning")
    print("=" * 80)
    
    test_service_areas_database()
    test_service_area_integration()
    test_break_enforcement()
    test_route_optimization_with_breaks()
    
    print("\n" + "=" * 80)
    print("Service areas testing completed!")

if __name__ == "__main__":
    main()
