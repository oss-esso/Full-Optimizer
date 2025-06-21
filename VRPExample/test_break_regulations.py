#!/usr/bin/env python3
"""
Test Break Enforcement with Long Routes

This script creates scenarios with longer routes that will definitely trigger
the 4.5-hour driving time limit for heavy trucks, forcing break planning.
"""

import os
import sys

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from vrp_data_models import VRPInstance, Location, Vehicle, RideRequest, VRPObjective
from vrp_optimizer_fixed import VRPQuantumOptimizer
from service_areas_db import service_areas_db, service_area_to_location

def create_long_distance_scenario():
    """Create a scenario with routes that definitely require breaks."""
    instance = VRPInstance("Long Distance Heavy Truck Test")
    
    # Add depot in Milan
    depot = Location("depot_milan", 9.1896, 45.4642, demand=0)
    depot.address = "Milano Centrale Hub"
    depot.lat = 45.4642
    depot.lon = 9.1896
    instance.add_location(depot)
    
    # Add very distant locations to force long driving times
    long_distance_locations = [
        # Northern Italy extreme points
        ("pickup_torino", 7.6869, 45.0703, "Torino, Italy"),
        ("dropoff_trieste", 13.7760, 45.6495, "Trieste, Italy"),  # ~500km from Turin
        ("pickup_genova", 8.9463, 44.4056, "Genova, Italy"),
        ("dropoff_bologna", 11.3426, 44.4949, "Bologna, Italy"),  # ~200km from Genoa
        ("pickup_aosta", 7.3154, 45.7372, "Aosta, Italy"),
        ("dropoff_venice", 12.3155, 45.4408, "Venezia, Italy"),  # ~450km from Aosta
    ]
    
    for loc_id, lon, lat, address in long_distance_locations:
        location = Location(loc_id, lon, lat, demand=0, service_time=30)  # 30 min service
        location.address = address
        location.lat = lat
        location.lon = lon
        instance.add_location(location)
    
    # Add service areas along the major routes
    key_service_areas = [
        "SA_A1_Fiorenzuola_N",  # A1 Milan-Bologna route
        "SA_A1_Secchia_N",      # A1 continued
        "SA_A4_Bergamo_E",      # A4 Milan-Venice route
        "SA_A4_Limenella_E",    # A4 continued
        "SA_A7_Serravalle_N",   # A7 Milan-Genoa route
        "SA_A6_Mondovi_N",      # A6 Turin route
    ]
    
    added_service_areas = 0
    for sa_id in key_service_areas:
        if sa_id in service_areas_db.service_areas:
            sa = service_areas_db.service_areas[sa_id]
            instance.locations[sa_id] = service_area_to_location(sa)
            added_service_areas += 1
    
    print(f"Added {added_service_areas} strategic service areas")
    
    # Add heavy trucks with strict regulations
    heavy_trucks = [
        ("heavy_truck_1", "depot_milan"),
        ("heavy_truck_2", "depot_milan"),
    ]
    
    for truck_id, depot_id in heavy_trucks:
        vehicle = Vehicle(truck_id, capacity=24000, depot_id=depot_id, max_time=540)
        vehicle.vehicle_type = "heavy"
        vehicle.max_driving_time = 270.0  # 4.5 hours = 270 minutes
        vehicle.required_break_time = 45.0  # 45-minute break
        vehicle.max_total_work_time = 540.0  # 9 hours total
        instance.add_vehicle(vehicle)
    
    # Add long-distance cargo requests
    long_requests = [
        ("request_1", "pickup_torino", "dropoff_trieste", 8000),  # Cross country
        ("request_2", "pickup_genova", "dropoff_bologna", 6000),  # Medium distance
        ("request_3", "pickup_aosta", "dropoff_venice", 9000),    # Very long distance
    ]
    
    for req_id, pickup, dropoff, weight in long_requests:
        request = RideRequest(req_id, pickup, dropoff, passengers=weight)
        instance.add_ride_request(request)
    
    # Calculate distance matrix
    instance.calculate_distance_matrix(distance_method="euclidean")
    
    return instance

def test_break_enforcement_detailed():
    """Test break enforcement with detailed analysis."""
    print("=== Testing Break Enforcement with Long Routes ===")
    
    # Create long distance scenario
    scenario = create_long_distance_scenario()
    
    print(f"Long distance scenario created:")
    print(f"  Total locations: {len(scenario.locations)}")
    print(f"  Heavy trucks: {len([v for v in scenario.vehicles.values() if getattr(v, 'vehicle_type', 'standard') == 'heavy'])}")
    print(f"  Long-distance requests: {len(scenario.ride_requests)}")
    
    service_areas = [loc_id for loc_id in scenario.locations.keys() if loc_id.startswith('SA_')]
    print(f"  Service areas available: {len(service_areas)}")
    
    # Test optimization
    optimizer = VRPQuantumOptimizer(scenario, VRPObjective.MINIMIZE_DISTANCE)
    
    print("\nTesting quantum solver with break enforcement:")
    result = optimizer.optimize_with_quantum_benders()
    
    print(f"  Status: {result.status}")
    print(f"  Objective: {result.objective_value}")
    print(f"  Runtime: {result.runtime:.2f}ms")
    
    # Analyze routes in detail
    total_breaks = 0
    for vehicle_id, route in result.routes.items():
        if len(route) > 2:  # Vehicle is used
            vehicle = scenario.vehicles[vehicle_id]
            is_heavy = getattr(vehicle, 'vehicle_type', 'standard') == 'heavy'
            
            # Count different types of stops
            pickup_stops = [loc for loc in route if loc.startswith('pickup_')]
            dropoff_stops = [loc for loc in route if loc.startswith('dropoff_')]
            service_stops = [loc for loc in route if loc.startswith('SA_')]
            depot_stops = [loc for loc in route if loc.startswith('depot_')]
            
            print(f"\n  Vehicle {vehicle_id} ({'heavy' if is_heavy else 'standard'}):")
            print(f"    Route: {route}")
            print(f"    Total stops: {len(route)}")
            print(f"    Pickup stops: {len(pickup_stops)}")
            print(f"    Dropoff stops: {len(dropoff_stops)}")
            print(f"    Service area breaks: {len(service_stops)}")
            print(f"    Depot visits: {len(depot_stops)}")
            
            if service_stops:
                print(f"    Break locations used:")
                for sa_id in service_stops:
                    if sa_id in scenario.locations:
                        sa_location = scenario.locations[sa_id]
                        print(f"      {sa_id}: {getattr(sa_location, 'address', 'N/A')}")
                total_breaks += len(service_stops)
            
            # Calculate approximate route distance and time
            if is_heavy and len(route) > 2:
                total_distance = 0
                total_time = 0
                for i in range(len(route) - 1):
                    try:
                        distance = scenario.get_distance(route[i], route[i+1])
                        total_distance += distance
                        # Assuming 1 degree ≈ 111 km and average speed of 60 km/h
                        time_hours = (distance * 111) / 60
                        total_time += time_hours
                    except:
                        pass
                
                print(f"    Estimated total distance: {total_distance:.2f} units")
                print(f"    Estimated driving time: {total_time:.2f} hours")
                
                if total_time > 4.5:  # More than 4.5 hours
                    if len(service_stops) == 0:
                        print("    ⚠️ WARNING: Long route without breaks detected!")
                    else:
                        print("    ✅ Break enforcement working correctly")
    
    print(f"\nTotal service area breaks used: {total_breaks}")
    
    if total_breaks > 0:
        print("✅ Break enforcement is working - service areas being used!")
    else:
        print("ℹ️ No breaks inserted - routes may be short enough or enforcement needs tuning")

def test_manual_route_with_breaks():
    """Test manual route creation with forced break insertion."""
    print("\n=== Testing Manual Route with Break Insertion ===")
    
    # Test the break insertion logic directly
    scenario = create_long_distance_scenario()
    optimizer = VRPQuantumOptimizer(scenario, VRPObjective.MINIMIZE_DISTANCE)
    
    # Create a manual long route
    test_route = ["depot_milan", "pickup_torino", "pickup_aosta", "dropoff_trieste", "dropoff_venice", "depot_milan"]
    
    print(f"Original test route: {test_route}")
    
    # Get a heavy truck for testing
    heavy_truck = None
    for vehicle in scenario.vehicles.values():
        if getattr(vehicle, 'vehicle_type', 'standard') == 'heavy':
            heavy_truck = vehicle
            break
    
    if heavy_truck:
        print(f"Testing with heavy truck: {heavy_truck.id}")
        print(f"  Max driving time: {heavy_truck.max_driving_time} minutes")
        print(f"  Required break time: {heavy_truck.required_break_time} minutes")
        
        # Test break optimization
        optimized_route = optimizer._optimize_route_with_breaks(test_route, heavy_truck)
        
        print(f"Optimized route: {optimized_route}")
        
        service_stops = [loc for loc in optimized_route if loc.startswith('SA_')]
        if service_stops:
            print(f"Break locations inserted: {service_stops}")
            for sa_id in service_stops:
                if sa_id in scenario.locations:
                    sa_loc = scenario.locations[sa_id]
                    print(f"  {sa_id}: {getattr(sa_loc, 'address', 'N/A')}")
        else:
            print("No break locations inserted")

def main():
    """Run break enforcement tests."""
    print("Testing Break Enforcement for Heavy Trucks")
    print("=" * 60)
    
    test_break_enforcement_detailed()
    test_manual_route_with_breaks()
    
    print("\n" + "=" * 60)
    print("Break enforcement testing completed!")

if __name__ == "__main__":
    main()
