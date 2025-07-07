#!/usr/bin/env python3

import sys
import os
import random

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vrp_data_models import VRPInstance, RideRequest, Vehicle, Location

def create_moda_first_larger_windows() -> VRPInstance:
    """Create MODA_first scenario with much larger time windows for better feasibility."""
    
    instance = VRPInstance("MODA_first_larger_windows")
    
    # Add depots in Northern Italy
    depot_1 = Location("depot_1", 8.6267, 45.0703, 0)  # Asti
    depot_1.address = "AST Logistics Hub, Asti"
    depot_1.lat = 45.0703
    depot_1.lon = 8.6267
    depot_1.time_window_start = 0
    depot_1.time_window_end = 2880  # 48 hours
    depot_1.service_time = 5
    instance.add_location(depot_1)
    
    depot_2 = Location("depot_2", 9.1895, 45.4654, 0)  # Milan
    depot_2.address = "MIL Central Depot, Milan"
    depot_2.lat = 45.4654
    depot_2.lon = 9.1895
    depot_2.time_window_start = 0
    depot_2.time_window_end = 2880  # 48 hours
    depot_2.service_time = 5
    instance.add_location(depot_2)
    
    # Expanded Northern Italy city coordinates
    northern_italy_areas = [
        (45.4654, 9.1895, "Milan", "Via della Spiga"),
        (45.0703, 8.6267, "Asti", "Via Roma"),
        (45.0732, 7.6869, "Turin", "Via Po"),
        (44.4949, 11.3426, "Bologna", "Via dell'Indipendenza"),
        (45.4384, 10.9916, "Verona", "Via Mazzini"),
        (45.6495, 9.6988, "Bergamo", "Via XX Settembre"),
        (45.2775, 9.1278, "Pavia", "Corso Strada Nuova"),
        (45.1847, 9.1582, "Lodi", "Corso Umberto I"),
        (45.6906, 9.6692, "Lecco", "Via Nazario Sauro"),
        (45.8205, 9.0807, "Como", "Via Vittorio Emanuele II"),
        (45.5415, 10.2118, "Brescia", "Via dei Musei"),
        (45.9205, 8.8512, "Varese", "Corso Matteotti"),
        (45.0559, 9.7026, "Cremona", "Via Solferino"),
        (45.1564, 10.7916, "Mantova", "Via Roma"),
        (44.8015, 10.3279, "Parma", "Via della Repubblica"),
    ]
    
    pickup_locations = []
    dropoff_locations = []
    
    print("Creating MODA_first scenario with larger time windows...")
    
    # Bounds for Northern Italy
    MIN_LAT, MAX_LAT = 44.0, 46.5
    MIN_LON, MAX_LON = 7.0, 12.0
    
    # Generate 100 pickup-dropoff pairs distributed across areas
    areas_per_pair = len(northern_italy_areas)
    pairs_per_area = max(1, 100 // areas_per_pair)
    
    location_counter = 1
    
    for area_idx, (center_lat, center_lon, area_name, sample_street) in enumerate(northern_italy_areas):
        pairs_to_generate = min(pairs_per_area, 100 - len(pickup_locations))
        if pairs_to_generate <= 0:
            break
            
        spread = 0.015  # approximately 1.5 km
        
        for i in range(pairs_to_generate):
            pickup_lat = center_lat + random.uniform(-spread, spread)
            pickup_lon = center_lon + random.uniform(-spread, spread)
            pickup_id = f"pickup_{location_counter}"
            pickup_addr = f"{100+i*10} {sample_street}, {area_name}"
            
            # *** LARGER TIME WINDOWS ***
            # Pickup time windows: much longer and more flexible
            pickup_start = random.randint(0, 1440)  # Start anytime in 24 hours (0-1440 min)
            pickup_window_length = random.randint(480, 1440)  # 8-24 hours (480-1440 min)
            pickup_end = min(pickup_start + pickup_window_length, 2880)  # Cap at 48 hours
            
            pickup = Location(pickup_id, pickup_lon, pickup_lat, demand=0,
                            time_window_start=pickup_start, time_window_end=pickup_end, service_time=15)
            pickup.address = pickup_addr
            pickup.lat = pickup_lat
            pickup.lon = pickup_lon
            instance.add_location(pickup)
            pickup_locations.append(pickup_id)
            
            # Dropoff location
            target_areas = [a for a in northern_italy_areas if a != (center_lat, center_lon, area_name, sample_street)]
            if target_areas:
                target = random.choice(target_areas)
                dropoff_lat = target[0] + random.uniform(-spread, spread)
                dropoff_lon = target[1] + random.uniform(-spread, spread)
                dropoff_area = target[2]
                dropoff_street = target[3]
            else:
                dropoff_lat = center_lat + random.uniform(-spread, spread)
                dropoff_lon = center_lon + random.uniform(-spread, spread)
                dropoff_area = area_name
                dropoff_street = sample_street
            
            dropoff_id = f"dropoff_{location_counter}"
            dropoff_addr = f"{200+i*10} {dropoff_street}, {dropoff_area}"
            
            # *** MUCH LARGER DROPOFF WINDOWS ***
            # Allow much more time between pickup and dropoff
            min_travel_time = 30  # Minimum 30 minutes
            min_dropoff_start = max(0, pickup_end + min_travel_time)
            max_dropoff_start = min(2400, min_dropoff_start + 720)  # Allow up to 12 hours flexibility
            if min_dropoff_start >= max_dropoff_start:
                dropoff_start = min_dropoff_start
            else:
                dropoff_start = random.randint(min_dropoff_start, max_dropoff_start)
            
            dropoff_window_length = random.randint(480, 1440)  # 8-24 hours (much longer)
            dropoff_end = min(dropoff_start + dropoff_window_length, 2880)  # Cap at 48 hours
            
            dropoff = Location(dropoff_id, dropoff_lon, dropoff_lat, demand=0,
                             time_window_start=dropoff_start, time_window_end=dropoff_end, service_time=10)
            dropoff.address = dropoff_addr
            dropoff.lat = dropoff_lat
            dropoff.lon = dropoff_lon
            instance.add_location(dropoff)
            dropoff_locations.append(dropoff_id)
            
            location_counter += 1
    
    # Add remaining locations to reach 100 pairs
    while len(pickup_locations) < 100:
        area = random.choice(northern_italy_areas)
        center_lat, center_lon, area_name, sample_street = area
        spread = 0.015
        
        pickup_lat = center_lat + random.uniform(-spread, spread)
        pickup_lon = center_lon + random.uniform(-spread, spread)
        pickup_id = f"pickup_{location_counter}"
        pickup_addr = f"{300+location_counter} {sample_street}, {area_name}"
        
        # Large time windows
        pickup_start = random.randint(0, 1440)
        pickup_window_length = random.randint(480, 1440)
        pickup_end = min(pickup_start + pickup_window_length, 2880)
        
        pickup = Location(pickup_id, pickup_lon, pickup_lat, demand=0,
                        time_window_start=pickup_start, time_window_end=pickup_end, service_time=15)
        pickup.address = pickup_addr
        pickup.lat = pickup_lat
        pickup.lon = pickup_lon
        instance.add_location(pickup)
        pickup_locations.append(pickup_id)
        
        # Dropoff
        target = random.choice([a for a in northern_italy_areas if a != area])
        dropoff_lat = target[0] + random.uniform(-spread, spread)
        dropoff_lon = target[1] + random.uniform(-spread, spread)
        dropoff_id = f"dropoff_{location_counter}"
        dropoff_addr = f"{400+location_counter} {target[3]}, {target[2]}"
        
        min_dropoff_start = max(0, pickup_end + 30)
        max_dropoff_start = min(2400, min_dropoff_start + 720)
        dropoff_start = random.randint(min_dropoff_start, max_dropoff_start) if min_dropoff_start < max_dropoff_start else min_dropoff_start
        dropoff_window_length = random.randint(480, 1440)
        dropoff_end = min(dropoff_start + dropoff_window_length, 2880)
        
        dropoff = Location(dropoff_id, dropoff_lon, dropoff_lat, demand=0,
                         time_window_start=dropoff_start, time_window_end=dropoff_end, service_time=10)
        dropoff.address = dropoff_addr
        dropoff.lat = dropoff_lat
        dropoff.lon = dropoff_lon
        instance.add_location(dropoff)
        dropoff_locations.append(dropoff_id)
        
        location_counter += 1
    
    # Add service areas for driver breaks
    try:
        from service_areas_db import service_areas_db
        relevant_areas = []
        for area_id, area in service_areas_db.service_areas.items():
            if (MIN_LAT <= area.lat <= MAX_LAT and MIN_LON <= area.lon <= MAX_LON):
                relevant_areas.append(area_id)
        
        for area_id in relevant_areas:
            from service_areas_db import service_area_to_location
            sa_location = service_area_to_location(service_areas_db.service_areas[area_id])
            sa_location.time_window_start = 0
            sa_location.time_window_end = 2880  # 48 hours
            instance.locations[area_id] = sa_location
        
        print(f"  Added {len(relevant_areas)} service areas for driver breaks across Northern Italy")
        
    except ImportError:
        print(f"  Service areas database not available")
    
    # Add 60 vehicles with larger capacity and flexible schedules
    total_vehicles = 60
    standard_trucks = 40  # 4-ton capacity
    heavy_trucks = 20     # 24-ton capacity
    
    # Distribute vehicles across depots
    depot_assignments = ["depot_1", "depot_2"] * (total_vehicles // 2)
    if total_vehicles % 2:
        depot_assignments.append("depot_1")
    
    total_capacity = 0
    
    for i in range(total_vehicles):
        vehicle_id = f"vehicle_{i+1}"
        depot_id = depot_assignments[i % len(depot_assignments)]
        
        if i < standard_trucks:
            capacity = 4000  # 4-ton trucks (in kg)
            vehicle_type = "heavy"
            max_work_time = 540  # 9 hours for heavy trucks (regulations)
        else:
            capacity = 24000  # 24-ton trucks (in kg)
            vehicle_type = "standard"
            max_work_time = 600  # 10 hours for standard trucks
        
        total_capacity += capacity
        
        vehicle = Vehicle(vehicle_id, capacity, depot_id)
        vehicle.vehicle_type = vehicle_type
        vehicle.max_total_work_time = max_work_time
        instance.add_vehicle(vehicle)
    
    # Generate ride requests
    cargo_weights = [50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 4000]
    total_demand = 0
    
    for i in range(len(pickup_locations)):
        pickup_loc = pickup_locations[i]
        dropoff_loc = dropoff_locations[i]
        
        cargo_weight = random.choice(cargo_weights)
        total_demand += cargo_weight
        
        request = RideRequest(f"request_{i+1}", pickup_loc, dropoff_loc, cargo_weight)
        instance.add_ride_request(request)
    
    print(f"MODA_first with larger windows total demand: {total_demand} kg, total fleet capacity: {total_capacity} kg")
    print("Note: Trucks can do multiple pickup/dropoff cycles, so demand can exceed total capacity")
    print(f"Fleet composition: {standard_trucks} x 4t trucks + {heavy_trucks} x 24t trucks")
    print(f"Created MODA_first scenario with larger time windows in Northern Italy:")
    print(f"  - {len(instance.location_ids)} total locations")
    print(f"  - 2 depots (Asti & Milan)")
    print(f"  - {len(pickup_locations)} pickup locations")
    print(f"  - {len(dropoff_locations)} dropoff locations")
    print(f"  - {len(instance.vehicles)} vehicles")
    print(f"  - {len(instance.ride_requests)} ride requests")
    print(f"  - Coverage: Northern Italy cities and metropolitan areas")
    print(f"  - Depots: Asti and Milan transportation hubs")
    print(f"  - Time windows: up to 48 hours span, 8-24 hour individual windows")
    
    return instance

if __name__ == "__main__":
    scenario = create_moda_first_larger_windows()
    print(f"Created scenario with {len(scenario.ride_requests)} ride requests")
