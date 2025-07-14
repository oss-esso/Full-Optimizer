import numpy as np
from typing import List, Dict, Tuple
from vrp_data_models import VRPInstance, Location, Vehicle, RideRequest

# Default truck speed profiles for different vehicle types
DEFAULT_TRUCK_SPEED_RATIOS = {
    'standard': {
        'motorway': 130 / 130,      # Light trucks: 80 km/h, Cars: ~130 km/h
        'trunk': 100 / 100,         # Light trucks: 70 km/h, Cars: ~100 km/h
        'primary': 90 / 90,        # Light trucks: 60 km/h, Cars: ~90 km/h
        'secondary': 70 / 70,      # Light trucks: 50 km/h, Cars: ~70 km/h
        'tertiary': 60 / 60,       # Light trucks: 45 km/h, Cars: ~60 km/h
        'residential': 40 / 50,    # Light trucks: 30 km/h, Cars: ~50 km/h
        'service': 25 / 30,        # Light trucks: 25 km/h, Cars: ~30 km/h
        'default': 0.80            # Light trucks ~20% slower than cars
    },
    'heavy': {
        'motorway': 90 / 130,      # Heavy trucks: 70 km/h, Cars: ~130 km/h
        'trunk': 80 / 100,         # Heavy trucks: 60 km/h, Cars: ~100 km/h
        'primary': 70 / 90,        # Heavy trucks: 50 km/h, Cars: ~90 km/h
        'secondary': 70 / 70,      # Heavy trucks: 40 km/h, Cars: ~70 km/h
        'tertiary': 60 / 60,       # Heavy trucks: 35 km/h, Cars: ~60 km/h
        'residential': 25 / 50,    # Heavy trucks: 25 km/h, Cars: ~50 km/h
        'service': 15 / 30,        # Heavy trucks: 20 km/h, Cars: ~30 km/h
        'default': 0.65            # Heavy trucks ~35% slower than cars
    }
}

def calculate_cargo_volume(weight_kg: float, cargo_density_kg_m3: float = 200.0) -> float:
    """
    Calculate cargo volume based on weight and estimated density.
    
    Args:
        weight_kg: Cargo weight in kilograms
        cargo_density_kg_m3: Cargo density in kg/m³ (default: 200 kg/m³ for mixed cargo)
    
    Returns:
        Volume in cubic meters
    
    Note:
        Typical cargo densities:
        - Dense cargo (machinery, metals): 500-800 kg/m³
        - General cargo (packages, boxes): 150-300 kg/m³
        - Light cargo (textiles, electronics): 50-150 kg/m³
        - Mixed cargo (default): ~200 kg/m³
    """
    return weight_kg / cargo_density_kg_m3





def create_furgoni_scenario(average_speed_kmh=70) -> VRPInstance:
    """Create VRP scenario from the furgoni Excel data with calculated weights and time windows.
    Fixed to match MODA_small format for solvability, now with unique depot bays for all pickup-to-depot requests."
    """
    import random
    random.seed(42)  # For reproducibility
    
    instance = VRPInstance("Furgoni Delivery Scenario")
    instance.is_realistic = True  # Added missing attribute
    
    # Create depot location (Milan coordinates)
    depot = Location("depot", 9.1896, 45.4642, demand=0, 
                    time_window_start=0, time_window_end=1440, service_time=5)
    depot.address = "Milan Distribution Center (labeled as Asti)"
    depot.lat = 45.4642  # Milan latitude
    depot.lon = 9.1896   # Milan longitude
    instance.add_location(depot)
    


    # Vehicle data with calculated capacities - matching MODA format with max_time
    vehicles_data = {
        "FURGONE 1": 3500,
        "FURGONE 2": 3500, 
        "FURGONE 3": 3500,
        "CAMION 16 T.": 16000,
        "FURGONE 6": 3500,
        "CAMION 7,5 T.": 7500,
        "FURGONE 8": 3500,
        "FURGONE 9": 3500,
        "FURGONE 10": 3500,
        "FURGONE 11": 3500,
    }
    
    # Add vehicles to the scenario with proper attributes
    for vehicle_name, capacity in vehicles_data.items():
        vehicle_id = vehicle_name.lower().replace(" ", "_").replace(",", "").replace(".", "")
        # Set max_time and volume constraints based on vehicle type (matching MODA logic)
        if "CAMION" in vehicle_name:
            max_time =  900 # Heavy vehicles: 9 hours (EU regulations)
            vehicle_type = "heavy"
            max_driving_time = 450 # 4.5 hours
            required_break_time = 45.0
            max_total_work_time = 900
            # Volume capacity for heavy trucks based on their weight capacity
            if capacity >= 15000:  # 16T truck
                volume_capacity = 67.0  # 67 m³ for 16t trucks (same as 24t trucks)
            else:  # 7.5T truck
                volume_capacity = 40.0  # 40 m³ for 7.5t trucks
        else:
            max_time = 900  # Light vehicles: 10 hours
            vehicle_type = "standard"
            max_driving_time = 900.0
            required_break_time = 0.0
            max_total_work_time = 900.0
            volume_capacity = 20.0  # 20 m³ for furgoni (3.5t trucks)
            
        vehicle = Vehicle(vehicle_id, capacity=capacity, volume_capacity=volume_capacity, 
                         depot_id="depot", max_time=max_time)
        vehicle.vehicle_type = vehicle_type
        vehicle.max_driving_time = max_driving_time
        vehicle.required_break_time = required_break_time
        vehicle.max_total_work_time = max_total_work_time
        instance.add_vehicle(vehicle)
    
    # Delivery and pickup addresses with time windows from Excel
    # Format: (location_id, address, lon, lat, tw_start, tw_end, service_time, is_pickup)
    locations_data = [
        # FURGONE 1 (1 delivery - 750kg total)
        ("schonaich_de", "gentile gusto carl Zeiss Str. 4 71101 Schonaich DE", 9.0000, 48.6667, 0, 1440, 30, False),
        ("malmoe_sweden", "malmoe sweden menarini diag.slr", 13.0038, 55.6050, 0, 1440, 30, False),
        # FURGONE 2 (4 deliveries - 831kg each for 3.3T total)
        ("salvazzano_pd", "Via Pelosa 20 35030 Salvazzano Dentro PD", 11.7833, 45.3833, 0, 1440, 20, False),
        ("badia_polesine_ro", "Via L. Da Vinci 537 45021 Badia Polesine RO", 11.5000, 45.1167, 0, 1440, 20, False),
        ("badia_polesine_ro_2", "Via L. Da Vinci 537 45021 Badia Polesine RO", 11.5000, 45.1167, 0, 1440, 20, False),
        ("marostica_vi", "via Milano 1 36063 Marostica VI", 11.6500, 45.7500, 0, 1440, 20, False),
        
        # FURGONE 3 (1 delivery - 1500kg total)
        ("st_martin_crau", "Rue Gay Lussac 3 St. Martin de Crau 13310", 4.8000, 43.6333, 0, 1440, 30, False),
        
        # CAMION 16 T. (2 deliveries)
        ("sant_olcese_ge", "Piazza Guglielmo Marconi 40 16010 Sant'Olcese GE", 8.9333, 44.5167, 0, 1440, 45, False),
        ("casarza_ligure_ge", "Via Tangoni 30/32 I-16030 Casarza Ligure GE", 9.4667, 44.2833, 0, 1440, 45, False),
        
        # FURGONE 5 (2 deliveries to same location - 750kg each for 1.5T total)
        ("quinto_do_anjo", "Quinta De Marquesa 1CCI Portoga 10218 29050-557 Quinto Do Anjo", -8.9000, 38.5667, 0, 1440, 30, False),
        ("quinto_do_anjo_2", "Quinta De Marquesa 1CCI Portoga 10218 29050-557 Quinto Do Anjo", -8.9000, 38.5667, 0, 1440, 30, False),
        # FURGONE 6 (1 delivery - 1500kg total)
        ("paris_fr", "Rue de L'Abbaye 10 PARIGI FR", 2.3522, 48.8566, 0, 1440, 30, False),
        
        # CAMION 7.5 T. (1 delivery - 2755kg)
        ("chiva_spagna", "Poligono I La Pamilla 196 46370 Chiva Spagna", -0.7167, 39.4667, 0, 1440, 45, False),
        
        # FURGONE 8 (1 delivery + 3 pickups with time constraints)
        ("cerro_al_lambro_mi", "Via Autosole 7 20070 Cerro Al Lambro MI", 9.3000, 45.3333, 0, 1440, 20, False),
        ("cormano_mi", "via dell'Artigianato 1 20032 Cormano MI", 9.1667, 45.5333, 0, 1440, 20, True),
        ("sicor_caronno", "Sicor Caronno", 8.9000, 45.6000, 0, 840, 15, True),  # Pickup before 14:00
        ("ferno_va", "superstrada per Malpensa uscita Cargocity Torre D 5 piano 21010 Ferno VA", 8.7167, 45.6167, 0, 510, 15, True),  # Pickup before 8:30
        
        # FURGONE 9 (4 deliveries + 1 pickup)
        ("imperia_1", "via Filippo Airenti 2 18100 Imperia IM", 8.0333, 43.8833, 0, 1440, 20, False),
        ("imperia_2", "via Nazionale 356/3 18100 Imperia IM", 8.0333, 43.8833, 0, 1440, 20, False),
        ("villar_cuneo", "via I Maggio 43 Cuneo 12020 Villar San Costanzo", 7.5333, 44.3833, 0, 1440, 20, False),
        ("villar_cuneo_2", "via I Maggio 43 Cuneo 12020 Villar San Costanzo", 7.5333, 44.3833, 0, 1440, 20, False),
        ("castellalfero_at", "Via Statale 25/A Castell'Alfero 14033 AT", 8.2167, 44.9667, 0, 1440, 15, True),
        
        # FURGONE 10 (1 delivery + 1 pickup)
        ("osimo_an", "via Francesco Crispi 2 60027 Osimo AN", 13.4833, 43.4833, 0, 1440, 20, False),
        ("castelfidardo", "via Jesina 27/P 60022 Castelfidardo", 13.5500, 43.4667, 0, 1440, 15, True),
        
        # FURGONE 11 (2 deliveries + 1 pickup)
        ("somaglia_lo", "strada Provinciale 223 26867 Località Cantonale Somaglia LO", 9.6667, 45.1667, 0, 1440, 20, False),
        ("capriate_bg", "via Bergamo 61/63 24042 Capriate San Gervasio BG", 9.5333, 45.6000, 0, 1440, 20, False),
        ("cazzano_bg", "via Cavalier Pietro Radici 19 24026 Cazzano Sant'andrea BG", 9.8000, 45.7500, 0, 1440, 15, True),
    ]
    
    # Add all locations with proper attributes (matching MODA format)
    for location_id, address, lon, lat, tw_start, tw_end, service_time, is_pickup in locations_data:
        location = Location(location_id, lon, lat, demand=0,
                           time_window_start=tw_start, time_window_end=tw_end, 
                           service_time=service_time)
        location.address = address
        location.lat = lat  # Added missing lat/lon attributes
        location.lon = lon
        location.is_pickup = is_pickup
        instance.add_location(location)
    
    # Create delivery/pickup requests using MODA depot bay pattern
    # Separate delivery cargo (depot bays → destinations) from pickup cargo (pickups → depot bays)
    
    # Delivery cargo data: create depot bays for each delivery
    delivery_cargo = [
        # FURGONE 1
        ("schonaich_de", 750),
        
        # FURGONE 2
        ("salvazzano_pd", 831), 
        ("badia_polesine_ro", 750), 
        ("badia_polesine_ro_2", 810), 
        ("marostica_vi", 950),
        
        # FURGONE 3
        ("st_martin_crau", 1500),
        
        # CAMION 16 T.
        ("sant_olcese_ge", 4300), 
        ("casarza_ligure_ge", 4300),
        
        # FURGONE 6
        ("paris_fr", 1500),
        
        # CAMION 7.5 T.
        ("chiva_spagna", 2700),
        
        # FURGONE 5 deliveries (2 to same location)
        ("quinto_do_anjo", 750),
        ("quinto_do_anjo_2", 750),


        # FURGONE 8 delivery
        ("cerro_al_lambro_mi", 800),
        
        # FURGONE 9 deliveries
        ("imperia_1", 600),
        ("imperia_2", 700),
        ("villar_cuneo", 550),
        ("villar_cuneo_2", 665),
        
        # FURGONE 10 delivery
        ("osimo_an", 750),
        
        # FURGONE 11 deliveries
        ("somaglia_lo", 1100),
        ("capriate_bg", 900),
    ]
    
    # Pickup cargo data: direct pickup → depot requests
    pickup_cargo = [
        # FURGONE 8 pickups
        ("cormano_mi", 700),
        ("sicor_caronno", 750),
        ("ferno_va", 820),
        
        # FURGONE 9 pickup
        ("castellalfero_at", 650),
        
        # FURGONE 10 pickup
        ("castelfidardo", 550),
        
        # FURGONE 11 pickup
        ("cazzano_bg", 1200),
    ]
    
    # Create depot bays for delivery cargo (following MODA pattern)
    depot_requests = []
    request_id = 1
    depot_location = instance.locations["depot"]

    for dropoff_id, cargo_weight in delivery_cargo:
        # Create virtual pickup bay at depot for this delivery
        pickup_bay_id = f"depot_bay_{request_id}"
        pickup_bay = Location(
            pickup_bay_id, 
            depot_location.x,  # Same longitude as main depot
            depot_location.y,  # Same latitude as main depot
            demand=0,
            service_time=5     # Minimal loading time at depot bay
        )
        pickup_bay.lat = depot_location.lat
        pickup_bay.lon = depot_location.lon
        pickup_bay.address = f"Depot Loading Bay {request_id}"
        instance.add_location(pickup_bay)
        # Create request from depot bay to dropoff location
        cargo_volume = calculate_cargo_volume(cargo_weight)  # Calculate volume based on weight  
        depot_request = RideRequest(f"depot_request_{request_id}", pickup_bay_id, dropoff_id, passengers=cargo_weight, volume=cargo_volume)
        instance.add_ride_request(depot_request)
        depot_requests.append(depot_request)
        request_id += 1

    # Create pickup requests (pickup location → unique depot bay)
    pickup_requests = []
    pickup_bay_counter = 1
    for pickup_id, cargo_weight in pickup_cargo:
        # Create a unique depot bay for each pickup
        pickup_bay_id = f"pickup_bay_{pickup_bay_counter}"
        pickup_bay = Location(
            pickup_bay_id,
            depot_location.x,
            depot_location.y,
            demand=0,
            service_time=5
        )
        pickup_bay.lat = depot_location.lat
        pickup_bay.lon = depot_location.lon
        pickup_bay.address = f"Depot Pickup Bay {pickup_bay_counter}"
        instance.add_location(pickup_bay)
        # Create request from pickup location to unique depot bay
        cargo_volume = calculate_cargo_volume(cargo_weight)  # Calculate volume based on weight
        pickup_request = RideRequest(f"pickup_request_{request_id}", pickup_id, pickup_bay_id, passengers=cargo_weight, volume=cargo_volume)
        instance.add_ride_request(pickup_request)
        pickup_requests.append(pickup_request)
        request_id += 1
        pickup_bay_counter += 1
    
    # Calculate distance matrix using same method as MODA
    instance.calculate_distance_matrix(distance_method="manhattan")
    
    # Add summary statistics (matching MODA format)
    total_capacity = sum(vehicle.capacity for vehicle in instance.vehicles.values())
    total_delivery_weight = sum(r.passengers for r in depot_requests)  # Changed from delivery_requests
    total_pickup_weight = sum(r.passengers for r in pickup_requests)
    total_cargo_weight = total_delivery_weight + total_pickup_weight
    
    print(f"Created Furgoni scenario with depot bay pattern (unique bays for all pickups and deliveries):")
    print(f"  - Total locations: {len(instance.locations)}")
    print(f"  - Main depot: 1")
    print(f"  - Depot bays: {len(depot_requests)} (for deliveries)")
    print(f"  - Pickup bays: {len(pickup_requests)} (for pickups)")
    print(f"  - Delivery destinations: {len([l for l in instance.locations.values() if not getattr(l, 'is_pickup', False) and l.id != 'depot' and not l.id.startswith('depot_bay') and not l.id.startswith('pickup_bay')])}")
    print(f"  - Field pickup locations: {len([l for l in instance.locations.values() if getattr(l, 'is_pickup', False)])}")
    print(f"  - Vehicles: {len(instance.vehicles)}")
    print(f"  - Ride requests: {len(instance.ride_requests)} ({len(depot_requests)} depot→delivery + {len(pickup_requests)} pickup→depot)")
    print(f"Fleet composition: {len([v for v in instance.vehicles.values() if v.vehicle_type == 'standard'])} light vehicles + {len([v for v in instance.vehicles.values() if v.vehicle_type == 'heavy'])} heavy vehicles")
    
    print(f"\nCapacity utilization:")
    print(f"  - Total fleet capacity: {total_capacity:,} kg")
    print(f"  - Depot cargo (via bays): {total_delivery_weight:,} kg")
    print(f"  - Field pickup cargo: {total_pickup_weight:,} kg")
    print(f"  - Total cargo: {total_cargo_weight:,} kg = {total_cargo_weight/total_capacity*100:.1f}% of fleet capacity")
    
    # Add volume constraint summary
    total_volume_capacity = sum(vehicle.volume_capacity for vehicle in instance.vehicles.values())
    total_delivery_volume = sum(r.volume for r in depot_requests)
    total_pickup_volume = sum(r.volume for r in pickup_requests)
    total_cargo_volume = total_delivery_volume + total_pickup_volume
    
    print(f"\nVolume constraints:")
    print(f"  - Total fleet volume capacity: {total_volume_capacity:.1f} m³")
    print(f"  - Depot cargo volume: {total_delivery_volume:.1f} m³")
    print(f"  - Field pickup volume: {total_pickup_volume:.1f} m³")
    print(f"  - Total cargo volume: {total_cargo_volume:.1f} m³ = {total_cargo_volume/total_volume_capacity*100:.1f}% of fleet volume capacity")
    print(f"  - Vehicle types: Furgoni (20m³), Camion 7.5T (40m³), Camion 16T (67m³)")
    
    print(f"\n📦 Depot Bay Strategy (matching MODA_small, now for all pickups):")
    print(f"  - {len(depot_requests)} depot bays at same coordinates as main depot (for deliveries)")
    print(f"  - {len(pickup_requests)} unique pickup bays at depot (for pickups)")
    print(f"  - Each bay serves 1 request (delivery or pickup)")
    print(f"  - 💡 Optimizer decides which bays to visit per vehicle")
    
    return instance