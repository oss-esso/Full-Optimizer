"""
Offline geographic data for VRP scenarios - no API calls required.
Uses pre-compiled coordinate databases and realistic address data.
"""

import os
import json
import random
from typing import Dict, List, Tuple
from vrp_data_models import VRPInstance, Location, Vehicle, RideRequest

# Real business coordinates for major US cities (manually curated)
CITY_BUSINESS_DATA = {
    "new_york": {
        "center": (-74.0060, 40.7128),
        "businesses": [
            (-74.0060, 40.7128, "Times Square Distribution Center"),
            (-74.0104, 40.7069, "Wall Street Financial Services"),
            (-73.9857, 40.7484, "5th Avenue Retail Store"),
            (-73.9776, 40.7831, "Broadway Theater Supply"),
            (-74.0134, 40.7335, "Tribeca Art Gallery"),
            (-73.9903, 40.7505, "Park Avenue Medical Center"),
            (-73.9665, 40.7781, "Columbus Circle Restaurant"),
            (-74.0047, 40.7505, "Theater District Hotel"),
            (-73.9924, 40.7157, "Nolita Boutique"),
            (-73.9730, 40.7831, "Central Park Cafe"),
            (-74.0041, 40.7282, "Chelsea Market Vendor"),
            (-73.9865, 40.7731, "Lincoln Center Office"),
        ]
    },
    "los_angeles": {
        "center": (-118.2437, 34.0522),
        "businesses": [
            (-118.2437, 34.0522, "Downtown LA Distribution"),
            (-118.2917, 34.0928, "Hollywood Entertainment"),
            (-118.3964, 34.0195, "West Hollywood Design Studio"),
            (-118.4912, 34.0194, "Santa Monica Pier Restaurant"),
            (-118.2649, 34.0928, "East Hollywood Production"),
            (-118.3370, 34.0823, "Beverly Hills Boutique"),
            (-118.2074, 34.0522, "Arts District Warehouse"),
            (-118.3431, 34.0669, "Miracle Mile Museum"),
            (-118.4039, 34.0259, "West LA Medical Center"),
            (-118.2871, 34.1139, "Silver Lake Coffee Shop"),
        ]
    },
    "chicago": {
        "center": (-87.6298, 41.8781),
        "businesses": [
            (-87.6298, 41.8781, "Loop Business District"),
            (-87.6244, 41.8847, "Millennium Park Events"),
            (-87.6064, 41.8827, "Navy Pier Tourism"),
            (-87.6715, 41.9301, "Lincoln Park Services"),
            (-87.6039, 41.8962, "Gold Coast Retail"),
            (-87.6051, 41.8673, "Museum Campus Education"),
            (-87.6563, 41.9301, "DePaul University Area"),
            (-87.6169, 41.8919, "River North Gallery"),
            (-87.6431, 41.8895, "Old Town Restaurant"),
            (-87.6003, 41.8677, "Grant Park Vendor"),
        ]
    }
}

def create_realistic_city_scenario(city_name: str, scenario_type: str = "delivery") -> VRPInstance:
    """Create realistic scenario using offline coordinate database."""
    
    if city_name not in CITY_BUSINESS_DATA:
        raise ValueError(f"City {city_name} not in offline database")
    
    city_data = CITY_BUSINESS_DATA[city_name]
    center_lon, center_lat = city_data["center"]
    businesses = city_data["businesses"]
    
    if scenario_type == "delivery":
        instance = VRPInstance(f"{city_name.title()} Delivery (Offline Realistic)")
        instance.is_realistic = True
        
        # Add depot at city center
        depot = Location("depot", center_lon, center_lat, demand=0)
        depot.address = f"{city_name.title()} Distribution Center"
        depot.lat = center_lat
        depot.lon = center_lon
        instance.add_location(depot)
        
        # Add customer locations
        for i, (lon, lat, business_name) in enumerate(businesses[:10]):  # Limit to 10 customers
            customer = Location(f"customer_{i+1}", lon, lat, demand=random.randint(1, 5))
            customer.address = business_name
            customer.lat = lat
            customer.lon = lon
            instance.add_location(customer)
        
        # Add vehicles
        for i in range(3):
            instance.add_vehicle(Vehicle(f"vehicle_{i+1}", capacity=15, depot_id="depot"))
            
    elif scenario_type == "rideshare":
        instance = VRPInstance(f"{city_name.title()} Rideshare (Offline Realistic)")
        instance.is_realistic = True
        
        # Add depots
        depot1 = Location("depot_1", center_lon, center_lat, demand=0)
        depot1.address = f"{city_name.title()} Rideshare Hub 1"
        depot1.lat = center_lat
        depot1.lon = center_lon
        instance.add_location(depot1)
        
        # Second depot slightly offset
        depot2 = Location("depot_2", center_lon + 0.02, center_lat + 0.01, demand=0)
        depot2.address = f"{city_name.title()} Rideshare Hub 2"
        depot2.lat = center_lat + 0.01
        depot2.lon = center_lon + 0.02
        instance.add_location(depot2)
        
        # Add pickup/dropoff pairs
        for i in range(min(4, len(businesses) // 2)):
            pickup_lon, pickup_lat, pickup_business = businesses[i * 2]
            dropoff_lon, dropoff_lat, dropoff_business = businesses[i * 2 + 1]
            
            pickup = Location(f"pickup_{i+1}", pickup_lon, pickup_lat, demand=0)
            pickup.address = f"Pickup at {pickup_business}"
            pickup.lat = pickup_lat
            pickup.lon = pickup_lon
            instance.add_location(pickup)
            
            dropoff = Location(f"dropoff_{i+1}", dropoff_lon, dropoff_lat, demand=0)
            dropoff.address = f"Dropoff at {dropoff_business}"
            dropoff.lat = dropoff_lat
            dropoff.lon = dropoff_lon
            instance.add_location(dropoff)
            
            # Add ride request
            instance.add_ride_request(RideRequest(
                f"request_{i+1}", f"pickup_{i+1}", f"dropoff_{i+1}", 
                passengers=random.randint(1, 3)
            ))
        
        # Add vehicles
        instance.add_vehicle(Vehicle("vehicle_1", capacity=4, depot_id="depot_1"))
        instance.add_vehicle(Vehicle("vehicle_2", capacity=4, depot_id="depot_2"))
    
    # Calculate distance matrix using real geographic coordinates
    instance.calculate_distance_matrix(distance_method="euclidean")
    
    return instance

def create_all_offline_scenarios() -> Dict[str, VRPInstance]:
    """Create all offline realistic scenarios."""
    scenarios = {}
    
    # Create delivery scenarios for major cities
    for city in ["new_york", "los_angeles", "chicago"]:
        try:
            delivery_scenario = create_realistic_city_scenario(city, "delivery")
            scenarios[f"{city}_delivery_offline"] = delivery_scenario
            
            rideshare_scenario = create_realistic_city_scenario(city, "rideshare")
            scenarios[f"{city}_rideshare_offline"] = rideshare_scenario
            
        except Exception as e:
            print(f"Error creating offline scenario for {city}: {e}")
            continue
    
    print(f"Created {len(scenarios)} offline realistic scenarios")
    return scenarios
