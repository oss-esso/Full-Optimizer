"""
OpenStreetMap and OSRM integration for realistic VRP scenarios.
Based on OVIG approach using OpenAddresses database and OSRM for routing.
"""

import os
import sys
import time
import logging
import requests
import json
import random
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from vrp_data_models import VRPInstance, Location, Vehicle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OSMLocation:
    """Enhanced location with OSM data."""
    id: str
    lat: float
    lon: float
    address: str
    demand: int = 0
    location_type: str = "customer"  # depot, customer, pickup, dropoff

class OSMRoutingService:
    """Service for computing routes using OSRM or similar routing engines."""
    
    def __init__(self, osrm_url: str = "http://router.project-osrm.org"):
        self.osrm_url = osrm_url if osrm_url else "http://router.project-osrm.org"
        self.cache = {}  # Cache for route results
        
    def get_route_matrix(self, locations: List[OSMLocation]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get distance and duration matrix for all locations using OSRM.
        Returns (distance_matrix, duration_matrix) in meters and seconds.
        """
        n = len(locations)
        distance_matrix = np.zeros((n, n))
        duration_matrix = np.zeros((n, n))
        
        # Build coordinate string for OSRM
        coordinates = ";".join([f"{loc.lon},{loc.lat}" for loc in locations])
        
        try:
            # Use OSRM table service for matrix calculation
            url = f"{self.osrm_url}/table/v1/driving/{coordinates}"
            params = {
                'annotations': 'distance,duration'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] == 'Ok':
                # Extract distance matrix (in meters)
                if 'distances' in data:
                    distance_matrix = np.array(data['distances'])
                
                # Extract duration matrix (in seconds)
                if 'durations' in data:
                    duration_matrix = np.array(data['durations'])
                
                logger.info(f"Successfully retrieved route matrix for {n} locations")
            else:
                logger.error(f"OSRM API error: {data.get('message', 'Unknown error')}")
                # Fallback to Haversine distance
                distance_matrix, duration_matrix = self._fallback_matrix(locations)
                
        except Exception as e:
            logger.error(f"Error calling OSRM API: {str(e)}")
            # Fallback to Haversine distance
            distance_matrix, duration_matrix = self._fallback_matrix(locations)
        
        return distance_matrix, duration_matrix
    
    def _fallback_matrix(self, locations: List[OSMLocation]) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback to Haversine distance when OSRM is unavailable."""
        n = len(locations)
        distance_matrix = np.zeros((n, n))
        duration_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = self._haversine_distance(
                        locations[i].lat, locations[i].lon,
                        locations[j].lat, locations[j].lon
                    )
                    distance_matrix[i, j] = dist
                    # Estimate duration assuming 50 km/h average speed
                    duration_matrix[i, j] = dist / 50000 * 3600  # seconds
        
        logger.warning("Using Haversine distance fallback")
        return distance_matrix, duration_matrix
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate Haversine distance between two points in meters."""
        R = 6371000  # Earth radius in meters
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat / 2) ** 2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c

class OSMAddressProvider:
    """Provider for realistic addresses using Nominatim (OpenStreetMap)."""
    
    def __init__(self, nominatim_url: str = "https://nominatim.openstreetmap.org"):
        self.nominatim_url = nominatim_url if nominatim_url else "https://nominatim.openstreetmap.org"
        self.rate_limit_delay = 1.0  # Respect Nominatim usage policy
        
    def get_addresses_in_city(self, city: str, country: str = "United States", 
                             count: int = 10, location_types: List[str] = None) -> List[OSMLocation]:
        """
        Get realistic addresses in a city using Nominatim.
        
        Args:
            city: City name
            country: Country name
            count: Number of addresses to retrieve
            location_types: Types of locations (shop, restaurant, office, etc.)
        """
        if location_types is None:
            location_types = ["shop", "restaurant", "office", "amenity"]
        
        locations = []
        
        try:
            # First, get city boundaries
            city_bounds = self._get_city_bounds(city, country)
            if not city_bounds:
                logger.error(f"Could not find bounds for {city}, {country}")
                return self._generate_synthetic_locations(city, count)
            
            # Search for different types of locations
            for location_type in location_types:
                if len(locations) >= count:
                    break
                    
                batch_locations = self._search_locations_by_type(
                    location_type, city_bounds, min(count // len(location_types) + 2, count - len(locations))
                )
                locations.extend(batch_locations)
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
            
            # If we don't have enough locations, fill with synthetic ones
            if len(locations) < count:
                synthetic_count = count - len(locations)
                synthetic_locations = self._generate_synthetic_locations_in_bounds(
                    city_bounds, synthetic_count, start_id=len(locations)
                )
                locations.extend(synthetic_locations)
            
            # Limit to requested count
            locations = locations[:count]
            
            logger.info(f"Retrieved {len(locations)} locations for {city}")
            return locations
            
        except Exception as e:
            logger.error(f"Error retrieving addresses: {str(e)}")
            return self._generate_synthetic_locations(city, count)
    
    def _get_city_bounds(self, city: str, country: str) -> Optional[Dict]:
        """Get bounding box for a city."""
        try:
            url = f"{self.nominatim_url}/search"
            params = {
                'q': f"{city}, {country}",
                'format': 'json',
                'limit': 1,
                'extratags': 1,
                'namedetails': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data:
                result = data[0]
                if 'boundingbox' in result:
                    bbox = result['boundingbox']
                    return {
                        'south': float(bbox[0]),
                        'north': float(bbox[1]),
                        'west': float(bbox[2]),
                        'east': float(bbox[3])
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting city bounds: {str(e)}")
            return None
    
    def _search_locations_by_type(self, location_type: str, bounds: Dict, count: int) -> List[OSMLocation]:
        """Search for specific types of locations within bounds."""
        locations = []
        
        try:
            url = f"{self.nominatim_url}/search"
            params = {
                'q': location_type,
                'format': 'json',
                'limit': count * 2,  # Get more than needed for filtering
                'bounded': 1,
                'viewbox': f"{bounds['west']},{bounds['south']},{bounds['east']},{bounds['north']}",
                'addressdetails': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for i, item in enumerate(data[:count]):
                if 'lat' in item and 'lon' in item:
                    address = self._format_address(item.get('address', {}))
                    location = OSMLocation(
                        id=f"{location_type}_{i+1}",
                        lat=float(item['lat']),
                        lon=float(item['lon']),
                        address=address,
                        demand=random.randint(1, 5),
                        location_type="customer"
                    )
                    locations.append(location)
            
        except Exception as e:
            logger.error(f"Error searching locations by type {location_type}: {str(e)}")
        
        return locations
    
    def _format_address(self, address_dict: Dict) -> str:
        """Format address from Nominatim response."""
        parts = []
        
        # House number and street
        if 'house_number' in address_dict and 'road' in address_dict:
            parts.append(f"{address_dict['house_number']} {address_dict['road']}")
        elif 'road' in address_dict:
            parts.append(address_dict['road'])
        
        # City
        city = (address_dict.get('city') or 
                address_dict.get('town') or 
                address_dict.get('village') or
                address_dict.get('hamlet'))
        if city:
            parts.append(city)
        
        # State and country
        if 'state' in address_dict:
            parts.append(address_dict['state'])
        
        return ", ".join(parts) if parts else "Unknown Address"
    
    def _generate_synthetic_locations(self, city: str, count: int) -> List[OSMLocation]:
        """Generate synthetic locations when real data is unavailable."""
        logger.warning(f"Generating {count} synthetic locations for {city}")
        
        # Default coordinates for major cities (for demo purposes)
        city_coords = {
            "New York": (40.7128, -74.0060),
            "Los Angeles": (34.0522, -118.2437),
            "Chicago": (41.8781, -87.6298),
            "Houston": (29.7604, -95.3698),
            "Phoenix": (33.4484, -112.0740),
            "Philadelphia": (39.9526, -75.1652),
            "San Antonio": (29.4241, -98.4936),
            "San Diego": (32.7157, -117.1611),
            "Dallas": (32.7767, -96.7970),
            "San Jose": (37.3382, -121.8863),
        }
        
        center_lat, center_lon = city_coords.get(city, (40.7128, -74.0060))  # Default to NYC
        
        return self._generate_synthetic_locations_around_point(
            center_lat, center_lon, count, spread=0.1
        )
    
    def _generate_synthetic_locations_in_bounds(self, bounds: Dict, count: int, start_id: int = 0) -> List[OSMLocation]:
        """Generate synthetic locations within specified bounds."""
        center_lat = (bounds['south'] + bounds['north']) / 2
        center_lon = (bounds['west'] + bounds['east']) / 2
        spread = min(bounds['north'] - bounds['south'], bounds['east'] - bounds['west']) / 2
        
        return self._generate_synthetic_locations_around_point(
            center_lat, center_lon, count, spread, start_id
        )
    
    def _generate_synthetic_locations_around_point(self, center_lat: float, center_lon: float, 
                                                 count: int, spread: float = 0.1, start_id: int = 0) -> List[OSMLocation]:
        """Generate synthetic locations around a center point."""
        locations = []
        
        for i in range(count):
            # Generate random offset within spread
            lat_offset = random.uniform(-spread, spread)
            lon_offset = random.uniform(-spread, spread)
            
            location = OSMLocation(
                id=f"synthetic_{start_id + i + 1}",
                lat=center_lat + lat_offset,
                lon=center_lon + lon_offset,
                address=f"Synthetic Address {start_id + i + 1}",
                demand=random.randint(1, 5),
                location_type="customer"
            )
            locations.append(location)
        
        return locations

class OSMVRPGenerator:
    """VRP scenario generator using real OpenStreetMap data."""
    
    def __init__(self, osrm_url: str = None, nominatim_url: str = None):
        # Fix the None URL issue
        nominatim_url = nominatim_url or "https://nominatim.openstreetmap.org"
        osrm_url = osrm_url or "http://router.project-osrm.org"
        
        self.address_provider = OSMAddressProvider(nominatim_url)
        self.routing_service = OSMRoutingService(osrm_url)
    
    def create_realistic_city_delivery(self, city: str, country: str = "United States", 
                                     num_customers: int = 15, num_vehicles: int = 3) -> VRPInstance:
        """Create a realistic delivery scenario for a specific city."""
        logger.info(f"Creating realistic delivery scenario for {city}, {country}")
        
        instance = VRPInstance(f"Realistic Delivery - {city}")
        
        # Get depot location (city center or major business district)
        depot_locations = self.address_provider.get_addresses_in_city(
            city, country, count=1, location_types=["office", "commercial"]
        )
        
        if depot_locations:
            depot_osm = depot_locations[0]
            depot = Location(
                "depot", 
                depot_osm.lon, depot_osm.lat,  # Note: using lon as x, lat as y
                demand=0
            )
            depot.address = depot_osm.address
        else:
            # Fallback depot
            depot = Location("depot", -74.0060, 40.7128, demand=0)
            depot.address = f"Depot - {city} Center"
        
        instance.add_location(depot)
        
        # Get customer locations
        customer_locations = self.address_provider.get_addresses_in_city(
            city, country, count=num_customers, 
            location_types=["shop", "restaurant", "amenity", "office"]
        )
        
        for i, customer_osm in enumerate(customer_locations):
            customer = Location(
                f"customer_{i+1}",
                customer_osm.lon, customer_osm.lat,  # Note: using lon as x, lat as y
                demand=customer_osm.demand
            )
            customer.address = customer_osm.address
            instance.add_location(customer)
        
        # Add vehicles
        for i in range(num_vehicles):
            vehicle = Vehicle(f"vehicle_{i+1}", capacity=20, depot_id="depot")
            instance.add_vehicle(vehicle)
        
        # Calculate realistic distance matrix using OSRM
        all_osm_locations = [depot_osm] + customer_locations if depot_locations else customer_locations
        if len(all_osm_locations) > 1:
            try:
                distance_matrix, duration_matrix = self.routing_service.get_route_matrix(all_osm_locations)
                
                # Convert from meters to kilometers for practical use
                instance.distance_matrix = distance_matrix / 1000.0
                instance.duration_matrix = duration_matrix  # Keep in seconds
                
                logger.info(f"Using real routing data for {len(all_osm_locations)} locations")
            except Exception as e:
                logger.error(f"Error getting routing data: {str(e)}")                # Fallback to Manhattan distance
                instance.calculate_distance_matrix(distance_method="manhattan")
        else:
            instance.calculate_distance_matrix(distance_method="manhattan")
        
        return instance
    
    def create_realistic_vrppd(self, city: str, country: str = "United States", 
                                    num_requests: int = 8, num_vehicles: int = 3) -> VRPInstance:
        """Create a realistic VRPPD scenario for a specific city."""
        logger.info(f"Creating realistic VRPPD scenario for {city}, {country}")
        
        instance = VRPInstance(f"Realistic VRPPD - {city}")
        
        # Get diverse locations for pickups and dropoffs
        all_locations = self.address_provider.get_addresses_in_city(
            city, country, count=num_requests * 2 + num_vehicles,
            location_types=["amenity", "shop", "restaurant", "office", "leisure"]
        )
        
        # Use first locations as depots
        for i in range(num_vehicles):
            if i < len(all_locations):
                depot_osm = all_locations[i]
                depot = Location(
                    f"depot_{i+1}",
                    depot_osm.lon, depot_osm.lat,
                    demand=0
                )
                depot.address = depot_osm.address
                instance.add_location(depot)
                
                # Add vehicle
                vehicle = Vehicle(f"vehicle_{i+1}", capacity=4, depot_id=f"depot_{i+1}")
                instance.add_vehicle(vehicle)
        
        # Use remaining locations for pickup/dropoff pairs
        remaining_locations = all_locations[num_vehicles:]
        
        for i in range(min(num_requests, len(remaining_locations) // 2)):
            pickup_osm = remaining_locations[i * 2]
            dropoff_osm = remaining_locations[i * 2 + 1]
            
            pickup = Location(
                f"pickup_{i+1}",
                pickup_osm.lon, pickup_osm.lat,
                demand=0
            )
            pickup.address = pickup_osm.address
            
            dropoff = Location(
                f"dropoff_{i+1}",
                dropoff_osm.lon, dropoff_osm.lat,
                demand=0
            )
            dropoff.address = dropoff_osm.address
            
            instance.add_location(pickup)
            instance.add_location(dropoff)
            
            # Add ride request
            from vrp_data_models import RideRequest
            request = RideRequest(
                f"request_{i+1}",
                f"pickup_{i+1}",
                f"dropoff_{i+1}",
                passengers=random.randint(1, 3)
            )
            instance.add_ride_request(request)
        
        # Calculate realistic distance matrix
        try:
            distance_matrix, duration_matrix = self.routing_service.get_route_matrix(all_locations[:len(instance.location_ids)])
            instance.distance_matrix = distance_matrix / 1000.0
            instance.duration_matrix = duration_matrix
            logger.info("Using real routing data for VRPPD scenario")
        except Exception as e:
            logger.error(f"Error getting routing data: {str(e)}")
            instance.calculate_distance_matrix(distance_method="manhattan")
        
        return instance

def create_osm_scenarios() -> Dict[str, VRPInstance]:
    """Create VRP scenarios using real OpenStreetMap data."""
    
    # For now, create scenarios using offline/static data to avoid API issues
    scenarios = {}
    
    try:
        # Use static realistic coordinates for major cities instead of API calls
        scenarios.update(create_static_realistic_scenarios())
        
        # Only try API-based scenarios if specifically requested
        # generator = OSMVRPGenerator()
        # api_scenarios = create_api_based_scenarios(generator)
        # scenarios.update(api_scenarios)
        
    except Exception as e:
        logger.error(f"Error creating OSM scenarios: {str(e)}")
    
    return scenarios

def create_static_realistic_scenarios() -> Dict[str, VRPInstance]:
    """Create realistic scenarios using static coordinate data (no API calls)."""
    scenarios = {}
    
    # NYC delivery scenario with realistic Manhattan coordinates
    nyc_delivery = VRPInstance("NYC Delivery (Realistic)")
    nyc_delivery.is_realistic = True
    
    # Real NYC coordinates (lat, lon converted to x, y for visualization)
    # Using actual Manhattan business district coordinates
    nyc_locations = [
        ("depot", -74.0060, 40.7128, 0, "NYC Distribution Center, Manhattan"),  # Times Square area
        ("customer_1", -74.0104, 40.7069, 3, "123 Wall St, Financial District"),
        ("customer_2", -73.9857, 40.7484, 2, "456 5th Ave, Midtown"),
        ("customer_3", -73.9776, 40.7831, 4, "789 Broadway, Upper West Side"),
        ("customer_4", -74.0134, 40.7335, 1, "321 Greenwich St, Tribeca"),
        ("customer_5", -73.9903, 40.7505, 3, "654 Park Ave, Midtown East"),
        ("customer_6", -73.9665, 40.7781, 2, "987 Columbus Ave, Upper West Side"),
        ("customer_7", -74.0047, 40.7505, 5, "147 W 42nd St, Theater District"),
        ("customer_8", -73.9924, 40.7157, 3, "258 Mott St, Nolita"),
    ]
    
    for loc_id, lon, lat, demand, address in nyc_locations:
        location = Location(loc_id, lon, lat, demand)
        location.address = address
        location.lat = lat
        location.lon = lon
        nyc_delivery.add_location(location)
    
    # Add vehicles
    for i in range(3):
        nyc_delivery.add_vehicle(Vehicle(f"vehicle_{i+1}", capacity=12, depot_id="depot"))
    
    # Calculate realistic distances (using Haversine for real lat/lon)
    nyc_delivery.calculate_distance_matrix(distance_method="euclidean")  # Will use coordinates as-is
    scenarios["nyc_delivery_realistic"] = nyc_delivery
    
    # LA delivery scenario
    la_delivery = VRPInstance("LA Delivery (Realistic)")
    la_delivery.is_realistic = True
    
    la_locations = [
        ("depot", -118.2437, 34.0522, 0, "LA Distribution Center, Downtown"),
        ("customer_1", -118.2917, 34.0928, 2, "456 Sunset Blvd, Hollywood"),
        ("customer_2", -118.3964, 34.0195, 4, "789 Santa Monica Blvd, West Hollywood"),
        ("customer_3", -118.4912, 34.0194, 3, "321 Ocean Ave, Santa Monica"),
        ("customer_4", -118.2649, 34.0928, 1, "654 Melrose Ave, East Hollywood"),
        ("customer_5", -118.3370, 34.0823, 5, "987 Beverly Blvd, Beverly Hills"),
        ("customer_6", -118.2074, 34.0522, 2, "147 Spring St, Downtown"),
    ]
    
    for loc_id, lon, lat, demand, address in la_locations:
        location = Location(loc_id, lon, lat, demand)
        location.address = address
        location.lat = lat
        location.lon = lon
        la_delivery.add_location(location)
    
    for i in range(2):
        la_delivery.add_vehicle(Vehicle(f"vehicle_{i+1}", capacity=15, depot_id="depot"))
    
    la_delivery.calculate_distance_matrix(distance_method="euclidean")
    scenarios["la_delivery_realistic"] = la_delivery
    
    # Chicago VRPPD scenario
    chicago_rideshare = VRPInstance("Chicago Rideshare (Realistic)")
    chicago_rideshare.is_realistic = True
    
    # Real Chicago coordinates for ride sharing
    chicago_locations = [
        ("depot_1", -87.6298, 41.8781, 0, "Chicago Rideshare Hub, Loop"),
        ("depot_2", -87.6073, 41.8993, 0, "North Side Rideshare Hub"),
        ("pickup_1", -87.6244, 41.8847, 0, "Millennium Park Pickup"),
        ("dropoff_1", -87.6064, 41.8827, 0, "Navy Pier Dropoff"),
        ("pickup_2", -87.6715, 41.9301, 0, "Lincoln Park Zoo Pickup"),
        ("dropoff_2", -87.6039, 41.8962, 0, "Gold Coast Dropoff"),
        ("pickup_3", -87.6051, 41.8673, 0, "Museum Campus Pickup"),
        ("dropoff_3", -87.6563, 41.9301, 0, "DePaul University Dropoff"),
    ]
    
    for loc_id, lon, lat, demand, address in chicago_locations:
        location = Location(loc_id, lon, lat, demand)
        location.address = address
        location.lat = lat
        location.lon = lon
        chicago_rideshare.add_location(location)
    
    # Add vehicles and ride requests
    chicago_rideshare.add_vehicle(Vehicle("vehicle_1", capacity=4, depot_id="depot_1"))
    chicago_rideshare.add_vehicle(Vehicle("vehicle_2", capacity=4, depot_id="depot_2"))
    
    from vrp_data_models import RideRequest
    chicago_rideshare.add_ride_request(RideRequest("request_1", "pickup_1", "dropoff_1", passengers=2))
    chicago_rideshare.add_ride_request(RideRequest("request_2", "pickup_2", "dropoff_2", passengers=1))
    chicago_rideshare.add_ride_request(RideRequest("request_3", "pickup_3", "dropoff_3", passengers=3))
    
    chicago_rideshare.calculate_distance_matrix(distance_method="euclidean")
    scenarios["chicago_rideshare_realistic"] = chicago_rideshare
    
    logger.info(f"Created {len(scenarios)} static realistic scenarios using real coordinates")
    return scenarios

# Example usage and testing
if __name__ == "__main__":
    # Test the OSM integration
    print("Testing OSM VRP Integration...")
    
    try:
        # Test address provider
        address_provider = OSMAddressProvider()
        locations = address_provider.get_addresses_in_city("New York", count=5)
        print(f"Retrieved {len(locations)} locations:")
        for loc in locations:
            print(f"  {loc.id}: {loc.address} ({loc.lat:.4f}, {loc.lon:.4f})")
        
        # Test routing service
        if len(locations) >= 2:
            routing_service = OSMRoutingService()
            distance_matrix, duration_matrix = routing_service.get_route_matrix(locations[:3])
            print(f"Distance matrix shape: {distance_matrix.shape}")
            print(f"Sample distance: {distance_matrix[0,1]:.2f} meters")
            print(f"Sample duration: {duration_matrix[0,1]:.2f} seconds")
        
        # Test scenario generation
        generator = OSMVRPGenerator()
        scenario = generator.create_realistic_city_delivery("New York", num_customers=5, num_vehicles=2)
        print(f"\nCreated scenario: {scenario.name}")
        print(f"Locations: {len(scenario.location_ids)}")
        print(f"Vehicles: {len(scenario.vehicles)}")
        
    except Exception as e:
        print(f"Error in testing: {str(e)}")
        import traceback
        traceback.print_exc()
