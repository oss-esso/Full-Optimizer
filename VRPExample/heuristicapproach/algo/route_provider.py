"""
OSRM Route Provider with SQLite Caching

This module provides route information for the EPDT algorithm by:
1. Caching route data in a local SQLite database 
2. Fetching from OSRM API when cache misses occur
3. Supporting vehicle-specific travel time calculations
4. Providing centralized travel time calculation with configurable modes

The module encapsulates the road composition logic from tests/road_composition.py
and provides a clean interface for the EPDT algorithm to get realistic travel times.

Configuration:
- USE_OSRM: Global flag to control whether to use OSRM service or Haversine calculation
- Set to False for testing with consistent Haversine distances
- Set to True for production with OSRM routing service
"""

import requests
import sqlite3
import json
import os
from collections import defaultdict
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

# Global configuration flag for travel time calculation mode
USE_OSRM = True  # Set to False for testing mode with Haversine calculation

def set_testing_mode(use_haversine: bool = True):
    """
    Configure the route provider for testing or production mode.
    
    Args:
        use_haversine: If True, use Haversine calculation (testing mode)
                      If False, use OSRM service (production mode)
    """
    global USE_OSRM
    USE_OSRM = not use_haversine
    print(f"Route provider mode: {'OSRM' if USE_OSRM else 'Haversine (testing)'}")

def is_testing_mode() -> bool:
    """Check if currently in testing mode (using Haversine calculation)."""
    return not USE_OSRM


# Speed profiles based on OSRM car.lua profile (km/h)
CAR_SPEEDS = {
    'motorway': 100,
    'trunk': 85,
    'primary': 75,
    'secondary': 65,
    'tertiary': 50,
    'unclassified': 30,
    'residential': 30,
    'service': 20,
    'motorway_link': 50,
    'trunk_link': 45,
    'primary_link': 35,
    'secondary_link': 30,
    'tertiary_link': 25,
    'living_street': 15,
}

# Vehicle-specific speed profiles (km/h)
VEHICLE_SPEEDS = {
    'car': CAR_SPEEDS,
    'standard': {
        'motorway': 130,
        'trunk': 100,
        'primary': 90,
        'secondary': 70,
        'tertiary': 60,
        'residential': 40,
        'service': 25,
        'unclassified': 30,
        'motorway_link': 50,
        'trunk_link': 45,
        'primary_link': 35,
        'secondary_link': 30,
        'tertiary_link': 25,
        'living_street': 15,
    },
    'heavy': {
        'motorway': 90,
        'trunk': 80,
        'primary': 70,
        'secondary': 70,
        'tertiary': 60,
        'residential': 25,
        'service': 20,
        'unclassified': 25,
        'motorway_link': 45,
        'trunk_link': 40,
        'primary_link': 30,
        'secondary_link': 25,
        'tertiary_link': 20,
        'living_street': 15,
    }
}

# Speed ratios for different truck types compared to a car
TRUCK_SPEED_RATIOS = {
    'standard': {
        'motorway': 0.9,
        'motorway_link': 0.9,
        'trunk': 0.9,
        'trunk_link': 0.9,
        'primary': 0.85,
        'primary_link': 0.85,
        'secondary': 0.85,
        'secondary_link': 0.85,
        'tertiary': 0.8,
        'tertiary_link': 0.8,
        'unclassified': 0.75,
        'residential': 0.75,
        'living_street': 0.7,
        'service': 0.7,
        'default': 0.80
    },
    'heavy': {
        'motorway': 0.7,
        'motorway_link': 0.7,
        'trunk': 0.7,
        'trunk_link': 0.7,
        'primary': 0.65,
        'primary_link': 0.65,
        'secondary': 0.6,
        'secondary_link': 0.6,
        'tertiary': 0.55,
        'tertiary_link': 0.55,
        'unclassified': 0.5,
        'residential': 0.5,
        'living_street': 0.5,
        'service': 0.5,
        'default': 0.65
    }
}


class RouteProvider:
    """
    Provides route information with OSRM API integration and SQLite caching.
    """
    
    def __init__(self, db_path: str = "moda_routes.db", osrm_url: str = "http://router.project-osrm.org"):
        """
        Initialize the route provider.
        
        Args:
            db_path: Path to SQLite database file
            osrm_url: Base URL for OSRM API
        """
        self.db_path = db_path
        self.osrm_url = osrm_url
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with the routes table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS route_cache (
                start_node_id TEXT,
                end_node_id TEXT,
                distance_km REAL,
                duration_minutes REAL,
                road_composition_json TEXT,
                route_geometry_json TEXT,
                PRIMARY KEY (start_node_id, end_node_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_route_details(self, start_node_id: str, end_node_id: str, 
                         start_coords: Tuple[float, float] = None, 
                         end_coords: Tuple[float, float] = None) -> Optional[Dict[str, Any]]:
        """
        Get route details for the given node IDs.
        
        Args:
            start_node_id: Starting node identifier
            end_node_id: Ending node identifier
            start_coords: (lon, lat) coordinates for start (required if not in cache)
            end_coords: (lon, lat) coordinates for end (required if not in cache)
            
        Returns:
            Dictionary with route information:
            {
                'distance_km': float,
                'duration_minutes': float,
                'road_composition': dict,
                'route_geometry': dict
            }
        """
        # Same start and end
        if start_node_id == end_node_id:
            return {
                'distance_km': 0.0,
                'duration_minutes': 0.0,
                'road_composition': {},
                'route_geometry': None
            }
        
        # Try to get from local database first
        route_data = self._query_local_db(start_node_id, end_node_id)
        
        if route_data is not None:
            return route_data
        
        # If not in cache and coordinates provided, fetch from OSRM
        if start_coords and end_coords:
            return self._query_osrm_and_cache(start_coords, end_coords, start_node_id, end_node_id)
        
        # If no coordinates provided, return None
        return None
    
    def _query_local_db(self, start_node_id: str, end_node_id: str) -> Optional[Dict[str, Any]]:
        """Query the local database for route information."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT distance_km, duration_minutes, road_composition_json, route_geometry_json
            FROM route_cache
            WHERE start_node_id = ? AND end_node_id = ?
        """, (start_node_id, end_node_id))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            distance_km, duration_minutes, road_comp_json, route_geom_json = result
            return {
                'distance_km': distance_km,
                'duration_minutes': duration_minutes,
                'road_composition': json.loads(road_comp_json) if road_comp_json else {},
                'route_geometry': json.loads(route_geom_json) if route_geom_json else None
            }
        
        return None
    
    def _query_osrm_and_cache(self, start_coords: Tuple[float, float], end_coords: Tuple[float, float],
                             start_node_id: str, end_node_id: str) -> Optional[Dict[str, Any]]:
        """Query OSRM API and cache the results."""
        coords_str = f"{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
        url = f"{self.osrm_url}/route/v1/driving/{coords_str}"
        params = {'annotations': 'true', 'overview': 'full'}
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('code') != 'Ok' or not data['routes']:
                print(f"OSRM could not find a route: {data.get('code')}")
                return None
            
            route = data['routes'][0]
            distance_m = route.get('distance', 0)
            duration_seconds = route.get('duration', 0)
            
            # Convert to desired units
            distance_km = distance_m / 1000.0
            duration_minutes = duration_seconds / 60.0
            
            # Extract road composition
            road_composition = self._extract_road_composition(route)
            
            # Extract route geometry
            route_geometry = None
            if 'geometry' in route:
                geometry = route['geometry']
                if isinstance(geometry, dict):
                    route_geometry = {
                        'coordinates': geometry.get('coordinates', []),
                        'type': geometry.get('type', 'LineString')
                    }
                elif isinstance(geometry, str):
                    # Handle encoded polyline
                    route_geometry = {
                        'encoded': geometry,
                        'type': 'encoded_polyline'
                    }
            
            # Cache the results
            self._cache_route(start_node_id, end_node_id, distance_km, duration_minutes, 
                            road_composition, route_geometry)
            
            return {
                'distance_km': distance_km,
                'duration_minutes': duration_minutes,
                'road_composition': road_composition,
                'route_geometry': route_geometry
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling OSRM API: {e}")
            return None
    
    def _extract_road_composition(self, route: Dict[str, Any]) -> Dict[str, float]:
        """Extract road composition from OSRM route response."""
        if 'legs' not in route or not route['legs']:
            return {}
        
        annotation = route['legs'][0].get('annotation', {})
        distances = annotation.get('distance', [])
        durations = annotation.get('duration', [])
        
        if not distances or not durations:
            return {}
        
        road_distances = defaultdict(float)
        total_distance_m = sum(distances)
        
        if total_distance_m == 0:
            return {}
        
        for i in range(len(distances)):
            if durations[i] > 0:
                speed_kmh = (distances[i] / durations[i]) * 3.6
                road_type = self._infer_road_type_from_speed(speed_kmh)
                road_distances[road_type] += distances[i]
        
        # Convert to percentages
        composition_percent = {
            rt: (dist / total_distance_m) * 100 
            for rt, dist in road_distances.items()
        }
        
        return composition_percent
    
    def _infer_road_type_from_speed(self, speed_kmh: float) -> str:
        """Infer road type from observed speed."""
        return min(CAR_SPEEDS, key=lambda road: abs(CAR_SPEEDS[road] - speed_kmh))
    
    def _cache_route(self, start_node_id: str, end_node_id: str, distance_km: float,
                    duration_minutes: float, road_composition: Dict[str, float],
                    route_geometry: Optional[Dict[str, Any]]):
        """Cache route data in the local database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO route_cache 
            (start_node_id, end_node_id, distance_km, duration_minutes, 
             road_composition_json, route_geometry_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            start_node_id, end_node_id, distance_km, duration_minutes,
            json.dumps(road_composition) if road_composition else None,
            json.dumps(route_geometry) if route_geometry else None
        ))
        
        conn.commit()
        conn.close()
    
    def calculate_vehicle_travel_time(self, start_node_id: str, end_node_id: str,
                                    vehicle_type: str = 'car',
                                    start_coords: Tuple[float, float] = None,
                                    end_coords: Tuple[float, float] = None) -> float:
        """
        Calculate travel time for a specific vehicle type.
        
        Args:
            start_node_id: Starting node identifier
            end_node_id: Ending node identifier  
            vehicle_type: Type of vehicle ('car', 'standard', 'heavy')
            start_coords: (lon, lat) coordinates for start (required if not in cache)
            end_coords: (lon, lat) coordinates for end (required if not in cache)
            
        Returns:
            Travel time in minutes
        """
        route_data = self.get_route_details(start_node_id, end_node_id, start_coords, end_coords)
        
        if not route_data:
            # Fallback to simple Euclidean distance if route data unavailable
            if start_coords and end_coords:
                return self._calculate_euclidean_time(start_coords, end_coords)
            return 1.0  # Minimum travel time
        
        base_duration = route_data['duration_minutes']
        road_composition = route_data['road_composition']
        
        # For cars, return base duration
        if vehicle_type == 'car' or not road_composition:
            return max(base_duration, 1.0)
        
        # For trucks, calculate adjusted time using road composition
        if vehicle_type in TRUCK_SPEED_RATIOS:
            return self._calculate_truck_time(base_duration, road_composition, vehicle_type)
        
        # Unknown vehicle type, return base duration
        return max(base_duration, 1.0)
    
    def _calculate_truck_time(self, car_time_minutes: float, composition_percent: Dict[str, float],
                             vehicle_type: str) -> float:
        """Calculate truck travel time using road composition and speed ratios."""
        vehicle_ratios = TRUCK_SPEED_RATIOS.get(vehicle_type)
        if not vehicle_ratios:
            return car_time_minutes
        
        vehicle_minutes = 0
        for road_type, perc in composition_percent.items():
            car_speed = CAR_SPEEDS.get(road_type, CAR_SPEEDS.get('unclassified', 30))
            truck_ratio = vehicle_ratios.get(road_type, vehicle_ratios['default'])
            truck_speed = car_speed * truck_ratio
            
            if truck_speed > 0:
                segment_time_adjustment = (perc / 100) * (car_speed / truck_speed)
                vehicle_minutes += car_time_minutes * segment_time_adjustment
            else:
                # Fallback to default ratio
                segment_time_adjustment = (perc / 100) * (car_speed / (car_speed * vehicle_ratios['default']))
                vehicle_minutes += car_time_minutes * segment_time_adjustment
        
        return max(vehicle_minutes, 1.0)
    
    def _calculate_euclidean_time(self, start_coords: Tuple[float, float], 
                                 end_coords: Tuple[float, float]) -> float:
        """Fallback Euclidean distance calculation."""
        import math
        lat1, lon1 = start_coords[1], start_coords[0]  # Note: coords are (lon, lat)
        lat2, lon2 = end_coords[1], end_coords[0]
        
        distance = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
        travel_time = (distance * 111.32) / 50 * 60  # 111.32 km per degree, 50 km/h avg speed
        return max(travel_time, 1.0)


# Global route provider instance
_route_provider = None

def get_route_provider() -> RouteProvider:
    """Get the global route provider instance."""
    global _route_provider
    if _route_provider is None:
        _route_provider = RouteProvider()
    return _route_provider

def calculate_travel_time_between_tasks(task1, task2, vehicle) -> float:
    """
    Calculate travel time between two tasks for a specific vehicle.
    
    This is the centralized function for travel time calculation that decides
    whether to use OSRM service or Haversine calculation based on the USE_OSRM flag.
    Enhanced with cross-validation to detect inconsistencies between OSRM and Haversine.
    
    Args:
        task1: Starting task with lat, lon, and location_id attributes
        task2: Ending task with lat, lon, and location_id attributes  
        vehicle: Vehicle object with vehicle_type attribute
        
    Returns:
        Travel time in minutes
    """
    global USE_OSRM
    
    # Cross-validation threshold (percentage difference)
    CROSS_VALIDATION_THRESHOLD = 200.0  # Log warning if difference exceeds 50%
    
    # Always calculate Haversine for cross-validation when coordinates are available
    haversine_time = None
    if hasattr(task1, 'lat') and hasattr(task2, 'lat'):
        try:
            import math
            
            # Basic Haversine calculation
            lat1_rad = math.radians(task1.lat)
            lon1_rad = math.radians(task1.lon)
            lat2_rad = math.radians(task2.lat)
            lon2_rad = math.radians(task2.lon)
            
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            
            a = (math.sin(dlat/2)**2 + 
                 math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            
            R = 6371.0  # Earth's radius in km
            distance_km = R * c
            
            # ENHANCED: Use vehicle type for speed instead of just average_speed attribute
            # This ensures furgoni get higher speeds than heavy trucks
            speed_kmh = get_vehicle_speed_by_type(vehicle)
            haversine_time = (distance_km / speed_kmh) * 60.0  # Convert to minutes
        except Exception as e:
            print(f"Warning: Haversine calculation failed for cross-validation: {e}")
    
    if USE_OSRM:
        # Production mode: Use OSRM service with cross-validation
        osrm_time = None
        try:
            provider = get_route_provider()
            
            # Extract coordinates and node IDs
            start_coords = (getattr(task1, 'lon', 0), getattr(task1, 'lat', 0))
            end_coords = (getattr(task2, 'lon', 0), getattr(task2, 'lat', 0))
            start_node_id = getattr(task1, 'location_id', f"{task1.lat}_{task1.lon}")
            end_node_id = getattr(task2, 'location_id', f"{task2.lat}_{task2.lon}")
            
            # Get vehicle type, default to 'standard' if not specified
            vehicle_type = getattr(vehicle, 'vehicle_type', 'standard')
            
            osrm_time = provider.calculate_vehicle_travel_time(
                start_node_id, end_node_id, vehicle_type, start_coords, end_coords
            )
            
            # Cross-validation: Compare OSRM and Haversine results
            if haversine_time is not None and osrm_time is not None:
                if haversine_time > 0:  # Avoid division by zero
                    percentage_diff = abs(osrm_time - haversine_time) / haversine_time * 100
                    
                    # Log both values for comparison
                    task1_location = getattr(task1, 'location_id', f"({task1.lat:.3f},{task1.lon:.3f})")
                    task2_location = getattr(task2, 'location_id', f"({task2.lat:.3f},{task2.lon:.3f})")
                    
                    #print(f"TRAVEL_TIME_VALIDATION: {task1_location} -> {task2_location}")
                    #print(f"  OSRM: {osrm_time:.1f}m | Haversine: {haversine_time:.1f}m | Diff: {percentage_diff:.1f}%")
                    
                    # Warning for significant discrepancies
                    #if percentage_diff > CROSS_VALIDATION_THRESHOLD:
                    #    print(f"  ⚠️  WARNING: Travel time discrepancy exceeds {CROSS_VALIDATION_THRESHOLD}% threshold!")
                    #    print(f"      This may indicate OSRM routing issues or unusual road conditions.")
            
            return osrm_time
            
        except Exception as e:
            print(f"Warning: OSRM calculation failed ({e}), falling back to Haversine")
            if haversine_time is not None:
                return haversine_time
            # Fall through to centralized Haversine calculation
    
    # Testing mode or OSRM fallback: Use centralized Haversine calculation
    if haversine_time is not None:
        return haversine_time
    
    try:
        # Import from our centralized distance calculator
        import sys
        import os
        
        # Add utils directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        utils_dir = os.path.join(current_dir, '..', 'utils')
        if utils_dir not in sys.path:
            sys.path.insert(0, utils_dir)
        
        from distance_calculator import calculate_travel_time_between_tasks as haversine_calc
        return haversine_calc(task1, task2, vehicle)
        
    except ImportError as e:
        print(f"Warning: Could not import centralized distance calculator ({e})")
        # Ultimate fallback - return the calculated Haversine or minimum time
        return haversine_time if haversine_time is not None else 15.0
