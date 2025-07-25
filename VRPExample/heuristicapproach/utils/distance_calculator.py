"""
Central Distance and Travel Time Calculator

This module provides a single source of truth for all distance and travel time calculations
across the EPDT algorithm. It centralizes the logic to avoid inconsistencies between
different calculation methods used throughout the codebase.

Key Features:
- Haversine distance calculation for accurate geodesic distances
- Configurable speed profiles for different vehicle types
- Consistent travel time calculations
- Easy switching between testing and production modes
"""

import math
from typing import Optional

# Global distance calculation counter for performance monitoring
_distance_calculation_count = 0

def reset_distance_calculation_count():
    """Reset the global distance calculation counter."""
    global _distance_calculation_count
    _distance_calculation_count = 0

def get_distance_calculation_count() -> int:
    """Get the current distance calculation count."""
    return _distance_calculation_count

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth using the Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of the first point in decimal degrees
        lat2, lon2: Latitude and longitude of the second point in decimal degrees
    
    Returns:
        Distance in kilometers
    """
    global _distance_calculation_count
    _distance_calculation_count += 1
    
    # Convert decimal degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (math.sin(dlat/2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    # Earth's radius in kilometers
    R = 6371.0
    distance = R * c
    
    return distance

def calculate_travel_time_haversine(lat1: float, lon1: float, lat2: float, lon2: float, 
                                  speed_kmh: float = 60.0) -> float:
    """
    Calculate travel time between two points using Haversine distance and given speed.
    
    Args:
        lat1, lon1: Starting point coordinates in decimal degrees
        lat2, lon2: Ending point coordinates in decimal degrees
        speed_kmh: Average speed in km/h (default: 60 km/h for realistic truck speeds)
    
    Returns:
        Travel time in minutes
    """
    distance_km = haversine_distance(lat1, lon1, lat2, lon2)
    travel_time_hours = distance_km / speed_kmh
    travel_time_minutes = travel_time_hours * 60.0
    return travel_time_minutes

def calculate_travel_time_between_tasks(task1, task2, vehicle, speed_kmh: Optional[float] = None) -> float:
    """
    Calculate travel time between two tasks using Haversine distance.
    
    This is the standardized interface for travel time calculation that matches
    the expected signature used throughout the EPDT algorithm.
    
    Args:
        task1: Starting task with lat, lon attributes
        task2: Ending task with lat, lon attributes
        vehicle: Vehicle object (used for speed profile if speed_kmh not provided)
        speed_kmh: Optional override speed in km/h
    
    Returns:
        Travel time in minutes
    """
    # Validate coordinates
    if not all(hasattr(task, attr) for task in [task1, task2] for attr in ['lat', 'lon']):
        return 15.0  # Default fallback for missing coordinates
    
    # Determine speed
    if speed_kmh is None:
        # Use vehicle-specific speed or default to realistic truck speed
        speed_kmh = getattr(vehicle, 'average_speed', 60.0)
    
    return calculate_travel_time_haversine(
        task1.lat, task1.lon, 
        task2.lat, task2.lon, 
        speed_kmh
    )

# Vehicle speed profiles (km/h) for different vehicle types
VEHICLE_SPEED_PROFILES = {
    'standard': 60.0,     # Standard trucks - realistic city + highway average
    'heavy': 55.0,        # Heavy trucks - slightly slower
    'light': 65.0,        # Light vehicles - slightly faster
    'urban': 40.0,        # Urban delivery vehicles
    'highway': 80.0,      # Highway-optimized vehicles
    'default': 60.0       # Default fallback speed
}

def get_vehicle_speed(vehicle) -> float:
    """
    Get appropriate speed for a vehicle based on its type.
    
    Args:
        vehicle: Vehicle object with optional vehicle_type or average_speed attributes
    
    Returns:
        Speed in km/h
    """
    # Check for explicit speed setting
    if hasattr(vehicle, 'average_speed') and vehicle.average_speed:
        return vehicle.average_speed
    
    # Check vehicle type
    if hasattr(vehicle, 'vehicle_type'):
        vehicle_type = vehicle.vehicle_type.lower() if vehicle.vehicle_type else 'default'
        return VEHICLE_SPEED_PROFILES.get(vehicle_type, VEHICLE_SPEED_PROFILES['default'])
    
    # Fallback to default
    return VEHICLE_SPEED_PROFILES['default']
