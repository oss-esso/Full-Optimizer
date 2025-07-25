"""
Utilities package for the EPDT algorithm.

This package contains shared utility functions and modules used across
the EPDT algorithm implementation.
"""

from .distance_calculator import (
    haversine_distance,
    calculate_travel_time_haversine,
    calculate_travel_time_between_tasks,
    get_vehicle_speed,
    reset_distance_calculation_count,
    get_distance_calculation_count,
    VEHICLE_SPEED_PROFILES
)

__all__ = [
    'haversine_distance',
    'calculate_travel_time_haversine', 
    'calculate_travel_time_between_tasks',
    'get_vehicle_speed',
    'reset_distance_calculation_count',
    'get_distance_calculation_count',
    'VEHICLE_SPEED_PROFILES'
]
