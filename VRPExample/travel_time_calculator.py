#!/usr/bin/env python3
"""
Travel Time Calculator using OSRM for Real Road-Based Routing

This module provides functions to calculate real travel times using OSRM,
with caching to avoid repeated API calls for the same route pairs.
"""

import requests
import time
import logging
from typing import Dict, List, Tuple, Optional
import json
import os
import hashlib

logger = logging.getLogger(__name__)

class TravelTimeCalculator:
    """Calculate travel times using OSRM with caching."""
    
    def __init__(self, osrm_base_url: str = "https://router.project-osrm.org"):
        self.osrm_base_url = osrm_base_url
        self.cache = {}
        self.cache_file = "travel_time_cache.json"
        self.load_cache()
    
    def load_cache(self):
        """Load cached travel times from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached travel times")
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
                self.cache = {}
        else:
            self.cache = {}
    
    def save_cache(self):
        """Save cached travel times to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
            logger.info(f"Saved {len(self.cache)} travel times to cache")
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def _make_cache_key(self, from_coords: Tuple[float, float], to_coords: Tuple[float, float]) -> str:
        """Create a cache key for coordinate pairs."""
        # Round coordinates to 6 decimal places for consistent caching
        from_rounded = (round(from_coords[0], 6), round(from_coords[1], 6))
        to_rounded = (round(to_coords[0], 6), round(to_coords[1], 6))
        return f"{from_rounded[1]},{from_rounded[0]};{to_rounded[1]},{to_rounded[0]}"
    
    def get_travel_time_single(self, from_coords: Tuple[float, float], to_coords: Tuple[float, float]) -> Tuple[int, float]:
        """
        Get travel time between two coordinates using OSRM.
        
        Args:
            from_coords: (latitude, longitude) of starting point
            to_coords: (latitude, longitude) of destination
            
        Returns:
            Tuple of (travel_time_minutes, distance_km)
        """
        # Check cache first
        cache_key = self._make_cache_key(from_coords, to_coords)
        if cache_key in self.cache:
            return self.cache[cache_key]['time_minutes'], self.cache[cache_key]['distance_km']
        
        # Same location
        if from_coords == to_coords:
            result = (0, 0.0)
            self.cache[cache_key] = {'time_minutes': 0, 'distance_km': 0.0}
            return result
        
        try:
            # OSRM expects longitude,latitude format
            from_lon, from_lat = from_coords[1], from_coords[0]
            to_lon, to_lat = to_coords[1], to_coords[0]
            
            # OSRM route API call
            url = f"{self.osrm_base_url}/route/v1/driving/{from_lon},{from_lat};{to_lon},{to_lat}"
            params = {
                'overview': 'false',  # We don't need geometry
                'alternatives': 'false',
                'steps': 'false'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['code'] == 'Ok' and 'routes' in data and len(data['routes']) > 0:
                route = data['routes'][0]
                duration_seconds = route['duration']
                distance_meters = route['distance']
                
                time_minutes = int(duration_seconds / 60)
                distance_km = distance_meters / 1000.0
                
                # Ensure minimum 1 minute for non-zero distances
                if distance_km > 0 and time_minutes == 0:
                    time_minutes = 1
                
                # Cache the result
                self.cache[cache_key] = {
                    'time_minutes': time_minutes,
                    'distance_km': distance_km
                }
                
                return time_minutes, distance_km
            else:
                logger.warning(f"OSRM API returned error for {from_coords} -> {to_coords}: {data.get('message', 'Unknown error')}")
                return self._fallback_calculation(from_coords, to_coords)
                
        except Exception as e:
            logger.warning(f"OSRM API call failed for {from_coords} -> {to_coords}: {e}")
            return self._fallback_calculation(from_coords, to_coords)
    
    def _fallback_calculation(self, from_coords: Tuple[float, float], to_coords: Tuple[float, float]) -> Tuple[int, float]:
        """Fallback to Haversine distance calculation if OSRM fails."""
        import math
        
        lat1, lon1 = math.radians(from_coords[0]), math.radians(from_coords[1])
        lat2, lon2 = math.radians(to_coords[0]), math.radians(to_coords[1])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        distance_km = 6371 * 2 * math.asin(math.sqrt(a))  # Earth radius in km
        
        # Use conservative 40 km/h average speed for road travel (slower than straight-line 60 km/h)
        time_minutes = max(1, int(distance_km * 60 / 40)) if distance_km > 0 else 0
        
        logger.info(f"Using fallback calculation: {distance_km:.2f}km -> {time_minutes}min")
        return time_minutes, distance_km
    
    def get_travel_time_matrix(self, coordinates: List[Tuple[float, float]]) -> List[List[int]]:
        """
        Get a full travel time matrix for a list of coordinates.
        
        Args:
            coordinates: List of (latitude, longitude) tuples
            
        Returns:
            2D matrix where matrix[i][j] is travel time from coordinates[i] to coordinates[j]
        """
        n = len(coordinates)
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        
        logger.info(f"Calculating travel time matrix for {n} locations...")
        
        # Calculate pairwise travel times
        api_calls = 0
        cache_hits = 0
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 0
                else:
                    cache_key = self._make_cache_key(coordinates[i], coordinates[j])
                    if cache_key in self.cache:
                        matrix[i][j] = self.cache[cache_key]['time_minutes']
                        cache_hits += 1
                    else:
                        time_minutes, _ = self.get_travel_time_single(coordinates[i], coordinates[j])
                        matrix[i][j] = time_minutes
                        api_calls += 1
                        
                        # Add small delay to avoid overwhelming OSRM
                        if api_calls % 10 == 0:
                            time.sleep(0.1)
        
        logger.info(f"Travel time matrix complete: {cache_hits} cache hits, {api_calls} API calls")
        
        # Save cache after building matrix
        if api_calls > 0:
            self.save_cache()
        
        return matrix
    
    def print_matrix_summary(self, matrix: List[List[int]], coordinates: List[Tuple[float, float]]):
        """Print a summary of the travel time matrix."""
        n = len(matrix)
        total_time = sum(sum(row) for row in matrix)
        non_zero_count = sum(1 for i in range(n) for j in range(n) if matrix[i][j] > 0)
        
        if non_zero_count > 0:
            avg_time = total_time / non_zero_count
            max_time = max(max(row) for row in matrix)
            min_time = min(min(val for val in row if val > 0) for row in matrix if any(val > 0 for val in row))
            
            logger.info(f"Travel Time Matrix Summary:")
            logger.info(f"  Locations: {n}")
            logger.info(f"  Total routes: {n * (n-1)}")
            logger.info(f"  Average travel time: {avg_time:.1f} minutes")
            logger.info(f"  Min travel time: {min_time} minutes")
            logger.info(f"  Max travel time: {max_time} minutes")
        else:
            logger.info("Travel time matrix is empty")

# Global instance for easy access
travel_calculator = TravelTimeCalculator()
