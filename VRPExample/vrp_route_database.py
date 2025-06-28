#!/usr/bin/env python3
"""
VRP Route Database with Road Composition Caching
===============================================

This module creates and manages a persistent database of routes with road composition
analysis for VRP optimization. It caches results to avoid repeated OSRM API calls.

Features:
- Persistent JSON cache for route data and road compositions
- Smart route filtering to reduce unnecessary API calls
- Bulk OSRM calls where possible
- Road composition analysis with truck speed adjustments
- Distance-based filtering to skip very long/unlikely routes
"""

import os
import json
import requests
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VRPRouteDatabase:
    """
    Manages a database of routes with road composition for VRP optimization.
    """
    
    def __init__(self, cache_file: str = "vrp_route_cache.json", 
                 osrm_url: str = "https://router.project-osrm.org"):
        # Always use absolute path for cache file, relative to this module's directory
        if not os.path.isabs(cache_file):
            # Get the directory where this script is located (VRPExample folder)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.cache_file = os.path.join(script_dir, cache_file)
        else:
            self.cache_file = cache_file
        
        self.osrm_url = osrm_url
        self.cache = self._load_cache()
        self.session = requests.Session()
        
        # Default truck speed ratios (can be overridden)
        self.truck_speed_ratios = {
            'motorway': 90 / 130,      # Heavy trucks: 70 km/h, Cars: ~130 km/h
            'trunk': 80 / 100,         # Heavy trucks: 60 km/h, Cars: ~100 km/h
            'primary': 70 / 90,        # Heavy trucks: 50 km/h, Cars: ~90 km/h
            'secondary': 70 / 70,      # Heavy trucks: 40 km/h, Cars: ~70 km/h
            'tertiary': 60 / 60,       # Heavy trucks: 35 km/h, Cars: ~60 km/h
            'residential': 25 / 50,    # Heavy trucks: 25 km/h, Cars: ~50 km/h
            'service': 15 / 30,        # Heavy trucks: 20 km/h, Cars: ~30 km/h
            'default': 0.65            # Heavy trucks ~35% slower than cars
        }
    
    def _load_cache(self) -> Dict:
        """Load existing cache from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                    logger.info(f"📂 Loaded {len(cache)} cached routes from {self.cache_file}")
                    return cache
            except Exception as e:
                logger.warning(f"Failed to load cache from {self.cache_file}: {e}")
        else:
            logger.info(f"📂 No existing cache found at {self.cache_file}, starting fresh")
        
        return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
            logger.info(f"💾 Saved {len(self.cache)} routes to cache at {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache to {self.cache_file}: {e}")
    
    def _get_route_key(self, origin_id: str, dest_id: str) -> str:
        """Generate cache key for a route."""
        return f"{origin_id}→{dest_id}"
    
    def _should_compute_route(self, origin: Dict, destination: Dict, max_distance_km: float = 500) -> bool:
        """
        Determine if we should compute this route based on distance and practicality.
        
        Args:
            origin: Origin location dict
            destination: Destination location dict
            max_distance_km: Maximum reasonable route distance
        """
        # Don't compute self-routes
        if origin['id'] == destination['id']:
            return False
        
        # Estimate straight-line distance
        dx = float(destination['x']) - float(origin['x'])
        dy = float(destination['y']) - float(origin['y'])
        straight_distance = np.sqrt(dx*dx + dy*dy)
        
        # Convert to rough km estimate (this depends on coordinate system)
        # For lat/lon coordinates, 1 degree ≈ 111 km
        # Adjust this factor based on your coordinate system
        coord_to_km_factor = 111 if abs(float(origin['x'])) < 180 else 1
        estimated_km = straight_distance * coord_to_km_factor
        
        # Skip routes that are too long to be practical
        if estimated_km > max_distance_km:
            logger.debug(f"Skipping long route {origin['id']} → {destination['id']}: {estimated_km:.1f} km")
            return False
        
        return True
    
    def get_route_with_composition(self, origin: Dict, destination: Dict) -> Optional[Dict]:
        """
        Get route data with road composition, using cache when available.
        
        Returns:
            Dict with keys: distance_km, car_time_minutes, truck_time_minutes, composition
        """
        route_key = self._get_route_key(origin['id'], destination['id'])
        
        # Check cache first
        if route_key in self.cache:
            return self.cache[route_key]
        
        # Check if we should compute this route
        if not self._should_compute_route(origin, destination):
            return None
        
        # Get route from OSRM
        route_data = self._fetch_route_from_osrm(origin, destination)
        
        if route_data:
            # Cache the result
            self.cache[route_key] = route_data
            
            # Save cache periodically (every 50 routes)
            if len(self.cache) % 50 == 0:
                self._save_cache()
        
        return route_data
    
    def _fetch_route_from_osrm(self, origin: Dict, destination: Dict) -> Optional[Dict]:
        """Fetch detailed route information from OSRM."""
        try:
            # Build coordinates
            if 'lat' in origin and 'lon' in origin:
                origin_coords = f"{origin['lon']},{origin['lat']}"
                dest_coords = f"{destination['lon']},{destination['lat']}"
            else:
                origin_coords = f"{origin['x']},{origin['y']}"
                dest_coords = f"{destination['x']},{destination['y']}"
            
            # Call OSRM route service
            url = f"{self.osrm_url}/route/v1/driving/{origin_coords};{dest_coords}"
            params = {
                'overview': 'full',
                'steps': 'true',
                'annotations': 'true'
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] == 'Ok' and 'routes' in data and len(data['routes']) > 0:
                route = data['routes'][0]
                
                # Basic route information
                distance_km = route['distance'] / 1000.0
                car_time_minutes = route['duration'] / 60.0
                
                # Analyze road composition
                composition = self._analyze_road_composition(route)
                
                # Calculate truck time with road composition
                truck_ratio = self._calculate_weighted_truck_ratio(composition)
                truck_time_minutes = car_time_minutes / truck_ratio
                
                return {
                    'distance_km': distance_km,
                    'car_time_minutes': car_time_minutes,
                    'truck_time_minutes': truck_time_minutes,
                    'truck_ratio': truck_ratio,
                    'composition': composition,
                    'cached_at': datetime.now().isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error fetching route {origin['id']} → {destination['id']}: {e}")
            return None
    
    def _analyze_road_composition(self, route: Dict) -> Dict[str, float]:
        """Analyze road composition from OSRM route."""
        composition = {}
        
        if 'legs' not in route:
            return composition
        
        for leg in route['legs']:
            if 'steps' not in leg:
                continue
                
            for step in leg['steps']:
                step_distance_km = step.get('distance', 0) / 1000.0
                road_type = self._extract_road_type_from_step(step)
                
                if road_type not in composition:
                    composition[road_type] = 0
                composition[road_type] += step_distance_km
        
        return composition
    
    def _extract_road_type_from_step(self, step: Dict) -> str:
        """Extract road type from OSRM route step."""
        # Try multiple methods to determine road type
        
        # Method 1: Check intersections for road classes
        if 'intersections' in step and len(step['intersections']) > 0:
            intersection = step['intersections'][0]
            if 'classes' in intersection and len(intersection['classes']) > 0:
                road_class = intersection['classes'][0]
                
                class_mapping = {
                    'motorway': 'motorway',
                    'highway': 'motorway',
                    'trunk': 'trunk',
                    'primary': 'primary',
                    'secondary': 'secondary',
                    'tertiary': 'tertiary',
                    'residential': 'residential',
                    'service': 'service'
                }
                
                for osrm_class, our_type in class_mapping.items():
                    if osrm_class in road_class.lower():
                        return our_type
        
        # Method 2: Check step name/ref
        if 'name' in step and step['name']:
            name = step['name'].lower()
            if any(keyword in name for keyword in ['highway', 'motorway', 'autobahn']):
                return 'motorway'
            elif any(keyword in name for keyword in ['trunk', 'expressway']):
                return 'trunk'
            elif any(keyword in name for keyword in ['primary', 'hauptstrasse']):
                return 'primary'
            elif any(keyword in name for keyword in ['secondary', 'landstrasse']):
                return 'secondary'
        
        # Method 3: Use ref (road number)
        if 'ref' in step and step['ref']:
            ref = step['ref'].upper()
            if any(prefix in ref for prefix in ['A', 'E']):
                return 'motorway'
            elif any(prefix in ref for prefix in ['B']):
                return 'primary'
        
        return 'default'
    
    def _calculate_weighted_truck_ratio(self, composition: Dict[str, float]) -> float:
        """Calculate weighted truck speed ratio based on road composition."""
        if not composition:
            return self.truck_speed_ratios.get('default', 1.0)
        
        total_distance = sum(composition.values())
        if total_distance == 0:
            return self.truck_speed_ratios.get('default', 1.0)
        
        weighted_ratio = 0
        for road_type, distance in composition.items():
            weight = distance / total_distance
            ratio = self.truck_speed_ratios.get(road_type, self.truck_speed_ratios.get('default', 1.0))
            weighted_ratio += weight * ratio
        
        return weighted_ratio
    
    def build_distance_matrices(self, locations: List[Dict], 
                              max_distance_km: float = 500,
                              delay_between_calls: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build optimized distance and time matrices for a set of locations.
        
        Args:
            locations: List of location dictionaries
            max_distance_km: Maximum route distance to consider
            delay_between_calls: Delay between OSRM calls to avoid rate limiting
            
        Returns:
            Tuple of (distance_matrix, time_matrix) as numpy arrays
        """
        n = len(locations)
        distance_matrix = np.zeros((n, n))
        time_matrix = np.zeros((n, n))
        
        total_possible_routes = n * (n - 1)
        routes_to_compute = []
        
        # First pass: identify which routes we actually need to compute
        logger.info(f"🔍 Analyzing {total_possible_routes} possible routes...")
        
        for i, origin in enumerate(locations):
            for j, destination in enumerate(locations):
                if i != j and self._should_compute_route(origin, destination, max_distance_km):
                    routes_to_compute.append((i, j, origin, destination))
        
        logger.info(f"📊 Will compute {len(routes_to_compute)} routes (filtered from {total_possible_routes})")
        
        # Second pass: get route data (from cache or OSRM)
        computed_routes = 0
        cache_hits = 0
        
        for i, j, origin, destination in routes_to_compute:
            route_key = self._get_route_key(origin['id'], destination['id'])
            was_cached = route_key in self.cache
            
            route_data = self.get_route_with_composition(origin, destination)
            
            if route_data:
                distance_matrix[i, j] = route_data['distance_km']
                time_matrix[i, j] = route_data['truck_time_minutes']
                computed_routes += 1
                
                if was_cached:
                    cache_hits += 1
                else:
                    # Add delay only for new API calls
                    time.sleep(delay_between_calls)
            else:
                # Use a high penalty for impossible routes
                distance_matrix[i, j] = 9999
                time_matrix[i, j] = 9999
        
        # Save final cache
        self._save_cache()
        
        logger.info(f"✅ Route computation complete:")
        logger.info(f"   • Computed routes: {computed_routes}/{len(routes_to_compute)}")
        logger.info(f"   • Cache hits: {cache_hits}")
        logger.info(f"   • New API calls: {computed_routes - cache_hits}")
        logger.info(f"   • Total cached routes: {len(self.cache)}")
        
        return distance_matrix, time_matrix
    
    def get_route_summary(self) -> Dict:
        """Get summary statistics about cached routes."""
        if not self.cache:
            return {"total_routes": 0}
        
        total_distance = sum(route['distance_km'] for route in self.cache.values() if 'distance_km' in route)
        total_routes = len(self.cache)
        
        # Analyze road composition across all routes
        road_type_totals = {}
        for route in self.cache.values():
            if 'composition' in route:
                for road_type, distance in route['composition'].items():
                    if road_type not in road_type_totals:
                        road_type_totals[road_type] = 0
                    road_type_totals[road_type] += distance
        
        return {
            "total_routes": total_routes,
            "total_distance_km": round(total_distance, 1),
            "average_distance_km": round(total_distance / total_routes, 1) if total_routes > 0 else 0,
            "road_type_distribution": {k: round(v, 1) for k, v in road_type_totals.items()},
            "cache_file": self.cache_file,
            "cache_size_kb": round(os.path.getsize(self.cache_file) / 1024, 1) if os.path.exists(self.cache_file) else 0
        }
    
    def clear_cache(self):
        """Clear the route cache."""
        self.cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        logger.info("🗑️ Route cache cleared")


def demo_route_database():
    """Demonstrate the route database functionality."""
    print("🗃️ VRP Route Database Demo")
    print("=" * 40)
    
    # Create test locations (smaller set for demo)
    locations = [
        {'id': 'depot', 'x': 47.3769, 'y': 8.5417, 'address': 'Zurich Depot'},
        {'id': 'basel', 'x': 47.5596, 'y': 7.5886, 'address': 'Basel'},
        {'id': 'bern', 'x': 46.9481, 'y': 7.4474, 'address': 'Bern'},
        {'id': 'geneva', 'x': 46.2044, 'y': 6.1432, 'address': 'Geneva'},
        {'id': 'lausanne', 'x': 46.5197, 'y': 6.6323, 'address': 'Lausanne'},
        {'id': 'lucerne', 'x': 47.0502, 'y': 8.3093, 'address': 'Lucerne'},
    ]
    
    # Create route database
    route_db = VRPRouteDatabase(cache_file="demo_routes.json")
    
    print(f"📍 Demo locations: {len(locations)}")
    for loc in locations:
        print(f"   • {loc['id']}: ({loc['x']}, {loc['y']})")
    
    # Build matrices
    print(f"\n🚀 Building distance matrices...")
    distance_matrix, time_matrix = route_db.build_distance_matrices(locations, max_distance_km=300)
    
    print(f"\n📊 Results:")
    print(f"Distance Matrix (km):")
    print(distance_matrix.round(1))
    print(f"\nTime Matrix (minutes):")
    print(time_matrix.round(1))
    
    # Show summary
    summary = route_db.get_route_summary()
    print(f"\n📈 Route Database Summary:")
    for key, value in summary.items():
        print(f"   • {key}: {value}")


if __name__ == "__main__":
    demo_route_database()
