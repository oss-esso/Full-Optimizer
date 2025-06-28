"""
OSRM Route Database for VRP Optimization
========================================

This module provides persistent caching of OSRM routing results using SQLite.
Routes are cached by coordinates and can include truck speed adjustments and 
road composition analysis. This dramatically speeds up repeated VRP solving
by avoiding redundant API calls.

Features:
- SQLite database for persistence across runs
- Coordinate-based route lookup with precision handling
- Road composition storage for truck speed analysis
- Cache statistics and maintenance utilities
- Thread-safe operations
- Automatic schema migration
"""

import sqlite3
import json
import time
import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import os
import threading
import requests
from contextlib import contextmanager

class RouteDatabase:
    """
    Persistent database for caching OSRM routing results.
    
    This class manages a SQLite database that stores route information including:
    - Source and destination coordinates
    - Distance and duration
    - Road composition for truck speed analysis
    - Timestamps for cache management
    """
    
    def __init__(self, db_path: str = "osrm_routes_cache.db"):
        """
        Initialize the route database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        
        # Create database and tables
        self._init_database()
        
        # OSRM service configuration
        self.osrm_url = "http://router.project-osrm.org"
        self.request_timeout = 15
        self.coordinate_precision = 6  # ~10cm precision
        
        print(f"📁 Route database initialized: {db_path}")
        self._print_cache_stats()
    
    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            # Create routes table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS routes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_lat REAL NOT NULL,
                    from_lon REAL NOT NULL,
                    from_id TEXT,
                    to_lat REAL NOT NULL,
                    to_lon REAL NOT NULL,
                    to_id TEXT,
                    distance_km REAL NOT NULL,
                    duration_minutes REAL NOT NULL,
                    road_composition TEXT,
                    osrm_success BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(from_lat, from_lon, to_lat, to_lon)
                )
            """)
            
            # Create indices for fast lookup
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_route_coords 
                ON routes(from_lat, from_lon, to_lat, to_lon)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON routes(created_at)
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def _round_coordinates(self, lat: float, lon: float) -> Tuple[float, float]:
        """Round coordinates to handle floating point precision."""
        return (
            round(lat, self.coordinate_precision),
            round(lon, self.coordinate_precision)
        )
    
    def get_route(self, from_lat: float, from_lon: float, to_lat: float, to_lon: float,
                  from_id: str = None, to_id: str = None) -> Optional[Dict]:
        """
        Get route from database or fetch from OSRM if not cached.
        
        Args:
            from_lat, from_lon: Source coordinates
            to_lat, to_lon: Destination coordinates  
            from_id, to_id: Optional location IDs for logging
            
        Returns:
            Dict with keys: distance_km, duration_minutes, road_composition, cached
            None if route cannot be found or fetched
        """
        # Round coordinates for consistent lookup
        from_lat_r, from_lon_r = self._round_coordinates(from_lat, from_lon)
        to_lat_r, to_lon_r = self._round_coordinates(to_lat, to_lon)
        
        # Try database lookup first
        with self._lock:
            cached_route = self._lookup_cached_route(from_lat_r, from_lon_r, to_lat_r, to_lon_r)
            
        if cached_route:
            return {
                'distance_km': cached_route['distance_km'],
                'duration_minutes': cached_route['duration_minutes'],
                'road_composition': json.loads(cached_route['road_composition']) if cached_route['road_composition'] else {},
                'cached': True
            }
        
        # Not in cache, fetch from OSRM
        route_data = self._fetch_osrm_route(from_lat, from_lon, to_lat, to_lon, from_id, to_id)
        
        if route_data:
            # Store in database for future use
            with self._lock:
                self._store_route(from_lat_r, from_lon_r, to_lat_r, to_lon_r,
                                from_id, to_id, route_data)
            
            route_data['cached'] = False
            return route_data
        
        return None
    
    def _lookup_cached_route(self, from_lat: float, from_lon: float, 
                           to_lat: float, to_lon: float) -> Optional[sqlite3.Row]:
        """Lookup route in database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM routes 
                    WHERE from_lat = ? AND from_lon = ? AND to_lat = ? AND to_lon = ?
                    AND osrm_success = 1
                    LIMIT 1
                """, (from_lat, from_lon, to_lat, to_lon))
                
                return cursor.fetchone()
        except Exception as e:
            self.logger.error(f"Database lookup error: {e}")
            return None
    
    def _fetch_osrm_route(self, from_lat: float, from_lon: float, 
                         to_lat: float, to_lon: float,
                         from_id: str = None, to_id: str = None) -> Optional[Dict]:
        """Fetch route from OSRM API with road composition analysis."""
        try:
            # Build OSRM request URL
            from_coords = f"{from_lon},{from_lat}"
            to_coords = f"{to_lon},{to_lat}"
            url = f"{self.osrm_url}/route/v1/driving/{from_coords};{to_coords}"
            
            params = {
                'overview': 'full',
                'geometries': 'geojson',
                'alternatives': 'false',
                'annotations': 'true',
                'steps': 'true'  # Get detailed step information for road composition
            }
            
            response = requests.get(url, params=params, timeout=self.request_timeout)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] == 'Ok' and 'routes' in data and len(data['routes']) > 0:
                route = data['routes'][0]
                
                # Extract basic route info
                distance_km = route['distance'] / 1000.0
                duration_minutes = route['duration'] / 60.0
                
                # Extract road composition
                road_composition = self._extract_road_composition(route)
                
                # Calculate travel times for different truck types to show speed ratio impact
                # Create road composition summary
                road_summary = ", ".join([f"{road_type}:{dist:.1f}km" for road_type, dist in road_composition.items()]) if road_composition else "no_data"
                
                try:
                    from vrp_scenarios import DEFAULT_TRUCK_SPEED_RATIOS
                    
                    # Calculate adjusted times for different truck types
                    standard_time = self._calculate_truck_adjusted_time(duration_minutes, road_composition, 
                                                                        DEFAULT_TRUCK_SPEED_RATIOS['standard'])
                    heavy_time = self._calculate_truck_adjusted_time(duration_minutes, road_composition, 
                                                                     DEFAULT_TRUCK_SPEED_RATIOS['heavy'])
                    
                    route_key = f"{from_id or 'unknown'}→{to_id or 'unknown'}"
                    print(f"  🛣️ OSRM: {route_key} = {distance_km:.1f}km, furgone:{standard_time:.1f}min, camion:{heavy_time:.1f}min")
                    print(f"    📍 Road composition: {road_summary}")
                except ImportError:
                    # Fallback to original format if truck speed ratios not available
                    route_key = f"{from_id or 'unknown'}→{to_id or 'unknown'}"
                    print(f"  🛣️ OSRM: {route_key} = {distance_km:.1f}km, {duration_minutes:.1f}min")
                    print(f"    📍 Road composition: {road_summary}")
                
                return {
                    'distance_km': distance_km,
                    'duration_minutes': duration_minutes,
                    'road_composition': road_composition
                }
            else:
                self.logger.warning(f"OSRM API error: {data.get('message', 'Unknown error')}")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.warning(f"OSRM API timeout for route {from_id}→{to_id}")
            return None
        except Exception as e:
            self.logger.error(f"OSRM API error: {e}")
            return None
    
    def _extract_road_composition(self, route: Dict) -> Dict[str, float]:
        """Extract road type composition from OSRM route data."""
        road_composition = {}
        
        try:
            if 'legs' in route:
                for leg in route['legs']:
                    if 'steps' in leg:
                        for step in leg['steps']:
                            # Extract distance for this step (in km)
                            step_distance_km = step.get('distance', 0) / 1000.0
                            
                            # Classify road type from step data
                            road_type = self._classify_road_type(step)
                            
                            # Add to composition
                            if road_type in road_composition:
                                road_composition[road_type] += step_distance_km
                            else:
                                road_composition[road_type] = step_distance_km
        except Exception as e:
            self.logger.warning(f"Road composition extraction error: {e}")
        
        return road_composition
    
    def _calculate_truck_adjusted_time(self, base_time_minutes: float, road_composition: Dict[str, float], 
                                     truck_speed_ratios: Dict[str, float]) -> float:
        """Calculate truck-adjusted travel time based on road composition and speed ratios."""
        if not road_composition or not truck_speed_ratios:
            # No road composition data, apply default ratio
            default_ratio = truck_speed_ratios.get('default', 0.80)
            return base_time_minutes / default_ratio
        
        # Calculate weighted ratio based on road composition
        weighted_ratio = 0.0
        total_distance = 0.0
        
        for road_type, distance_km in road_composition.items():
            if road_type in truck_speed_ratios:
                ratio = truck_speed_ratios[road_type]
                weighted_ratio += ratio * distance_km
                total_distance += distance_km
        
        if total_distance > 0:
            final_ratio = weighted_ratio / total_distance
        else:
            final_ratio = truck_speed_ratios.get('default', 0.80)
        
        # Apply the speed ratio to get adjusted time
        # Lower ratio means slower speed, so higher travel time
        adjusted_time = base_time_minutes / final_ratio
        
        return adjusted_time
    
    def _classify_road_type(self, step: Dict) -> str:
        """Classify road type from OSRM step data."""
        # Get step name and maneuver information
        name = step.get('name', '').lower()
        
        # Road type classification based on common patterns
        if any(keyword in name for keyword in ['autostrada', 'highway', 'motorway', 'freeway', 'a1', 'a4', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15']):
            return 'motorway'
        elif any(keyword in name for keyword in ['statale', 'trunk', 'ss', 'sr']):
            return 'trunk'
        elif any(keyword in name for keyword in ['provinciale', 'primary', 'sp']):
            return 'primary'
        elif any(keyword in name for keyword in ['secondary', 'comunale']):
            return 'secondary'
        elif any(keyword in name for keyword in ['via', 'street', 'strada', 'tertiary']):
            return 'tertiary'
        elif any(keyword in name for keyword in ['residential', 'vicolo', 'piazza']):
            return 'residential'
        elif any(keyword in name for keyword in ['service', 'parking', 'access']):
            return 'service'
        else:
            # Default classification
            return 'secondary'
    
    def _store_route(self, from_lat: float, from_lon: float, to_lat: float, to_lon: float,
                    from_id: str, to_id: str, route_data: Dict):
        """Store route in database."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO routes 
                    (from_lat, from_lon, from_id, to_lat, to_lon, to_id,
                     distance_km, duration_minutes, road_composition, osrm_success)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    from_lat, from_lon, from_id, to_lat, to_lon, to_id,
                    route_data['distance_km'], route_data['duration_minutes'],
                    json.dumps(route_data['road_composition']), True
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error storing route in database: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) as total FROM routes WHERE osrm_success = 1")
                total_routes = cursor.fetchone()['total']
                
                cursor = conn.execute("SELECT COUNT(*) as recent FROM routes WHERE osrm_success = 1 AND created_at > datetime('now', '-7 days')")
                recent_routes = cursor.fetchone()['recent']
                
                cursor = conn.execute("SELECT MIN(created_at) as oldest, MAX(created_at) as newest FROM routes WHERE osrm_success = 1")
                dates = cursor.fetchone()
                
                return {
                    'total_routes': total_routes,
                    'recent_routes': recent_routes,
                    'oldest_route': dates['oldest'],
                    'newest_route': dates['newest'],
                    'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
                }
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {'total_routes': 0, 'recent_routes': 0, 'database_size_mb': 0}
    
    def _print_cache_stats(self):
        """Print cache statistics."""
        stats = self.get_cache_stats()
        print(f"📊 Route Cache Stats:")
        print(f"  - Total cached routes: {stats['total_routes']}")
        print(f"  - Recent routes (7 days): {stats['recent_routes']}")
        print(f"  - Database size: {stats['database_size_mb']:.2f} MB")
        if stats['oldest_route']:
            print(f"  - Cache period: {stats['oldest_route']} to {stats['newest_route']}")
    
    def clear_old_routes(self, days_old: int = 30):
        """Clear routes older than specified days."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM routes 
                    WHERE created_at < datetime('now', '-{} days')
                """.format(days_old))
                deleted = cursor.rowcount
                conn.commit()
                
                print(f"🗑️ Deleted {deleted} routes older than {days_old} days")
                return deleted
        except Exception as e:
            self.logger.error(f"Error clearing old routes: {e}")
            return 0

class CachedOSRMDistanceCalculator:
    """
    OSRM distance calculator with persistent database caching and truck speed adjustments.
    
    This calculator uses the RouteDatabase to cache OSRM results and applies
    truck speed adjustments based on road composition analysis.
    """
    
    def __init__(self, locations: List[Dict], truck_speed_ratios: Optional[Dict] = None,
                 use_truck_speeds: bool = False, use_road_composition: bool = True,
                 db_path: str = "osrm_routes_cache.db"):
        """
        Initialize the cached OSRM distance calculator.
        
        Args:
            locations: List of location dictionaries with 'id', 'x', 'y' keys
            truck_speed_ratios: Dictionary of road type to speed ratio adjustments
            use_truck_speeds: Whether to apply truck speed adjustments
            use_road_composition: Whether to use detailed road composition
            db_path: Path to route database
        """
        self.locations = locations
        self.truck_speed_ratios = truck_speed_ratios or {}
        self.use_truck_speeds = use_truck_speeds
        self.use_road_composition = use_road_composition
        self.logger = logging.getLogger(__name__)
        
        # Initialize route database
        self.route_db = RouteDatabase(db_path)
        
        # Pre-calculate distance and time matrices
        self.distance_matrix = None
        self.time_matrix = None
        self._calculate_cached_matrices()
    
    def _calculate_cached_matrices(self):
        """Calculate distance and time matrices using cached OSRM routing."""
        n = len(self.locations)
        self.distance_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        self.time_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        print(f"🗺️ Calculating cached OSRM matrices for {n} locations...")
        
        cache_hits = 0
        api_calls = 0
        errors = 0
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue  # Same location
                
                loc1 = self.locations[i]
                loc2 = self.locations[j]
                
                try:
                    # Get route from database or OSRM API
                    route_data = self.route_db.get_route(
                        loc1['y'], loc1['x'], loc2['y'], loc2['x'],
                        loc1['id'], loc2['id']
                    )
                    
                    if route_data:
                        if route_data['cached']:
                            cache_hits += 1
                        else:
                            api_calls += 1
                        
                        distance_km = route_data['distance_km']
                        duration_minutes = route_data['duration_minutes']
                        
                        # Apply truck speed adjustments if enabled
                        if self.use_truck_speeds and self.truck_speed_ratios:
                            if self.use_road_composition and route_data['road_composition']:
                                # Use road composition for weighted adjustment
                                truck_factor = self._calculate_weighted_truck_ratio(route_data['road_composition'])
                                duration_minutes *= (1.0 / truck_factor)
                            else:
                                # Use default truck speed ratio
                                default_ratio = self.truck_speed_ratios.get('default', 0.80)
                                duration_minutes *= (1.0 / default_ratio)
                        
                        # Add service time at origin
                        service_time = loc1.get('service_time', 0)
                        total_time = duration_minutes + service_time
                        
                        self.distance_matrix[i][j] = distance_km
                        self.time_matrix[i][j] = total_time
                    else:
                        errors += 1
                        # Fallback to approximate distance
                        distance_km = self._haversine_distance(loc1, loc2)
                        time_minutes = distance_km * 60 / 50  # 50 km/h average
                        
                        self.distance_matrix[i][j] = distance_km
                        self.time_matrix[i][j] = time_minutes
                        
                except Exception as e:
                    errors += 1
                    self.logger.warning(f"Route calculation error {loc1['id']}→{loc2['id']}: {e}")
                    
                    # Fallback to Haversine
                    distance_km = self._haversine_distance(loc1, loc2)
                    time_minutes = distance_km * 60 / 50
                    
                    self.distance_matrix[i][j] = distance_km
                    self.time_matrix[i][j] = time_minutes
        
        total_routes = n * (n - 1)
        print(f"✅ OSRM matrix calculation complete:")
        print(f"  - Cache hits: {cache_hits}/{total_routes} ({cache_hits/total_routes*100:.1f}%)")
        print(f"  - API calls: {api_calls}/{total_routes} ({api_calls/total_routes*100:.1f}%)")
        print(f"  - Errors: {errors}/{total_routes}")
        
        if self.use_truck_speeds:
            print(f"  🚛 Truck speed adjustments applied")
    
    def _calculate_weighted_truck_ratio(self, road_composition: Dict[str, float]) -> float:
        """Calculate weighted truck speed ratio based on road composition."""
        weighted_ratio = 0.0
        total_distance = 0.0
        
        for road_type, distance_km in road_composition.items():
            if road_type in self.truck_speed_ratios:
                ratio = self.truck_speed_ratios[road_type]
                weighted_ratio += ratio * distance_km
                total_distance += distance_km
        
        if total_distance > 0:
            return weighted_ratio / total_distance
        else:
            return self.truck_speed_ratios.get('default', 0.80)
    
    def _haversine_distance(self, loc1: Dict, loc2: Dict) -> float:
        """Calculate Haversine distance between two locations."""
        import math
        
        lat1, lon1 = loc1['y'], loc1['x']
        lat2, lon2 = loc2['y'], loc2['x']
        
        R = 6371  # Earth's radius in kilometers
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def get_distance(self, from_idx: int, to_idx: int) -> float:
        """Get distance between location indices."""
        return self.distance_matrix[from_idx][to_idx]
    
    def get_travel_time(self, from_idx: int, to_idx: int) -> float:
        """Get travel time between location indices."""
        return self.time_matrix[from_idx][to_idx]
    
    def get_cache_stats(self) -> Dict:
        """Get route database cache statistics."""
        return self.route_db.get_cache_stats()

if __name__ == "__main__":
    # Test the route database
    print("🧪 Testing Route Database")
    
    # Test locations (Italian cities)
    test_locations = [
        {'id': 'milan', 'x': 9.1900, 'y': 45.4642},
        {'id': 'rome', 'x': 12.4964, 'y': 41.9028},
        {'id': 'venice', 'x': 12.3155, 'y': 45.4408},
        {'id': 'florence', 'x': 11.2558, 'y': 43.7696}
    ]
    
    # Test calculator
    calculator = CachedOSRMDistanceCalculator(test_locations, use_truck_speeds=False)
    
    print(f"\n📊 Test Results:")
    for i, loc1 in enumerate(test_locations):
        for j, loc2 in enumerate(test_locations):
            if i != j:
                distance = calculator.get_distance(i, j)
                time = calculator.get_travel_time(i, j)
                print(f"  {loc1['id']} → {loc2['id']}: {distance:.1f}km, {time:.1f}min")
