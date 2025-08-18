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

# Configure logging
logger = logging.getLogger(__name__)
# Set to WARNING to suppress most debug messages
logger.setLevel(logging.WARNING)

class RouteDatabase:
    """
    Persistent database for caching OSRM routing results.
    
    This class manages a SQLite database that stores route information including:
    - Source and destination coordinates
    - Distance and duration
    - Road composition for truck speed analysis
    - Timestamps for cache management
    """
    
    def __init__(self, db_path: str = "moda_routes.db"):
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
        
        # Auto-consolidate existing databases on first run
        if self.db_path == "moda_routes.db":
            self._auto_consolidate_databases()
        
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
                    route_geometry TEXT,
                    osrm_success BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(from_lat, from_lon, to_lat, to_lon)
                )
            """)
            
            # Add route_geometry column if it doesn't exist (for existing databases)
            try:
                conn.execute("ALTER TABLE routes ADD COLUMN route_geometry TEXT")
                conn.commit()
            except sqlite3.OperationalError:
                pass  # Column already exists
            
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
            # Parse geometry if available
            route_geometry = None
            if cached_route['route_geometry']:
                try:
                    route_geometry = json.loads(cached_route['route_geometry'])
                except json.JSONDecodeError:
                    pass
            
            return {
                'distance_km': cached_route['distance_km'],
                'duration_minutes': cached_route['duration_minutes'],
                'road_composition': json.loads(cached_route['road_composition']) if cached_route['road_composition'] else {},
                'route_geometry': route_geometry,
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
    
    def _fetch_osrm_route(self, from_lat: float, from_lon: float, to_lat: float, to_lon: float,
                         from_id: str = None, to_id: str = None) -> Optional[Dict]:
        """Fetch route from OSRM API."""
        try:
            # Build OSRM request
            coords = f"{from_lon},{from_lat};{to_lon},{to_lat}"
            url = f"{self.osrm_url}/route/v1/driving/{coords}"
            
            params = {
                'overview': 'full',
                'geometries': 'geojson',
                'steps': 'true'
            }
            
            response = requests.get(url, params=params, timeout=self.request_timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if data['code'] == 'Ok' and 'routes' in data and len(data['routes']) > 0:
                route = data['routes'][0]
                
                # Extract basic route information
                distance_km = route['distance'] / 1000.0  # Convert to km
                duration_minutes = route['duration'] / 60.0  # Convert to minutes
                
                # Extract road composition
                road_composition = self._extract_road_composition(route)
                
                # Extract route geometry
                route_geometry = None
                if 'geometry' in route:
                    route_geometry = route['geometry']
                
                # Print concise route summary (only for new routes)
                if road_composition:
                    road_summary = ", ".join([f"{k}: {v:.1f}km" for k, v in road_composition.items() if v > 0.1])
                else:
                    road_summary = "unknown"
                
                try:
                    from vrp_scenarios import DEFAULT_TRUCK_SPEED_RATIOS
                    
                    # Calculate adjusted times for different truck types
                    standard_time = self._calculate_truck_adjusted_time(duration_minutes, road_composition, 
                                                                        DEFAULT_TRUCK_SPEED_RATIOS['standard'])
                    heavy_time = self._calculate_truck_adjusted_time(duration_minutes, road_composition, 
                                                                     DEFAULT_TRUCK_SPEED_RATIOS['heavy'])
                    
                    route_key = f"{from_id or 'unknown'}->{to_id or 'unknown'}"
                    print(f"  🛣️ OSRM: {route_key} = {distance_km:.1f}km, furgone:{standard_time:.1f}min, camion:{heavy_time:.1f}min")
                    print(f"    📍 Road composition: {road_summary}")
                except ImportError:
                    # Fallback to original format if truck speed ratios not available
                    route_key = f"{from_id or 'unknown'}->{to_id or 'unknown'}"
                    print(f"  🛣️ OSRM: {route_key} = {distance_km:.1f}km, {duration_minutes:.1f}min")
                    print(f"    📍 Road composition: {road_summary}")
                
                return {
                    'distance_km': distance_km,
                    'duration_minutes': duration_minutes,
                    'road_composition': road_composition,
                    'route_geometry': route_geometry
                }
            else:
                self.logger.warning(f"OSRM API error: {data.get('message', 'Unknown error')}")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.warning(f"OSRM API timeout for route {from_id}->{to_id}")
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
        elif any(keyword in name for keyword in ['comunale', 'secondary', 'sc']):
            return 'secondary'
        elif any(keyword in name for keyword in ['residential', 'locale']):
            return 'residential'
        else:
            return 'tertiary'  # Default for unnamed/other roads
    
    def _store_route(self, from_lat: float, from_lon: float, to_lat: float, to_lon: float,
                    from_id: str, to_id: str, route_data: Dict):
        """Store route in database."""
        try:
            with self._get_connection() as conn:
                # Serialize complex data as JSON
                road_composition_json = json.dumps(route_data.get('road_composition', {}))
                route_geometry_json = json.dumps(route_data.get('route_geometry')) if route_data.get('route_geometry') else None
                
                conn.execute("""
                    INSERT OR REPLACE INTO routes 
                    (from_lat, from_lon, from_id, to_lat, to_lon, to_id, 
                     distance_km, duration_minutes, road_composition, route_geometry, osrm_success)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """, (from_lat, from_lon, from_id, to_lat, to_lon, to_id,
                      route_data['distance_km'], route_data['duration_minutes'], 
                      road_composition_json, route_geometry_json))
                
                conn.commit()
        except Exception as e:
            self.logger.error(f"Database storage error: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) as total_routes FROM routes WHERE osrm_success = 1")
                total_routes = cursor.fetchone()['total_routes']
                
                # Get database file size
                db_size_bytes = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                db_size_mb = db_size_bytes / (1024 * 1024)
                
                return {
                    'total_routes': total_routes,
                    'database_size_mb': db_size_mb,
                    'database_path': self.db_path
                }
        except Exception as e:
            self.logger.error(f"Stats retrieval error: {e}")
            return {'total_routes': 0, 'database_size_mb': 0, 'database_path': self.db_path}
    
    def _print_cache_stats(self):
        """Print current cache statistics."""
        stats = self.get_cache_stats()
        print(f"  📊 {stats['total_routes']} routes cached ({stats['database_size_mb']:.2f} MB)")
    
    def consolidate_databases(self, db_paths: List[str]):
        """Consolidate multiple route databases into this one."""
        print(f"🔄 Consolidating {len(db_paths)} databases into {self.db_path}")
        
        routes_added = 0
        routes_skipped = 0
        
        for source_db in db_paths:
            if source_db == self.db_path:
                continue  # Skip self
                
            try:
                source_conn = sqlite3.connect(source_db, timeout=30.0)
                source_conn.row_factory = sqlite3.Row
                
                # Check if source database has the expected schema
                cursor = source_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='routes'")
                if not cursor.fetchone():
                    print(f"  ⚠️ Skipping {source_db}: No routes table found")
                    source_conn.close()
                    continue
                
                # Get all routes from source database
                cursor = source_conn.execute("SELECT * FROM routes WHERE osrm_success = 1")
                
                with self._get_connection() as target_conn:
                    for row in cursor:
                        try:
                            target_conn.execute("""
                                INSERT OR IGNORE INTO routes 
                                (from_lat, from_lon, from_id, to_lat, to_lon, to_id, 
                                 distance_km, duration_minutes, road_composition, route_geometry, osrm_success)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (row['from_lat'], row['from_lon'], row.get('from_id'), 
                                  row['to_lat'], row['to_lon'], row.get('to_id'),
                                  row['distance_km'], row['duration_minutes'], 
                                  row.get('road_composition'), row.get('route_geometry'), 1))
                            routes_added += 1
                        except sqlite3.IntegrityError:
                            routes_skipped += 1
                    
                    target_conn.commit()
                
                source_conn.close()
                print(f"  ✅ Consolidated {source_db}: {routes_added} routes added, {routes_skipped} duplicates skipped")
                
            except Exception as e:
                print(f"  ❌ Error consolidating {source_db}: {e}")
        
        print(f"✅ Consolidation complete: {routes_added} total routes added")
        self._print_cache_stats()
    
    def _auto_consolidate_databases(self):
        """Automatically consolidate existing databases in current and parent directories."""
        current_dir = os.getcwd()
        parent_dir = os.path.dirname(current_dir)
        
        # Find databases in current directory
        current_dir_dbs = []
        for file in os.listdir(current_dir):
            if file.endswith('.db') and file != self.db_path and 'routes' in file.lower():
                current_dir_dbs.append(os.path.join(current_dir, file))
        
        # Find databases in parent directory
        parent_dir_dbs = []
        if os.path.exists(parent_dir):
            for file in os.listdir(parent_dir):
                if file.endswith('.db') and 'routes' in file.lower():
                    parent_dir_dbs.append(os.path.join(parent_dir, file))
        
        all_dbs = current_dir_dbs + parent_dir_dbs
        
        if all_dbs:
            print(f"🔍 Found {len(all_dbs)} existing database files to consolidate")
            self.consolidate_databases(all_dbs)

class CachedOSRMDistanceCalculator:
    """
    OSRM distance calculator with persistent database caching and truck speed adjustments.
    
    This calculator uses the RouteDatabase to cache OSRM results and applies
    truck speed adjustments based on road composition analysis.
    """
    
    def __init__(self, locations: List[Dict], truck_speed_ratios: Optional[Dict] = None,
                 use_truck_speeds: bool = False, use_road_composition: bool = True,
                 db_path: str = "moda_routes.db"):
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
        print(f"🎯 Target routes to cache: {n * (n - 1)} (all pairs)")
        
        cache_hits = 0
        api_calls = 0
        errors = 0
        missing_routes = []
        
        # First pass: check cache and collect missing routes
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue  # Same location
                
                loc1 = self.locations[i]
                loc2 = self.locations[j]
                
                # Check if route exists in cache
                from_lat_r, from_lon_r = self.route_db._round_coordinates(loc1['y'], loc1['x'])
                to_lat_r, to_lon_r = self.route_db._round_coordinates(loc2['y'], loc2['x'])
                
                cached_route = self.route_db._lookup_cached_route(from_lat_r, from_lon_r, to_lat_r, to_lon_r)
                
                if cached_route:
                    cache_hits += 1
                    # Use cached data
                    route_geometry = None
                    if cached_route['route_geometry']:
                        try:
                            route_geometry = json.loads(cached_route['route_geometry'])
                        except json.JSONDecodeError:
                            pass
                    
                    distance_km = cached_route['distance_km']
                    duration_minutes = cached_route['duration_minutes']
                    
                    # Apply truck speed adjustments if enabled
                    if self.use_truck_speeds and self.truck_speed_ratios:
                        road_composition = json.loads(cached_route['road_composition']) if cached_route['road_composition'] else {}
                        if self.use_road_composition and road_composition:
                            truck_factor = self._calculate_weighted_truck_ratio(road_composition)
                            duration_minutes *= (1.0 / truck_factor)
                        else:
                            default_ratio = self.truck_speed_ratios.get('default', 0.80)
                            duration_minutes *= (1.0 / default_ratio)
                    
                    # Add service time at origin
                    service_time = loc1.get('service_time', 0)
                    total_time = duration_minutes + service_time
                    
                    self.distance_matrix[i][j] = distance_km
                    self.time_matrix[i][j] = total_time
                else:
                    # Route not cached - add to missing list
                    missing_routes.append((i, j, loc1, loc2))
        
        # Second pass: fetch missing routes from OSRM API
        if missing_routes:
            print(f"🌐 Fetching {len(missing_routes)} missing routes from OSRM API...")
            
            for route_idx, (i, j, loc1, loc2) in enumerate(missing_routes):
                if route_idx % 50 == 0:
                    print(f"  📡 Progress: {route_idx}/{len(missing_routes)} routes fetched...")
                
                try:
                    # Force OSRM API call for missing route
                    route_data = self.route_db.get_route(
                        loc1['y'], loc1['x'], loc2['y'], loc2['x'],
                        loc1['id'], loc2['id']
                    )
                    
                    if route_data:
                        if not route_data['cached']:  # Should be a new API call
                            api_calls += 1
                        
                        distance_km = route_data['distance_km']
                        duration_minutes = route_data['duration_minutes']
                        
                        # Apply truck speed adjustments if enabled
                        if self.use_truck_speeds and self.truck_speed_ratios:
                            if self.use_road_composition and route_data['road_composition']:
                                truck_factor = self._calculate_weighted_truck_ratio(route_data['road_composition'])
                                duration_minutes *= (1.0 / truck_factor)
                            else:
                                default_ratio = self.truck_speed_ratios.get('default', 0.80)
                                duration_minutes *= (1.0 / default_ratio)
                        
                        # Add service time at origin
                        service_time = loc1.get('service_time', 0)
                        total_time = duration_minutes + service_time
                        
                        self.distance_matrix[i][j] = distance_km
                        self.time_matrix[i][j] = total_time
                    else:
                        errors += 1
                        # Fallback to Haversine only if OSRM fails
                        distance_km = self._haversine_distance(loc1, loc2)
                        time_minutes = distance_km * 60 / 50  # 50 km/h average
                        
                        self.distance_matrix[i][j] = distance_km
                        self.time_matrix[i][j] = time_minutes
                        
                except Exception as e:
                    errors += 1
                    self.logger.warning(f"Route calculation error {loc1['id']}->{loc2['id']}: {e}")
                    
                    # Fallback to Haversine
                    distance_km = self._haversine_distance(loc1, loc2)
                    time_minutes = distance_km * 60 / 50
                    
                    self.distance_matrix[i][j] = distance_km
                    self.time_matrix[i][j] = time_minutes
        
        total_routes = n * (n - 1)
        final_cache_stats = self.route_db.get_cache_stats()
        
        print(f"✅ OSRM matrix calculation complete:")
        print(f"  - Cache hits: {cache_hits}/{total_routes} ({cache_hits/total_routes*100:.1f}%)")
        print(f"  - API calls: {api_calls}/{total_routes} ({api_calls/total_routes*100:.1f}%)")
        print(f"  - Errors: {errors}/{total_routes}")
        print(f"  - Final cached routes: {final_cache_stats['total_routes']}")
        print(f"  - Database size: {final_cache_stats['database_size_mb']:.2f} MB")
        
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
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def get_distance(self, from_location_id: str, to_location_id: str) -> int:
        """Get distance between two locations for OR-Tools (integer required)."""
        try:
            from_idx = self._get_location_index(from_location_id)
            to_idx = self._get_location_index(to_location_id)
            
            if from_idx is not None and to_idx is not None:
                # Return distance in meters (integer) for OR-Tools
                distance_km = self.distance_matrix[from_idx][to_idx]
                return int(distance_km * 1000)
            else:
                self.logger.warning(f"Location not found: {from_location_id} or {to_location_id}")
                return 0
        except Exception as e:
            self.logger.error(f"Distance calculation error: {e}")
            return 0
    
    def get_time(self, from_location_id: str, to_location_id: str) -> int:
        """Get travel time between two locations for OR-Tools (integer required)."""
        try:
            from_idx = self._get_location_index(from_location_id)
            to_idx = self._get_location_index(to_location_id)
            
            if from_idx is not None and to_idx is not None:
                # Return time in seconds (integer) for OR-Tools
                time_minutes = self.time_matrix[from_idx][to_idx]
                return int(time_minutes * 60)
            else:
                self.logger.warning(f"Location not found: {from_location_id} or {to_location_id}")
                return 0
        except Exception as e:
            self.logger.error(f"Time calculation error: {e}")
            return 0
    
    def _get_location_index(self, location_id: str) -> Optional[int]:
        """Get the index of a location by its ID."""
        for i, location in enumerate(self.locations):
            if location['id'] == location_id:
                return i
        return None
