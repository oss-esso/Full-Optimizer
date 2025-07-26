"""
OSRM Utilities Module

This module contains utilities for OSRM route querying and caching that can be 
used both by the main route provider and the pre-computation script.

Extracted from route_provider.py to make OSRM functionality more accessible
for batch pre-computation operations.
"""

import requests
import sqlite3
import json
import os
from typing import Dict, Optional, Tuple, Any

def init_route_cache_db(db_path: str):
    """Initialize the route cache database with proper schema."""
    conn = sqlite3.connect(db_path)
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

def cache_route_data(db_path: str, start_node_id: str, end_node_id: str, 
                    distance_km: float, duration_minutes: float, 
                    road_composition: Dict[str, float],
                    route_geometry: Optional[Dict[str, Any]]):
    """Cache route data in the local database."""
    conn = sqlite3.connect(db_path)
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

def is_route_cached(db_path: str, start_node_id: str, end_node_id: str) -> bool:
    """Check if route is already cached in database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 1 FROM route_cache 
        WHERE start_node_id = ? AND end_node_id = ?
    """, (start_node_id, end_node_id))
    
    result = cursor.fetchone() is not None
    conn.close()
    return result

def extract_road_composition(route_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract road composition from OSRM route data.
    
    This function analyzes the route annotations to determine the percentage
    of different road types used in the route.
    """
    road_composition = {}
    
    try:
        if 'legs' not in route_data:
            return road_composition
            
        for leg in route_data['legs']:
            if 'annotation' not in leg:
                continue
                
            annotations = leg['annotation']
            
            # Extract road classes if available
            if 'metadata' in annotations and 'datasource_names' in annotations['metadata']:
                # OSRM provides road class information in metadata
                datasources = annotations['metadata']['datasource_names']
                
                # Simple heuristic: count different types based on duration
                durations = annotations.get('duration', [])
                total_duration = sum(durations) if durations else 1
                
                # Categorize based on speed (rough approximation)
                distances = annotations.get('distance', [])
                
                for i, (distance, duration) in enumerate(zip(distances, durations)):
                    if duration > 0:
                        speed_kmh = (distance / 1000.0) / (duration / 3600.0)
                        
                        # Categorize road type based on speed
                        if speed_kmh >= 80:
                            road_type = 'motorway'
                        elif speed_kmh >= 60:
                            road_type = 'trunk'
                        elif speed_kmh >= 40:
                            road_type = 'primary'
                        elif speed_kmh >= 25:
                            road_type = 'secondary'
                        else:
                            road_type = 'residential'
                            
                        road_composition[road_type] = road_composition.get(road_type, 0) + duration
                
                # Normalize to percentages
                total_time = sum(road_composition.values())
                if total_time > 0:
                    for road_type in road_composition:
                        road_composition[road_type] = road_composition[road_type] / total_time
                        
    except Exception as e:
        print(f"Warning: Could not extract road composition: {e}")
        
    return road_composition

def query_osrm_and_cache(osrm_url: str, db_path: str,
                        start_coords: Tuple[float, float], end_coords: Tuple[float, float],
                        start_node_id: str, end_node_id: str) -> Optional[Dict[str, Any]]:
    """
    Query OSRM API and cache the results.
    
    Args:
        osrm_url: OSRM server URL
        db_path: Path to SQLite cache database
        start_coords: (lon, lat) coordinates for start
        end_coords: (lon, lat) coordinates for end
        start_node_id: Node ID for start location
        end_node_id: Node ID for end location
        
    Returns:
        Dictionary with route data or None if failed
    """
    coords_str = f"{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
    url = f"{osrm_url}/route/v1/driving/{coords_str}"
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
        road_composition = extract_road_composition(route)
        
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
        cache_route_data(db_path, start_node_id, end_node_id, distance_km, 
                        duration_minutes, road_composition, route_geometry)
        
        return {
            'distance_km': distance_km,
            'duration_minutes': duration_minutes,
            'road_composition': road_composition,
            'route_geometry': route_geometry
        }
        
    except requests.exceptions.RequestException as e:
        print(f"OSRM request failed: {e}")
        return None
    except Exception as e:
        print(f"Error processing OSRM response: {e}")
        return None

def get_cache_stats(db_path: str) -> Dict[str, int]:
    """Get statistics about the route cache."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT COUNT(*) FROM route_cache")
        total_routes = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT start_node_id) FROM route_cache")
        unique_start_nodes = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT end_node_id) FROM route_cache")
        unique_end_nodes = cursor.fetchone()[0]
        
        return {
            'total_routes': total_routes,
            'unique_start_nodes': unique_start_nodes,
            'unique_end_nodes': unique_end_nodes
        }
    finally:
        conn.close()

def clear_cache(db_path: str):
    """Clear all cached routes."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM route_cache")
    conn.commit()
    conn.close()
    
    print(f"Cache cleared: {db_path}")
