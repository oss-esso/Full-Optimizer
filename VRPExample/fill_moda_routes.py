#!/usr/bin/env python3
"""
Fill moda_routes.db with routes for Furgoni scenario.

This script:
1. Creates a clean moda_routes.db 
2. Loads the Furgoni scenario locations
3. Fetches and caches all routes between locations
4. Reports on cache statistics and coverage

This ensures we have complete route coverage for interactive maps.
"""

import os
import sys
import time
import logging
from itertools import permutations
import sqlite3
from datetime import datetime

# Add the parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import modules
from route_database import RouteDatabase
from vrp_scenarios import create_furgoni_scenario

def fill_scenario_cache():
    """
    Fill the moda_routes.db with routes for Furgoni scenario.
    
    We'll fetch and cache all possible routes between locations,
    ensuring the interactive map has all necessary route data.
    """
    # Always remove any existing database first to start fresh
    print("\n🔄 Creating fresh moda_routes.db...")
    if os.path.exists("moda_routes.db"):
        os.remove("moda_routes.db")
    
    # Initialize the route database
    db = RouteDatabase("moda_routes.db")
    
    # Create the Furgoni scenario
    print("\n📦 Loading Furgoni scenario...")
    scenario = create_furgoni_scenario()
    
    # Extract all locations
    locations = list(scenario.locations.values())
    location_count = len(locations)
    
    # Calculate all possible routes (permutations of 2)
    routes_to_check = list(permutations(range(location_count), 2))
    total_routes = len(routes_to_check)
    
    logger.info(f"Scenario loaded with {location_count} locations.")
    logger.info(f"Generated {total_routes} possible routes (permutations of 2) to check.")
    
    # Track progress
    start_time = time.time()
    routes_cached = 0
    api_calls = 0
    cache_hits = 0
    errors = 0
    
    # Progress tracking
    print(f"\n🗺️ Fetching routes for {location_count} locations ({total_routes} total routes)...")
    print(f"📊 Estimated time: ~{total_routes * 0.1:.1f} seconds at 10 routes/second")
    
    # Process all routes in batches to avoid rate limiting
    batch_size = 50  # Process 50 API calls at a time then pause
    
    for i, (start_idx, end_idx) in enumerate(routes_to_check):
        # Get location objects
        start_loc = locations[start_idx]
        end_loc = locations[end_idx]
        
        start_loc_id = start_loc.id
        end_loc_id = end_loc.id
        
        # Skip if start and end are the same
        if start_loc_id == end_loc_id:
            continue
            
        # Get coordinates
        start_lat, start_lon = start_loc.y, start_loc.x
        end_lat, end_lon = end_loc.y, end_loc.x
        
        try:
            # Check if we already have this route cached
            route_data = db.get_route(start_lat, start_lon, end_lat, end_lon, start_loc_id, end_loc_id)
            
            if route_data and route_data.get('cached', False):
                cache_hits += 1
                if i % 100 == 0:  # Only log occasionally to avoid spam
                    print(f"  ✅ Cache hit: {start_loc_id} → {end_loc_id}")
            else:
                # Route is not in cache or had an error, insert it
                api_calls += 1
                routes_cached += 1
                if i % 20 == 0:  # Only log occasionally to avoid spam
                    print(f"  🌐 API call: {start_loc_id} → {end_loc_id} = "
                          f"{route_data['distance_km']:.1f}km, {route_data['duration_minutes']:.1f}min")
                
                # If we've made a batch of API calls, pause briefly to avoid rate limiting
                if api_calls % batch_size == 0:
                    time.sleep(2)  # 2 second pause every batch_size API calls
                    print(f"  ⏱️ Progress: {i+1}/{total_routes} routes processed "
                          f"({(i+1)/total_routes*100:.1f}%) - "
                          f"API calls: {api_calls}")
            
        except Exception as e:
            errors += 1
            logger.error(f"Error fetching route {start_loc_id} → {end_loc_id}: {e}")
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Print cache statistics
    print("\n✅ ROUTE CACHING COMPLETE")
    print("=" * 50)
    print(f"⏱️ Processing time: {elapsed:.1f} seconds ({total_routes/elapsed:.1f} routes/second)")
    print(f"📊 Routes checked: {len(routes_to_check)}")
    print(f"📌 Cache hits: {cache_hits}")
    print(f"🌐 API calls: {api_calls}")
    print(f"❌ Errors: {errors}")
    
    # Get database stats
    cache_stats = db.get_cache_stats()
    print(f"\n📂 CACHE STATISTICS")
    print("=" * 50)
    print(f"📊 Total cached routes: {cache_stats['total_routes']}")
    print(f"💾 Database size: {cache_stats['database_size_mb']:.2f} MB")
    
    # Check for route geometry coverage
    with sqlite3.connect("moda_routes.db") as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT COUNT(*) as count FROM routes WHERE route_geometry IS NOT NULL")
        geometry_count = cursor.fetchone()['count']
    
    print(f"🗺️ Routes with geometry: {geometry_count} ({geometry_count/cache_stats['total_routes']*100:.1f}%)")
    
    # Check for coverage percentage
    coverage = (cache_stats['total_routes'] / total_routes) * 100
    print(f"\n🎯 COVERAGE: {coverage:.1f}%")
    
    return cache_stats['total_routes'], total_routes

if __name__ == "__main__":
    print("🗃️ FILLING MODA ROUTES DATABASE")
    print("=" * 50)
    
    try:
        cached_routes, total_routes = fill_scenario_cache()
        
        # Final output
        print("\n🏁 DATABASE CREATION SUMMARY")
        print("=" * 50)
        print(f"✅ Created moda_routes.db with {cached_routes}/{total_routes} routes")
        print(f"🗺️ Database ready for interactive map generation")
        print(f"💾 Use this database in all future VRP runs with:")
        print(f"    db_path='moda_routes.db'")
        
    except Exception as e:
        logger.exception(f"Error in fill_scenario_cache: {e}")
        print(f"❌ Failed to fill route cache: {e}")
        sys.exit(1)
