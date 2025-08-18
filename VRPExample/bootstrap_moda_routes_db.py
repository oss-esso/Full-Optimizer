#!/usr/bin/env python3
"""
Bootstrap script to initialize a fresh moda_routes.db and populate it with routes from the Furgoni scenario.
This ensures that we have a clean starting point with proper schema including route_geometry.
"""

import os
import sys
import time
import sqlite3

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from route_database import RouteDatabase
from vrp_scenarios import create_furgoni_scenario

def initialize_moda_routes_db():
    """Initialize a fresh moda_routes.db database."""
    # Remove any existing database
    if os.path.exists("moda_routes.db"):
        print(f"🗑️ Removing existing moda_routes.db")
        try:
            os.remove("moda_routes.db")
        except Exception as e:
            print(f"⚠️ Failed to remove existing database: {e}")
            return False
    
    # Create fresh database
    try:
        db = RouteDatabase("moda_routes.db")
        print(f"✅ Created fresh moda_routes.db")
        return db
    except Exception as e:
        print(f"❌ Failed to create database: {e}")
        return None

def get_all_location_coordinates(scenario):
    """Get all coordinates from the scenario locations."""
    coordinates = []
    
    for loc_id, loc in scenario.locations.items():
        coordinates.append({
            'id': loc_id,
            'x': loc.x,  # longitude
            'y': loc.y   # latitude
        })
    
    print(f"📍 Extracted {len(coordinates)} unique locations")
    return coordinates

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate Euclidean distance between two lat/lon points."""
    import math
    # This is a simple approximation, not great for real distances but ok for ranking
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    dy = lat2 - lat1
    return math.sqrt(dx * dx + dy * dy) * 111.32  # rough km conversion
    
def populate_routes_for_locations(db, locations, max_pairs=None, limit_k=None):
    """Populate the database with routes between location pairs."""
    print("\n🔄 Starting route population...")
    n = len(locations)
    total_possible_pairs = n ** 2
    
    if max_pairs is not None:
        # Limit total number of pairs to process
        print(f"⚠️ Limiting to {max_pairs} random pairs for testing")
        import random
        pairs_to_process = []
        for i in range(min(max_pairs, total_possible_pairs)):
            idx1 = random.randint(0, n - 1)
            idx2 = random.randint(0, n - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, n - 1)
            pairs_to_process.append((idx1, idx2))
    elif limit_k is not None:
        # Process only k nearest neighbors for each location
        print(f"⚠️ Processing only {limit_k} nearest neighbors for each location")
        pairs_to_process = []
        
        # Calculate distance from each location to all others
        for i in range(n):
            loc1 = locations[i]
            distances = []
            
            for j in range(n):
                if i == j:
                    continue
                    
                loc2 = locations[j]
                dist = calculate_distance(loc1['y'], loc1['x'], loc2['y'], loc2['x'])
                distances.append((j, dist))
            
            # Sort by distance and take k nearest
            distances.sort(key=lambda x: x[1])
            nearest_neighbors = [(i, j) for j, _ in distances[:limit_k]]
            pairs_to_process.extend(nearest_neighbors)
            
            if i % 10 == 0:
                print(f"  📊 Computed nearest neighbors for {i}/{n} locations")
                
    else:
        # Process all pairs (n^2 - n total routes)
        print(f"⚠️ Processing ALL {total_possible_pairs} possible pairs - this may take a long time")
        pairs_to_process = [(i, j) for i in range(n) for j in range(n) if i != j]
    
    print(f"🎯 Will process {len(pairs_to_process)} route pairs")
    
    # Process all defined pairs
    routes_added = 0
    routes_existing = 0
    start_time = time.time()
    
    # Estimate API rate limits
    api_limit_per_minute = 60  # Standard OSRM public API limit
    
    # For API rate limiting
    new_routes_in_last_minute = 0
    last_minute_check = time.time()
    
    for idx, (i, j) in enumerate(pairs_to_process):
        loc1 = locations[i]
        loc2 = locations[j]
        
        # Check if we need to slow down for API rate limits
        current_time = time.time()
        if current_time - last_minute_check >= 60:
            # Reset the counter every minute
            new_routes_in_last_minute = 0
            last_minute_check = current_time
        elif new_routes_in_last_minute >= api_limit_per_minute:
            # Sleep until a minute has passed since we started counting
            sleep_time = 60 - (current_time - last_minute_check)
            if sleep_time > 0:
                print(f"  ⏱️ Pausing for {sleep_time:.1f}s to respect API rate limits...")
                time.sleep(sleep_time)
                new_routes_in_last_minute = 0
                last_minute_check = time.time()
        
        # Call get_route to fetch from OSRM and store in database
        route_data = db.get_route(
            from_lat=loc1['y'], from_lon=loc1['x'],
            to_lat=loc2['y'], to_lon=loc2['x'],
            from_id=loc1['id'], to_id=loc2['id']
        )
        
        if route_data:
            if route_data['cached']:
                routes_existing += 1
            else:
                # This is a new route we just added
                routes_added += 1
                new_routes_in_last_minute += 1
                
                # Only print detailed info for newly added routes
                print(f"  ✅ Added route #{routes_added}: {loc1['id']} -> {loc2['id']} "
                      f"({loc1['y']:.4f},{loc1['x']:.4f} -> {loc2['y']:.4f},{loc2['x']:.4f})")
                
                # Check if geometry exists
                has_geometry = 'route_geometry' in route_data and route_data['route_geometry'] is not None
                if not has_geometry:
                    print(f"  ⚠️ Warning: No geometry data for this route")
        
        # Print progress
        if (idx + 1) % 20 == 0 or idx == len(pairs_to_process) - 1:
            elapsed = time.time() - start_time
            progress = (idx + 1) / len(pairs_to_process) * 100
            
            # Calculate ETA
            pairs_processed = idx + 1
            remaining_pairs = len(pairs_to_process) - pairs_processed
            if pairs_processed > 0:
                avg_time_per_pair = elapsed / pairs_processed
                eta_seconds = remaining_pairs * avg_time_per_pair
                
                # Format ETA nicely
                eta_str = ""
                if eta_seconds > 3600:
                    eta_str = f"{eta_seconds/3600:.1f} hours"
                elif eta_seconds > 60:
                    eta_str = f"{eta_seconds/60:.1f} minutes"
                else:
                    eta_str = f"{eta_seconds:.1f} seconds"
                
                print(f"  📊 Progress: {pairs_processed}/{len(pairs_to_process)} ({progress:.1f}%) - "
                      f"Added: {routes_added}, Existing: {routes_existing}, ETA: {eta_str}")
    
    total_time = time.time() - start_time
    print(f"\n✅ Finished populating routes:")
    print(f"  - New routes added: {routes_added}")
    print(f"  - Existing routes: {routes_existing}")
    print(f"  - Total processed: {routes_added + routes_existing}")
    print(f"  - Total time: {total_time:.1f} seconds")
    
    # Calculate average only for meaningful operations
    if routes_added > 0:
        print(f"  - Average time per new route: {total_time / routes_added:.3f} seconds")
    else:
        print("  - All routes were already in the database")
    
    # Get stats from the database
    stats = db.get_cache_stats()
    print(f"\n📊 Final Database Stats:")
    print(f"  - Total cached routes: {stats['total_routes']}")
    print(f"  - Recent routes: {stats['recent_routes']}")
    print(f"  - Database size: {stats['database_size_mb']:.2f} MB")
    
    return routes_added

def verify_geometry_data(db):
    """Verify that routes have geometry data and check database integrity."""
    try:
        with sqlite3.connect("moda_routes.db") as conn:
            conn.row_factory = sqlite3.Row
            
            # Get total route count
            cursor = conn.execute("SELECT COUNT(*) as count FROM routes")
            row = cursor.fetchone()
            total_routes = row['count'] if row else 0
            
            # Get routes with geometry
            cursor = conn.execute("SELECT COUNT(*) as count FROM routes WHERE route_geometry IS NOT NULL")
            row = cursor.fetchone()
            routes_with_geometry = row['count'] if row else 0
            
            # Get routes without geometry
            cursor = conn.execute("SELECT COUNT(*) as count FROM routes WHERE route_geometry IS NULL")
            row = cursor.fetchone()
            routes_without_geometry = row['count'] if row else 0
            
            # Get unique coordinate pairs
            cursor = conn.execute("""
                SELECT COUNT(*) as count FROM 
                (SELECT DISTINCT from_lat, from_lon, to_lat, to_lon FROM routes)
            """)
            row = cursor.fetchone()
            unique_coord_pairs = row['count'] if row else 0
            
            print(f"\n🗺️ Route Database Verification:")
            print(f"  - Total routes: {total_routes}")
            print(f"  - Unique coordinate pairs: {unique_coord_pairs}")
            print(f"  - Routes with geometry data: {routes_with_geometry} ({routes_with_geometry/total_routes*100:.1f}%)")
            print(f"  - Routes without geometry data: {routes_without_geometry}")
            
            if routes_with_geometry > 0:
                # Fetch a sample route with geometry
                cursor = conn.execute("""
                    SELECT from_id, to_id, from_lat, from_lon, to_lat, to_lon, route_geometry 
                    FROM routes 
                    WHERE route_geometry IS NOT NULL 
                    LIMIT 1
                """)
                sample = cursor.fetchone()
                if sample:
                    import json
                    try:
                        geometry = json.loads(sample['route_geometry'])
                        print(f"\n  - Sample route coordinates: "
                              f"({sample['from_lat']:.6f}, {sample['from_lon']:.6f}) -> "
                              f"({sample['to_lat']:.6f}, {sample['to_lon']:.6f})")
                        print(f"  - Sample location IDs: {sample['from_id']} -> {sample['to_id']}")
                        print(f"  - Sample geometry points: {len(geometry)}")
                        
                        # Validate geometry matches the endpoints
                        if len(geometry) >= 2:
                            first_point = geometry[0]
                            last_point = geometry[-1]
                            print(f"  - First geometry point: ({first_point[0]:.6f}, {first_point[1]:.6f})")
                            print(f"  - Last geometry point: ({last_point[0]:.6f}, {last_point[1]:.6f})")
                    except Exception as e:
                        print(f"  ⚠️ Failed to parse geometry JSON: {e}")
            
            # Database has valid geometry if at least 1 route has geometry data
            return routes_with_geometry > 0
    except Exception as e:
        print(f"⚠️ Error verifying geometry: {e}")
        return False

def main():
    print("🔄 INITIALIZING MODA_ROUTES.DB FOR FURGONI SCENARIO")
    print("=" * 60)
    
    # Initialize fresh database
    db = initialize_moda_routes_db()
    if not db:
        print("❌ Failed to initialize database. Exiting.")
        return
    
    # Load the Furgoni scenario
    print("\n📦 Loading Furgoni scenario...")
    try:
        scenario = create_furgoni_scenario()
        print(f"✅ Loaded Furgoni scenario with {len(scenario.locations)} locations")
    except Exception as e:
        print(f"❌ Failed to load Furgoni scenario: {e}")
        return
    
    # Get all location coordinates
    locations = get_all_location_coordinates(scenario)
    
    # Ask the user how to proceed
    import sys
    print("\n" + "=" * 60)
    print("ROUTE POPULATION OPTIONS:")
    print("-" * 60)
    print("1) Process ALL pairs (fully populate the cache) - this will take a long time")
    print(f"   Total pairs: {len(locations) * (len(locations) - 1)} routes")
    print(f"   Estimated time: {len(locations) * (len(locations) - 1) * 0.5 / 60:.1f} minutes assuming 0.5s per route")
    print("\n2) Process a LIMITED number of pairs (faster for testing)")
    print("\n3) Process only NEAREST NEIGHBORS for each location (efficient)")
    
    choice = input("\nSelect option (1, 2, or 3) [default=2]: ").strip() or "2"
    
    if choice == "1":
        # Process all pairs
        populate_routes_for_locations(db, locations, max_pairs=None, limit_k=None)
    elif choice == "3":
        # Process k nearest neighbors
        k = input("Enter number of nearest neighbors to process for each location [default=10]: ").strip()
        k = int(k) if k.isdigit() else 10
        populate_routes_for_locations(db, locations, max_pairs=None, limit_k=k)
    else:
        # Process limited random pairs
        limit = input("Enter number of random pairs to process [default=100]: ").strip()
        limit = int(limit) if limit.isdigit() else 100
        populate_routes_for_locations(db, locations, max_pairs=limit, limit_k=None)
    
    # Verify that we have geometry data
    has_geometry = verify_geometry_data(db)
    
    print("\n" + "=" * 60)
    print("✅ DATABASE INITIALIZATION COMPLETE")
    print("=" * 60)
    print(f"Route database ready at: moda_routes.db")
    print(f"Route geometry data: {'Available' if has_geometry else 'Missing - check OSRM API'}")
    print("\nYou can now use this database with the VRP optimizer scripts.")
    
if __name__ == "__main__":
    main()
