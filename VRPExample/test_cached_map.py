#!/usr/bin/env python3
"""
Test script to verify that the interactive map uses cached OSRM routes.
This should be much faster for subsequent runs since routes are already cached.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from VRPExample.vrp_multiday_sequential_old import SequentialMultiDayVRP, test_sequential_multiday
from test_optimizer_clean_overnight import create_interactive_vrp_map
from vrp_scenarios import create_furgoni_scenario
import time

def test_cached_map_performance():
    """Test the performance difference of using cached routes for map generation."""
    print("🧪 TESTING CACHED ROUTE MAP PERFORMANCE")
    print("=" * 50)
    
    # Load Furgoni scenario
    print("📊 Loading Furgoni scenario...")
    scenario = create_furgoni_scenario()
    
    # Solve using sequential multi-day VRP
    print("🚛 Solving with Sequential Multi-Day VRP...")
    
    # Convert VRPInstance to the format expected by SequentialMultiDayVRP
    vehicles_list = []
    for vehicle in scenario.vehicles.values():
        vehicle_dict = {
            'id': vehicle.id,
            'capacity': vehicle.capacity,
            'volume_capacity': vehicle.volume_capacity,
            'cost_per_km': vehicle.cost_per_km,
            'truck_speed_ratios': getattr(vehicle, 'truck_speed_ratios', None)
        }
        vehicles_list.append(vehicle_dict)
    
    locations_list = []
    for location in scenario.locations.values():
        location_dict = {
            'id': location.id,
            'x': location.x,
            'y': location.y,
            'demand': location.demand,
            'service_time': location.service_time,
            'address': getattr(location, 'address', ''),
        }
        locations_list.append(location_dict)
    
    sequential_vrp = SequentialMultiDayVRP(
        vehicles=vehicles_list, 
        locations=locations_list, 
        use_truck_speeds=True,
        db_path="moda_routes.db"
    )
    
    solution_start = time.time()
    solution = sequential_vrp.solve_sequential_multiday(max_days=3)
    solution_time = time.time() - solution_start
    
    if not solution:
        print("❌ Failed to solve VRP")
        return
    
    print(f"✅ VRP solved in {solution_time:.2f} seconds")
    
    # Check cache statistics
    if hasattr(sequential_vrp.distance_calculator, 'route_db'):
        cache_stats = sequential_vrp.distance_calculator.route_db.get_cache_stats()
        print(f"📊 Cache Stats: {cache_stats['total_routes']} routes cached, {cache_stats['database_size_mb']:.2f} MB")
    
    # Generate interactive map using cached routes
    print("\n🗺️ Generating interactive map with cached routes...")
    map_start = time.time()
    
    try:
        html_file = create_interactive_vrp_map(scenario, solution, sequential_vrp)
        map_time = time.time() - map_start
        
        if html_file:
            print(f"✅ Interactive map generated in {map_time:.2f} seconds")
            print(f"🌐 Map saved as: {html_file}")
            print(f"💾 Total time saved by using cached routes instead of new OSRM calls!")
        else:
            print("❌ Failed to generate interactive map")
            
    except Exception as e:
        print(f"❌ Error generating map: {e}")
        import traceback
        traceback.print_exc()

def verify_cache_database():
    """Verify that the route database has geometry data."""
    print("\n🔍 VERIFYING CACHE DATABASE STRUCTURE")
    print("=" * 50)
    
    try:
        from route_database import RouteDatabase
        
        # Initialize database
        db = RouteDatabase("moda_routes.db")
        
        # Check if routes exist with geometry
        import sqlite3
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT COUNT(*) as total_routes,
                       COUNT(route_geometry) as routes_with_geometry,
                       AVG(LENGTH(route_geometry)) as avg_geometry_size
                FROM routes 
                WHERE osrm_success = 1
            """)
            
            stats = cursor.fetchone()
            print(f"📊 Database Statistics:")
            print(f"  - Total cached routes: {stats['total_routes']}")
            print(f"  - Routes with geometry: {stats['routes_with_geometry']}")
            if stats['avg_geometry_size'] is not None:
                print(f"  - Average geometry size: {stats['avg_geometry_size']:.1f} chars")
            else:
                print(f"  - Average geometry size: N/A")
            
            # Get a sample route with geometry
            cursor = conn.execute("""
                SELECT from_id, to_id, distance_km, 
                       LENGTH(route_geometry) as geom_size,
                       route_geometry IS NOT NULL as has_geometry
                FROM routes 
                WHERE osrm_success = 1 
                LIMIT 5
            """)
            
            print(f"\n📍 Sample Routes:")
            for row in cursor.fetchall():
                geom_status = "✅" if row['has_geometry'] else "❌"
                print(f"  {geom_status} {row['from_id']} → {row['to_id']}: {row['distance_km']:.1f}km, geometry: {row['geom_size']} chars")
                
    except Exception as e:
        print(f"❌ Database verification error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_cache_database()
    test_cached_map_performance()
