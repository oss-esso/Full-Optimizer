#!/usr/bin/env python3
"""
Force cache all missing routes for the Furgoni scenario.
This will complete the route database with all 54² = 2,916 route combinations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from route_database import CachedOSRMDistanceCalculator
from vrp_scenarios import create_furgoni_scenario
import time

def force_cache_missing_routes():
    """Force cache all missing routes for complete coverage."""
    print("🎯 FORCE CACHING MISSING ROUTES")
    print("=" * 50)
    
    # Load Furgoni scenario
    scenario = create_furgoni_scenario()
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
    
    n_locations = len(locations_list)
    expected_routes = n_locations * (n_locations - 1)
    
    print(f"📍 Scenario: {n_locations} locations")
    print(f"🎯 Expected routes: {expected_routes}")
    
    # Get initial cache stats
    from route_database import RouteDatabase
    db = RouteDatabase("moda_routes.db")
    initial_stats = db.get_cache_stats()
    
    print(f"📊 Initial cache: {initial_stats['total_routes']} routes")
    print(f"🔄 Missing routes: {expected_routes - initial_stats['total_routes']}")
    
    if initial_stats['total_routes'] >= expected_routes:
        print("✅ All routes already cached!")
        return
    
    # Force calculation of all routes (this will cache missing ones)
    print(f"\n🌐 Starting OSRM route caching...")
    start_time = time.time()
    
    calculator = CachedOSRMDistanceCalculator(
        locations=locations_list,
        use_truck_speeds=False,  # Disable truck speeds for faster caching
        db_path="moda_routes.db"
    )
    
    calculation_time = time.time() - start_time
    
    # Get final stats
    final_stats = calculator.get_cache_stats()
    
    print(f"\n🎉 ROUTE CACHING COMPLETE!")
    print(f"⏱️ Total time: {calculation_time:.1f} seconds")
    print(f"📊 Final cache: {final_stats['total_routes']} routes")
    print(f"💾 Database size: {final_stats['database_size_mb']:.1f} MB")
    print(f"✅ Coverage: {final_stats['total_routes']}/{expected_routes} ({final_stats['total_routes']/expected_routes*100:.1f}%)")
    
    # Calculate how many routes were added
    routes_added = final_stats['total_routes'] - initial_stats['total_routes']
    print(f"🚀 Routes added: {routes_added}")
    
    if final_stats['total_routes'] >= expected_routes * 0.95:
        print("🎯 Excellent! Route database is now complete for fast map generation!")
    else:
        print("⚠️ Some routes may have failed - but most are cached")

if __name__ == "__main__":
    force_cache_missing_routes()
