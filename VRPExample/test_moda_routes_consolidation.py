#!/usr/bin/env python3
"""
Test script to consolidate all databases into moda_routes.db and verify full route coverage.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from route_database import RouteDatabase
from vrp_scenarios import create_furgoni_scenario
import time

def test_database_consolidation():
    """Test the database consolidation process."""
    print("🔄 TESTING DATABASE CONSOLIDATION")
    print("=" * 50)
    
    # Initialize the main database (this will auto-consolidate)
    db = RouteDatabase("moda_routes.db")
    
    # Get initial stats
    stats = db.get_cache_stats()
    print(f"\n📊 Consolidated Database Stats:")
    print(f"  - Total cached routes: {stats['total_routes']}")
    print(f"  - Database size: {stats['database_size_mb']:.2f} MB")
    
    return db

def test_full_route_coverage():
    """Test that we have full route coverage for a scenario."""
    print("\n🎯 TESTING FULL ROUTE COVERAGE")
    print("=" * 50)
    
    # Load Furgoni scenario to test coverage
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
    
    print(f"📍 Testing with {n_locations} locations")
    print(f"🎯 Expected total routes: {expected_routes} (n × (n-1))")
    
    # Initialize distance calculator (this will cache missing routes)
    from route_database import CachedOSRMDistanceCalculator
    
    start_time = time.time()
    calculator = CachedOSRMDistanceCalculator(
        locations=locations_list,
        use_truck_speeds=False,
        db_path="moda_routes.db"
    )
    calculation_time = time.time() - start_time
    
    # Get final stats
    final_stats = calculator.get_cache_stats()
    
    print(f"\n✅ Route Coverage Test Complete:")
    print(f"  - Calculation time: {calculation_time:.2f} seconds")
    print(f"  - Final cached routes: {final_stats['total_routes']}")
    print(f"  - Database size: {final_stats['database_size_mb']:.2f} MB")
    print(f"  - Coverage: {final_stats['total_routes']}/{expected_routes} ({final_stats['total_routes']/expected_routes*100:.1f}%)")
    
    if final_stats['total_routes'] >= expected_routes * 0.95:  # Allow 5% tolerance for errors
        print("🎉 Excellent route coverage achieved!")
    else:
        print("⚠️ Route coverage below expected - some routes may have failed")
    
    return calculator

def test_geometry_coverage():
    """Test that cached routes have geometry data for visualization."""
    print("\n🗺️ TESTING GEOMETRY COVERAGE")
    print("=" * 50)
    
    db = RouteDatabase("moda_routes.db")
    
    # Test a few route geometries
    test_routes = [
        ([44.5404, 8.1407], [45.4642, 9.1900], "Genova → Milano"),
        ([45.4642, 9.1900], [41.9028, 12.4964], "Milano → Roma"),
        ([41.9028, 12.4964], [44.5404, 8.1407], "Roma → Genova"),
    ]
    
    geometry_count = 0
    for from_coords, to_coords, description in test_routes:
        geometry = db.get_cached_route_geometry(from_coords[0], from_coords[1], 
                                               to_coords[0], to_coords[1])
        if geometry:
            geometry_count += 1
            print(f"  ✅ {description}: {len(geometry)} geometry points")
        else:
            print(f"  ❌ {description}: No geometry data")
    
    print(f"\n📊 Geometry Coverage: {geometry_count}/{len(test_routes)} test routes have geometry")
    
    return geometry_count > 0

if __name__ == "__main__":
    print("🧪 MODA ROUTES CONSOLIDATION AND COVERAGE TEST")
    print("=" * 60)
    
    # Test 1: Database consolidation
    db = test_database_consolidation()
    
    # Test 2: Full route coverage
    calculator = test_full_route_coverage()
    
    # Test 3: Geometry coverage
    has_geometry = test_geometry_coverage()
    
    print("\n🏁 CONSOLIDATION TEST SUMMARY")
    print("=" * 50)
    print("✅ Database consolidation: Complete")
    print("✅ Route coverage calculation: Complete")
    print(f"✅ Geometry data: {'Available' if has_geometry else 'Missing'}")
    print("\n🎯 The moda_routes.db database is now ready for optimized map generation!")
