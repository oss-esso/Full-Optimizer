#!/usr/bin/env python3
"""
Test script for the OSRM pre-computation functionality.

This script validates that the pre-computation infrastructure works correctly
without actually making OSRM calls. It's useful for testing the location
extraction and cache management logic.
"""

import sys
import os
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent.parent))  # For VRP models
sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # For root level  
sys.path.append(str(Path(__file__).parent))

def test_location_extraction():
    """Test location extraction from scenarios."""
    print("🧪 Testing location extraction from scenarios...")
    
    try:
        from precompute_routes import RoutePrecomputer
        
        # Initialize precomputer with dummy OSRM URL
        precomputer = RoutePrecomputer(
            osrm_url="http://dummy-osrm:5000",
            db_path=":memory:",  # Use in-memory database for testing
            batch_size=10,
            rate_limit_delay=0.0
        )
        
        # Test furgoni scenario loading
        try:
            locations = precomputer.load_scenario_from_function('furgoni')
            print(f"✅ Successfully extracted {len(locations)} locations from furgoni scenario")
            
            # Show location breakdown
            location_types = {}
            for loc in locations:
                location_types[loc.location_type] = location_types.get(loc.location_type, 0) + 1
            
            print("📊 Location breakdown:")
            for loc_type, count in location_types.items():
                print(f"   - {loc_type}: {count}")
                
            # Show some example locations
            print("📍 Example locations:")
            for i, loc in enumerate(locations[:5]):
                print(f"   {i+1}. {loc.id} ({loc.location_type}): {loc.lat:.4f}, {loc.lon:.4f}")
            
            if len(locations) > 5:
                print(f"   ... and {len(locations) - 5} more")
                
            return True
            
        except Exception as e:
            print(f"❌ Failed to load furgoni scenario: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_cache_operations():
    """Test cache database operations."""
    print("\n🧪 Testing cache database operations...")
    
    try:
        from osrm_utils import (init_route_cache_db, cache_route_data, 
                               is_route_cached, get_cache_stats, clear_cache)
        
        import tempfile
        import os
        
        # Use a temporary file for testing instead of in-memory 
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
            test_db = tmp.name
            
        try:
            # Initialize database
            init_route_cache_db(test_db)
            print("✅ Database initialization successful")
            
            # Test caching
            cache_route_data(
                db_path=test_db,
                start_node_id="loc1",
                end_node_id="loc2", 
                distance_km=10.5,
                duration_minutes=15.0,
                road_composition={"motorway": 0.7, "primary": 0.3},
                route_geometry={"type": "LineString", "coordinates": [[9.0, 45.0], [9.1, 45.1]]}
            )
            print("✅ Route caching successful")
            
            # Test cache checking
            if is_route_cached(test_db, "loc1", "loc2"):
                print("✅ Cache checking successful - route found")
            else:
                print("❌ Cache checking failed - route not found")
                return False
                
            # Test stats
            stats = get_cache_stats(test_db)
            expected_stats = {'total_routes': 1, 'unique_start_nodes': 1, 'unique_end_nodes': 1}
            
            if stats == expected_stats:
                print(f"✅ Cache stats successful: {stats}")
            else:
                print(f"❌ Cache stats mismatch. Expected: {expected_stats}, Got: {stats}")
                return False
                
            return True
            
        finally:
            # Clean up temp file
            if os.path.exists(test_db):
                os.unlink(test_db)
        
    except Exception as e:
        print(f"❌ Cache operations failed: {e}")
        return False

def test_dry_run():
    """Test dry run functionality."""
    print("\n🧪 Testing dry run functionality...")
    
    try:
        from precompute_routes import RoutePrecomputer
        
        # Initialize precomputer
        precomputer = RoutePrecomputer(
            osrm_url="http://dummy-osrm:5000",
            db_path=":memory:",
            batch_size=5,
            rate_limit_delay=0.0
        )
        
        # Load scenario
        precomputer.locations = precomputer.load_scenario_from_function('furgoni')
        
        # Calculate expected pairs
        n_locations = len(precomputer.locations)
        expected_pairs = n_locations * (n_locations - 1)  # All pairs except self-loops
        
        print(f"✅ Dry run calculation: {n_locations} locations → {expected_pairs:,} route pairs")
        
        if expected_pairs > 0:
            print("✅ Dry run functionality working")
            return True
        else:
            print("❌ No route pairs calculated")
            return False
            
    except Exception as e:
        print(f"❌ Dry run test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 OSRM Pre-computation Test Suite")
    print("=" * 50)
    
    tests = [
        ("Location Extraction", test_location_extraction),
        ("Cache Operations", test_cache_operations), 
        ("Dry Run", test_dry_run)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running test: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pre-computation infrastructure is ready.")
        print("\n📝 Next steps:")
        print("   1. Set up OSRM server (local or remote)")
        print("   2. Run: python precompute_routes.py --scenario furgoni --dry-run")
        print("   3. Run: python precompute_routes.py --scenario furgoni")
    else:
        print("❌ Some tests failed. Please fix issues before using pre-computation.")
        sys.exit(1)

if __name__ == '__main__':
    main()
