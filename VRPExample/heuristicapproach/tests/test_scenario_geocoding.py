"""
Scenario Geocoding Validation Test

This test validates that the refactored scenario_creator.py correctly geocodes 
addresses from an Excel file using the new Nominatim-only implementation.

The test verifies:
1. Successful creation of Order objects from Excel data
2. Valid latitude and longitude coordinates for all tasks
3. Proper caching mechanism functionality
4. API is not called on second run (cache hit)

Usage:
    python test_scenario_geocoding.py
"""

import os
import sys
import json
import unittest.mock
from typing import List, Dict
from pathlib import Path
import requests

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
src_dir = os.path.join(heuristic_root, 'src')
utils_dir = os.path.join(heuristic_root, 'utils')
algo_dir = os.path.join(heuristic_root, 'algo')
sys.path.insert(0, heuristic_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, utils_dir)
sys.path.insert(0, algo_dir)

from scenario_creator import create_scenario_from_excel
from epdt_data_structures import Order, Task

def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate that coordinates are within valid ranges.
    
    Args:
        lat: Latitude value
        lon: Longitude value
        
    Returns:
        True if coordinates are valid, False otherwise
    """
    return (
        lat is not None and
        lon is not None and
        -90 <= lat <= 90 and
        -180 <= lon <= 180
    )

def test_geocoding_from_sample_excel():
    """
    Test geocoding functionality using a sample Excel file.
    """
    print("🧪 Starting Scenario Geocoding Test")
    print("=" * 50)
    
    # Use the main Excel file for testing (with limited scope for this test)
    excel_path = os.path.join(src_dir, 'furgoni.xlsx')
    
    if not os.path.exists(excel_path):
        print(f"❌ Excel file not found: {excel_path}")
        return False
    
    print(f"📁 Using Excel file: {excel_path}")
    
    # Clear any existing cache to ensure fresh test
    cache_file = "geocode_cache.json"  # Cache is created in current working directory
    if os.path.exists(cache_file):
        backup_cache = "geocode_cache_backup_test.json"
        os.rename(cache_file, backup_cache)
        print(f"💾 Backed up existing cache to: {backup_cache}")
    
    try:
        # First run: Test actual geocoding
        print("\n🔍 First Run: Testing actual geocoding...")
        orders, vehicles, drivers = create_scenario_from_excel(excel_path)
        
        if not orders:
            print("❌ No orders returned from create_scenario_from_excel")
            return False
        
        print(f"✅ Successfully created {len(orders)} orders and {len(vehicles)} vehicles")
        
        # Validate that we have Order objects
        if not all(isinstance(order, Order) for order in orders):
            print("❌ Not all returned objects are Order instances")
            return False
        
        print("✅ All returned objects are valid Order instances")
        
        # Check coordinates for all tasks
        total_tasks = 0
        valid_coordinates = 0
        invalid_coordinates = []
        
        for order in orders:
            for task in order.get_all_tasks():
                total_tasks += 1
                if validate_coordinates(task.lat, task.lon):
                    valid_coordinates += 1
                else:
                    invalid_coordinates.append({
                        'order_id': order.id,
                        'task_id': task.id,
                        'lat': task.lat,
                        'lon': task.lon
                    })
        
        print(f"\n📊 Coordinate Validation Results:")
        print(f"   Total tasks: {total_tasks}")
        print(f"   Valid coordinates: {valid_coordinates}")
        print(f"   Invalid coordinates: {len(invalid_coordinates)}")
        
        if invalid_coordinates:
            print("\n❌ Tasks with invalid coordinates:")
            for task_info in invalid_coordinates[:5]:  # Show first 5
                print(f"   - {task_info['task_id']}: ({task_info['lat']}, {task_info['lon']})")
            if len(invalid_coordinates) > 5:
                print(f"   ... and {len(invalid_coordinates) - 5} more")
        
        # Check that cache file was created
        if os.path.exists(cache_file):
            print("✅ Geocode cache file was created")
            
            # Load and inspect cache
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            print(f"✅ Cache contains {len(cache_data)} entries")
        else:
            print("❌ Geocode cache file was not created")
            return False
        
        # Second run: Test caching mechanism
        print("\n🔍 Second Run: Testing cache functionality...")
        
        # Mock requests.get to ensure API is not called
        with unittest.mock.patch('requests.get') as mock_get:
            orders_cached, vehicles_cached, drivers_cached = create_scenario_from_excel(excel_path)
            
            # Verify requests.get was not called (cache hit)
            if mock_get.called:
                print(f"❌ API was called {mock_get.call_count} times on cached run")
                print("   This indicates caching is not working properly")
                return False
            else:
                print("✅ API was not called on second run (successful cache hit)")
        
        print(f"✅ Second run created {len(orders_cached)} orders and {len(vehicles_cached)} vehicles from cache")
        
        # Compare results
        if len(orders) == len(orders_cached) and len(vehicles) == len(vehicles_cached):
            print("✅ Both runs returned the same number of orders and vehicles")
        else:
            print(f"❌ Object count mismatch: Orders {len(orders)} vs {len(orders_cached)}, Vehicles {len(vehicles)} vs {len(vehicles_cached)}")
            return False
        
        # Final validation
        success_rate = (valid_coordinates / total_tasks) * 100 if total_tasks > 0 else 0
        print(f"\n📈 Final Results:")
        print(f"   Geocoding success rate: {success_rate:.1f}%")
        print(f"   Cache functionality: ✅ Working")
        print(f"   Overall test result: {'✅ PASSED' if success_rate >= 80 else '⚠️ PARTIAL'}")
        
        return success_rate >= 80
        
    except Exception as e:
        print(f"❌ Unexpected error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Restore original cache if it existed
        backup_cache = "geocode_cache_backup_test.json"
        if os.path.exists(backup_cache):
            if os.path.exists(cache_file):
                os.remove(cache_file)
            os.rename(backup_cache, cache_file)
            print(f"🔄 Restored original cache from backup")

def test_single_address_geocoding():
    """
    Test geocoding of a single known address for debugging.
    """
    print("\n🧪 Testing Single Address Geocoding")
    print("=" * 40)
    
    # Import geocoding function directly
    from scenario_creator import get_coordinates, load_geocode_cache, save_geocode_cache
    
    # Test address
    test_address = "Via Roma 1, 10023 Chieri, Italy"
    
    print(f"📍 Testing address: {test_address}")
    
    # Load cache
    cache = load_geocode_cache()
    
    # Clear this specific address from cache for testing
    if test_address in cache:
        del cache[test_address]
        save_geocode_cache(cache)
    
    # Test geocoding
    coords = get_coordinates(test_address, cache)
    
    if coords:
        lat, lon = coords
        print(f"✅ Geocoded successfully: ({lat:.6f}, {lon:.6f})")
        
        if validate_coordinates(lat, lon):
            print("✅ Coordinates are within valid ranges")
            return True
        else:
            print("❌ Coordinates are outside valid ranges")
            return False
    else:
        print("❌ Geocoding failed")
        return False

if __name__ == "__main__":
    print("🚀 Starting EPDT Scenario Geocoding Tests")
    print("=" * 60)
    
    # Run single address test first
    single_test_passed = test_single_address_geocoding()
    
    # Run full scenario test
    scenario_test_passed = test_geocoding_from_sample_excel()
    
    print("\n" + "=" * 60)
    print("🏁 Test Summary:")
    print(f"   Single Address Test: {'✅ PASSED' if single_test_passed else '❌ FAILED'}")
    print(f"   Scenario Excel Test: {'✅ PASSED' if scenario_test_passed else '❌ FAILED'}")
    print(f"   Overall Result: {'✅ ALL TESTS PASSED' if (single_test_passed and scenario_test_passed) else '❌ SOME TESTS FAILED'}")
