"""
Test the corrected scenario_creator with Nominatim

This script tests the scenario_creator after switching from Photon to Nominatim
to verify that Cambiano now gets correct coordinates.

Usage:
    python test_corrected_scenario_creator.py
"""

import os
import sys
import json

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
utils_dir = os.path.join(heuristic_root, 'utils')
sys.path.insert(0, heuristic_root)
sys.path.insert(0, utils_dir)

try:
    from scenario_creator import get_coordinates, load_geocode_cache, save_geocode_cache
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Available in utils:", os.listdir(utils_dir) if os.path.exists(utils_dir) else "Utils dir not found")
    sys.exit(1)

def test_nominatim_geocoding():
    """Test the corrected scenario_creator with Nominatim."""
    print("🧪 TESTING CORRECTED SCENARIO_CREATOR WITH NOMINATIM")
    print("="*60)
    
    # Load existing cache
    cache = load_geocode_cache()
    print(f"📁 Loaded cache with {len(cache)} entries")
    
    # Test addresses that were problematic
    test_addresses = [
        "VIA NAZIONALE 11 CAMBIANO, 10020, ITALY",
        "VIA NAZIONALE 11 CAMBIANO, 10026, ITALY",  # Corrected postal code
    ]
    
    results = []
    
    for address in test_addresses:
        print(f"\n🔍 Testing: {address}")
        
        # Remove from cache to force fresh geocoding
        if address in cache:
            print(f"   📋 Removing cached result: {cache[address]}")
            del cache[address]
        
        # Get fresh coordinates using Nominatim
        coords = get_coordinates(address, cache)
        
        if coords:
            lat, lon = coords
            # Check if coordinates are in Northern Italy (Cambiano region)
            is_northern_italy = 44.5 <= lat <= 45.5 and 7.0 <= lon <= 8.5
            region = "Northern Italy ✅" if is_northern_italy else "Southern Italy ❌"
            
            print(f"   ✅ Result: ({lat:.6f}, {lon:.6f}) - {region}")
            
            results.append({
                'address': address,
                'lat': lat,
                'lon': lon,
                'region': region,
                'success': True
            })
        else:
            print(f"   ❌ Failed to geocode")
            results.append({
                'address': address,
                'success': False
            })
    
    # Save updated cache
    save_geocode_cache(cache)
    print(f"\n💾 Updated cache saved with {len(cache)} entries")
    
    # Results summary
    print(f"\n📊 RESULTS SUMMARY:")
    print("-" * 60)
    
    for result in results:
        if result['success']:
            print(f"✅ {result['address']}")
            print(f"   🌍 ({result['lat']:.6f}, {result['lon']:.6f}) - {result['region']}")
        else:
            print(f"❌ {result['address']} - Failed")
    
    # Compare with old cached result
    print(f"\n🔄 COMPARISON:")
    print("-" * 60)
    print(f"❌ Old (Photon): (40.639823, 15.806227) - Southern Italy")
    
    successful_results = [r for r in results if r['success']]
    if successful_results:
        new_result = successful_results[0]  # Take first successful result
        print(f"✅ New (Nominatim): ({new_result['lat']:.6f}, {new_result['lon']:.6f}) - {new_result['region']}")
        
        # Calculate distance between old and new coordinates
        from math import radians, cos, sin, asin, sqrt
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate haversine distance between two points in km."""
            R = 6371  # Earth's radius in km
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            return R * c
        
        distance = haversine_distance(40.639823, 15.806227, new_result['lat'], new_result['lon'])
        print(f"📏 Distance correction: {distance:.1f} km")
    
    return results

if __name__ == "__main__":
    test_nominatim_geocoding()
