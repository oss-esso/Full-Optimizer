"""
Cambiano Geocoding Test

This script tests geocoding specifically for the Cambiano address to debug why it's showing up in Africa.
We'll test multiple variations of the address to see if it's a formatting issue.

Usage:
    python test_cambiano_geocoding.py
"""

import os
import sys
import json
import requests
import time
from typing import Dict, Optional

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
src_dir = os.path.join(heuristic_root, 'src')
utils_dir = os.path.join(heuristic_root, 'utils')
sys.path.insert(0, heuristic_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, utils_dir)

def test_nominatim_geocoding(address: str) -> Optional[Dict]:
    """Test geocoding using Nominatim (OpenStreetMap) API."""
    print(f"🔍 Testing Nominatim geocoding for: '{address}'")
    
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': address,
        'format': 'json',
        'limit': 5,
        'countrycodes': 'it',  # Restrict to Italy
        'addressdetails': 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()
        
        if results:
            print(f"✅ Found {len(results)} results:")
            for i, result in enumerate(results):
                lat = float(result['lat'])
                lon = float(result['lon'])
                display_name = result.get('display_name', 'Unknown')
                print(f"  {i+1}. {display_name}")
                print(f"     📍 Lat: {lat:.6f}, Lon: {lon:.6f}")
                print(f"     🏷️  Type: {result.get('type', 'Unknown')}")
                print(f"     📊 Importance: {result.get('importance', 'Unknown')}")
                print()
            return results[0]  # Return best result
        else:
            print("❌ No results found")
            return None
            
    except Exception as e:
        print(f"❌ Error with Nominatim: {e}")
        return None

def test_internal_geocoding(address: str) -> Optional[Dict]:
    """Test the internal geocoding cache/system."""
    print(f"🔍 Testing internal geocoding for: '{address}'")
    
    try:
        # Check if there's a geocoding cache file
        cache_file = os.path.join(heuristic_root, 'geocode_cache.json')
        if os.path.exists(cache_file):
            print(f"📁 Found geocoding cache: {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            
            # Look for the address in cache
            for cached_addr, coords in cache.items():
                if 'cambiano' in cached_addr.lower():
                    print(f"✅ Found in cache: '{cached_addr}'")
                    print(f"   📍 Cached coordinates: {coords}")
                    
                    if isinstance(coords, dict) and 'lat' in coords and 'lon' in coords:
                        lat, lon = coords['lat'], coords['lon']
                    elif isinstance(coords, (list, tuple)) and len(coords) >= 2:
                        lat, lon = coords[0], coords[1]
                    else:
                        print(f"   ⚠️  Unexpected coordinate format: {coords}")
                        continue
                        
                    print(f"   🌍 Lat: {lat:.6f}, Lon: {lon:.6f}")
                    
                    # Check if coordinates are reasonable for Italy
                    if 35.0 <= lat <= 47.0 and 6.0 <= lon <= 19.0:
                        print("   ✅ Coordinates are within Italy bounds")
                    else:
                        print("   ❌ Coordinates are OUTSIDE Italy bounds!")
                        
                        # Check if it's in Africa
                        if -35.0 <= lat <= 35.0 and -20.0 <= lon <= 55.0:
                            print("   🌍 These coordinates appear to be in Africa!")
                    
                    return {'lat': lat, 'lon': lon, 'source': 'cache', 'address': cached_addr}
            
            print("❌ Address not found in geocoding cache")
        else:
            print("❌ No geocoding cache file found")
            
    except Exception as e:
        print(f"❌ Error checking internal geocoding: {e}")
        
    return None

def test_scenario_geocoding():
    """Test by loading the actual scenario and extracting Cambiano coordinates."""
    print(f"🔍 Testing scenario geocoding extraction...")
    
    try:
        from scenario_creator import create_scenario_from_excel
        
        excel_path = os.path.join(src_dir, 'furgoni.xlsx')
        orders, vehicles, drivers = create_scenario_from_excel(excel_path)
        
        print(f"📊 Loaded {len(orders)} orders from scenario")
        
        # Look for Cambiano in the orders
        cambiano_coords = []
        for order in orders:
            order_id = getattr(order, 'id', getattr(order, 'order_id', 'unknown'))
            
            # Check pickup tasks
            if hasattr(order, 'pickup_tasks') and order.pickup_tasks:
                for task in order.pickup_tasks:
                    if hasattr(task, 'location_id') and 'cambiano' in str(task.location_id).lower():
                        print(f"✅ Found Cambiano PICKUP in order {order_id}")
                        print(f"   📍 Address: {task.location_id}")
                        print(f"   🌍 Coordinates: ({task.lat:.6f}, {task.lon:.6f})")
                        cambiano_coords.append({
                            'type': 'pickup',
                            'order_id': order_id,
                            'address': str(task.location_id),
                            'lat': task.lat,
                            'lon': task.lon
                        })
            
            # Check delivery tasks
            if hasattr(order, 'delivery_tasks') and order.delivery_tasks:
                for task in order.delivery_tasks:
                    if hasattr(task, 'location_id') and 'cambiano' in str(task.location_id).lower():
                        print(f"✅ Found Cambiano DELIVERY in order {order_id}")
                        print(f"   📍 Address: {task.location_id}")
                        print(f"   🌍 Coordinates: ({task.lat:.6f}, {task.lon:.6f})")
                        cambiano_coords.append({
                            'type': 'delivery',
                            'order_id': order_id,
                            'address': str(task.location_id),
                            'lat': task.lat,
                            'lon': task.lon
                        })
        
        if cambiano_coords:
            print(f"\n📊 CAMBIANO ANALYSIS ({len(cambiano_coords)} instances found):")
            for coord in cambiano_coords:
                print(f"  🎯 {coord['type'].title()} - Order {coord['order_id']}")
                print(f"     📍 Address: {coord['address']}")
                print(f"     🌍 Coordinates: ({coord['lat']:.6f}, {coord['lon']:.6f})")
                
                # Check location
                if 35.0 <= coord['lat'] <= 47.0 and 6.0 <= coord['lon'] <= 19.0:
                    print("     ✅ Within Italy bounds")
                else:
                    print("     ❌ OUTSIDE Italy bounds!")
                    
                    # Try to identify the region
                    if -35.0 <= coord['lat'] <= 35.0 and -20.0 <= coord['lon'] <= 55.0:
                        print("     🌍 These coordinates appear to be in AFRICA!")
                    elif coord['lat'] == 0.0 and coord['lon'] == 0.0:
                        print("     🌍 These are NULL ISLAND coordinates (0,0) - geocoding failed!")
                print()
        else:
            print("❌ No Cambiano coordinates found in scenario")
            
    except Exception as e:
        print(f"❌ Error loading scenario: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function."""
    print("🔍 CAMBIANO GEOCODING DEBUG TEST")
    print("="*50)
    
    # Test different variations of the Cambiano address
    cambiano_variations = [
        "CAMBIANO",
        "Cambiano",
        "Cambiano, Italy",
        "Cambiano, Torino",
        "Cambiano, TO",
        "Cambiano, Piemonte",
        "VIA NAZIONALE 11 CAMBIANO, 10020, ITALY",
        "VIA NAZIONALE 11, CAMBIANO, 10020, ITALY",
        "Cambiano 10020",
        "10020 Cambiano TO"
    ]
    
    print("\n🌐 TESTING EXTERNAL GEOCODING (Nominatim):")
    print("-" * 50)
    
    for variation in cambiano_variations:
        test_nominatim_geocoding(variation)
        time.sleep(1)  # Be nice to the API
        print()
    
    print("\n📁 TESTING INTERNAL GEOCODING CACHE:")
    print("-" * 50)
    test_internal_geocoding("cambiano")
    
    print("\n📊 TESTING SCENARIO EXTRACTION:")
    print("-" * 50)
    test_scenario_geocoding()
    
    print("\n🎯 ANALYSIS SUMMARY:")
    print("="*50)
    print("1. Check if external geocoding gives correct Italy coordinates")
    print("2. Check if internal cache has incorrect coordinates")
    print("3. Identify the exact address format causing the issue")
    print("4. Determine if it's a caching issue or geocoding API issue")

if __name__ == "__main__":
    main()
