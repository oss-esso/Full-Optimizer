"""
Photon Geocoding Test for Cambiano

This script tests Photon geocoding service specifically for the Cambiano address
to understand if the coordinate issue is due to geocoding service differences.

Usage:
    python test_photon_geocoding.py
"""

import os
import sys
import json
import requests
import time
from typing import Dict, List, Tuple, Optional

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
src_dir = os.path.join(heuristic_root, 'src')
utils_dir = os.path.join(heuristic_root, 'utils')
sys.path.insert(0, heuristic_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, utils_dir)

def test_photon_geocoding(address: str, timeout: int = 10) -> Dict:
    """
    Test Photon geocoding service for a specific address.
    
    Args:
        address: Address to geocode
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with geocoding results
    """
    print(f"🔍 Testing Photon geocoding for: {address}")
    
    # Photon API endpoint
    photon_url = "https://photon.komoot.io/api/"
    
    try:
        # Make request to Photon
        params = {
            'q': address,
            'limit': 5,  # Get top 5 results
            'lang': 'en'
        }
        
        print(f"📡 Making request to Photon: {photon_url}")
        print(f"📋 Parameters: {params}")
        
        response = requests.get(photon_url, params=params, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        
        print(f"✅ Photon response received")
        print(f"📊 Number of results: {len(data.get('features', []))}")
        
        results = []
        for i, feature in enumerate(data.get('features', []), 1):
            coords = feature['geometry']['coordinates']  # [lon, lat]
            props = feature.get('properties', {})
            
            result = {
                'rank': i,
                'lat': coords[1],
                'lon': coords[0],
                'name': props.get('name', 'Unknown'),
                'city': props.get('city', 'Unknown'),
                'state': props.get('state', 'Unknown'),
                'country': props.get('country', 'Unknown'),
                'postcode': props.get('postcode', 'Unknown'),
                'osm_type': props.get('osm_type', 'Unknown'),
                'osm_id': props.get('osm_id', 'Unknown'),
                'confidence': props.get('confidence', 'Unknown')
            }
            results.append(result)
            
            print(f"\n📍 Result #{i}:")
            print(f"   🌍 Coordinates: ({result['lat']:.6f}, {result['lon']:.6f})")
            print(f"   📍 Name: {result['name']}")
            print(f"   🏙️  City: {result['city']}")
            print(f"   🏛️  State: {result['state']}")
            print(f"   🏳️  Country: {result['country']}")
            print(f"   📮 Postcode: {result['postcode']}")
            print(f"   🗺️  OSM Type: {result['osm_type']}")
        
        return {
            'success': True,
            'service': 'Photon',
            'address': address,
            'results': results,
            'total_results': len(results)
        }
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error with Photon request: {e}")
        return {
            'success': False,
            'service': 'Photon',
            'address': address,
            'error': str(e)
        }
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return {
            'success': False,
            'service': 'Photon',
            'address': address,
            'error': str(e)
        }

def test_nominatim_geocoding(address: str, timeout: int = 10) -> Dict:
    """
    Test Nominatim geocoding service for comparison.
    
    Args:
        address: Address to geocode
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with geocoding results
    """
    print(f"\n🔍 Testing Nominatim geocoding for: {address}")
    
    # Nominatim API endpoint
    nominatim_url = "https://nominatim.openstreetmap.org/search"
    
    try:
        # Make request to Nominatim
        params = {
            'q': address,
            'format': 'json',
            'limit': 5,
            'addressdetails': 1,
            'accept-language': 'en'
        }
        
        headers = {
            'User-Agent': 'EPDT-Coordinate-Test/1.0'
        }
        
        print(f"📡 Making request to Nominatim: {nominatim_url}")
        print(f"📋 Parameters: {params}")
        
        response = requests.get(nominatim_url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        
        print(f"✅ Nominatim response received")
        print(f"📊 Number of results: {len(data)}")
        
        results = []
        for i, item in enumerate(data, 1):
            result = {
                'rank': i,
                'lat': float(item['lat']),
                'lon': float(item['lon']),
                'display_name': item.get('display_name', 'Unknown'),
                'importance': item.get('importance', 'Unknown'),
                'place_rank': item.get('place_rank', 'Unknown'),
                'osm_type': item.get('osm_type', 'Unknown'),
                'osm_id': item.get('osm_id', 'Unknown')
            }
            results.append(result)
            
            print(f"\n📍 Result #{i}:")
            print(f"   🌍 Coordinates: ({result['lat']:.6f}, {result['lon']:.6f})")
            print(f"   📍 Display Name: {result['display_name']}")
            print(f"   ⭐ Importance: {result['importance']}")
            print(f"   🎯 Place Rank: {result['place_rank']}")
        
        return {
            'success': True,
            'service': 'Nominatim',
            'address': address,
            'results': results,
            'total_results': len(results)
        }
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error with Nominatim request: {e}")
        return {
            'success': False,
            'service': 'Nominatim',
            'address': address,
            'error': str(e)
        }
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return {
            'success': False,
            'service': 'Nominatim',
            'address': address,
            'error': str(e)
        }

def load_cached_coordinates():
    """Load cached coordinates from geocode_cache.json"""
    try:
        cache_path = os.path.join(heuristic_root, 'geocode_cache.json')
        print(f"📁 Loading cached coordinates from: {cache_path}")
        
        if not os.path.exists(cache_path):
            print(f"❌ Cache file not found: {cache_path}")
            return {}
        
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        
        print(f"✅ Loaded {len(cache)} cached entries")
        return cache
        
    except Exception as e:
        print(f"❌ Error loading cache: {e}")
        return {}

def compare_geocoding_results(address: str):
    """Compare geocoding results from different services and cache."""
    print("🔄 COMPREHENSIVE GEOCODING COMPARISON")
    print("="*60)
    
    # Load cached coordinates
    cache = load_cached_coordinates()
    cached_result = cache.get(address)
    
    if cached_result:
        print(f"\n💾 CACHED RESULT:")
        print(f"   🌍 Coordinates: ({cached_result['lat']:.6f}, {cached_result['lon']:.6f})")
        print(f"   📅 Source: Cache")
    else:
        print(f"\n❌ No cached result found for: {address}")
    
    # Test Photon
    photon_result = test_photon_geocoding(address)
    
    # Add delay to be respectful to servers
    time.sleep(1)
    
    # Test Nominatim
    nominatim_result = test_nominatim_geocoding(address)
    
    # Compare results
    print(f"\n📊 COMPARISON SUMMARY")
    print("="*60)
    
    all_results = []
    
    if cached_result:
        all_results.append({
            'service': 'Cache',
            'lat': cached_result['lat'],
            'lon': cached_result['lon'],
            'description': 'Cached coordinates'
        })
    
    if photon_result['success'] and photon_result['results']:
        best_photon = photon_result['results'][0]
        all_results.append({
            'service': 'Photon',
            'lat': best_photon['lat'],
            'lon': best_photon['lon'],
            'description': f"{best_photon['name']}, {best_photon['city']}, {best_photon['country']}"
        })
    
    if nominatim_result['success'] and nominatim_result['results']:
        best_nominatim = nominatim_result['results'][0]
        all_results.append({
            'service': 'Nominatim',
            'lat': best_nominatim['lat'],
            'lon': best_nominatim['lon'],
            'description': best_nominatim['display_name']
        })
    
    # Print comparison
    for i, result in enumerate(all_results, 1):
        print(f"{i}. {result['service']:12s} | ({result['lat']:9.6f}, {result['lon']:9.6f}) | {result['description']}")
    
    # Calculate distances between results
    if len(all_results) > 1:
        print(f"\n📏 DISTANCE ANALYSIS:")
        print("-" * 60)
        
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
        
        for i in range(len(all_results)):
            for j in range(i+1, len(all_results)):
                r1, r2 = all_results[i], all_results[j]
                distance = haversine_distance(r1['lat'], r1['lon'], r2['lat'], r2['lon'])
                print(f"   {r1['service']} <-> {r2['service']}: {distance:.2f} km apart")
    
    return {
        'address': address,
        'cached': cached_result,
        'photon': photon_result,
        'nominatim': nominatim_result,
        'comparison': all_results
    }

def main():
    """Main function to test Cambiano geocoding."""
    print("🧭 PHOTON GEOCODING TEST FOR CAMBIANO")
    print("="*50)
    
    # Test addresses
    test_addresses = [
        "VIA NAZIONALE 11 CAMBIANO, 10020, ITALY",
        "Cambiano, Italy",
        "Cambiano, TO, Italy",
        "Via Nazionale 11, Cambiano, Torino, Italy"
    ]
    
    results = []
    
    for address in test_addresses:
        print(f"\n{'='*80}")
        print(f"🔍 TESTING ADDRESS: {address}")
        print(f"{'='*80}")
        
        result = compare_geocoding_results(address)
        results.append(result)
        
        # Add delay between requests
        time.sleep(2)
    
    # Save all results
    output_file = os.path.join(current_dir, 'cambiano_geocoding_test_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Test results saved to: {output_file}")
    
    print(f"\n🎯 CONCLUSIONS:")
    print("1. Compare coordinates from different services")
    print("2. Check if Photon gives different results than Nominatim")
    print("3. Verify which service matches the expected Cambiano location (near Turin)")
    print("4. Expected coordinates for Cambiano should be around: (45.0°N, 7.7°E)")

if __name__ == "__main__":
    main()
