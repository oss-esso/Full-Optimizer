"""
Simple Geocoding Service Test

Quick test to check if geocoding services are accessible and compare results.
"""

import requests
import json
import time

def test_simple_geocoding():
    """Simple test of geocoding services."""
    
    address = "Cambiano, Italy"
    print(f"🔍 Testing geocoding for: {address}")
    
    # Test Photon
    print("\n📡 Testing Photon...")
    try:
        photon_url = "https://photon.komoot.io/api/"
        params = {'q': address, 'limit': 1}
        response = requests.get(photon_url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('features'):
                coords = data['features'][0]['geometry']['coordinates']
                print(f"✅ Photon: ({coords[1]:.6f}, {coords[0]:.6f})")
            else:
                print("❌ Photon: No results")
        else:
            print(f"❌ Photon: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Photon error: {e}")
    
    time.sleep(1)
    
    # Test Nominatim
    print("\n📡 Testing Nominatim...")
    try:
        nominatim_url = "https://nominatim.openstreetmap.org/search"
        params = {'q': address, 'format': 'json', 'limit': 1}
        headers = {'User-Agent': 'Test/1.0'}
        response = requests.get(nominatim_url, params=params, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                print(f"✅ Nominatim: ({float(data[0]['lat']):.6f}, {float(data[0]['lon']):.6f})")
            else:
                print("❌ Nominatim: No results")
        else:
            print(f"❌ Nominatim: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Nominatim error: {e}")
    
    # Show cached result
    print(f"\n💾 Cached result: (40.639823, 15.806227)")
    print(f"📍 Expected (Cambiano near Turin): (~45.0, ~7.7)")
    
    print(f"\n🎯 Analysis:")
    print("- The cached coordinates (40.6, 15.8) point to southern Italy")
    print("- Cambiano should be near Turin in northern Italy")
    print("- This suggests the geocoding service returned wrong results")

if __name__ == "__main__":
    test_simple_geocoding()
