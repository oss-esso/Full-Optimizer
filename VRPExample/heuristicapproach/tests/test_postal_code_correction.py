"""
Postal Code Correction Test for Cambiano

This script tests if correcting the postal code from 10020 to 10026 
fixes the Photon geocoding issue for Cambiano.

Usage:
    python test_postal_code_correction.py
"""

import requests
import json
import time

def test_photon_geocoding(address: str, timeout: int = 10) -> dict:
    """Test Photon geocoding for a specific address."""
    print(f"🔍 Testing Photon: {address}")
    
    try:
        photon_url = "https://photon.komoot.io/api/"
        params = {
            'q': address,
            'limit': 3,
            'lang': 'en'
        }
        
        response = requests.get(photon_url, params=params, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
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
                'display_name': f"{props.get('name', '')}, {props.get('city', '')}, {props.get('state', '')}, {props.get('country', '')}"
            }
            results.append(result)
            
            print(f"   #{i}: ({result['lat']:.6f}, {result['lon']:.6f}) - {result['display_name']}")
        
        return {
            'success': True,
            'address': address,
            'results': results,
            'service': 'Photon'
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {
            'success': False,
            'address': address,
            'error': str(e),
            'service': 'Photon'
        }

def test_nominatim_geocoding(address: str, timeout: int = 10) -> dict:
    """Test Nominatim geocoding for comparison."""
    print(f"🔍 Testing Nominatim: {address}")
    
    try:
        nominatim_url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': address,
            'format': 'json',
            'limit': 3,
            'addressdetails': 1,
            'accept-language': 'en'
        }
        
        headers = {
            'User-Agent': 'EPDT-PostalCode-Test/1.0'
        }
        
        response = requests.get(nominatim_url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for i, item in enumerate(data, 1):
            result = {
                'rank': i,
                'lat': float(item['lat']),
                'lon': float(item['lon']),
                'display_name': item.get('display_name', 'Unknown'),
                'importance': item.get('importance', 'Unknown'),
                'place_rank': item.get('place_rank', 'Unknown'),
                'address': item.get('address', {})
            }
            results.append(result)
            
            print(f"   #{i}: ({result['lat']:.6f}, {result['lon']:.6f}) - {result['display_name']}")
        
        return {
            'success': True,
            'address': address,
            'results': results,
            'service': 'Nominatim'
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {
            'success': False,
            'address': address,
            'error': str(e),
            'service': 'Nominatim'
        }

def main():
    """Test postal code correction for Cambiano."""
    print("🧭 POSTAL CODE CORRECTION TEST FOR CAMBIANO")
    print("="*60)
    
    # Test addresses with different postal codes
    test_addresses = [
        "VIA NAZIONALE 11 CAMBIANO, 10020, ITALY",  # Current (wrong)
        "VIA NAZIONALE 11 CAMBIANO, 10026, ITALY",  # Corrected
        "Cambiano, 10026, Italy",                   # Simplified with correct postal code
        "Cambiano, TO, Italy",                      # With province code
    ]
    
    # Expected coordinates for Cambiano (near Turin)
    expected_cambiano = {
        'lat': 45.0,  # Approximate
        'lon': 7.7,   # Approximate
        'description': 'Near Turin, Northern Italy'
    }
    
    print(f"📍 Expected Cambiano coordinates: ~({expected_cambiano['lat']}, {expected_cambiano['lon']}) - {expected_cambiano['description']}")
    print(f"💾 Cached (wrong) coordinates: (40.639823, 15.806227) - Southern Italy")
    print()
    
    all_results = []
    
    for i, address in enumerate(test_addresses, 1):
        print(f"\n{'='*80}")
        print(f"🧪 TEST {i}: {address}")
        print(f"{'='*80}")
        
        # Test with Photon
        photon_result = test_photon_geocoding(address)
        time.sleep(1)  # Rate limiting
        
        # Test with Nominatim
        nominatim_result = test_nominatim_geocoding(address)
        time.sleep(1)  # Rate limiting
        
        all_results.append({
            'address': address,
            'photon': photon_result,
            'nominatim': nominatim_result
        })
    
    # Analysis
    print(f"\n{'='*80}")
    print(f"📊 ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\n🎯 Results Summary:")
    print("-" * 60)
    
    for i, result in enumerate(all_results, 1):
        address = result['address']
        print(f"\n{i}. {address}")
        
        # Analyze Photon results
        if result['photon']['success'] and result['photon']['results']:
            best_photon = result['photon']['results'][0]
            lat, lon = best_photon['lat'], best_photon['lon']
            
            # Check if coordinates are in Northern Italy (Cambiano region)
            is_northern_italy = 44.5 <= lat <= 45.5 and 7.0 <= lon <= 8.5
            region = "Northern Italy ✅" if is_northern_italy else "Southern Italy ❌"
            
            print(f"   📡 Photon: ({lat:.6f}, {lon:.6f}) - {region}")
            print(f"      🏙️ {best_photon['display_name']}")
        else:
            print(f"   📡 Photon: ❌ Failed")
        
        # Analyze Nominatim results
        if result['nominatim']['success'] and result['nominatim']['results']:
            best_nominatim = result['nominatim']['results'][0]
            lat, lon = best_nominatim['lat'], best_nominatim['lon']
            
            # Check if coordinates are in Northern Italy (Cambiano region)
            is_northern_italy = 44.5 <= lat <= 45.5 and 7.0 <= lon <= 8.5
            region = "Northern Italy ✅" if is_northern_italy else "Southern Italy ❌"
            
            print(f"   🌍 Nominatim: ({lat:.6f}, {lon:.6f}) - {region}")
            print(f"      🏙️ {best_nominatim['display_name']}")
        else:
            print(f"   🌍 Nominatim: ❌ Failed")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    print("-" * 60)
    
    # Check if any service consistently gets Northern Italy coordinates
    photon_success = any(
        r['photon']['success'] and r['photon']['results'] and 
        44.5 <= r['photon']['results'][0]['lat'] <= 45.5 and 
        7.0 <= r['photon']['results'][0]['lon'] <= 8.5
        for r in all_results
    )
    
    nominatim_success = any(
        r['nominatim']['success'] and r['nominatim']['results'] and 
        44.5 <= r['nominatim']['results'][0]['lat'] <= 45.5 and 
        7.0 <= r['nominatim']['results'][0]['lon'] <= 8.5
        for r in all_results
    )
    
    if photon_success:
        print("✅ Photon shows improvement with corrected postal code")
        print("   -> Keep using Photon but correct the postal code in data")
    elif nominatim_success:
        print("✅ Nominatim provides better results for Cambiano")
        print("   -> Switch scenario_creator.py to use Nominatim instead of Photon")
    else:
        print("⚠️  Both services show issues with Cambiano geocoding")
        print("   -> Consider manual coordinate correction for Cambiano")
    
    # Save results
    output_file = "postal_code_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Test results saved to: {output_file}")

if __name__ == "__main__":
    main()
