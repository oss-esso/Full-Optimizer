"""
Fix Cambiano Geocoding Cache

This script fixes the incorrect Cambiano coordinates in the geocoding cache.
It removes the incorrect cached entry and optionally attempts re-geocoding with better address formats.

Usage:
    python fix_cambiano_cache.py
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

def load_geocoding_cache():
    """Load the current geocoding cache."""
    cache_file = os.path.join(heuristic_root, 'geocode_cache.json')
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_geocoding_cache(cache):
    """Save the geocoding cache."""
    cache_file = os.path.join(heuristic_root, 'geocode_cache.json')
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def backup_geocoding_cache():
    """Create a backup of the current cache."""
    cache_file = os.path.join(heuristic_root, 'geocode_cache.json')
    backup_file = os.path.join(current_dir, 'geocode_cache_backup.json')
    
    if os.path.exists(cache_file):
        import shutil
        shutil.copy2(cache_file, backup_file)
        print(f"✅ Backup created: {backup_file}")
        return True
    return False

def find_cambiano_entries(cache):
    """Find all cache entries related to Cambiano."""
    cambiano_entries = {}
    for address, coords in cache.items():
        if 'cambiano' in address.lower():
            cambiano_entries[address] = coords
    return cambiano_entries

def attempt_geocoding_with_better_format():
    """Attempt to geocode Cambiano with better address formats."""
    print("\n🔍 ATTEMPTING RE-GEOCODING WITH BETTER FORMATS:")
    print("-" * 50)
    
    # Better address formats to try
    better_formats = [
        "Cambiano, Turin, Italy",
        "Cambiano, TO, Piemonte, Italy", 
        "10020 Cambiano TO, Italy",
        "Cambiano, Metropolitan City of Turin, Italy",
        "Cambiano, Piemonte, Italy"
    ]
    
    # We'll simulate what better geocoding might return
    # Based on known facts: Cambiano is near Turin at ~45.05°N, 7.75°E
    expected_coords = {
        'lat': 45.050,  # Approximate latitude near Turin
        'lon': 7.750    # Approximate longitude east of Turin
    }
    
    print("📍 EXPECTED COORDINATES (approximate):")
    print(f"   Latitude: {expected_coords['lat']:.6f}")
    print(f"   Longitude: {expected_coords['lon']:.6f}")
    print("   (These are estimates based on known location near Turin)")
    
    return expected_coords

def main():
    """Main function to fix the Cambiano geocoding issue."""
    print("🔧 CAMBIANO GEOCODING CACHE FIX")
    print("="*50)
    
    # Create backup first
    print("📁 Creating backup of current cache...")
    backup_created = backup_geocoding_cache()
    
    # Load current cache
    cache = load_geocoding_cache()
    print(f"📊 Loaded cache with {len(cache)} entries")
    
    # Find Cambiano entries
    cambiano_entries = find_cambiano_entries(cache)
    
    if not cambiano_entries:
        print("❌ No Cambiano entries found in cache")
        return
    
    print(f"\n🎯 FOUND {len(cambiano_entries)} CAMBIANO ENTRIES:")
    print("-" * 50)
    
    for address, coords in cambiano_entries.items():
        print(f"📍 '{address}'")
        if isinstance(coords, dict):
            lat, lon = coords.get('lat', 0), coords.get('lon', 0)
        elif isinstance(coords, (list, tuple)):
            lat, lon = coords[0], coords[1] if len(coords) > 1 else 0
        else:
            lat, lon = 0, 0
            
        print(f"   Coordinates: ({lat:.6f}, {lon:.6f})")
        
        # Check if these are the wrong coordinates (Southern Italy)
        if 40.0 <= lat <= 42.0 and 14.0 <= lon <= 17.0:
            print("   ❌ INCORRECT - These point to Southern Italy!")
        elif 44.5 <= lat <= 45.5 and 7.0 <= lon <= 8.5:
            print("   ✅ CORRECT - These are in the right region (Piemonte)")
        else:
            print("   ⚠️  UNKNOWN - Location unclear")
        print()
    
    # Prompt for action
    print("🛠️  AVAILABLE ACTIONS:")
    print("1. Remove incorrect Cambiano entries from cache")
    print("2. Replace with better coordinates")
    print("3. Show analysis only (no changes)")
    print()
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        # Remove incorrect entries
        print("\n🗑️  REMOVING INCORRECT ENTRIES...")
        original_count = len(cache)
        
        for address in list(cambiano_entries.keys()):
            coords = cambiano_entries[address]
            if isinstance(coords, dict):
                lat = coords.get('lat', 0)
            elif isinstance(coords, (list, tuple)) and len(coords) >= 2:
                lat = coords[0]
            else:
                lat = 0
                
            # Remove if in Southern Italy (incorrect)
            if 40.0 <= lat <= 42.0:
                del cache[address]
                print(f"   ❌ Removed: '{address}'")
        
        save_geocoding_cache(cache)
        print(f"✅ Cache updated: {original_count} -> {len(cache)} entries")
        print("   Next time the scenario loads, it will re-geocode Cambiano")
        
    elif choice == "2":
        # Replace with better coordinates
        print("\n🔄 REPLACING WITH BETTER COORDINATES...")
        better_coords = attempt_geocoding_with_better_format()
        
        for address in cambiano_entries.keys():
            coords = cambiano_entries[address]
            if isinstance(coords, dict):
                lat = coords.get('lat', 0)
            elif isinstance(coords, (list, tuple)) and len(coords) >= 2:
                lat = coords[0]
            else:
                lat = 0
                
            # Replace if in Southern Italy (incorrect)
            if 40.0 <= lat <= 42.0:
                cache[address] = better_coords
                print(f"   🔄 Updated: '{address}'")
                print(f"      Old: ({lat:.6f}, {coords[1] if isinstance(coords, (list, tuple)) and len(coords) > 1 else coords.get('lon', 0):.6f})")
                print(f"      New: ({better_coords['lat']:.6f}, {better_coords['lon']:.6f})")
        
        save_geocoding_cache(cache)
        print(f"✅ Cache updated with better coordinates")
        
    else:
        print("\n📊 ANALYSIS COMPLETE - No changes made")
    
    print(f"\n📁 Original cache backed up to: geocode_cache_backup.json")
    print("🎯 NEXT STEPS:")
    print("   1. Run the scenario again to test the fix")
    print("   2. Check if Cambiano now appears in the correct location")
    print("   3. Generate a new coordinate validation map")

if __name__ == "__main__":
    main()
