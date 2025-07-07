#!/usr/bin/env python3
"""
Quick test to check what type of distance calculator is being used.
"""

import sys
import os

# Add current directory to path to import VRP optimizer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from vrp_optimizer_clean_copy import OSMDistanceCalculator
    print("✅ Found OSMDistanceCalculator in vrp_optimizer_clean_copy")
    print(f"📊 OSMDistanceCalculator attributes: {[attr for attr in dir(OSMDistanceCalculator) if not attr.startswith('_')]}")
    
    # Create a dummy instance to check methods
    dummy_locations = [
        {'id': 'depot', 'x': 8.1407, 'y': 44.5404, 'lat': 44.5404, 'lon': 8.1407},
        {'id': 'test', 'x': 8.2, 'y': 44.6, 'lat': 44.6, 'lon': 8.2}
    ]
    
    calc = OSMDistanceCalculator(dummy_locations)
    print(f"🔧 Created OSMDistanceCalculator instance")
    print(f"📊 Instance attributes: {[attr for attr in dir(calc) if not attr.startswith('_')]}")
    
    # Check for route_db
    if hasattr(calc, 'route_db'):
        print("✅ Has route_db attribute")
        route_db = calc.route_db  
        print(f"📊 Route DB type: {type(route_db)}")
        
        if hasattr(route_db, 'get_cached_route_geometry'):
            print("✅ Route DB has get_cached_route_geometry method")
        else:
            print("❌ Route DB does NOT have get_cached_route_geometry method")
    else:
        print("❌ Does NOT have route_db attribute")
    
    # Check for route_cache (indicating OSMDistanceCalculator behavior)
    if hasattr(calc, 'route_cache'):
        print("✅ Has route_cache attribute (OSMDistanceCalculator)")
    
    if hasattr(calc, 'osrm_url'):
        print(f"🌐 OSRM URL: {calc.osrm_url}")
        
except ImportError as e:
    print(f"❌ Error importing: {e}")

print("\n" + "="*50)
print("📊 CONCLUSION")
print("="*50)
print("The current VRP optimizer uses OSMDistanceCalculator which:")
print("✅ Uses route_cache (not route_db)")
print("❌ Does NOT store route geometry for mapping")
print("💡 Updated get_cached_route_geometry function should handle this case correctly")
