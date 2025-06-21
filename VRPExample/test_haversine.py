#!/usr/bin/env python3
"""
Test script to verify Haversine distance calculations
"""

import math
from vrp_scenarios import get_all_scenarios

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance between two points in km"""
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    distance_km = 6371 * 2 * math.asin(math.sqrt(a))  # Earth radius in km
    return distance_km

def simplified_distance(lat1, lon1, lat2, lon2):
    """Old simplified distance calculation"""
    dx = abs(lon1 - lon2)  # Longitude difference in degrees
    dy = abs(lat1 - lat2)  # Latitude difference in degrees
    
    # Convert degrees to kilometers using approximate conversion factors
    avg_lat = (lat1 + lat2) / 2
    lat_km = dy * 111.0
    lon_km = dx * 111.0 * math.cos(math.radians(avg_lat))
    
    # Calculate Euclidean distance in km
    distance_km = math.sqrt(lat_km**2 + lon_km**2)
    return distance_km

def main():
    print("🧪 TESTING HAVERSINE VS SIMPLIFIED DISTANCE CALCULATIONS")
    print("=" * 80)
    
    # Load a test scenario
    scenarios = get_all_scenarios()
    instance = scenarios['MODA_small']
    locations = list(instance.locations.values())
    
    print(f"📍 Testing with {len(locations)} locations from MODA_small scenario")
    print()
    
    # Test distance between first few locations
    test_pairs = [
        (0, 1),   # depot to pickup
        (2, 3),   # pickup to dropoff
        (4, 5),   # another pickup to dropoff
        (0, 10),  # depot to far location
        (10, 20)  # two far locations
    ]
    
    print("📏 DISTANCE COMPARISON:")
    print("-" * 80)
    print(f"{'From':<12} {'To':<12} {'Lat1':<8} {'Lon1':<8} {'Lat2':<8} {'Lon2':<8} {'Haversine':<10} {'Simplified':<10} {'Diff %':<8}")
    print("-" * 80)
    
    total_haversine = 0
    total_simplified = 0
    
    for i, j in test_pairs:
        if i < len(locations) and j < len(locations):
            loc1 = locations[i]
            loc2 = locations[j]
            
            # Get coordinates (y=latitude, x=longitude)
            lat1, lon1 = loc1.y, loc1.x
            lat2, lon2 = loc2.y, loc2.x
            
            # Calculate distances
            haversine_km = haversine_distance(lat1, lon1, lat2, lon2)
            simplified_km = simplified_distance(lat1, lon1, lat2, lon2)
            
            total_haversine += haversine_km
            total_simplified += simplified_km
            
            # Calculate percentage difference
            if simplified_km > 0:
                diff_pct = ((haversine_km - simplified_km) / simplified_km) * 100
            else:
                diff_pct = 0
            
            print(f"{i:<12} {j:<12} {lat1:<8.3f} {lon1:<8.3f} {lat2:<8.3f} {lon2:<8.3f} {haversine_km:<10.3f} {simplified_km:<10.3f} {diff_pct:<8.1f}")
    
    print("-" * 80)
    print(f"{'TOTALS':<12} {'':12} {'':8} {'':8} {'':8} {'':8} {total_haversine:<10.3f} {total_simplified:<10.3f}")
    
    if total_simplified > 0:
        total_diff_pct = ((total_haversine - total_simplified) / total_simplified) * 100
        print(f"Total difference: {total_diff_pct:.1f}%")
    
    print()
    print("🔍 ANALYSIS:")
    print(f"   • Haversine formula is more accurate for longer distances")
    print(f"   • For small distances (< 1 km), both methods are similar")
    print(f"   • For the MODA_small scenario, coordinates are in Northern Italy")
    print(f"   • If distances are very small, check coordinate spacing")
    
    # Check coordinate range
    lats = [loc.y for loc in locations]
    lons = [loc.x for loc in locations]
    
    print()
    print("📐 COORDINATE ANALYSIS:")
    print(f"   • Latitude range: {min(lats):.4f} to {max(lats):.4f} (span: {max(lats)-min(lats):.4f}°)")
    print(f"   • Longitude range: {min(lons):.4f} to {max(lons):.4f} (span: {max(lons)-min(lons):.4f}°)")
    print(f"   • Expected distance span: ~{(max(lats)-min(lats)) * 111:.1f} km lat x {(max(lons)-min(lons)) * 111 * math.cos(math.radians(sum(lats)/len(lats))):.1f} km lon")

if __name__ == "__main__":
    main()
