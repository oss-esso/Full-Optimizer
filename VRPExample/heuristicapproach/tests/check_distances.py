#!/usr/bin/env python3
"""
Simple distance calculation check
"""

import math

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Haversine distance between two points."""
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (math.sin(dlat/2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    # Earth's radius in kilometers
    R = 6371.0
    distance_km = R * c
    
    return distance_km

print("🔍 Checking coordinate-based distances from Asti")
print("="*50)

# Coordinates from moda_scenarios.py
asti_coords = (44.9009, 8.2057)  # Depot (Asti)
locations = {
    'chiva_spagna': (39.4667, -0.7167),   # From line 153
    'villar_cuneo': (44.3833, 7.5333),    # From line 171  
    'schonaich_de': (48.6667, 9.0000),    # From line 123
}

print(f"📍 Depot (Asti): {asti_coords}")
print()

for name, coords in locations.items():
    distance_km = haversine_distance(asti_coords[0], asti_coords[1], coords[0], coords[1])
    
    # Calculate travel times at different speeds
    time_60kmh = distance_km / 60.0  # hours
    time_80kmh = distance_km / 80.0  # hours
    time_100kmh = distance_km / 100.0  # hours
    
    print(f"📏 {name}: {coords}")
    print(f"   Distance: {distance_km:.0f} km")
    print(f"   @ 60 km/h: {time_60kmh:.1f} hours")
    print(f"   @ 80 km/h: {time_80kmh:.1f} hours") 
    print(f"   @ 100 km/h: {time_100kmh:.1f} hours")
    print()

print("Expected vs Reality check:")
print("- Chiva (Spain): Should be ~8h -> Calculated:", f"{locations['chiva_spagna']} = {haversine_distance(asti_coords[0], asti_coords[1], locations['chiva_spagna'][0], locations['chiva_spagna'][1])/80:.1f}h @ 80km/h")
print("- Villar (Cuneo): Should be ~1h -> Calculated:", f"{locations['villar_cuneo']} = {haversine_distance(asti_coords[0], asti_coords[1], locations['villar_cuneo'][0], locations['villar_cuneo'][1])/80:.1f}h @ 80km/h")  
print("- Schonaich (DE): Should be ~6h -> Calculated:", f"{locations['schonaich_de']} = {haversine_distance(asti_coords[0], asti_coords[1], locations['schonaich_de'][0], locations['schonaich_de'][1])/80:.1f}h @ 80km/h")
