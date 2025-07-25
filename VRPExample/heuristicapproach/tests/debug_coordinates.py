#!/usr/bin/env python3
"""
Debug script to check actual coordinates and travel time calculations.
"""

import sys
import os
import math

# Add paths
current_dir = os.path.dirname(__file__)
algo_dir = os.path.join(current_dir, '..', 'algo')
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, algo_dir)
sys.path.insert(0, src_dir)

print("🔍 DEBUG: Checking actual coordinates and travel times")
print("="*60)

try:
    from moda_scenarios import create_furgoni_scenario
    from data_adapter import convert_instance_to_epdt_input
    
    print("1️⃣ Loading scenario and checking coordinates...")
    instance = create_furgoni_scenario()
    orders, vehicles = convert_instance_to_epdt_input(instance)
    
    # Find specific locations mentioned
    locations_to_check = ['chiva_spagna', 'villar_cuneo', 'schonaich_de', 'asti']
    depot_coords = None
    
    print("\n📍 Key location coordinates:")
    
    # Check orders for these locations
    for order in orders:
        pickup_task = order.pickup_task
        delivery_task = order.delivery_task
        
        # Check pickup task
        if pickup_task and pickup_task.location_id in locations_to_check:
            print(f"   📦 {pickup_task.location_id}: lat={pickup_task.lat:.6f}, lon={pickup_task.lon:.6f}")
            
        # Check delivery task  
        if delivery_task and delivery_task.location_id in locations_to_check:
            print(f"   🏪 {delivery_task.location_id}: lat={delivery_task.lat:.6f}, lon={delivery_task.lon:.6f}")
            
        # Find depot coordinates (usually where pickup tasks start)
        if pickup_task and 'depot' in pickup_task.location_id and depot_coords is None:
            depot_coords = (pickup_task.lat, pickup_task.lon)
            print(f"   🏭 DEPOT: lat={pickup_task.lat:.6f}, lon={pickup_task.lon:.6f}")
    
    print("\n2️⃣ Testing Haversine calculations with realistic expectations:")
    
    # Test known distances from Asti (depot should be near Asti)
    if depot_coords:
        depot_lat, depot_lon = depot_coords
        
        # Manual coordinate checks for major cities
        test_locations = {
            'chiva_spagna': (39.4699, -0.7584),  # Valencia area, Spain - ~8h drive from Asti
            'villar_cuneo': (44.4378, 7.5702),   # Cuneo area, Italy - ~1h drive from Asti  
            'schonaich_de': (48.7078, 9.0536),   # Baden-Württemberg, Germany - ~6h drive from Asti
            'asti_reference': (44.9009, 8.2065)  # Asti, Italy - reference point
        }
        
        from second_level import haversine_distance, calculate_travel_time_haversine
        
        print(f"\n   From depot ({depot_lat:.4f}, {depot_lon:.4f}):")
        
        for location, (lat, lon) in test_locations.items():
            distance_km = haversine_distance(depot_lat, depot_lon, lat, lon)
            time_80kmh = calculate_travel_time_haversine(depot_lat, depot_lon, lat, lon, 80.0)
            time_60kmh = calculate_travel_time_haversine(depot_lat, depot_lon, lat, lon, 60.0)
            
            print(f"   📏 To {location}:")
            print(f"      Distance: {distance_km:.0f} km")
            print(f"      Time @ 80km/h: {time_80kmh/60:.1f} hours")
            print(f"      Time @ 60km/h: {time_60kmh/60:.1f} hours")
        
        print("\n3️⃣ Checking actual coordinates from scenario for discrepancies:")
        
        # Find the actual coordinates from the scenario
        actual_coords = {}
        for order in orders:
            if order.delivery_task:
                loc_id = order.delivery_task.location_id
                if any(target in loc_id for target in ['chiva', 'villar', 'schonaich']):
                    actual_coords[loc_id] = (order.delivery_task.lat, order.delivery_task.lon)
                    distance_km = haversine_distance(depot_lat, depot_lon, order.delivery_task.lat, order.delivery_task.lon)
                    time_80kmh = calculate_travel_time_haversine(depot_lat, depot_lon, order.delivery_task.lat, order.delivery_task.lon, 80.0)
                    
                    print(f"   🎯 ACTUAL {loc_id}:")
                    print(f"      Coords: ({order.delivery_task.lat:.6f}, {order.delivery_task.lon:.6f})")
                    print(f"      Distance: {distance_km:.0f} km")
                    print(f"      Time @ 80km/h: {time_80kmh/60:.1f} hours")
                    
                    # Check if coordinates look suspicious
                    if distance_km < 50:
                        print(f"      ⚠️ WARNING: Very short distance - coordinates might be wrong!")
                    elif distance_km > 2000:
                        print(f"      ⚠️ WARNING: Very long distance - coordinates might be wrong!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("🏁 Debug complete!")
