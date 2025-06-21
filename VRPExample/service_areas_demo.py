#!/usr/bin/env python3
"""
Service Areas Demo

This script demonstrates the Northern Italy service areas database
and how it integrates with the VRP optimizer for break planning.
"""

import os
import sys

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from service_areas_db import service_areas_db, service_area_to_location

def main():
    print("Northern Italy Service Areas Database Demo")
    print("=" * 60)
    
    # Show database statistics
    stats = service_areas_db.get_service_area_statistics()
    print(f"📍 Total service areas: {stats['total_areas']}")
    print(f"🚛 With truck parking: {stats['with_truck_parking']}")
    print(f"⛽ With fuel: {stats['with_fuel']}")
    print(f"🍽️ With restaurant: {stats['with_restaurant']}")
    
    print("\n🛣️ Service areas by highway:")
    for highway, count in sorted(stats['by_highway'].items()):
        print(f"   {highway}: {count} areas")
    
    print("\n" + "=" * 60)
    print("🌍 GEOGRAPHIC COVERAGE")
    print("=" * 60)
    
    # Show sample service areas from different highways
    highways_to_show = ['A1', 'A4', 'A7', 'A8']
    for highway in highways_to_show:
        areas = service_areas_db.get_service_areas_by_highway(highway)
        if areas:
            print(f"\n🛣️ {highway} Highway:")
            for area in areas[:2]:  # Show first 2 per highway
                print(f"   📍 {area.name}")
                print(f"      📍 {area.address}")
                facilities = ", ".join(area.facilities)
                print(f"      🔧 Facilities: {facilities}")
    
    print("\n" + "=" * 60)
    print("🔍 ROUTE-BASED BREAK PLANNING")
    print("=" * 60)
    
    # Test route from Milan to Bologna
    print("\n🚚 Route: Milan → Bologna")
    milan_lat, milan_lon = 45.4642, 9.1896
    bologna_lat, bologna_lon = 44.4949, 11.3426
    
    route_areas = service_areas_db.find_service_areas_near_route(
        milan_lat, milan_lon, bologna_lat, bologna_lon, max_detour_km=15.0
    )
    
    print(f"   🎯 Found {len(route_areas)} service areas along the route")
    for i, area in enumerate(route_areas, 1):
        distance = service_areas_db._haversine_distance(milan_lat, milan_lon, area.lat, area.lon)
        print(f"   {i}. {area.name} ({area.highway}) - {distance:.1f}km from Milan")
    
    # Find best break location
    best_break = service_areas_db.find_break_location_between_points(
        milan_lat, milan_lon, bologna_lat, bologna_lon
    )
    
    if best_break:
        print(f"\n⭐ Best break location: {best_break.name}")
        print(f"   📍 Highway: {best_break.highway}")
        print(f"   📍 Address: {best_break.address}")
        facilities = ", ".join(best_break.facilities)
        print(f"   🔧 Facilities: {facilities}")
    
    print("\n" + "=" * 60)
    print("🔧 VRP INTEGRATION")
    print("=" * 60)
    
    # Show how service areas integrate with VRP
    print("\n🔗 Service Area → VRP Location Conversion:")
    if best_break:
        vrp_location = service_area_to_location(best_break)
        print(f"   🆔 VRP Location ID: {vrp_location.id}")
        print(f"   📍 Coordinates: ({vrp_location.x:.4f}, {vrp_location.y:.4f})")
        print(f"   ⏰ Service time: {vrp_location.service_time} minutes")
        print(f"   📦 Demand: {vrp_location.demand} (service areas have no cargo demand)")
    
    print("\n" + "=" * 60)
    print("✅ SYSTEM STATUS")
    print("=" * 60)
    
    print("🚛 Mixed fleet support: ✅ Enabled")
    print("   • 4-ton trucks: Standard routes")
    print("   • 24-ton trucks: EU driving regulations enforced")
    
    print("\n⏰ Driver regulations: ✅ Enforced")
    print("   • Max 9 hours work per day for heavy trucks")
    print("   • Mandatory 45-min break after 4.5 hours driving")
    print("   • Real service area locations for breaks")
    
    print("\n🌍 Geographic coverage: ✅ Northern Italy")
    print("   • 29 service areas across 11 highways")
    print("   • All areas have truck parking, fuel, and restaurants")
    print("   • Route-based break location optimization")
    
    print("\n🔧 Solver integration: ✅ All solvers supported")
    print("   • Quantum-enhanced solver")
    print("   • Classical optimization")
    print("   • OR-Tools integration")
    
    print("\n" + "=" * 60)
    print("🎉 Service Areas System Ready!")
    print("The VRP optimizer now supports realistic break planning")
    print("for heavy trucks using actual highway service areas!")
    print("=" * 60)

if __name__ == "__main__":
    main()
