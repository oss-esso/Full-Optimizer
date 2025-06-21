#!/usr/bin/env python3
"""
Quick test of our mixed fleet and service areas implementation.
"""

from vrp_scenarios import create_moda_small_scenario

def test_scenario_features():
    print("Testing mixed fleet and service areas implementation...")
    
    # Create scenario
    instance = create_moda_small_scenario()
    
    # Analysis
    total_locations = len(instance.locations)
    total_vehicles = len(instance.vehicles)
    service_areas = [l for l in instance.locations.keys() if l.startswith("service_")]
    heavy_trucks = [v for v in instance.vehicles.values() if getattr(v, 'vehicle_type', None) == "heavy"]
    standard_trucks = [v for v in instance.vehicles.values() if getattr(v, 'vehicle_type', None) == "standard"]
    
    print(f"\n✅ SCENARIO FEATURES:")
    print(f"   Total locations: {total_locations}")
    print(f"   Total vehicles: {total_vehicles}")
    print(f"   Service areas: {len(service_areas)}")
    print(f"   Heavy trucks (24t): {len(heavy_trucks)}")
    print(f"   Standard trucks (4t): {len(standard_trucks)}")
    
    print(f"\n🛣️ SERVICE AREAS:")
    for area_id in service_areas[:5]:  # Show first 5
        area = instance.locations[area_id]
        print(f"   {area_id}: {getattr(area, 'address', 'Unknown location')}")
    
    print(f"\n🚛 DRIVER REGULATIONS:")
    if heavy_trucks:
        ht = heavy_trucks[0]
        print(f"   Heavy trucks: {getattr(ht, 'max_total_work_time', 'N/A')}min work, {getattr(ht, 'max_driving_time', 'N/A')}min driving")
    if standard_trucks:
        st = standard_trucks[0]
        print(f"   Standard trucks: {getattr(st, 'max_total_work_time', 'N/A')}min work, {getattr(st, 'max_driving_time', 'N/A')}min driving")
    
    print(f"\n📦 CARGO ANALYSIS:")
    cargo_weights = [req.passengers for req in instance.ride_requests]
    total_cargo = sum(cargo_weights)
    total_capacity = sum(v.capacity for v in instance.vehicles.values())
    print(f"   Total cargo: {total_cargo:,} kg")
    print(f"   Total capacity: {total_capacity:,} kg")
    print(f"   Utilization: {total_cargo/total_capacity:.1%}")
    print(f"   Feasible: {'✅' if total_cargo <= total_capacity else '❌'}")

if __name__ == "__main__":
    test_scenario_features()
