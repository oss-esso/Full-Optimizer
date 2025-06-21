#!/usr/bin/env python3
"""
Quick test to verify the modified MODA_small scenario capacity logic.
"""

from vrp_scenarios import create_moda_small_scenario

def test_capacity_scenario():
    """Test the modified scenario capacity distribution."""
    print("🧪 TESTING CAPACITY SCENARIO LOGIC")
    print("=" * 50)
    
    scenario = create_moda_small_scenario()
      # Analyze fleet composition
    total_capacity = 0
    preloaded_cargo = 0
    
    print("\n🚛 FLEET ANALYSIS (Strategic Loading):")
    for vehicle_id, vehicle in scenario.vehicles.items():
        preloaded = getattr(vehicle, 'preloaded_cargo', 0)
        total_capacity += vehicle.capacity
        preloaded_cargo += preloaded
        utilization = (preloaded / vehicle.capacity * 100) if vehicle.capacity > 0 else 0
        
        print(f"  {vehicle_id}: {vehicle.capacity}kg capacity, {preloaded}kg pre-loaded ({utilization:.1f}%)")
    
    print(f"\n💡 Loading Strategy Analysis:")
    heavy_trucks = [v for v in scenario.vehicles.values() if getattr(v, 'vehicle_type', '') == 'heavy']
    standard_trucks = [v for v in scenario.vehicles.values() if getattr(v, 'vehicle_type', '') == 'standard']
    
    if heavy_trucks:
        heavy_avg_util = sum(getattr(v, 'preloaded_cargo', 0) / v.capacity for v in heavy_trucks) / len(heavy_trucks) * 100
        print(f"  Heavy trucks average utilization: {heavy_avg_util:.1f}%")
    
    if standard_trucks:
        standard_avg_util = sum(getattr(v, 'preloaded_cargo', 0) / v.capacity for v in standard_trucks) / len(standard_trucks) * 100
        print(f"  Standard trucks average utilization: {standard_avg_util:.1f}%")    # Analyze pickup requests by type
    depot_pickup_cargo = 0
    field_pickup_cargo = 0
    print(f"\n📦 PICKUP REQUESTS BY TYPE:")
    for request in scenario.ride_requests:
        cargo_weight = request.passengers  # passengers = cargo weight
        if request.pickup_location == 'depot_1':
            depot_pickup_cargo += cargo_weight
            print(f"  {request.id}: {cargo_weight}kg from {request.pickup_location} (DEPOT) to {request.dropoff_location}")
        else:
            field_pickup_cargo += cargo_weight
            print(f"  {request.id}: {cargo_weight}kg from {request.pickup_location} (FIELD) to {request.dropoff_location}")
    
    total_cargo = depot_pickup_cargo + field_pickup_cargo
      # Summary
    print(f"\n📊 CAPACITY SUMMARY:")
    print(f"  Total fleet capacity: {total_capacity:,}kg")
    print(f"  Pre-loaded at depot: {preloaded_cargo:,}kg ({preloaded_cargo/total_capacity*100:.1f}%)")
    print(f"  Depot pickup requests: {depot_pickup_cargo:,}kg ({depot_pickup_cargo/total_capacity*100:.1f}%)")
    print(f"  Field pickup requests: {field_pickup_cargo:,}kg ({field_pickup_cargo/total_capacity*100:.1f}%)")
    print(f"  Total cargo: {total_cargo:,}kg ({total_cargo/total_capacity*100:.1f}%)")
    
    # Verification
    expected_depot = total_capacity * 0.7
    expected_field = total_capacity * 0.3
    
    print(f"\n✅ VERIFICATION:")
    print(f"  Expected 70% depot cargo: {expected_depot:,.0f}kg")
    print(f"  Actual depot cargo: {depot_pickup_cargo:,}kg ({'✅' if abs(depot_pickup_cargo - expected_depot) < 100 else '❌'})")
    print(f"  Expected 30% field cargo: {expected_field:,.0f}kg")
    print(f"  Actual field cargo: {field_pickup_cargo:,}kg ({'✅' if abs(field_pickup_cargo - expected_field) < 100 else '❌'})")
    
    # Check depot count
    depots = [loc for loc in scenario.locations.values() if 'depot' in loc.id]
    print(f"  Single depot: {'✅' if len(depots) == 1 else f'❌ ({len(depots)} depots found)'}")
    
    print(f"\n🎯 SCENARIO VALIDATION: {'✅ PASSED' if len(depots) == 1 and abs(depot_pickup_cargo - expected_depot) < 100 else '❌ FAILED'}")
    print(f"💡 Strategy: Virtual depot pickups allow optimizer to decide optimal loading!")

if __name__ == "__main__":
    test_capacity_scenario()
