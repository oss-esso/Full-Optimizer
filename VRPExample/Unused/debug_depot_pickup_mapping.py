#!/usr/bin/env python3
"""
Debug script to check how depot pickup requests are being mapped to node indices.
"""

from vrp_scenarios import create_moda_small_scenario
from vrp_optimizer_enhanced_full import VRPOptimizerEnhanced

def debug_depot_pickup_mapping():
    """Debug the depot pickup request mapping."""
    
    print("🔍 DEBUGGING DEPOT PICKUP REQUEST MAPPING")
    print("=" * 60)
    
    # Load scenario
    scenario = create_moda_small_scenario()
    
    print(f"\n📍 LOCATIONS ({len(scenario.locations)}):")
    location_list = list(scenario.locations.values())
    location_ids = [loc.id for loc in location_list]
    
    for i, loc in enumerate(location_list):
        print(f"  [{i}] {loc.id} - {loc.lat:.4f}, {loc.lon:.4f}")
    
    print(f"\n🚛 VEHICLES ({len(scenario.vehicles)}):")
    for vehicle in scenario.vehicles.values():
        print(f"  {vehicle.id}: {vehicle.capacity}kg, depot: {vehicle.depot_id}")
    
    print(f"\n📦 RIDE REQUESTS ({len(scenario.ride_requests)}):")
    depot_requests = []
    field_requests = []
    
    for req in scenario.ride_requests:
        if req.pickup_location == "depot_1":
            depot_requests.append(req)
        else:
            field_requests.append(req)
        
        try:
            pickup_idx = location_ids.index(req.pickup_location)
            dropoff_idx = location_ids.index(req.dropoff_location)
            print(f"  {req.id}: {req.pickup_location}[{pickup_idx}] → {req.dropoff_location}[{dropoff_idx}] ({req.passengers}kg)")
        except ValueError as e:
            print(f"  ❌ {req.id}: ERROR - {e}")
            print(f"     Pickup: {req.pickup_location} (exists: {req.pickup_location in location_ids})")
            print(f"     Dropoff: {req.dropoff_location} (exists: {req.dropoff_location in location_ids})")
    
    print(f"\n📊 REQUEST BREAKDOWN:")
    print(f"  Depot requests (virtual pickups): {len(depot_requests)}")
    print(f"  Field requests: {len(field_requests)}")
    
    # Check for duplicate or missing nodes
    print(f"\n🔍 LOCATION VALIDATION:")
    for req in scenario.ride_requests:
        if req.pickup_location not in location_ids:
            print(f"  ❌ Missing pickup location: {req.pickup_location}")
        if req.dropoff_location not in location_ids:
            print(f"  ❌ Missing dropoff location: {req.dropoff_location}")
    
    # Check depot configuration
    depot_locations = [loc for loc in location_list if "depot" in loc.id]
    print(f"\n🏭 DEPOT ANALYSIS:")
    print(f"  Found {len(depot_locations)} depot locations:")
    for depot in depot_locations:
        print(f"    {depot.id}: index {location_ids.index(depot.id)}")
        
        # Count requests from this depot
        depot_req_count = len([r for r in scenario.ride_requests if r.pickup_location == depot.id])
        print(f"    Requests from {depot.id}: {depot_req_count}")
    
    print(f"\n🎯 POTENTIAL ISSUES:")
    
    # Check if depot is index 0 (required for OR-Tools)
    if location_ids[0] != "depot_1":
        print(f"  ⚠️  Depot is not at index 0! Current: {location_ids[0]} at index 0")
        print(f"      Depot 'depot_1' is at index: {location_ids.index('depot_1')}")
    else:
        print(f"  ✅ Depot is correctly at index 0")
    
    # Check for time window conflicts
    depot = scenario.locations["depot_1"]
    print(f"\n🕐 DEPOT TIME WINDOW: {depot.time_window_start}-{depot.time_window_end}min")
    
    for req in depot_requests[:3]:  # Check first 3 depot requests
        dropoff_loc = scenario.locations[req.dropoff_location]
        print(f"  {req.id}: depot → {req.dropoff_location} ({dropoff_loc.time_window_start}-{dropoff_loc.time_window_end}min)")
        
        if dropoff_loc.time_window_start > depot.time_window_end:
            print(f"    ⚠️  Impossible: dropoff starts after depot closes!")

if __name__ == "__main__":
    debug_depot_pickup_mapping()
