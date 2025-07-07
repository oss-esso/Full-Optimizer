#!/usr/bin/env python3

from vrp_scenarios import create_moda_small_scenario

def debug_time_calculation():
    """Debug the time calculation for MODA_small scenario."""
    
    print("Debugging time calculation for MODA_small...")
    scenario = create_moda_small_scenario()
    
    print(f"\nScenario details:")
    print(f"- Ride requests: {len(scenario.ride_requests)}")
    print(f"- Total locations: {len(scenario.location_ids)}")
    print(f"- Vehicles: {len(scenario.vehicles)}")
    
    # Check service times
    print(f"\nService times by location:")
    total_service_time = 0
    pickup_count = 0
    dropoff_count = 0
    
    for location_id, location in scenario.locations.items():
        service_time = getattr(location, 'service_time', 0)
        if service_time > 0:
            print(f"  {location_id}: {service_time} minutes")
            total_service_time += service_time
            if location_id.startswith('pickup'):
                pickup_count += 1
            elif location_id.startswith('dropoff'):
                dropoff_count += 1
    
    print(f"\nService time summary:")
    print(f"- Pickup locations: {pickup_count}")
    print(f"- Dropoff locations: {dropoff_count}")
    print(f"- Total service time: {total_service_time} minutes ({total_service_time/60:.1f} hours)")
    
    # Check distances and calculate travel time
    print(f"\nDistance analysis:")
    sample_locations = list(scenario.locations.keys())[:5]  # Sample first 5 locations
    
    for i in range(len(sample_locations) - 1):
        from_loc = sample_locations[i]
        to_loc = sample_locations[i + 1]
        distance = scenario.get_distance(from_loc, to_loc)
        
        # Apply the same calculation as in the optimizer
        distance_km = distance * 111  # GPS degrees to km conversion
        avg_speed_kmh = 50
        travel_time_hours = distance_km / avg_speed_kmh
        travel_time_minutes = travel_time_hours * 60
        
        print(f"  {from_loc} -> {to_loc}:")
        print(f"    Distance: {distance:.6f} units ({distance_km:.2f} km)")
        print(f"    Travel time: {travel_time_minutes:.1f} minutes")
    
    # Estimate total travel time for 20 stops
    print(f"\nEstimated total time for 20 stops:")
    # Average distance between locations
    total_distance = 0
    num_pairs = 0
    for i, loc1 in enumerate(sample_locations):
        for j, loc2 in enumerate(sample_locations):
            if i != j:
                total_distance += scenario.get_distance(loc1, loc2)
                num_pairs += 1
    
    avg_distance = total_distance / num_pairs if num_pairs > 0 else 0
    avg_distance_km = avg_distance * 111
    avg_travel_time_minutes = (avg_distance_km / 50) * 60
    
    print(f"- Average distance between locations: {avg_distance:.6f} units ({avg_distance_km:.2f} km)")
    print(f"- Average travel time: {avg_travel_time_minutes:.1f} minutes")
    print(f"- Estimated travel time for 19 segments: {19 * avg_travel_time_minutes:.1f} minutes")
    print(f"- Total service time: {total_service_time} minutes")
    print(f"- TOTAL ESTIMATED TIME: {(19 * avg_travel_time_minutes) + total_service_time:.1f} minutes ({((19 * avg_travel_time_minutes) + total_service_time)/60:.1f} hours)")
    
    # Check vehicle time limits
    print(f"\nVehicle time limits:")
    for vehicle_id, vehicle in scenario.vehicles.items():
        max_time = getattr(vehicle, 'max_total_work_time', None)
        if max_time:
            print(f"- {vehicle_id}: {max_time} minutes ({max_time/60:.1f} hours)")

if __name__ == "__main__":
    debug_time_calculation()
