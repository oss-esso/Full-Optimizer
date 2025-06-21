import os
import sys

# Add parent directory to path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vrp_scenarios import VRPScenarioGenerator, create_moda_small_scenario

def analyze_scenario_constraints():
    """Analyze the key differences between old and MODA scenarios."""
    
    gen = VRPScenarioGenerator()
    
    print("ANALYZING SCENARIO CONSTRAINTS")
    print("=" * 60)
    
    # Test old scenario
    print("\n1. OLD DELIVERY SCENARIO:")
    old_scenario = gen.create_small_delivery_scenario()
    
    print(f"   Name: {old_scenario.name}")
    print(f"   Locations: {len(old_scenario.locations)}")
    print(f"   Vehicles: {len(old_scenario.vehicles)}")
    print(f"   Ride requests: {len(old_scenario.ride_requests)}")
    
    # Check time windows in old scenario
    time_windowed_locations = 0
    for loc_id, location in old_scenario.locations.items():
        if hasattr(location, 'time_window') and location.time_window:
            if location.time_window.start != 0 or location.time_window.end != 1440:
                time_windowed_locations += 1
                print(f"   Location {loc_id} has time window: {location.time_window.start}-{location.time_window.end}")
    
    print(f"   Locations with restricted time windows: {time_windowed_locations}")
    
    # Check vehicle time windows
    vehicle_time_constraints = 0
    for vehicle in old_scenario.vehicles:
        if hasattr(vehicle, 'time_window') and vehicle.time_window:
            if vehicle.time_window.start != 0 or vehicle.time_window.end != 1440:
                vehicle_time_constraints += 1
                print(f"   Vehicle {vehicle.id} has time window: {vehicle.time_window.start}-{vehicle.time_window.end}")
    
    print(f"   Vehicles with time constraints: {vehicle_time_constraints}")
    
    # Test MODA scenario
    print("\n2. MODA_SMALL SCENARIO:")
    moda_scenario = create_moda_small_scenario()
    
    print(f"   Name: {moda_scenario.name}")
    print(f"   Locations: {len(moda_scenario.locations)}")
    print(f"   Vehicles: {len(moda_scenario.vehicles)}")
    print(f"   Ride requests: {len(moda_scenario.ride_requests)}")
    
    # Check time windows in MODA scenario
    time_windowed_locations = 0
    total_window_time = 0
    for loc_id, location in moda_scenario.locations.items():
        if hasattr(location, 'time_window') and location.time_window:
            if location.time_window.start != 0 or location.time_window.end != 1440:
                time_windowed_locations += 1
                window_duration = location.time_window.end - location.time_window.start
                total_window_time += window_duration
                print(f"   Location {loc_id} has time window: {location.time_window.start}-{location.time_window.end} ({window_duration} min)")
    
    if time_windowed_locations > 0:
        avg_window = total_window_time / time_windowed_locations
        print(f"   Locations with restricted time windows: {time_windowed_locations}")
        print(f"   Average time window duration: {avg_window:.1f} minutes")
    else:
        print(f"   Locations with restricted time windows: {time_windowed_locations}")
    
    # Check vehicle time windows in MODA
    vehicle_time_constraints = 0
    for vehicle in moda_scenario.vehicles:
        if hasattr(vehicle, 'time_window') and vehicle.time_window:
            if vehicle.time_window.start != 0 or vehicle.time_window.end != 1440:
                vehicle_time_constraints += 1
                print(f"   Vehicle {vehicle.id} has time window: {vehicle.time_window.start}-{vehicle.time_window.end}")
    
    print(f"   Vehicles with time constraints: {vehicle_time_constraints}")
    
    # Test service times
    print("\n3. SERVICE TIMES:")
    print("   Old scenario locations:")
    for i, (loc_id, location) in enumerate(list(old_scenario.locations.items())[:3]):
        service_time = getattr(location, 'service_time', 'No service time')
        print(f"     {loc_id}: {service_time}")
    
    print("   MODA scenario locations:")
    for i, (loc_id, location) in enumerate(list(moda_scenario.locations.items())[:3]):
        service_time = getattr(location, 'service_time', 'No service time')
        print(f"     {loc_id}: {service_time}")

if __name__ == "__main__":
    analyze_scenario_constraints()
