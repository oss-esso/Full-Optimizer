import os
import sys

# Add parent directory to path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vrp_scenarios_backup import VRPScenarioGenerator

def debug_time_windows():
    """Debug what time windows are actually being set."""
    
    print("DEBUGGING TIME WINDOW VALUES")
    print("=" * 40)
    
    gen = VRPScenarioGenerator()
    
    # Test small delivery scenario
    scenario = gen.create_small_delivery_scenario()
    
    print(f"\nSMALL DELIVERY SCENARIO:")
    for loc_id, location in scenario.locations.items():
        print(f"  {loc_id}:")
        print(f"    time_window_start: {location.time_window_start}")
        print(f"    time_window_end: {location.time_window_end}")
        print(f"    service_time: {location.service_time}")
        
        # Check if it has actual time window constraints
        has_constraint = False
        if location.time_window_start is not None and location.time_window_end is not None:
            if location.time_window_start != 0 or location.time_window_end != 1440:
                has_constraint = True
        print(f"    has_time_constraint: {has_constraint}")
    
    # Test time window scenario
    print(f"\nTIME WINDOW SCENARIO:")
    tw_scenario = gen.create_time_window_scenario()
    
    for loc_id, location in tw_scenario.locations.items():
        print(f"  {loc_id}:")
        print(f"    time_window_start: {location.time_window_start}")
        print(f"    time_window_end: {location.time_window_end}")
        print(f"    service_time: {location.service_time}")
        
        # Check if it has actual time window constraints
        has_constraint = False
        if location.time_window_start is not None and location.time_window_end is not None:
            if location.time_window_start != 0 or location.time_window_end != 1440:
                has_constraint = True
        print(f"    has_time_constraint: {has_constraint}")

if __name__ == "__main__":
    debug_time_windows()
