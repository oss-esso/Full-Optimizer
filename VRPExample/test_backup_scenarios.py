import os
import sys

# Add parent directory to path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vrp_scenarios_backup import VRPScenarioGenerator
from vrp_optimizer_rolling_window import VRPOptimizerRollingWindow

def test_backup_scenarios():
    """Test backup scenarios with rolling window optimizer."""
    
    print("TESTING BACKUP SCENARIOS WITH ROLLING WINDOW OPTIMIZER")
    print("=" * 60)
    
    gen = VRPScenarioGenerator()
    
    # Test scenarios from backup
    test_scenarios = [
        ("Small Delivery", gen.create_small_delivery_scenario()),
        ("VRPPD", gen.create_vrppd_scenario()),
        ("Medium Delivery", gen.create_medium_delivery_scenario()),
        ("Time Window", gen.create_time_window_scenario()),
        ("Multi Depot", gen.create_multi_depot_scenario()),
    ]
    
    for scenario_name, scenario in test_scenarios:
        print(f"\n{scenario_name.upper()} SCENARIO:")
        print(f"  Locations: {len(scenario.locations)}")
        print(f"  Vehicles: {len(scenario.vehicles)}")
        print(f"  Ride requests: {len(scenario.ride_requests)}")
        
        # Check time windows
        time_windowed_locations = 0
        for loc_id, location in scenario.locations.items():
            if hasattr(location, 'time_window_start') and hasattr(location, 'time_window_end'):
                if location.time_window_start != 0 or location.time_window_end != 1440:
                    time_windowed_locations += 1
        print(f"  Time windowed locations: {time_windowed_locations}")
        
        # Check service times
        service_times = []
        for loc_id, location in list(scenario.locations.items())[:3]:
            service_time = getattr(location, 'service_time', 0)
            service_times.append(service_time)
        print(f"  Sample service times: {service_times}")
        
        # Test with rolling window optimizer
        try:
            optimizer = VRPOptimizerRollingWindow(scenario)
            result = optimizer.optimize_with_rolling_window()
            print(f"  RESULT: {result.status}")
            print(f"  Objective: {result.objective_value:.1f}")
            print(f"  Active routes: {len([r for r in result.routes.values() if len(r) > 2])}")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Testing which scenario types work with the rolling window optimizer")
    print("and whether time windows or service times are the main constraint.")

if __name__ == "__main__":
    test_backup_scenarios()
