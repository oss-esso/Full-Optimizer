import os
import sys

# Add parent directory to path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vrp_scenarios import VRPScenarioGenerator
from vrp_optimizer_rolling_window import VRPOptimizerRollingWindow

def test_service_time_hypothesis():
    """Test if service times are causing the infeasibility."""
    
    print("TESTING SERVICE TIME HYPOTHESIS")
    print("=" * 50)
    
    gen = VRPScenarioGenerator()
    
    # Create old scenario (no service times)
    old_scenario = gen.create_small_delivery_scenario()
    
    print("\n1. ORIGINAL OLD SCENARIO (no service times):")
    print(f"   Service times: depot={old_scenario.locations['depot'].service_time}, "
          f"customer={old_scenario.locations['customer_1'].service_time}")
    
    optimizer = VRPOptimizerRollingWindow(old_scenario)
    result = optimizer.optimize()
    print(f"   Result: {result.status}")
    
    # Add service times to old scenario
    print("\n2. OLD SCENARIO WITH SERVICE TIMES ADDED:")
    for loc_id, location in old_scenario.locations.items():
        if loc_id == 'depot':
            location.service_time = 5  # Same as MODA depots
        else:
            location.service_time = 15  # Same as MODA pickups
    
    print(f"   Service times: depot={old_scenario.locations['depot'].service_time}, "
          f"customer={old_scenario.locations['customer_1'].service_time}")
    
    optimizer2 = VRPOptimizerRollingWindow(old_scenario)
    result2 = optimizer2.optimize()
    print(f"   Result: {result2.status}")
    
    # Test with zero service times on MODA
    print("\n3. MODA SCENARIO WITH SERVICE TIMES REMOVED:")
    from vrp_scenarios import create_moda_small_scenario
    moda_scenario = create_moda_small_scenario()
    
    # Remove service times
    for loc_id, location in moda_scenario.locations.items():
        location.service_time = 0
    
    print(f"   Service times removed from all {len(moda_scenario.locations)} locations")
    
    optimizer3 = VRPOptimizerRollingWindow(moda_scenario)
    result3 = optimizer3.optimize()
    print(f"   Result: {result3.status}")

if __name__ == "__main__":
    test_service_time_hypothesis()
