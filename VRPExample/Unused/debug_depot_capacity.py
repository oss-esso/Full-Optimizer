#!/usr/bin/env python3
"""
Test to debug capacity callback logic for depot virtual pickups
"""

import logging
from vrp_scenarios import get_all_scenarios
from vrp_optimizer_enhanced_full import VRPOptimizerEnhanced

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_depot_capacity_callback():
    """Test the capacity callback logic for depot virtual pickups"""
    print("\n🔍 DEPOT CAPACITY CALLBACK DEBUG TEST")
    print("=" * 70)    # Load scenario
    scenarios = get_all_scenarios()
    instance = scenarios['MODA_small']
    scenario_name = 'MODA_small'
    logger.info(f"Loaded scenario: {len(instance.locations)} locations, {len(instance.vehicles)} vehicles")
    
    # Print depot info
    depot_location = list(instance.locations.values())[0]  # First location is depot
    logger.info(f"Depot location: {depot_location.id}")
    
    # Print depot requests
    depot_requests = [req for req in instance.ride_requests if req.pickup_location == depot_location.id]
    field_requests = [req for req in instance.ride_requests if req.pickup_location != depot_location.id]
    
    logger.info(f"Found {len(depot_requests)} depot requests:")
    total_depot_cargo = 0
    for i, req in enumerate(depot_requests[:5]):  # Show first 5
        logger.info(f"  {req.id}: {req.pickup_location} → {req.dropoff_location} ({req.passengers}kg)")
        total_depot_cargo += req.passengers
    
    logger.info(f"Total depot cargo if all loaded at once: {total_depot_cargo}kg")
    
    logger.info(f"Found {len(field_requests)} field requests:")
    for i, req in enumerate(field_requests[:3]):  # Show first 3
        logger.info(f"  {req.id}: {req.pickup_location} → {req.dropoff_location} ({req.passengers}kg)")
    
    # Check vehicle capacities
    vehicle_capacities = [int(v.capacity) for v in instance.vehicles.values()]
    logger.info(f"Vehicle capacities: {vehicle_capacities}")
    max_capacity = max(vehicle_capacities)
    logger.info(f"Max single vehicle capacity: {max_capacity}kg")
    
    print("\n🚨 PROBLEM IDENTIFIED:")
    if total_depot_cargo > max_capacity:
        print(f"  - ALL depot cargo ({total_depot_cargo}kg) > max vehicle capacity ({max_capacity}kg)")
        print(f"  - When visiting depot (index 0), capacity callback adds ALL {total_depot_cargo}kg at once!")
        print(f"  - This violates capacity constraint before any routes are even planned")
        print(f"  - Solution: Depot requests need to be handled differently")
    else:
        print(f"  - Depot cargo would fit in largest vehicle")
    
    print("\n💡 ROOT CAUSE:")
    print("  - Depot location is both the START depot AND pickup location for virtual requests")
    print("  - OR-Tools capacity callback assumes pickup location = add cargo immediately")
    print("  - But depot 'virtual pickups' should be handled differently")
    print("  - Each depot request should only affect the vehicle that is assigned to it")

if __name__ == "__main__":
    test_depot_capacity_callback()
