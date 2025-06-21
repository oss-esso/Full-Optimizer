#!/usr/bin/env python3
"""
Test ultra-simple capacity scenario: 1 small depot request that fits in any vehicle
"""

import logging
from vrp_optimizer_enhanced_full import VRPOptimizerEnhanced
from vrp_scenarios import create_moda_small_scenario

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ultra_simple_capacity():
    """Test with just one small depot request to isolate capacity issues."""
    print("\n🧪 ULTRA SIMPLE CAPACITY TEST")
    print("=" * 60)
    
    scenario = create_moda_small_scenario()
    
    # Keep only ONE small depot request and ONE field request
    depot_requests = [req for req in scenario.ride_requests if 'depot_request' in req.id]
    field_requests = [req for req in scenario.ride_requests if 'field_request' in req.id]
    
    # Find the smallest depot request
    smallest_depot = min(depot_requests, key=lambda r: r.passengers)
    smallest_field = min(field_requests, key=lambda r: r.passengers)
    
    # Keep only these two requests
    scenario.ride_requests = [smallest_depot, smallest_field]
    
    logger.info(f"Simplified to 2 requests:")
    logger.info(f"  {smallest_depot.id}: {smallest_depot.pickup_location} → {smallest_depot.dropoff_location} ({smallest_depot.passengers}kg)")
    logger.info(f"  {smallest_field.id}: {smallest_field.pickup_location} → {smallest_field.dropoff_location} ({smallest_field.passengers}kg)")
    
    # Vehicle capacities
    vehicle_capacities = [int(v.capacity) for v in scenario.vehicles.values()]
    logger.info(f"Vehicle capacities: {vehicle_capacities}")
    
    logger.info(f"✅ Both requests fit in ALL vehicles (smallest capacity: {min(vehicle_capacities)}kg)")
    
    optimizer = VRPOptimizerEnhanced()
    
    # Test just capacity level
    logger.info("🧪 Testing ULTRA SIMPLE Level 4: Capacity")
    try:
        result = optimizer._solve_with_constraint_level(scenario, "capacity", 30)
        if result['success']:
            logger.info("🎉 ULTRA SIMPLE CAPACITY SUCCESS!")
            logger.info(f"   🎯 Objective: {result['objective_value']}")
            logger.info(f"   🚛 Vehicles used: {result['vehicles_used']}/{len(scenario.vehicles)}")
        else:
            logger.error("❌ ULTRA SIMPLE CAPACITY FAILED!")
            logger.error(f"   Error: {result.get('error', 'Unknown')}")
            logger.error("   🚨 This suggests fundamental capacity constraint implementation issue!")
            
    except Exception as e:
        logger.error(f"❌ EXCEPTION: {e}")

if __name__ == "__main__":
    test_ultra_simple_capacity()
