#!/usr/bin/env python3
"""
Test the capacity constraints with the new fixes (no return-to-depot + capacity debugging)
"""

import logging
from vrp_optimizer_enhanced_full import VRPOptimizerEnhanced
from vrp_scenarios import create_moda_small_scenario

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_capacity_with_fixes():
    """Test capacity constraints with the new fixes."""
    print("\n🔧 TESTING CAPACITY CONSTRAINTS WITH FIXES")
    print("=" * 70)
    print("Fixes applied:")
    print("✅ Removed return-to-depot constraint (no more 0→0 connections)")
    print("✅ Added capacity constraint debugging")
    print("✅ Using virtual depot pickup bays")
    
    scenario = create_moda_small_scenario()
    logger.info(f"✅ Loaded scenario: {len(scenario.locations)} locations, {len(scenario.vehicles)} vehicles")
    
    optimizer = VRPOptimizerEnhanced()
    
    # Test just the capacity level to isolate the issue
    logger.info("🧪 Testing Level 4: Distance + Time + Time Windows + Capacity")
    try:
        result = optimizer._solve_with_constraint_level(scenario, "capacity", 60)
        if result['success']:
            logger.info("✅ CAPACITY CONSTRAINTS SUCCESS!")
            logger.info(f"   🎯 Objective: {result['objective_value']}")
            logger.info(f"   🚛 Vehicles used: {result['vehicles_used']}/{len(scenario.vehicles)}")
            logger.info(f"   ⏱️  Solve time: {result['solve_time']:.2f}s")
            
            # Show the solution details
            if 'route_analysis' in result:
                logger.info("\n📊 ROUTE ANALYSIS:")
                analysis = result['route_analysis']
                if isinstance(analysis, list):
                    for vehicle_info in analysis:
                        if isinstance(vehicle_info, dict):
                            vid = vehicle_info.get('vehicle_id', 'N/A')
                            stops = vehicle_info.get('stops', 0)
                            max_load = vehicle_info.get('max_load', 0)
                            logger.info(f"   🚛 Vehicle {vid}: {stops} stops, max load: {max_load}kg")
                            
        else:
            logger.error("❌ CAPACITY CONSTRAINTS FAILED!")
            logger.error(f"   Error: {result.get('error', 'Unknown')}")
            logger.error(f"   Details: {result.get('details', 'No additional details')}")
            
    except Exception as e:
        logger.error(f"❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_capacity_with_fixes()
