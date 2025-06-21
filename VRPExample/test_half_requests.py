#!/usr/bin/env python3
"""
Test Level 1 with HALF the requests to see if service time accounting makes the problem too constrained.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from VRPExample.vrp_scenarios import get_all_scenarios
from VRPExample.vrp_optimizer_enhanced_full import VRPOptimizerEnhanced

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reduce_requests_by_half(scenario):
    """Reduce the number of requests by half to test scalability."""
    original_count = len(scenario.ride_requests)
    
    # Take only the first half of requests
    scenario.ride_requests = scenario.ride_requests[:original_count // 2]
    
    # Also need to remove locations that are no longer needed
    # Get the location IDs that are still referenced by remaining requests
    used_location_ids = set()
    used_location_ids.add('depot_asti')  # Always keep depot
    used_location_ids.add('depot_milan')  # Always keep depot
    
    for req in scenario.ride_requests:
        used_location_ids.add(req.pickup_location)
        used_location_ids.add(req.dropoff_location)
    
    # Filter locations to only those that are still used
    original_locations = dict(scenario.locations)
    scenario.locations = {loc_id: loc for loc_id, loc in original_locations.items() 
                         if loc_id in used_location_ids}
    
    logger.info(f"📉 REDUCED PROBLEM SIZE:")
    logger.info(f"   Requests: {original_count} → {len(scenario.ride_requests)} ({len(scenario.ride_requests)/original_count*100:.1f}%)")
    logger.info(f"   Locations: {len(original_locations)} → {len(scenario.locations)} ({len(scenario.locations)/len(original_locations)*100:.1f}%)")
    logger.info(f"   Vehicles: {len(scenario.vehicles)} (unchanged)")
    
    return scenario

def test_level1_half_requests():
    """Test Level 1 with half the requests."""
    logger.info("=" * 80)
    logger.info("LEVEL 1 TEST WITH HALF THE REQUESTS")
    logger.info("=" * 80)
    
    # Get the scenario - Use MODA_first for real scale testing
    try:
        scenarios = get_all_scenarios()
        scenario = scenarios['MODA_first']
        logger.info(f"✅ Successfully loaded scenario: {scenario.name}")
        logger.info(f"   📊 Original scale: {len(scenario.vehicles)} vehicles, {len(scenario.locations)} locations, {len(scenario.ride_requests)} requests")
        
        # Reduce problem size
        scenario = reduce_requests_by_half(scenario)
        
    except Exception as e:
        logger.error(f"❌ Failed to load scenario: {e}")
        return
    
    # Create optimizer and test ONLY Level 1
    optimizer = VRPOptimizerEnhanced()
    
    # Run optimization with Level 1 only
    try:
        logger.info("\n🚀 Testing Level 1 with HALF requests...")
        result = optimizer._solve_with_constraint_level(scenario, "distance_time", 120)  # Give more time
        
        logger.info(f"\n🎯 LEVEL 1 RESULTS (HALF REQUESTS):")
        if result['success']:
            logger.info(f"✅ SUCCESS!")
            logger.info(f"   Objective: {result['objective_value']}")
            logger.info(f"   Fleet utilization: {result['vehicles_used']}/{len(scenario.vehicles)} ({result['vehicles_used']/len(scenario.vehicles)*100:.1f}%)")
            logger.info(f"   Vehicles used: {result['vehicles_used']}")
            
            # Analyze time constraint violations
            time_violations = 0
            capacity_violations = 0
            
            if 'route_analysis' in result:
                logger.info(f"\n📊 ROUTE ANALYSIS (showing first 10 vehicles):")
                for i, analysis in enumerate(result['route_analysis'][:10]):
                    stops = analysis['stops']
                    total_time = analysis['total_time']
                    max_time = analysis['max_allowed_time']
                    service_time = analysis['service_time']
                    driving_time = analysis['driving_time']
                    max_load = analysis['max_load_reached']
                    capacity = analysis['capacity']
                    
                    # Check for violations
                    if total_time > max_time:
                        time_violations += 1
                        logger.error(f"   🚨 Vehicle {analysis['vehicle_id']}: TIME VIOLATION - {total_time}min > {max_time}min limit")
                    if max_load > capacity:
                        capacity_violations += 1
                        logger.error(f"   🚨 Vehicle {analysis['vehicle_id']}: CAPACITY VIOLATION - {max_load}kg > {capacity}kg limit")
                    
                    logger.info(f"   Vehicle {analysis['vehicle_id']}: {stops} stops, {total_time}min total ({total_time/max_time*100:.1f}% of {max_time}min)")
                    logger.info(f"      ⏱️  Breakdown: {driving_time}min driving + {service_time}min service = {driving_time + service_time}min")
                    logger.info(f"      📦 Load: {max_load}kg ({max_load/capacity*100:.1f}% of {capacity}kg capacity)")
                    
                    # Calculate expected service time
                    expected_service_time = stops * 15  # 15 minutes per stop
                    if abs(service_time - expected_service_time) > 20:  # Allow some tolerance
                        logger.warning(f"      ⚠️ Service time mismatch: expected ~{expected_service_time}min, got {service_time}min")
                    else:
                        logger.info(f"      ✅ Service time looks correct: {service_time}min for {stops} stops")
            
            # Summary
            logger.info(f"\n📋 CONSTRAINT VIOLATIONS:")
            logger.info(f"   Time violations: {time_violations}")
            logger.info(f"   Capacity violations: {capacity_violations} (expected for Level 1)")
            
            if time_violations == 0:
                logger.info("✅ All time constraints respected!")
            else:
                logger.error(f"🚨 {time_violations} time constraint violations found!")
                
        else:
            logger.error(f"❌ FAILED: {result.get('error', 'Unknown error')}")
            logger.info("💡 This suggests that even with half the requests, the time constraints are too tight")
            logger.info("💡 Possible solutions:")
            logger.info("   - Increase vehicle time limits")
            logger.info("   - Reduce service time per stop")
            logger.info("   - Add more vehicles")
            logger.info("   - Use faster vehicle speed")
            
    except Exception as e:
        logger.error(f"❌ Error in Level 1 test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_level1_half_requests()
