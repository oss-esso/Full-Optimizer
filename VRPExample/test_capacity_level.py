#!/usr/bin/env python3
"""
Test capacity constraints (Level 4) in isolation with enhanced debugging.
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

def reduce_requests_to_quarter(scenario):
    """Reduce the number of requests to 25% for easier capacity testing."""
    original_count = len(scenario.ride_requests)
    
    # Take only the first quarter of requests (25%)
    scenario.ride_requests = scenario.ride_requests[:original_count // 4]
    
    # Also need to remove locations that are no longer needed
    # Get the location IDs that are still referenced by remaining requests
    used_location_ids = set()
    used_location_ids.add('depot_asti')  # Always keep depot
    used_location_ids.add('depot_milan')  # Always keep depot
    
    for req in scenario.ride_requests:
        used_location_ids.add(req.pickup_location)
        used_location_ids.add(req.dropoff_location)    # Keep all service areas too (for driver breaks) - using naming convention
    for loc_id in scenario.locations.keys():
        if ('service' in loc_id.lower() or 'area' in loc_id.lower() or 
            'depot' in loc_id.lower()):
            # Keep service areas and depots
            used_location_ids.add(loc_id)
    
    # Filter locations to only those that are still used
    original_locations = dict(scenario.locations)
    scenario.locations = {loc_id: loc for loc_id, loc in original_locations.items() 
                         if loc_id in used_location_ids}
    
    logger.info(f"📉 REDUCED PROBLEM SIZE:")
    logger.info(f"   Requests: {original_count} → {len(scenario.ride_requests)} ({len(scenario.ride_requests)/original_count*100:.1f}%)")
    logger.info(f"   Locations: {len(original_locations)} → {len(scenario.locations)} ({len(scenario.locations)/len(original_locations)*100:.1f}%)")
    logger.info(f"   Vehicles: {len(scenario.vehicles)} (unchanged)")
    
    return scenario

def test_capacity_level():
    """Test Level 4 (capacity) with enhanced debugging."""
    logger.info("=" * 80)
    logger.info("CAPACITY LEVEL TEST (Level 4)")
    logger.info("=" * 80)
    
    # Get the scenario - Use MODA_first for real scale testing
    try:
        scenarios = get_all_scenarios()
        scenario = scenarios['MODA_first']
        logger.info(f"✅ Successfully loaded scenario: {scenario.name}")
        logger.info(f"   📊 Original scale: {len(scenario.vehicles)} vehicles, {len(scenario.locations)} locations, {len(scenario.ride_requests)} requests")
        
        # Reduce problem size to 25% for easier capacity solving
        scenario = reduce_requests_to_quarter(scenario)
        
        # Show demand vs capacity analysis
        total_demand = sum(req.passengers for req in scenario.ride_requests)
        total_capacity = sum(v.capacity for v in scenario.vehicles.values())
        logger.info(f"\n📊 DEMAND vs CAPACITY ANALYSIS:")
        logger.info(f"   Total demand: {total_demand} kg")
        logger.info(f"   Total fleet capacity: {total_capacity} kg")
        logger.info(f"   Demand/capacity ratio: {total_demand/total_capacity*100:.1f}%")
        
        # Show sample vehicle capacities
        sample_vehicles = list(scenario.vehicles.values())[:10]
        vehicle_caps = [v.capacity for v in sample_vehicles]
        logger.info(f"   Sample vehicle capacities: {vehicle_caps}")
        
        # Show sample request sizes
        sample_requests = scenario.ride_requests[:10]
        request_sizes = [req.passengers for req in sample_requests]
        logger.info(f"   Sample request sizes: {request_sizes} kg")
        
    except Exception as e:
        logger.error(f"❌ Failed to load scenario: {e}")
        return
    
    # Create optimizer and test Level 4 (capacity)
    optimizer = VRPOptimizerEnhanced()
    
    # Run optimization with Level 4 (capacity constraints)
    try:
        logger.info("\n🚀 Testing Level 4 (Capacity Constraints)...")
        result = optimizer._solve_with_constraint_level(scenario, "capacity", 300)  # Give 5 minutes
        
        logger.info(f"\n🎯 LEVEL 4 RESULTS (CAPACITY CONSTRAINTS):")
        if result['success']:
            logger.info(f"✅ SUCCESS!")
            logger.info(f"   Objective: {result['objective_value']}")
            logger.info(f"   Fleet utilization: {result['vehicles_used']}/{len(scenario.vehicles)} ({result['vehicles_used']/len(scenario.vehicles)*100:.1f}%)")
            logger.info(f"   Vehicles used: {result['vehicles_used']}")
            logger.info(f"   Solve time: {result['solve_time']:.2f}s")
            
            # Show constraints applied
            if 'constraints_applied' in result:
                logger.info(f"   Constraints applied: {result['constraints_applied']}")
            
            # Analyze capacity constraint compliance
            capacity_violations = 0
            time_violations = 0
            
            if 'route_analysis' in result:
                logger.info(f"\n📊 DETAILED ROUTE ANALYSIS:")
                for i, analysis in enumerate(result['route_analysis']):
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
                    else:
                        logger.info(f"   ✅ Vehicle {analysis['vehicle_id']}: {stops} stops, {total_time}min total ({total_time/max_time*100:.1f}% of {max_time}min)")
                    
                    if max_load > capacity:
                        capacity_violations += 1
                        logger.error(f"   🚨 Vehicle {analysis['vehicle_id']}: CAPACITY VIOLATION - {max_load}kg > {capacity}kg limit")
                    else:
                        logger.info(f"      ✅ Capacity OK: {max_load}kg ({max_load/capacity*100:.1f}% of {capacity}kg capacity)")
                    
                    logger.info(f"      ⏱️  Breakdown: {driving_time}min driving + {service_time}min service = {driving_time + service_time}min")
                    
                    # Show capacity utilization efficiency
                    if max_load > 0:
                        capacity_efficiency = max_load / capacity * 100
                        if capacity_efficiency > 90:
                            logger.info(f"      📦 EXCELLENT capacity efficiency: {capacity_efficiency:.1f}%")
                        elif capacity_efficiency > 70:
                            logger.info(f"      📦 Good capacity efficiency: {capacity_efficiency:.1f}%")
                        elif capacity_efficiency > 50:
                            logger.info(f"      📦 Moderate capacity efficiency: {capacity_efficiency:.1f}%")
                        else:
                            logger.info(f"      📦 Low capacity efficiency: {capacity_efficiency:.1f}%")
            
            # Summary
            logger.info(f"\n📋 CONSTRAINT COMPLIANCE:")
            logger.info(f"   Time violations: {time_violations}")
            logger.info(f"   Capacity violations: {capacity_violations}")
            
            if capacity_violations == 0:
                logger.info("✅ ALL CAPACITY CONSTRAINTS RESPECTED!")
                logger.info("✅ Level 4 working correctly - capacity constraints are enforced")
            else:
                logger.error(f"🚨 {capacity_violations} capacity constraint violations found!")
                logger.error("❌ Level 4 not working correctly - capacity constraints not enforced")
            
            if time_violations == 0:
                logger.info("✅ All time constraints respected!")
            else:
                logger.error(f"🚨 {time_violations} time constraint violations found!")
                
        else:
            logger.error(f"❌ FAILED: {result.get('error', 'Unknown error')}")
            logger.info("💡 This suggests that the capacity constraints make the problem infeasible")
            logger.info("💡 Possible solutions:")
            logger.info("   - Add more vehicles")
            logger.info("   - Use vehicles with higher capacity")
            logger.info("   - Reduce request sizes")
            logger.info("   - Allow multiple trips per vehicle")
            logger.info("   - Relax time constraints further")
            
            # Show debugging info
            total_demand = sum(req.passengers for req in scenario.ride_requests)
            total_capacity = sum(v.capacity for v in scenario.vehicles.values())
            logger.info(f"\n🔍 DEBUGGING INFO:")
            logger.info(f"   Total demand: {total_demand} kg")
            logger.info(f"   Total fleet capacity: {total_capacity} kg")
            logger.info(f"   Demand exceeds capacity: {total_demand > total_capacity}")
            logger.info(f"   Excess demand: {max(0, total_demand - total_capacity)} kg")
            
    except Exception as e:
        logger.error(f"❌ Error in Level 4 test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_capacity_level()
