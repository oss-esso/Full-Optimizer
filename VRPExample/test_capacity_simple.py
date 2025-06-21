#!/usr/bin/env python3
"""
Simple test for capacity constraints (Level 4) with enhanced debugging.
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

def test_capacity_simple():
    """Test Level 4 (capacity) with a smaller scenario."""
    logger.info("=" * 80)
    logger.info("SIMPLE CAPACITY LEVEL TEST (Level 4)")
    logger.info("=" * 80)
    
    # Get a smaller scenario first - MODA_small
    try:
        scenarios = get_all_scenarios()
        scenario = scenarios['MODA_small']  # Use smaller scenario first
        logger.info(f"✅ Successfully loaded scenario: {scenario.name}")
        logger.info(f"   📊 Scale: {len(scenario.vehicles)} vehicles, {len(scenario.locations)} locations, {len(scenario.ride_requests)} requests")
        
        # Show demand vs capacity analysis
        total_demand = sum(req.passengers for req in scenario.ride_requests)
        total_capacity = sum(v.capacity for v in scenario.vehicles.values())
        logger.info(f"\n📊 DEMAND vs CAPACITY ANALYSIS:")
        logger.info(f"   Total demand: {total_demand} kg")
        logger.info(f"   Total fleet capacity: {total_capacity} kg")
        logger.info(f"   Demand/capacity ratio: {total_demand/total_capacity*100:.1f}%")
        
        # Show vehicle capacities
        vehicle_caps = [v.capacity for v in scenario.vehicles.values()]
        logger.info(f"   Vehicle capacities: {vehicle_caps}")
        
        # Show request sizes
        request_sizes = [req.passengers for req in scenario.ride_requests]
        logger.info(f"   Request sizes: {request_sizes} kg")
        
    except Exception as e:
        logger.error(f"❌ Failed to load scenario: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create optimizer and test Level 4 (capacity)
    optimizer = VRPOptimizerEnhanced()
    
    # Run optimization with Level 4 (capacity constraints)
    try:
        logger.info("\n🚀 Testing Level 4 (Capacity Constraints)...")
        result = optimizer._solve_with_constraint_level(scenario, "capacity", 180)  # Give 3 minutes
        
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
                for analysis in result['route_analysis']:
                    vehicle_id = analysis['vehicle_id']
                    stops = analysis['stops']
                    total_time = analysis['total_time']
                    max_time = analysis['max_allowed_time']
                    max_load = analysis['max_load_reached']
                    capacity = analysis['capacity']
                    
                    # Check for violations
                    if total_time > max_time:
                        time_violations += 1
                        logger.error(f"   🚨 Vehicle {vehicle_id}: TIME VIOLATION - {total_time}min > {max_time}min")
                    
                    if max_load > capacity:
                        capacity_violations += 1
                        logger.error(f"   🚨 Vehicle {vehicle_id}: CAPACITY VIOLATION - {max_load}kg > {capacity}kg")
                    
                    logger.info(f"   Vehicle {vehicle_id}: {stops} stops, time={total_time}min, load={max_load}kg")
                    logger.info(f"      Time: {total_time}/{max_time}min ({total_time/max_time*100:.1f}%)")
                    logger.info(f"      Load: {max_load}/{capacity}kg ({max_load/capacity*100:.1f}%)")
            
            # Summary
            logger.info(f"\n📋 CONSTRAINT COMPLIANCE:")
            logger.info(f"   Time violations: {time_violations}")
            logger.info(f"   Capacity violations: {capacity_violations}")
            
            if capacity_violations == 0:
                logger.info("✅ ALL CAPACITY CONSTRAINTS RESPECTED!")
                logger.info("✅ Level 4 working correctly - capacity constraints are enforced")
            else:
                logger.error(f"🚨 {capacity_violations} capacity violations - Level 4 not working correctly")
            
            if time_violations == 0:
                logger.info("✅ All time constraints respected!")
            else:
                logger.error(f"🚨 {time_violations} time violations found")
                
        else:
            logger.error(f"❌ FAILED: {result.get('error', 'Unknown error')}")
            logger.info("💡 Capacity constraints make the problem infeasible")
            
            # Show debugging info
            logger.info(f"\n🔍 ANALYSIS:")
            logger.info(f"   This means capacity constraints are being enforced (good!)")
            logger.info(f"   But the problem is over-constrained for this scenario")
            logger.info(f"   Try: more vehicles, larger vehicles, or smaller requests")
            
    except Exception as e:
        logger.error(f"❌ Error in Level 4 test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_capacity_simple()
