#!/usr/bin/env python3
"""
Quick test for Level 1 only to debug time constraints faster.
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

def test_level1_quick():
    """Test only Level 1 for quick debugging."""
    logger.info("=" * 80)
    logger.info("QUICK LEVEL 1 TEST WITH STRICT TIME ENFORCEMENT")
    logger.info("=" * 80)
    
    # Get the scenario - Use MODA_first for real scale testing
    try:
        scenarios = get_all_scenarios()
        scenario = scenarios['MODA_first']
        logger.info(f"✅ Successfully loaded scenario: {scenario.name}")
        logger.info(f"   📊 Scale: {len(scenario.vehicles)} vehicles, {len(scenario.locations)} locations, {len(scenario.ride_requests)} requests")
    except Exception as e:
        logger.error(f"❌ Failed to load scenario: {e}")
        return
    
    # Create optimizer and test ONLY Level 1
    optimizer = VRPOptimizerEnhanced()
    
    # Run optimization with Level 1 only
    try:
        logger.info("\n🚀 Testing Level 1 with STRICT constraints...")
        result = optimizer._solve_with_constraint_level(scenario, "distance_time", 60)
        
        logger.info(f"\n🎯 LEVEL 1 RESULTS:")
        if result['success']:
            logger.info(f"✅ SUCCESS!")
            logger.info(f"   Objective: {result['objective_value']}")
            logger.info(f"   Fleet utilization: {result['vehicles_used']}/{len(scenario.vehicles)} ({result['vehicles_used']/len(scenario.vehicles)*100:.1f}%)")
            logger.info(f"   Vehicles used: {result['vehicles_used']}")
            
            if result['vehicles_used'] < len(scenario.vehicles) / 10:  # Less than 10% of fleet
                logger.warning(f"⚠️ POOR: Only using {result['vehicles_used']} vehicles - need better distribution")
            elif result['vehicles_used'] < len(scenario.vehicles) / 5:  # Less than 20% of fleet
                logger.warning(f"⚠️ MODERATE: Using {result['vehicles_used']} vehicles - could be better")
            else:
                logger.info(f"✅ GOOD: Using {result['vehicles_used']} vehicles")
                
            # Check for constraint violations
            violations = 0
            for analysis in result.get('route_analysis', []):
                if analysis['total_time'] > analysis['max_allowed_time']:
                    violations += 1
                    logger.error(f"🚨 VIOLATION: Vehicle {analysis['vehicle_id']} exceeds time limit!")
                if float(analysis['load_utilization'].replace('%', '')) > 100:
                    violations += 1
                    logger.error(f"🚨 VIOLATION: Vehicle {analysis['vehicle_id']} exceeds capacity!")
                    
            if violations == 0:
                logger.info("✅ NO CONSTRAINT VIOLATIONS!")
            else:
                logger.error(f"🚨 {violations} CONSTRAINT VIOLATIONS found!")
                
        else:
            logger.error(f"❌ FAILED: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"❌ EXCEPTION during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_level1_quick()
