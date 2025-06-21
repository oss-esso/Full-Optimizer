#!/usr/bin/env python3
"""
Quick test for fleet utilization improvements.
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

def test_fleet_utilization():
    """Test fleet utilization with MODA_first scenario."""
    logger.info("=" * 80)
    logger.info("TESTING FLEET UTILIZATION IMPROVEMENTS")
    logger.info("=" * 80)
    
    try:
        # Get the scenario
        scenarios = get_all_scenarios()
        scenario = scenarios['MODA_first']
        logger.info(f"✅ Successfully loaded scenario: {scenario.name}")
        logger.info(f"   📊 Scale: {len(scenario.vehicles)} vehicles, {len(scenario.locations)} locations, {len(scenario.ride_requests)} requests")
    except Exception as e:
        logger.error(f"❌ Failed to load scenario: {e}")
        return
    
    # Create optimizer and test only Level 1 (distance_only) for fleet utilization
    optimizer = VRPOptimizerEnhanced()
    
    try:
        logger.info("\n🚀 Testing Level 1 with fleet utilization improvements...")
        result = optimizer._solve_with_constraint_level(scenario, "distance_only", 120)
        
        if result['success']:
            logger.info(f"\n🎯 FLEET UTILIZATION RESULTS:")
            logger.info(f"✅ SUCCESS!")
            logger.info(f"   Objective: {result['objective_value']}")
            logger.info(f"   Fleet utilization: {result['vehicles_used']}/{len(scenario.vehicles)} ({result['vehicles_used']/len(scenario.vehicles)*100:.1f}%)")
            logger.info(f"   Vehicles used: {result['vehicles_used']}")
            
            if result['vehicles_used'] < len(scenario.vehicles) // 10:  # Less than 10% of fleet
                logger.warning(f"⚠️ POOR: Only using {result['vehicles_used']} vehicles - need better distribution")
            elif result['vehicles_used'] < len(scenario.vehicles) // 2:  # Less than 50% of fleet
                logger.info(f"🔶 MODERATE: Using {result['vehicles_used']} vehicles - could be better")
            else:
                logger.info(f"✅ GOOD: Using {result['vehicles_used']} vehicles - well distributed")
                
        else:
            logger.error(f"❌ FAILED: {result.get('error', 'Unknown')}")
            
    except Exception as e:
        logger.error(f"❌ EXCEPTION during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fleet_utilization()
