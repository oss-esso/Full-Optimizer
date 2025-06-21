#!/usr/bin/env python3
"""Test script to isolate Level 2 (time) issues"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vrp_optimizer_enhanced_full import *
from vrp_scenarios import get_all_scenarios
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_level_2_only():
    """Test only Level 2 with MODA_first scenario"""
    print("="*80)
    print("LEVEL 2 ISOLATED TEST")
    print("="*80)
    
    # Load MODA_first scenario
    scenarios = get_all_scenarios()
    scenario = scenarios["MODA_first"]
    
    logger.info(f"✅ Successfully loaded scenario: {scenario.name}")
    logger.info(f"   📊 Scale: {len(scenario.vehicles)} vehicles, {len(scenario.locations)} locations, {len(scenario.ride_requests)} requests")
    
    print("\n🚀 Testing Level 2 (time dimension) with detailed logging...")
    
    # Test Level 2 only
    optimizer = VRPOptimizerEnhanced()
    solution = optimizer.optimize_progressively(
        scenario, 
        max_levels=["time"],  # Only test Level 2
        time_limit_per_level=60  # 1 minute per level
    )
    
    if solution:
        logger.info(f"✅ LEVEL 2 SUCCESS!")
        logger.info(f"   Objective: {solution['objective']}")
        logger.info(f"   Vehicles used: {solution['vehicles_used']}")
        logger.info(f"   Fleet utilization: {solution['vehicles_used']}/{len(scenario.vehicles)} ({solution['vehicles_used']/len(scenario.vehicles)*100:.1f}%)")
    else:
        logger.error(f"❌ LEVEL 2 FAILED - No solution found")
    
    return solution

if __name__ == "__main__":
    test_level_2_only()
