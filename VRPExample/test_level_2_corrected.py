#!/usr/bin/env python3
"""
Test script for Level 2 (time windows) VRP optimization.
This tests time dimension AND time window constraints.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vrp_optimizer_enhanced_full import VRPOptimizerEnhanced
from vrp_scenarios import load_scenarios
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_level_2_corrected():
    """Test Level 2 constraints with correct method calls."""
    print("🚀 Testing Level 2 (time windows) with corrected method...")
    
    # Load scenarios
    scenarios = load_scenarios()
    scenario_name = "MODA_first"
    
    if scenario_name not in scenarios:
        raise ValueError(f"Scenario {scenario_name} not found")
    
    instance = scenarios[scenario_name]
    logger.info(f"✅ Successfully loaded scenario: {scenario_name}")
    logger.info(f"   📊 Scale: {len(instance.vehicles)} vehicles, {len(instance.locations)} locations, {len(instance.requests)} requests")
    logger.info("")
    
    # Create optimizer
    optimizer = VRPOptimizerEnhanced()
    
    # Test Level 2: time windows
    level = "time_windows"
    logger.info(f"🚀 Testing Level 2 ({level}) with detailed logging...")
    
    try:
        # Use the correct method name
        solution = optimizer.optimize(
            instance=instance,
            level=level,
            time_limit=60
        )
        
        if solution:
            logger.info("✅ LEVEL 2 SUCCESS!")
            logger.info(f"   Objective: {solution['objective']}")
            logger.info(f"   Time: {solution.get('solve_time', 'N/A')}s")
            logger.info(f"   Fleet utilization: {solution.get('vehicles_used', 'N/A')}/{len(instance.vehicles)}")
            
            # Analyze time windows compliance
            total_violations = 0
            if 'route_analysis' in solution:
                for analysis in solution['route_analysis']:
                    violations = analysis.get('time_window_violations', 0)
                    if violations > 0:
                        logger.error(f"🚨 Vehicle {analysis['vehicle_id']}: {violations} time window violations")
                        total_violations += violations
                        
            if total_violations == 0:
                logger.info("✅ All time windows respected!")
            else:
                logger.error(f"🚨 Total time window violations: {total_violations}")
                
        else:
            logger.error("❌ LEVEL 2 FAILED - No solution found")
            
    except Exception as e:
        logger.error(f"❌ LEVEL 2 ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_level_2_corrected()
