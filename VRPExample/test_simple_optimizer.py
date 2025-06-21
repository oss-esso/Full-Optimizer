#!/usr/bin/env python3
"""
Simple test for the rolling window optimizer with extended time limits.
"""

import sys
import os
import logging

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vrp_scenarios import create_moda_small_scenario, create_moda_first_scenario
from vrp_optimizer_rolling_window import VRPOptimizerRollingWindow
from vrp_data_models import VRPObjective

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_scenario(scenario_name: str, scenario_func, time_limits: list):
    """Test a scenario with different time limits."""
    logger.info(f"\n{'='*80}")
    logger.info(f"TESTING {scenario_name}")
    logger.info(f"{'='*80}")
    
    # Create scenario
    logger.info(f"Creating {scenario_name} scenario...")
    instance = scenario_func()
    
    # Print scenario info
    logger.info(f"Scenario info:")
    logger.info(f"  Locations: {len(instance.location_ids)}")
    logger.info(f"  Vehicles: {len(instance.vehicles)}")
    logger.info(f"  Requests: {len(instance.ride_requests)}")
    
    # Test with different time limits
    for time_limit in time_limits:
        logger.info(f"\n🕒 Testing with {time_limit} second time limit...")
        
        try:
            optimizer = VRPOptimizerRollingWindow(instance, VRPObjective.MINIMIZE_DISTANCE)
            
            result = optimizer.optimize_with_rolling_window(
                timeout_duration=time_limit + 30,  # Buffer
                time_limit_seconds=time_limit
            )
            
            logger.info(f"  Status: {result.status}")
            logger.info(f"  Objective: {result.objective_value}")
            logger.info(f"  Runtime: {result.runtime:.1f}ms")
            
            if result.status == "optimal" and result.objective_value < float('inf'):
                logger.info(f"  ✅ SOLUTION FOUND!")
                logger.info(f"  Routes: {len(result.routes)}")
                logger.info(f"  Distance: {result.metrics.get('total_distance', 0):.1f}")
                return True
            else:
                logger.info(f"  ❌ No solution found")
                
        except Exception as e:
            logger.error(f"  💥 Error: {str(e)}")
    
    return False

def main():
    """Run simple tests."""
    logger.info("SIMPLE VRP OPTIMIZER TESTS WITH EXTENDED TIME LIMITS")
    logger.info("="*80)
    
    # Progressive time limits (seconds)
    time_limits = [30, 60, 120, 300, 600]  # 30s, 1m, 2m, 5m, 10m
    
    # Test MODA_small first (should be easier)
    moda_small_solved = test_scenario("MODA_small", create_moda_small_scenario, time_limits)
    
    # Test MODA_first (more challenging)
    moda_first_solved = test_scenario("MODA_first", create_moda_first_scenario, time_limits)
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"MODA_small: {'✅ SOLVED' if moda_small_solved else '❌ NOT SOLVED'}")
    logger.info(f"MODA_first: {'✅ SOLVED' if moda_first_solved else '❌ NOT SOLVED'}")
    
    if not moda_small_solved:
        logger.warning("Even MODA_small could not be solved - there may be a fundamental issue")
    
    if not moda_first_solved:
        logger.info("MODA_first remains unsolved - likely due to complexity or modeling constraints")
    
    logger.info("Simple test completed!")

if __name__ == "__main__":
    main()
