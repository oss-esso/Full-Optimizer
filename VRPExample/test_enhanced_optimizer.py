#!/usr/bin/env python3
"""
Test the enhanced optimizer to see how it performs with progressive constraint addition.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from VRPExample.vrp_scenarios import get_all_scenarios
from VRPExample.vrp_optimizer_enhanced_full import VRPOptimizerEnhanced
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_optimizer():
    """Test the enhanced optimizer with MODA_small scenario."""
    logger.info("=" * 80)
    logger.info("TESTING ENHANCED OPTIMIZER WITH PROGRESSIVE CONSTRAINTS")
    logger.info("=" * 80)
      # Get the scenario
    try:
        scenarios = get_all_scenarios()
        scenario = scenarios['MODA_small']
        logger.info(f"✅ Successfully loaded scenario: {scenario.name}")
        logger.info(f"   Vehicles: {len(scenario.vehicles)}")
        logger.info(f"   Locations: {len(scenario.locations)}")
        logger.info(f"   Ride requests: {len(scenario.ride_requests)}")
    except Exception as e:
        logger.error(f"❌ Failed to load scenario: {e}")
        return
    
    # Create optimizer
    optimizer = VRPOptimizerEnhanced()
      # Run optimization
    try:
        logger.info("\n🚀 Starting optimization with progressive constraint addition...")
        result = optimizer.solve(scenario)
        
        if result.get('success', False):
            logger.info(f"✅ SUCCESS! Solution found using strategy: {result.get('strategy_used', 'unknown')}")
            logger.info(f"   Total cost: {result.get('objective_value', 'unknown')}")
            logger.info(f"   Routes found: {len(result.get('routes', {}))}")
            
            # Print route details if available
            routes = result.get('routes', {})
            for vehicle_name, route in routes.items():
                logger.info(f"   {vehicle_name}: {' -> '.join(map(str, route))}")
                
        else:
            logger.error("❌ FAILED: No solution found")
            logger.error(f"   Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"❌ EXCEPTION during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_optimizer()
