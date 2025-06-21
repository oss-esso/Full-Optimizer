#!/usr/bin/env python3
"""
Simple test to check if time callback is being called correctly
"""

import logging
from vrp_optimizer_enhanced_full import VRPOptimizerEnhanced
from vrp_scenarios import create_moda_small_scenario

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_time_callback():
    """Test if time callback is working correctly."""
    logger.info("🧪 TESTING TIME CALLBACK FUNCTIONALITY")
    logger.info("=" * 50)
    
    scenario = create_moda_small_scenario()
    optimizer = VRPOptimizerEnhanced()
    
    # Add debug logging to understand what's happening
    logger.info("📊 Travel time matrix sample:")
    if hasattr(optimizer, 'travel_time_matrix') and optimizer.travel_time_matrix:
        for i in range(min(3, len(optimizer.travel_time_matrix))):
            row = optimizer.travel_time_matrix[i][:5]  # First 5 elements
            logger.info(f"  Row {i}: {row}")
    else:
        logger.info("  Travel time matrix not yet created")
    
    # Test the optimizer with a simple configuration
    try:
        logger.info("🚀 Testing simple optimization...")
        result = optimizer.solve(scenario, max_time_seconds=30)
        
        if result.success:
            logger.info("✅ Optimization successful!")
            logger.info(f"   Objective: {result.objective_value}")
            logger.info(f"   Total time: {result.total_time}")
            logger.info(f"   Total distance: {result.total_distance}")
            
            # Check individual routes
            for i, route in enumerate(result.routes):
                logger.info(f"   Route {i}: {len(route)} stops")
                if len(route) > 2:  # More than just start-end
                    logger.info(f"     First few stops: {route[:min(5, len(route))]}")
        else:
            logger.error("❌ Optimization failed!")
            logger.error(f"   Error: {result.status}")
            
    except Exception as e:
        logger.error(f"❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_time_callback()
