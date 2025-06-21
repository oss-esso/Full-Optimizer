#!/usr/bin/env python3
"""
Test the updated VRP optimizer with real road-based travel times using OSRM.

This script tests the MODA_small scenario with the enhanced optimizer that uses
OSRM for realistic travel time calculations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vrp_scenarios import create_moda_small_scenario
from vrp_optimizer_enhanced_full import VRPOptimizerEnhanced
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_osrm_optimizer():
    """Test the VRP optimizer with OSRM-based travel times."""
    logger.info("🌍 TESTING VRP OPTIMIZER WITH REAL ROAD-BASED TRAVEL TIMES")
    logger.info("=" * 70)
    
    # Load scenario
    instance = create_moda_small_scenario()
    logger.info(f"Loaded scenario: {len(instance.locations)} locations, {len(instance.vehicles)} vehicles")
    
    # Initialize enhanced optimizer
    optimizer = VRPOptimizerEnhanced()
    
    # Solve with reasonable time limit
    result = optimizer.solve(instance, time_limit_seconds=120)
    
    # Print summary
    if result and 'best_solution' in result:
        solution = result['best_solution']
        logger.info("\n" + "=" * 50)
        logger.info("SOLUTION SUMMARY WITH REAL ROAD TRAVEL TIMES")
        logger.info("=" * 50)
        
        if 'routes' in solution:
            total_driving_time = 0
            total_service_time = 0
            total_distance = 0
            
            for route_idx, route in enumerate(solution['routes']):
                route_driving = route.get('total_driving_time', 0)
                route_service = route.get('total_service_time', 0)
                route_distance = route.get('total_distance', 0)
                
                total_driving_time += route_driving
                total_service_time += route_service
                total_distance += route_distance
                
                logger.info(f"Route {route_idx + 1}: {len(route.get('stops', []))} stops")
                logger.info(f"  Driving time: {route_driving} min")
                logger.info(f"  Service time: {route_service} min")
                logger.info(f"  Total time: {route_driving + route_service} min")
                logger.info(f"  Distance: {route_distance/1000:.1f} km")
                
                if 'stops' in route:
                    stop_names = [stop.get('location_id', 'Unknown') for stop in route['stops']]
                    logger.info(f"  Route: {' → '.join(stop_names)}")
                logger.info("")
            
            logger.info(f"TOTAL SUMMARY:")
            logger.info(f"  Total driving time: {total_driving_time} minutes")
            logger.info(f"  Total service time: {total_service_time} minutes")
            logger.info(f"  Total working time: {total_driving_time + total_service_time} minutes")
            logger.info(f"  Total distance: {total_distance/1000:.1f} km")
            
        # Compare with the result status
        if 'level_results' in result:
            logger.info(f"\nOptimization completed at level: {result.get('successful_level', 'Unknown')}")
            logger.info(f"Solution quality: {result.get('solution_quality', 'Unknown')}")
        
    else:
        logger.error("❌ No solution found!")
    
    return result

if __name__ == "__main__":
    test_osrm_optimizer()
