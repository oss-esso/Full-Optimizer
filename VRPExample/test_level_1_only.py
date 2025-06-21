#!/usr/bin/env python3
"""
Quick test script for Level 1 (distance only) constraints to debug time calculation issues.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from VRPExample.vrp_scenarios import get_all_scenarios
from VRPExample.vrp_optimizer_enhanced_full import VRPOptimizerEnhanced

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_level_1_only():
    """Test only Level 1 (distance + time dimension) for faster debugging."""
    logger.info("=" * 80)
    logger.info("TESTING LEVEL 1 ONLY - DISTANCE + TIME DIMENSION")
    logger.info("=" * 80)
    
    # Get the scenario - Use MODA_first for real scale testing
    try:
        scenarios = get_all_scenarios()
        scenario = scenarios['MODA_first']  # Real scale scenario
        logger.info(f"✅ Successfully loaded scenario: {scenario.name}")
        logger.info(f"   📊 Scale: {len(scenario.vehicles)} vehicles, {len(scenario.locations)} locations, {len(scenario.ride_requests)} requests")
    except Exception as e:
        logger.error(f"❌ Failed to load scenario: {e}")
        return
    
    # Create optimizer and test only Level 1
    class TestLevel1OnlyOptimizer(VRPOptimizerEnhanced):
        def solve(self, instance, time_limit_seconds=300):
            """Override to test only Level 1."""
            logger.info("🔧 TESTING LEVEL 1 ONLY (DISTANCE + TIME DIMENSION)")
            logger.info("=" * 70)
            
            # RELAX TIME WINDOWS GLOBALLY - Set all time windows to start at 0
            logger.info("🔄 RELAXING TIME WINDOWS - Setting all start times to 0")
            relaxed_count = 0
            for location_id, location in instance.locations.items():
                if hasattr(location, 'time_window_start') and location.time_window_start is not None:
                    original_start = location.time_window_start
                    if original_start > 0:
                        location.time_window_start = 0  # Set to 0 for maximum flexibility
                        relaxed_count += 1
                        if relaxed_count <= 5:  # Only show first 5 for brevity
                            logger.info(f"   Relaxed {location_id}: [{original_start}-{location.time_window_end}] → [0-{location.time_window_end}]")
            
            if relaxed_count > 0:
                logger.info(f"✅ Relaxed {relaxed_count} time windows to start at 0")
            
            # Comprehensive sanity check (abbreviated for Level 1)
            self._print_level_1_sanity_check(instance)
            
            if not ORTOOLS_AVAILABLE:
                return {'success': False, 'error': 'OR-Tools not available'}
            
            # Test only Level 1: distance + time dimension
            result = self._solve_with_constraint_level(instance, "distance_only", time_limit_seconds)
            
            if result['success']:
                logger.info(f"✅ SUCCESS WITH LEVEL 1!")
                result['strategy_used'] = "LEVEL 1"
                result['constraint_level_used'] = "distance_only"
                return result
            else:
                logger.warning(f"❌ FAILED at LEVEL 1: {result.get('error', 'Unknown')}")
                return result
        
        def _print_level_1_sanity_check(self, instance):
            """Abbreviated sanity check for Level 1 testing."""
            logger.info("📊 LEVEL 1 SANITY CHECK")
            logger.info("-" * 40)
            
            # Basic counts
            num_locations = len(instance.locations)
            num_vehicles = len(instance.vehicles)
            num_requests = len(instance.ride_requests) if instance.ride_requests else 0
            
            logger.info(f"📍 Locations: {num_locations}")
            logger.info(f"🚛 Vehicles: {num_vehicles}")
            logger.info(f"📦 Requests: {num_requests}")
            
            # Vehicle time analysis
            logger.info("\n🚛 VEHICLE TIME LIMITS:")
            vehicle_max_times = []
            for i, (vid, vehicle) in enumerate(instance.vehicles.items()):
                max_time = (getattr(vehicle, 'max_total_work_time', None) or 
                           getattr(vehicle, 'max_time', None) or 
                           getattr(vehicle, 'max_work_time', None) or 600)
                max_time = int(max_time) if max_time is not None else 600
                vehicle_max_times.append(max_time)
                
                if i < 3:  # Show first 3 vehicles
                    logger.info(f"   {vid}: {max_time}min max time")
            
            min_time = min(vehicle_max_times)
            max_time = max(vehicle_max_times)
            avg_time = sum(vehicle_max_times) / len(vehicle_max_times)
            
            logger.info(f"   📊 Time limits: {min_time}-{max_time}min (avg: {avg_time:.0f}min)")
            
            # Service time analysis
            service_times = []
            for loc in instance.locations.values():
                service_time = getattr(loc, 'service_time', 0)
                if service_time > 0:
                    service_times.append(service_time)
            
            if service_times:
                logger.info(f"   📊 Service times: {min(service_times)}-{max(service_times)}min (avg: {sum(service_times)/len(service_times):.1f}min)")
                logger.info(f"   📊 Total service time if all locations visited: {sum(service_times)}min")
            
            logger.info("-" * 40)
    
    optimizer = TestLevel1OnlyOptimizer()
    
    # Run optimization with shorter time limit for quick testing
    try:
        logger.info("\n🚀 Starting Level 1 optimization...")
        result = optimizer.solve(scenario, time_limit_seconds=60)  # Quick 1-minute test
        
        logger.info(f"\n🎯 LEVEL 1 RESULTS:")
        if result['success']:
            logger.info(f"   ✅ SUCCESS!")
            logger.info(f"   Objective: {result['objective_value']}")
            logger.info(f"   Vehicles used: {result['vehicles_used']}/{len(scenario.vehicles)}")
            logger.info(f"   Solve time: {result['solve_time']:.2f}s")
            
            # Detailed route analysis
            if 'route_analysis' in result:
                logger.info(f"\n📊 DETAILED ROUTE ANALYSIS:")
                for analysis in result['route_analysis']:
                    vid = analysis['vehicle_id']
                    stops = analysis['stops']
                    total_time = analysis['total_time']
                    driving_time = analysis['driving_time']
                    service_time = analysis['service_time']
                    max_time = analysis['max_allowed_time']
                    distance = analysis['total_distance']
                    
                    logger.info(f"   Vehicle {vid}:")
                    logger.info(f"      🛣️  {stops} stops, {distance:.1f}km")
                    logger.info(f"      ⏱️  {total_time}min total ({total_time/max_time*100:.1f}% of {max_time}min limit)")
                    logger.info(f"      🚗 {driving_time}min driving + {service_time}min service = {driving_time + service_time}min calculated")
                    
                    # Check for time constraint violations
                    if total_time > max_time:
                        logger.error(f"      ❌ TIME VIOLATION: {total_time}min > {max_time}min limit!")
                    elif driving_time + service_time != total_time:
                        logger.warning(f"      ⚠️ TIME MISMATCH: calculated {driving_time + service_time}min ≠ OR-Tools {total_time}min")
                    else:
                        logger.info(f"      ✅ Time calculation correct")
        else:
            logger.error(f"   ❌ FAILED: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"❌ EXCEPTION during Level 1 test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_level_1_only()
