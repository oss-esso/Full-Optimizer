#!/usr/bin/env python3
"""
Test multiple constraint levels with HALF requests to verify adaptive fleet utilization.
This will show the difference between levels with and without capacity constraints.
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

def reduce_requests_by_half(scenario):
    """Reduce the number of requests by half to test scalability."""
    original_count = len(scenario.ride_requests)
    
    # Take only the first half of requests
    scenario.ride_requests = scenario.ride_requests[:original_count // 2]
    
    # Also need to remove locations that are no longer needed
    # Get the location IDs that are still referenced by remaining requests
    used_location_ids = set()
    used_location_ids.add('depot_asti')  # Always keep depot
    used_location_ids.add('depot_milan')  # Always keep depot
    
    for req in scenario.ride_requests:
        used_location_ids.add(req.pickup_location)
        used_location_ids.add(req.dropoff_location)
    
    # Filter locations to only those that are still used
    original_locations = dict(scenario.locations)
    scenario.locations = {loc_id: loc for loc_id, loc in original_locations.items() 
                         if loc_id in used_location_ids}
    
    logger.info(f"📉 REDUCED PROBLEM SIZE:")
    logger.info(f"   Requests: {original_count} → {len(scenario.ride_requests)} ({len(scenario.ride_requests)/original_count*100:.1f}%)")
    logger.info(f"   Locations: {len(original_locations)} → {len(scenario.locations)} ({len(scenario.locations)/len(original_locations)*100:.1f}%)")
    logger.info(f"   Vehicles: {len(scenario.vehicles)} (unchanged)")
    
    return scenario

def test_multiple_levels():
    """Test multiple constraint levels to show adaptive fleet utilization."""
    logger.info("=" * 80)
    logger.info("MULTI-LEVEL TEST: ADAPTIVE FLEET UTILIZATION")
    logger.info("=" * 80)
    
    # Get the scenario - Use MODA_first for real scale testing
    try:
        scenarios = get_all_scenarios()
        scenario = scenarios['MODA_first']
        logger.info(f"✅ Successfully loaded scenario: {scenario.name}")
        logger.info(f"   📊 Original scale: {len(scenario.vehicles)} vehicles, {len(scenario.locations)} locations, {len(scenario.ride_requests)} requests")
        
        # Reduce problem size
        scenario = reduce_requests_by_half(scenario)
        
    except Exception as e:
        logger.error(f"❌ Failed to load scenario: {e}")
        return
    
    # Create optimizer
    optimizer = VRPOptimizerEnhanced()
    
    # Test specific levels to show the difference
    levels_to_test = [
        ("LEVEL 1", "distance_time", "Distance + Time + Service time (aggressive stop limits)"),
        ("LEVEL 4", "capacity", "Distance + Time + Service time + Capacity (capacity-driven)")
    ]
    
    results_summary = []
    
    for level_name, level_code, level_desc in levels_to_test:
        logger.info(f"\n🎯 TESTING {level_name}: {level_desc}")
        logger.info("-" * 60)
        
        try:
            result = optimizer._solve_with_constraint_level(scenario, level_code, 120)
            
            if result['success']:
                logger.info(f"✅ SUCCESS WITH {level_name}!")
                
                # Analyze fleet utilization
                vehicles_used = result['vehicles_used']
                fleet_utilization = vehicles_used / len(scenario.vehicles) * 100
                
                # Analyze capacity violations
                capacity_violations = 0
                time_violations = 0
                
                if 'route_analysis' in result:
                    for analysis in result['route_analysis']:
                        max_load = analysis['max_load_reached']
                        capacity = analysis['capacity']
                        total_time = analysis['total_time']
                        max_time = analysis['max_allowed_time']
                        
                        if max_load > capacity:
                            capacity_violations += 1
                        if total_time > max_time:
                            time_violations += 1
                
                summary = {
                    'level': level_name,
                    'success': True,
                    'vehicles_used': vehicles_used,
                    'fleet_utilization': f"{fleet_utilization:.1f}%",
                    'capacity_violations': capacity_violations,
                    'time_violations': time_violations,
                    'objective': result['objective_value'],
                    'constraints': result['constraints_applied']
                }
                
                logger.info(f"   📊 Fleet utilization: {vehicles_used}/{len(scenario.vehicles)} ({fleet_utilization:.1f}%)")
                logger.info(f"   📊 Capacity violations: {capacity_violations}")
                logger.info(f"   📊 Time violations: {time_violations}")
                logger.info(f"   📊 Objective: {result['objective_value']}")
                
                # Log fleet utilization strategy used
                if "aggressive_fleet_utilization" in result['constraints_applied']:
                    logger.info("   🚛 Used AGGRESSIVE fleet utilization (stop limits)")
                elif "moderate_fleet_utilization" in result['constraints_applied']:
                    logger.info("   🚛 Used MODERATE fleet utilization (capacity-driven)")
                
            else:
                logger.error(f"❌ FAILED: {result.get('error', 'Unknown error')}")
                summary = {
                    'level': level_name,
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                }
            
            results_summary.append(summary)
            
        except Exception as e:
            logger.error(f"❌ Error with {level_name}: {e}")
            results_summary.append({
                'level': level_name,
                'success': False,
                'error': str(e)
            })
    
    # Final summary
    logger.info(f"\n🎯 FINAL COMPARISON:")
    logger.info("=" * 60)
    
    for summary in results_summary:
        if summary['success']:
            logger.info(f"{summary['level']}:")
            logger.info(f"   Fleet utilization: {summary['fleet_utilization']}")
            logger.info(f"   Capacity violations: {summary['capacity_violations']}")
            logger.info(f"   Time violations: {summary['time_violations']}")
            
            # Show the fleet utilization strategy
            if "aggressive_fleet_utilization" in summary['constraints']:
                logger.info("   Strategy: AGGRESSIVE (artificial stop limits)")
            elif "moderate_fleet_utilization" in summary['constraints']:
                logger.info("   Strategy: MODERATE (capacity naturally limits stops)")
            
        else:
            logger.error(f"{summary['level']}: FAILED - {summary.get('error', 'Unknown')}")
    
    logger.info("\n💡 Expected behavior:")
    logger.info("   - Level 1: Aggressive stop limits (6 stops/vehicle) → many vehicles, capacity violations")
    logger.info("   - Level 4: Moderate fixed costs, capacity limits stops naturally → fewer vehicles, no capacity violations")

if __name__ == "__main__":
    test_multiple_levels()
