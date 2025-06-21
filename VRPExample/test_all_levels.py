#!/usr/bin/env python3
"""
Comprehensive test of all VRP constraint levels with realistic travel times.

This script tests all 5 constraint levels:
- Level 1: Distance + Time + Service time (no time windows, no capacity)
- Level 2: Distance + Time + Service time (with better search)
- Level 3: Distance + Time + Service time + Time windows
- Level 4: Distance + Time + Service time + Time windows + Capacity
- Level 5: Distance + Time + Service time + Time windows + Capacity + Pickup-delivery

Each level uses the hybrid travel time calculator for realistic road-based estimates.
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

def test_all_constraint_levels():
    """Test all 5 constraint levels with realistic travel times."""
    logger.info("🚀 COMPREHENSIVE CONSTRAINT LEVEL TESTING WITH REALISTIC TRAVEL TIMES")
    logger.info("=" * 80)
    
    # Load scenario
    instance = create_moda_small_scenario()
    logger.info(f"✅ Loaded scenario: {len(instance.locations)} locations, {len(instance.vehicles)} vehicles")
    
    # Create optimizer
    optimizer = VRPOptimizerEnhanced()
    
    # Test each level individually
    levels_to_test = [
        ("Level 1", "distance_time", "Distance + Time + Service time (no time windows, no capacity)"),
        ("Level 2", "time_dimension", "Distance + Time + Service time (with better search)"),
        ("Level 3", "time_windows", "Distance + Time + Service time + Time windows"),
        ("Level 4", "capacity", "Distance + Time + Service time + Time windows + Capacity"),
        ("Level 5", "pickup_delivery", "Distance + Time + Service time + Time windows + Capacity + Pickup-delivery")
    ]
    
    results = {}
    
    for level_name, level_code, description in levels_to_test:
        logger.info(f"\n" + "=" * 60)
        logger.info(f"🎯 TESTING {level_name.upper()}: {description}")
        logger.info("=" * 60)
        
        try:
            # Test this specific level with generous time limit
            result = optimizer._solve_with_constraint_level(instance, level_code, 300)  # 5 minutes per level
            
            if result['success']:
                logger.info(f"✅ {level_name} SUCCESS!")
                logger.info(f"   🎯 Objective: {result['objective_value']}")
                logger.info(f"   🚛 Vehicles used: {result['vehicles_used']}/{len(instance.vehicles)}")
                logger.info(f"   ⏱️  Solve time: {result['solve_time']:.2f}s")
                logger.info(f"   🔧 Constraints: {result.get('constraints_applied', [])}")
                
                # Detailed route analysis
                if 'route_analysis' in result:
                    logger.info(f"\n📊 DETAILED ROUTE ANALYSIS:")
                    total_realistic_driving = 0
                    total_service_time = 0
                    total_distance = 0
                    
                    for analysis in result['route_analysis']:
                        vehicle_id = analysis['vehicle_id']
                        stops = analysis['stops']
                        total_time = analysis['total_time']
                        max_time = analysis['max_allowed_time']
                        driving_time = analysis.get('driving_time', 0)
                        service_time = analysis.get('service_time', 0)
                        distance = analysis.get('distance', 0)
                        max_load = analysis.get('max_load_reached', 0)
                        capacity = analysis.get('capacity', 0)
                        
                        total_realistic_driving += driving_time
                        total_service_time += service_time
                        total_distance += distance
                        
                        # Time utilization analysis
                        time_efficiency = total_time / max_time * 100
                        if time_efficiency > 90:
                            time_status = "🔴 HIGH utilization"
                        elif time_efficiency > 70:
                            time_status = "🟡 GOOD utilization"
                        else:
                            time_status = "🟢 LOW utilization"
                        
                        # Capacity utilization analysis
                        if capacity > 0:
                            load_efficiency = max_load / capacity * 100
                            if load_efficiency > 90:
                                load_status = "🔴 HIGH load"
                            elif load_efficiency > 70:
                                load_status = "🟡 GOOD load"
                            else:
                                load_status = "🟢 LOW load"
                        else:
                            load_status = "N/A"
                        
                        logger.info(f"   🚛 {vehicle_id}: {stops} stops, {total_time}min ({time_efficiency:.1f}%) - {time_status}")
                        logger.info(f"      🚗 Realistic driving: {driving_time}min, Service: {service_time}min")
                        logger.info(f"      📏 Distance: {distance:.2f}km")
                        if capacity > 0:
                            logger.info(f"      📦 Load: {max_load}kg/{capacity}kg ({load_efficiency:.1f}%) - {load_status}")
                    
                    # Summary statistics
                    logger.info(f"\n📈 SUMMARY STATISTICS:")
                    logger.info(f"   🚗 Total realistic driving time: {total_realistic_driving} minutes")
                    logger.info(f"   🏢 Total service time: {total_service_time} minutes")
                    logger.info(f"   ⏱️  Total working time: {total_realistic_driving + total_service_time} minutes")
                    logger.info(f"   📏 Total distance: {total_distance:.2f} km")
                    
                    if total_distance > 0:
                        avg_speed = total_distance / (total_realistic_driving / 60) if total_realistic_driving > 0 else 0
                        logger.info(f"   🏎️  Average speed: {avg_speed:.1f} km/h")
                    
                    # Store results for comparison
                    results[level_name] = {
                        'success': True,
                        'objective': result['objective_value'],
                        'vehicles_used': result['vehicles_used'],
                        'solve_time': result['solve_time'],
                        'total_driving_time': total_realistic_driving,
                        'total_service_time': total_service_time,
                        'total_distance': total_distance,
                        'constraints': result.get('constraints_applied', [])
                    }
                
            else:
                logger.error(f"❌ {level_name} FAILED: {result.get('error', 'Unknown error')}")
                results[level_name] = {
                    'success': False,
                    'error': result.get('error', 'Unknown error'),
                    'solve_time': result.get('solve_time', 0)
                }
                
                # Provide specific guidance for failures
                if level_code == "time_windows":
                    logger.info("💡 Time windows might be too restrictive. Consider:")
                    logger.info("   - Relaxing time window constraints")
                    logger.info("   - Increasing vehicle work hours")
                    logger.info("   - Adding more vehicles")
                elif level_code == "capacity":
                    logger.info("💡 Capacity constraints might be too restrictive. Consider:")
                    logger.info("   - Using vehicles with higher capacity")
                    logger.info("   - Adding more vehicles")
                    logger.info("   - Reducing request sizes")
                elif level_code == "pickup_delivery":
                    logger.info("💡 Pickup-delivery constraints are very strict. Consider:")
                    logger.info("   - Relaxing time windows further")
                    logger.info("   - Increasing fleet size")
                    logger.info("   - Allowing longer routes")
                
        except Exception as e:
            logger.error(f"❌ {level_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[level_name] = {
                'success': False,
                'error': str(e),
                'solve_time': 0
            }
    
    # Generate comprehensive comparison report
    print_comprehensive_comparison(results, levels_to_test)
    
    return results

def print_comprehensive_comparison(results, levels_tested):
    """Print a comprehensive comparison of all tested levels."""
    print("\n" + "=" * 100)
    print(" COMPREHENSIVE CONSTRAINT LEVEL COMPARISON - WITH REALISTIC TRAVEL TIMES")
    print("=" * 100)
    
    # Success rate analysis
    successful_levels = [level for level, result in results.items() if result.get('success', False)]
    failed_levels = [level for level, result in results.items() if not result.get('success', False)]
    
    print(f"\n📊 SUCCESS RATE: {len(successful_levels)}/{len(results)} levels passed")
    print(f"✅ Successful: {', '.join(successful_levels) if successful_levels else 'None'}")
    print(f"❌ Failed: {', '.join(failed_levels) if failed_levels else 'None'}")
    
    if successful_levels:
        print(f"\n📈 PERFORMANCE COMPARISON (Successful Levels Only):")
        print("-" * 100)
        print(f"{'Level':<15} {'Objective':<12} {'Vehicles':<10} {'Solve Time':<12} {'Driving':<10} {'Distance':<10} {'Avg Speed':<12}")
        print("-" * 100)
        
        for level_name in successful_levels:
            result = results[level_name]
            obj = result.get('objective', 0)
            vehicles = result.get('vehicles_used', 0)
            solve_time = result.get('solve_time', 0)
            driving_time = result.get('total_driving_time', 0)
            distance = result.get('total_distance', 0)
            
            # Calculate average speed
            avg_speed = distance / (driving_time / 60) if driving_time > 0 and distance > 0 else 0
            
            print(f"{level_name:<15} {obj:<12.0f} {vehicles:<10} {solve_time:<12.2f}s {driving_time:<10}min {distance:<10.2f}km {avg_speed:<12.1f}km/h")
        
        # Find the best performing level
        best_level = min(successful_levels, key=lambda x: results[x].get('objective', float('inf')))
        fastest_level = min(successful_levels, key=lambda x: results[x].get('solve_time', float('inf')))
        most_efficient_level = min(successful_levels, key=lambda x: results[x].get('vehicles_used', float('inf')))
        
        print(f"\n� WINNERS:")
        print(f"   🎯 Best objective: {best_level} (objective: {results[best_level].get('objective', 0):.0f})")
        print(f"   ⚡ Fastest solve: {fastest_level} ({results[fastest_level].get('solve_time', 0):.2f}s)")
        print(f"   🚛 Most efficient: {most_efficient_level} ({results[most_efficient_level].get('vehicles_used', 0)} vehicles)")
        
        # Realistic travel time analysis
        print(f"\n🌍 REALISTIC TRAVEL TIME ANALYSIS:")
        for level_name in successful_levels:
            result = results[level_name]
            driving_time = result.get('total_driving_time', 0)
            service_time = result.get('total_service_time', 0)
            distance = result.get('total_distance', 0)
            
            if driving_time > 0 and distance > 0:
                print(f"   {level_name}: {driving_time}min driving for {distance:.2f}km = {distance/(driving_time/60):.1f}km/h avg speed")
    
    if failed_levels:
        print(f"\n❌ FAILURE ANALYSIS:")
        for level_name in failed_levels:
            result = results[level_name]
            error = result.get('error', 'Unknown error')
            solve_time = result.get('solve_time', 0)
            print(f"   {level_name}: {error} (attempted for {solve_time:.2f}s)")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if len(successful_levels) == len(levels_tested):
        print("   🎉 All constraint levels working perfectly with realistic travel times!")
        print("   ✅ The VRP optimizer handles all constraint combinations successfully")
        print("   🌍 Realistic travel times (2.48x correction factor) are properly integrated")
    elif len(successful_levels) >= 3:
        print("   ✅ Most constraint levels working well")
        print("   🔧 Consider tuning parameters for failed levels")
        print("   🌍 Realistic travel times working correctly for successful levels")
    else:
        print("   ⚠️  Multiple constraint levels failing")
        print("   🔧 Review scenario parameters (time windows, capacities, fleet size)")
        print("   ⏱️  Consider increasing solve time limits")
        print("   🌍 Realistic travel times may be making constraints tighter")
    
    print("=" * 100)

if __name__ == "__main__":
    test_all_constraint_levels()
