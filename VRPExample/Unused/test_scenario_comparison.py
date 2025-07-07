#!/usr/bin/env python3
"""
Test script to compare old vs new scenarios to identify the MODA_first issue.

This script tests:
1. Original scenarios with uniform capacity
2. MODA scenarios 
3. Simplified versions to isolate the root cause

The goal is to understand why MODA_first is infeasible while older scenarios work.
"""

import sys
import os
import time
import logging
from typing import Dict, Any

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vrp_scenarios import (
    VRPScenarioGenerator,
    create_moda_small_scenario,
    create_moda_first_scenario
)
from vrp_optimizer_rolling_window import VRPOptimizerRollingWindow
from vrp_data_models import VRPObjective

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def uniformize_vehicle_capacity(instance, uniform_capacity: int = 1000):
    """Set all vehicles to have the same capacity."""
    for vehicle in instance.vehicles.values():
        vehicle.capacity = uniform_capacity
    logger.info(f"Set all {len(instance.vehicles)} vehicles to uniform capacity: {uniform_capacity}")

def analyze_scenario(scenario_name: str, instance) -> Dict[str, Any]:
    """Analyze scenario characteristics."""
    analysis = {
        'scenario_name': scenario_name,
        'num_locations': len(instance.location_ids),
        'num_vehicles': len(instance.vehicles),
        'num_requests': len(instance.ride_requests),
        'has_time_windows': any(r.earliest_pickup is not None and r.latest_dropoff is not None for r in instance.ride_requests),
        'has_service_times': any(hasattr(r, 'service_time') and r.service_time > 0 for r in instance.ride_requests),
        'has_capacity_constraints': any(v.capacity > 0 for v in instance.vehicles.values()),
        'vehicle_capacities': [v.capacity for v in instance.vehicles.values()],
        'total_demand': sum(getattr(r, 'demand', 0) for r in instance.ride_requests)
    }
    
    # Calculate time window statistics if they exist
    if analysis['has_time_windows']:
        time_windows = []
        for request in instance.ride_requests:
            if request.earliest_pickup is not None and request.latest_dropoff is not None:
                window_size = request.latest_dropoff - request.earliest_pickup
                time_windows.append(window_size)
        
        if time_windows:
            analysis['avg_time_window_hours'] = sum(time_windows) / len(time_windows) / 60
            analysis['min_time_window_hours'] = min(time_windows) / 60
            analysis['max_time_window_hours'] = max(time_windows) / 60
    
    return analysis

def test_scenario(scenario_name: str, create_scenario_func, uniform_capacity: int = None):
    """Test a single scenario with optional capacity uniformization."""
    logger.info(f"\n{'='*80}")
    logger.info(f"TESTING SCENARIO: {scenario_name}")
    logger.info(f"{'='*80}")
    
    try:
        # Create scenario
        logger.info(f"Creating {scenario_name}...")
        instance = create_scenario_func()
        
        # Uniformize capacity if requested
        if uniform_capacity:
            uniformize_vehicle_capacity(instance, uniform_capacity)
        
        # Analyze scenario
        analysis = analyze_scenario(scenario_name, instance)
        logger.info(f"Scenario analysis:")
        for key, value in analysis.items():
            if key != 'vehicle_capacities':
                logger.info(f"  {key}: {value}")
        
        # Test with rolling window optimizer
        optimizer = VRPOptimizerRollingWindow(instance, VRPObjective.MINIMIZE_DISTANCE)
        
        # Test with 60 second time limit
        logger.info(f"\n🕒 Testing {scenario_name} with 60s time limit...")
        start_time = time.time()
        
        result = optimizer.optimize_with_rolling_window(
            timeout_duration=90,  # Buffer for timeout
            time_limit_seconds=60
        )
        
        elapsed_time = time.time() - start_time
        
        # Report results
        logger.info(f"Results for {scenario_name}:")
        logger.info(f"  Status: {result.status}")
        logger.info(f"  Objective: {result.objective_value}")
        logger.info(f"  Runtime: {elapsed_time:.1f}s")
        
        if result.status == 'optimal' and result.objective_value < float('inf'):
            logger.info(f"  ✅ SOLUTION FOUND!")
            logger.info(f"  Routes: {len(result.routes)}")
            if result.metrics and 'total_distance' in result.metrics:
                logger.info(f"  Distance: {result.metrics['total_distance']:.1f}")
            return True, analysis
        else:
            logger.info(f"  ❌ NO SOLUTION FOUND")
            return False, analysis
            
    except Exception as e:
        logger.error(f"Failed to test {scenario_name}: {str(e)}")
        return False, {'error': str(e)}

def main():
    """Test various scenarios to isolate the MODA_first issue."""
    logger.info("="*100)
    logger.info("SCENARIO COMPARISON TEST - ISOLATING MODA_FIRST ISSUE")
    logger.info("="*100)
    logger.info("Testing original scenarios vs MODA scenarios to find the root cause")
    
    results = {}
      # Test scenarios in order of complexity
    test_cases = [
        # Original simpler scenarios (from class)
        ("Small Delivery", VRPScenarioGenerator.create_small_delivery_scenario),
        ("VRPPD", VRPScenarioGenerator.create_vrppd_scenario), 
        ("Medium Delivery", VRPScenarioGenerator.create_medium_delivery_scenario),
        ("Time Window", VRPScenarioGenerator.create_time_window_scenario),
        ("Multi Depot", VRPScenarioGenerator.create_multi_depot_scenario),
        ("Large Delivery", VRPScenarioGenerator.create_large_delivery_scenario),
        ("Benchmark CVRP", VRPScenarioGenerator.create_benchmark_cvrp_scenario),
        
        # MODA scenarios (standalone functions)
        ("MODA Small", create_moda_small_scenario),
        ("MODA First", create_moda_first_scenario),
    ]
    
    # Test each scenario with original capacities
    logger.info(f"\n{'*'*100}")
    logger.info(f"PHASE 1: Testing scenarios with ORIGINAL vehicle capacities")
    logger.info(f"{'*'*100}")
    
    for scenario_name, create_func in test_cases:
        solved, analysis = test_scenario(scenario_name, create_func)
        results[f"{scenario_name}_original"] = {
            'solved': solved,
            'analysis': analysis
        }
    
    # Test scenarios with uniform capacity (1000 units)
    logger.info(f"\n{'*'*100}")
    logger.info(f"PHASE 2: Testing scenarios with UNIFORM capacity (1000 units)")
    logger.info(f"{'*'*100}")
    
    for scenario_name, create_func in test_cases:
        solved, analysis = test_scenario(scenario_name, create_func, uniform_capacity=1000)
        results[f"{scenario_name}_uniform1k"] = {
            'solved': solved,
            'analysis': analysis
        }
    
    # Test scenarios with very high uniform capacity (10000 units)
    logger.info(f"\n{'*'*100}")
    logger.info(f"PHASE 3: Testing scenarios with HIGH uniform capacity (10000 units)")
    logger.info(f"{'*'*100}")
    
    for scenario_name, create_func in test_cases:
        solved, analysis = test_scenario(scenario_name, create_func, uniform_capacity=10000)
        results[f"{scenario_name}_uniform10k"] = {
            'solved': solved,
            'analysis': analysis
        }
    
    # Summary
    logger.info(f"\n{'='*100}")
    logger.info("FINAL SUMMARY - ROOT CAUSE ANALYSIS")
    logger.info(f"{'='*100}")
    
    for phase in ["original", "uniform1k", "uniform10k"]:
        logger.info(f"\n{phase.upper()} CAPACITY RESULTS:")
        for scenario_name, _ in test_cases:
            key = f"{scenario_name}_{phase}"
            if key in results:
                status = "✅ SOLVED" if results[key]['solved'] else "❌ FAILED"
                logger.info(f"  {scenario_name}: {status}")
    
    # Identify patterns
    logger.info(f"\n🔍 PATTERN ANALYSIS:")
    
    # Check if any original scenarios fail
    original_failures = [k for k, v in results.items() if "_original" in k and not v['solved']]
    if original_failures:
        logger.info(f"  Original scenarios that failed: {[k.replace('_original', '') for k in original_failures]}")
    else:
        logger.info(f"  All original scenarios solved successfully")
    
    # Check if uniform capacity helps
    moda_first_original = results.get("MODA First_original", {}).get('solved', False)
    moda_first_uniform1k = results.get("MODA First_uniform1k", {}).get('solved', False)
    moda_first_uniform10k = results.get("MODA First_uniform10k", {}).get('solved', False)
    
    logger.info(f"\n📊 MODA_FIRST SPECIFIC ANALYSIS:")
    logger.info(f"  Original capacity: {'✅' if moda_first_original else '❌'}")
    logger.info(f"  Uniform 1k capacity: {'✅' if moda_first_uniform1k else '❌'}")
    logger.info(f"  Uniform 10k capacity: {'✅' if moda_first_uniform10k else '❌'}")
    
    if not any([moda_first_original, moda_first_uniform1k, moda_first_uniform10k]):
        logger.info(f"  💡 CONCLUSION: MODA_First issue is NOT related to vehicle capacity")
        logger.info(f"  🔍 Likely causes: Time constraints, scenario complexity, or OR-Tools search limitations")
    elif moda_first_uniform10k and not moda_first_original:
        logger.info(f"  💡 CONCLUSION: MODA_First issue IS related to vehicle capacity")
        logger.info(f"  🔧 Solution: Increase vehicle capacities or reduce demand")
    
    logger.info(f"\nDetailed results available in logs above.")
    logger.info("Test completed!")

if __name__ == "__main__":
    main()
