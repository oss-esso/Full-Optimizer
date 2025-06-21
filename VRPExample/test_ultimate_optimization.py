#!/usr/bin/env python3
"""
Ultimate VRP optimization test with maximum strategies and time limits.

This script tests the VRP optimizers with:
1. Extended time limits (up to 60 minutes)
2. Multiple OR-Tools strategies and metaheuristics
3. Progressive time increase approach
4. Detailed analysis and diagnostics

The goal is to determine if MODA_first is truly infeasible or if we just need
more computational resources and better search strategies.
"""

import sys
import os
import time
import logging
import json
from typing import Dict, Any

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vrp_scenarios import create_moda_small_scenario, create_moda_first_scenario
from vrp_optimizer_rolling_window import VRPOptimizerRollingWindow
from vrp_optimizer_enhanced import VRPOptimizerEnhanced
from vrp_data_models import VRPObjective

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ultimate_optimization.log')
    ]
)
logger = logging.getLogger(__name__)

def analyze_scenario(scenario_name: str, instance) -> Dict[str, Any]:
    """Analyze scenario characteristics."""
    analysis = {
        'scenario_name': scenario_name,
        'num_locations': len(instance.location_ids),
        'num_vehicles': len(instance.vehicles),
        'num_requests': len(instance.ride_requests),
        'has_time_windows': any(r.earliest_pickup is not None and r.latest_dropoff is not None for r in instance.ride_requests),
        'has_service_times': hasattr(instance.ride_requests[0], 'service_time') if instance.ride_requests else False,
        'has_capacity_constraints': any(v.capacity > 0 for v in instance.vehicles.values()),
    }
    
    # Calculate time window statistics 
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

def test_progressive_time_limits(optimizer, scenario_name: str) -> Dict[str, Any]:
    """Test optimizer with progressively increasing time limits."""
    logger.info(f"\n{'='*80}")
    logger.info(f"PROGRESSIVE TIME LIMIT TESTING: {scenario_name}")
    logger.info(f"{'='*80}")
    
    # Progressive time limits (in seconds)
    time_limits = [60, 300, 900, 1800, 3600]  # 1min, 5min, 15min, 30min, 60min
    
    results = {}
    
    for time_limit in time_limits:
        logger.info(f"\n🕒 Testing time limit: {time_limit} seconds ({time_limit/60:.1f} minutes)")
        logger.info("-" * 60)
        
        start_time = time.time()
        
        try:
            # Test with rolling window optimizer (extended timeout)
            result = optimizer.optimize_with_rolling_window(
                timeout_duration=time_limit + 60,  # Give extra buffer for timeout
                time_limit_seconds=time_limit
            )
            
            elapsed_time = time.time() - start_time
            
            result_info = {
                'time_limit': time_limit,
                'elapsed_time': elapsed_time,
                'status': result.status,
                'objective_value': result.objective_value,
                'feasible': result.status == 'optimal' and result.objective_value < float('inf'),
                'num_routes': len(result.routes) if result.routes else 0,
                'total_distance': result.metrics.get('total_distance', 0) if result.metrics else 0
            }
            
            results[f"{time_limit}s"] = result_info
            
            logger.info(f"  Status: {result.status}")
            logger.info(f"  Objective: {result.objective_value}")
            logger.info(f"  Feasible: {result_info['feasible']}")
            logger.info(f"  Elapsed: {elapsed_time:.1f}s")
            
            # If we found a feasible solution, we can stop early or continue to see if we improve
            if result_info['feasible']:
                logger.info(f"  ✅ FEASIBLE SOLUTION FOUND!")
                logger.info(f"  Routes: {result_info['num_routes']}")
                logger.info(f"  Distance: {result_info['total_distance']:.1f}")
                # Continue to see if longer time limits give better solutions
            else:
                logger.info(f"  ❌ No feasible solution found")
                
        except Exception as e:
            logger.error(f"  💥 Error with time limit {time_limit}s: {str(e)}")
            results[f"{time_limit}s"] = {
                'time_limit': time_limit,
                'elapsed_time': time.time() - start_time,
                'status': 'error',
                'error': str(e),
                'feasible': False
            }
    
    return results

def test_enhanced_strategies(instance, scenario_name: str) -> Dict[str, Any]:
    """Test with enhanced multi-strategy optimizer."""
    logger.info(f"\n{'='*80}")
    logger.info(f"ENHANCED MULTI-STRATEGY TESTING: {scenario_name}")
    logger.info(f"{'='*80}")
    
    try:
        optimizer = VRPOptimizerEnhanced(instance, VRPObjective.MINIMIZE_DISTANCE)
        
        # Test with 60-minute time limit
        start_time = time.time()
        result = optimizer.solve_with_enhanced_strategies(time_limit_seconds=3600)
        elapsed_time = time.time() - start_time
        
        result_info = {
            'elapsed_time': elapsed_time,
            'status': getattr(result, 'status', 'unknown'),
            'objective_value': result.objective_value,
            'feasible': result.objective_value < float('inf'),
            'num_routes': len(result.routes) if result.routes else 0,
            'total_distance': result.metadata.get('total_distance', 0) if result.metadata else 0,
            'metadata': result.metadata if hasattr(result, 'metadata') else {}
        }
        
        logger.info(f"Enhanced strategies result:")
        logger.info(f"  Status: {result_info['status']}")
        logger.info(f"  Objective: {result_info['objective_value']}")
        logger.info(f"  Feasible: {result_info['feasible']}")
        logger.info(f"  Elapsed: {elapsed_time:.1f}s")
        
        if result_info['feasible']:
            logger.info(f"  ✅ ENHANCED STRATEGIES FOUND SOLUTION!")
            logger.info(f"  Routes: {result_info['num_routes']}")
            logger.info(f"  Distance: {result_info['total_distance']:.1f}")
        
        return result_info
        
    except Exception as e:
        logger.error(f"Enhanced strategies failed: {str(e)}")
        return {
            'elapsed_time': 0,
            'status': 'error',
            'error': str(e),
            'feasible': False
        }

def main():
    """Run ultimate VRP optimization tests."""
    logger.info("="*100)
    logger.info("ULTIMATE VRP OPTIMIZATION TEST")
    logger.info("="*100)
    logger.info("Testing with maximum time limits and all available strategies")
    logger.info("Goal: Determine if MODA_first is truly infeasible or just computationally challenging")
    
    # Test scenarios
    scenarios = [
        ("MODA_small", "small"),
        ("MODA_first", "first")
    ]
    
    all_results = {}
    
    for scenario_name, scenario_type in scenarios:
        logger.info(f"\n{'*'*100}")
        logger.info(f"TESTING SCENARIO: {scenario_name}")
        logger.info(f"{'*'*100}")
        
        try:            # Create scenario
            logger.info(f"Creating {scenario_name} scenario...")
            if scenario_type == "small":
                instance = create_moda_small_scenario()
            elif scenario_type == "first":
                instance = create_moda_first_scenario()
            else:
                raise ValueError(f"Unknown scenario type: {scenario_type}")
            
            # Analyze scenario
            analysis = analyze_scenario(scenario_name, instance)
            logger.info(f"Scenario analysis: {json.dumps(analysis, indent=2)}")
            
            # Initialize optimizer
            optimizer = VRPOptimizerRollingWindow(instance, VRPObjective.MINIMIZE_DISTANCE)
            
            scenario_results = {
                'analysis': analysis,
                'progressive_time_limits': None,
                'enhanced_strategies': None
            }
            
            # Test 1: Progressive time limits with rolling window optimizer
            progressive_results = test_progressive_time_limits(optimizer, scenario_name)
            scenario_results['progressive_time_limits'] = progressive_results
            
            # Test 2: Enhanced multi-strategy optimizer
            enhanced_results = test_enhanced_strategies(instance, scenario_name)
            scenario_results['enhanced_strategies'] = enhanced_results
            
            all_results[scenario_name] = scenario_results
            
            # Summary for this scenario
            logger.info(f"\n📊 SUMMARY FOR {scenario_name}:")
            logger.info("-" * 50)
            
            # Check if any method found a feasible solution
            found_feasible = False
            
            # Check progressive results
            for time_limit, result in progressive_results.items():
                if result.get('feasible', False):
                    found_feasible = True
                    logger.info(f"  ✅ Progressive ({time_limit}): FEASIBLE (obj: {result['objective_value']:.0f})")
                    break
            
            # Check enhanced results
            if enhanced_results.get('feasible', False):
                found_feasible = True
                logger.info(f"  ✅ Enhanced strategies: FEASIBLE (obj: {enhanced_results['objective_value']:.0f})")
            
            if not found_feasible:
                logger.info(f"  ❌ No feasible solution found with any method or time limit")
                
        except Exception as e:
            logger.error(f"Failed to test {scenario_name}: {str(e)}")
            all_results[scenario_name] = {'error': str(e)}
    
    # Final summary
    logger.info(f"\n{'='*100}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*100}")
    
    for scenario_name, results in all_results.items():
        if 'error' in results:
            logger.info(f"{scenario_name}: ERROR - {results['error']}")
            continue
            
        logger.info(f"\n{scenario_name}:")
        
        # Progressive results summary
        prog_results = results.get('progressive_time_limits', {})
        feasible_progressive = [k for k, v in prog_results.items() if v.get('feasible', False)]
        
        if feasible_progressive:
            logger.info(f"  Progressive time limits: FEASIBLE with {feasible_progressive}")
        else:
            logger.info(f"  Progressive time limits: NO FEASIBLE SOLUTION")
        
        # Enhanced results summary
        enh_results = results.get('enhanced_strategies', {})
        if enh_results.get('feasible', False):
            logger.info(f"  Enhanced strategies: FEASIBLE (obj: {enh_results['objective_value']:.0f})")
        else:
            logger.info(f"  Enhanced strategies: NO FEASIBLE SOLUTION")
    
    # Save detailed results
    output_file = 'ultimate_optimization_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\nDetailed results saved to: {output_file}")
    logger.info("Ultimate optimization test completed!")

if __name__ == "__main__":
    main()
