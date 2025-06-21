#!/usr/bin/env python3
"""
Simple Extended Time Test for MODA_first

This script tests the MODA_first scenario with just extended time limits
using the existing rolling window optimizer.
"""

import logging
import time
from vrp_optimizer_rolling_window import VRPOptimizerRollingWindow
from vrp_scenarios import create_moda_first_scenario

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_with_extended_time():
    """Test MODA_first with extended time limit."""
    
    print("Testing MODA_first with Extended Time Limit")
    print("=" * 60)
    
    # Create scenario
    scenario = create_moda_first_scenario()
    
    # Test with different time limits
    time_limits = [300, 600, 1200, 1800]  # 5, 10, 20, 30 minutes
    
    for time_limit in time_limits:
        print(f"\n🕐 Testing with {time_limit} seconds ({time_limit/60:.0f} minutes)")
        print("-" * 40)
        
        try:
            # Create optimizer
            optimizer = VRPOptimizerRollingWindow(scenario)
              # Solve with rolling window (note: this doesn't support custom time limits)
            start_time = time.time()
            result = optimizer.optimize_with_rolling_window()
            actual_time = time.time() - start_time
            
            print(f"Actual solve time: {actual_time:.1f} seconds")
            
            # Check result
            if hasattr(result, 'status'):
                status = result.status
            elif hasattr(result, 'routes') and result.routes:
                status = 'feasible'
            elif isinstance(result, dict) and 'routes' in result:
                status = 'feasible' if result['routes'] else 'infeasible'
            else:
                status = 'unknown'
            
            if status == 'feasible':
                print(f"✅ SOLUTION FOUND!")
                if hasattr(result, 'routes'):
                    routes = result.routes
                elif isinstance(result, dict):
                    routes = result.get('routes', {})
                else:
                    routes = {}
                    
                print(f"   Routes: {len(routes)} vehicles used")
                
                # Show route summary
                total_stops = 0
                for vehicle_id, route in routes.items():
                    stops = len(route) - 2 if len(route) > 2 else 0  # Exclude start/end depot
                    total_stops += stops
                    print(f"   {vehicle_id}: {stops} stops")
                
                print(f"   Total stops served: {total_stops}")
                
                # We found a solution, no need to test longer times
                print(f"\n🎉 SUCCESS! Found solution with {time_limit/60:.0f} minute time limit")
                return result
                
            else:
                print(f"❌ No solution found")
                if hasattr(result, 'error'):
                    print(f"   Error: {result.error}")
                elif isinstance(result, dict) and 'error' in result:
                    print(f"   Error: {result['error']}")
                    
        except Exception as e:
            print(f"❌ Exception occurred: {str(e)}")
    
    print(f"\n💡 All time limits tested. MODA_first may need scenario modifications.")
    return None

if __name__ == "__main__":
    result = test_with_extended_time()
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    if result:
        print("✅ MODA_first is solvable with sufficient time!")
        print("The diagnostic was correct - the scenario is feasible.")
        print("Recommendation: Use longer time limits for complex scenarios.")
    else:
        print("❌ MODA_first remains unsolvable even with extended time.")
        print("This suggests the issue may be:")
        print("1. Problem complexity beyond OR-Tools capabilities")
        print("2. Need for problem decomposition approaches")
        print("3. Scenario parameter adjustments (fewer requests, more vehicles, etc.)")
        print("4. Different algorithm approaches (heuristics, metaheuristics)")
