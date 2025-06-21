#!/usr/bin/env python3
"""
Simple Test for MODA_first with Rolling Window Optimizer

This script tests whether MODA_first can be solved with the current 
rolling window optimizer implementation.
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

def test_current_implementation():
    """Test MODA_first with the current rolling window optimizer."""
    
    print("Testing MODA_first with Rolling Window Optimizer")
    print("=" * 60)
    
    # Create scenario
    scenario = create_moda_first_scenario()
    
    print(f"\n🕐 Testing with current implementation")
    print("-" * 40)
    
    try:
        # Create optimizer
        optimizer = VRPOptimizerRollingWindow(scenario)
        
        # Solve with rolling window
        start_time = time.time()
        result = optimizer.optimize_with_rolling_window()
        actual_time = time.time() - start_time
        
        print(f"Actual solve time: {actual_time:.1f} seconds")
        
        # Check result
        status = result.status
        
        if status == 'feasible':
            print(f"✅ SOLUTION FOUND!")
            routes = result.routes
            print(f"   Status: {status}")
            print(f"   Routes: {len(routes)} vehicles used")
            print(f"   Objective value: {result.objective_value}")
            
            # Show route summary
            total_stops = 0
            active_routes = 0
            for vehicle_id, route in routes.items():
                if len(route) > 0:
                    active_routes += 1
                    stops = len(route)
                    total_stops += stops
                    print(f"   {vehicle_id}: {stops} locations")
            
            print(f"   Active vehicles: {active_routes}")
            print(f"   Total locations visited: {total_stops}")
            
            return result
            
        else:
            print(f"❌ No solution found")
            print(f"   Status: {status}")
            if hasattr(result, 'metrics') and result.metrics:
                error = result.metrics.get('error', 'Unknown error')
                print(f"   Error: {error}")
                
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return None

if __name__ == "__main__":
    result = test_current_implementation()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if result and result.status == 'feasible':
        print("🎉 SUCCESS! MODA_first is solvable with the rolling window optimizer!")
        print("\nThis confirms our diagnostic analysis:")
        print("✅ The scenario constraints are feasible")
        print("✅ The rolling window optimizer can find solutions")
        print("✅ The issue was likely with the previous optimizer implementations")
        
        print(f"\nSolution details:")
        print(f"- Vehicles used: {len(result.routes)}")
        print(f"- Objective value: {result.objective_value}")
        print(f"- Optimization status: {result.status}")
        
    else:
        print("❌ MODA_first could not be solved with current implementation")
        print("\nThis suggests:")
        print("- The scenario may genuinely be too complex")
        print("- Need for algorithm improvements or longer timeouts")
        print("- Consider scenario simplification (fewer requests, more vehicles)")
        print("- Alternative optimization approaches may be needed")
