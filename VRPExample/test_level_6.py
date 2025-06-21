#!/usr/bin/env python3
"""
Test Level 6 specifically to debug the driver break constraints
"""

import logging
from vrp_optimizer_enhanced_full import VRPOptimizerEnhanced
from vrp_scenarios import create_moda_small_scenario

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_level_6_only():
    """Test only Level 6 to isolate the driver break issue."""
    print("🧪 TESTING LEVEL 6 DRIVER BREAKS ONLY")
    print("=" * 60)
    
    try:
        # Load scenario
        scenario = create_moda_small_scenario()
        optimizer = VRPOptimizerEnhanced()
        
        # Force Level 6 only
        result = optimizer._solve_with_constraint_level(scenario, "full", 120)
        
        print(f"\n🎯 LEVEL 6 RESULTS:")
        print(f"Success: {result['success']}")
        
        if result['success']:
            print(f"✅ LEVEL 6 SUCCEEDED!")
            print(f"   Objective: {result['objective_value']}")
            print(f"   Vehicles used: {result['vehicles_used']}")
            print(f"   Solve time: {result['solve_time']:.2f}s")
            print(f"   Constraints: {result['constraints_applied']}")
        else:
            print(f"❌ LEVEL 6 FAILED: {result.get('error', 'Unknown error')}")
            print(f"   Constraints attempted: {result.get('constraints_applied', [])}")
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_level_6_only()
