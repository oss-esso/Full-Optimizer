#!/usr/bin/env python3
"""
Ultra simple test: one depot pickup to see if the basic mechanics work
"""

from vrp_scenarios import create_moda_small_scenario
from vrp_optimizer_enhanced_full import VRPOptimizerEnhanced
import logging

logging.basicConfig(level=logging.WARNING)  # Minimal logging

def test_ultra_simple():
    """Test with just one depot pickup request."""
    
    print("🔧 ULTRA SIMPLE TEST: 1 depot pickup only")
    print("=" * 60)
    
    # Load scenario
    scenario = create_moda_small_scenario()
    
    # Keep only ONE depot pickup request
    depot_requests = [req for req in scenario.ride_requests if req.pickup_location == "depot_1"]
    scenario.ride_requests = [depot_requests[0]]  # Just the first one
    
    print(f"Ultra-simple scenario: {len(scenario.ride_requests)} request")
    req = scenario.ride_requests[0]
    print(f"  {req.id}: {req.pickup_location} → {req.dropoff_location} ({req.passengers}kg)")
    
    # Check time windows
    depot = scenario.locations[req.pickup_location]
    dropoff = scenario.locations[req.dropoff_location]
    print(f"  Depot window: {depot.time_window_start}-{depot.time_window_end}")
    print(f"  Dropoff window: {dropoff.time_window_start}-{dropoff.time_window_end}")
    
    optimizer = VRPOptimizerEnhanced()
    
    # Test levels progressively
    levels = ["distance", "time_dimension", "time_windows", "capacity", "pickup_delivery"]
    
    for level in levels:
        print(f"\n🧪 Testing {level}...")
        try:
            result = optimizer._solve_with_constraint_level(scenario, level, 30)
            if result['success']:
                print(f"✅ {level.upper()} SUCCESS - {result['vehicles_used']} vehicles")
            else:
                print(f"❌ {level.upper()} FAILED - {result.get('error', 'Unknown')}")
                break
        except Exception as e:
            print(f"❌ {level.upper()} EXCEPTION: {e}")
            break

if __name__ == "__main__":
    test_ultra_simple()
