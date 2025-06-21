#!/usr/bin/env python3
"""
Simple test: only field pickup requests to isolate the depot pickup issue
"""

from vrp_scenarios import create_moda_small_scenario
from vrp_optimizer_enhanced_full import VRPOptimizerEnhanced
import logging

logging.basicConfig(level=logging.WARNING)  # Reduce logging noise

def test_field_only():
    """Test with only field pickup requests."""
    
    print("🔧 TESTING FIELD PICKUP REQUESTS ONLY")
    print("=" * 60)
    
    # Load scenario
    scenario = create_moda_small_scenario()
    
    # Keep only field requests (remove all depot pickups)
    field_requests = [req for req in scenario.ride_requests if req.pickup_location != "depot_1"]
    scenario.ride_requests = field_requests
    
    print(f"Field-only scenario: {len(scenario.ride_requests)} requests")
    
    optimizer = VRPOptimizerEnhanced()
    
    # Test Level 5 (pickup-delivery)
    print("\n🧪 Testing Level 5 (pickup-delivery) - Field requests only...")
    try:
        result = optimizer._solve_with_constraint_level(scenario, "pickup_delivery", 30)
        if result['success']:
            print(f"✅ Level 5 SUCCESS - {result['vehicles_used']} vehicles used")
            print(f"   Solve time: {result['solve_time']:.2f}s")
            
            # Test Level 6 (full)
            print("\n🧪 Testing Level 6 (full) - Field requests only...")
            result = optimizer._solve_with_constraint_level(scenario, "full", 30)
            if result['success']:
                print(f"✅ Level 6 SUCCESS - {result['vehicles_used']} vehicles used")
                print(f"   Solve time: {result['solve_time']:.2f}s")
                return True
            else:
                print(f"❌ Level 6 FAILED - {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Level 5 FAILED - {result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

def test_depot_only():
    """Test with only depot pickup requests."""
    
    print("\n🔧 TESTING DEPOT PICKUP REQUESTS ONLY")
    print("=" * 60)
    
    # Load scenario
    scenario = create_moda_small_scenario()
    
    # Keep only depot requests
    depot_requests = [req for req in scenario.ride_requests if req.pickup_location == "depot_1"]
    scenario.ride_requests = depot_requests[:3]  # Only first 3 depot requests
    
    print(f"Depot-only scenario: {len(scenario.ride_requests)} requests")
    for req in scenario.ride_requests:
        print(f"  {req.id}: {req.pickup_location} → {req.dropoff_location}")
    
    optimizer = VRPOptimizerEnhanced()
    
    # Test Level 5 (pickup-delivery)
    print("\n🧪 Testing Level 5 (pickup-delivery) - Depot requests only...")
    try:
        result = optimizer._solve_with_constraint_level(scenario, "pickup_delivery", 30)
        if result['success']:
            print(f"✅ Level 5 SUCCESS - {result['vehicles_used']} vehicles used")
            return True
        else:
            print(f"❌ Level 5 FAILED - {result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

if __name__ == "__main__":
    field_success = test_field_only()
    depot_success = test_depot_only()
    
    print("\n🏁 SUMMARY:")
    print(f"Field-only requests: {'✅ SUCCESS' if field_success else '❌ FAILED'}")
    print(f"Depot-only requests: {'✅ SUCCESS' if depot_success else '❌ FAILED'}")
    
    if field_success and not depot_success:
        print("💡 CONCLUSION: Issue is specifically with depot pickup requests")
    elif not field_success:
        print("💡 CONCLUSION: Issue is with all pickup-delivery constraints")
    else:
        print("💡 CONCLUSION: Both work individually - issue may be with combined scenario")
