#!/usr/bin/env python3
"""
Test simplified scenarios to isolate the Level 6 constraint issue.
"""

from vrp_scenarios import create_moda_small_scenario
from vrp_optimizer_enhanced_full import VRPOptimizerEnhanced
import logging

def test_reduced_scenario():
    """Test with reduced number of requests to isolate the issue."""
    
    print("🧪 TESTING REDUCED SCENARIO TO ISOLATE ISSUE")
    print("=" * 60)
    
    # Load full scenario
    scenario = create_moda_small_scenario()
    
    print(f"Original scenario: {len(scenario.ride_requests)} requests")
    
    # Test 1: Only field pickup requests (no depot pickups)
    print("\n🔍 TEST 1: Only field pickup requests (no virtual depot pickups)")
    test_scenario = create_test_scenario_field_only(scenario)
    test_optimizer(test_scenario, "Field-only")
    
    # Test 2: Only 2 depot pickup requests
    print("\n🔍 TEST 2: Only 2 depot pickup requests")
    test_scenario = create_test_scenario_depot_only(scenario, max_requests=2)
    test_optimizer(test_scenario, "2 Depot-only")
    
    # Test 3: Mix of 2 depot + 2 field requests
    print("\n🔍 TEST 3: Mix of 2 depot + 2 field requests")
    test_scenario = create_test_scenario_mixed(scenario, depot_count=2, field_count=2)
    test_optimizer(test_scenario, "Mixed 2+2")

def create_test_scenario_field_only(original_scenario):
    """Create scenario with only field pickup requests."""
    scenario = create_moda_small_scenario()
    
    # Remove all depot pickup requests
    field_requests = [req for req in scenario.ride_requests if req.pickup_location != "depot_1"]
    scenario.ride_requests = field_requests
    
    print(f"  Reduced to {len(scenario.ride_requests)} field requests")
    return scenario

def create_test_scenario_depot_only(original_scenario, max_requests=2):
    """Create scenario with only depot pickup requests."""
    scenario = create_moda_small_scenario()
    
    # Keep only depot pickup requests
    depot_requests = [req for req in scenario.ride_requests if req.pickup_location == "depot_1"]
    scenario.ride_requests = depot_requests[:max_requests]
    
    print(f"  Reduced to {len(scenario.ride_requests)} depot requests")
    return scenario

def create_test_scenario_mixed(original_scenario, depot_count=2, field_count=2):
    """Create scenario with mixed requests."""
    scenario = create_moda_small_scenario()
    
    # Get subset of requests
    depot_requests = [req for req in scenario.ride_requests if req.pickup_location == "depot_1"][:depot_count]
    field_requests = [req for req in scenario.ride_requests if req.pickup_location != "depot_1"][:field_count]
    
    scenario.ride_requests = depot_requests + field_requests
    
    print(f"  Reduced to {len(scenario.ride_requests)} requests ({depot_count} depot + {field_count} field)")
    return scenario

def test_optimizer(scenario, test_name):
    """Test the optimizer with given scenario."""
    
    optimizer = VRPOptimizerEnhanced()
    
    # Test Level 5 first (pickup-delivery without breaks)
    print(f"  🔧 Testing {test_name} - Level 5 (pickup-delivery)...")
    
    try:
        solution = optimizer.solve(scenario, "pickup_delivery")
        if solution and solution.get('success', False):
            print(f"  ✅ Level 5 SUCCESS for {test_name}")
            
            # Now test Level 6 (full constraints)
            print(f"  🔧 Testing {test_name} - Level 6 (full)...")
            solution = optimizer.solve(scenario, "full")
            if solution and solution.get('success', False):
                print(f"  ✅ Level 6 SUCCESS for {test_name}")
            else:
                print(f"  ❌ Level 6 FAILED for {test_name}")
                print(f"     Error: {solution.get('error', 'Unknown error') if solution else 'No solution returned'}")
        else:
            print(f"  ❌ Level 5 FAILED for {test_name}")
            print(f"     Error: {solution.get('error', 'Unknown error') if solution else 'No solution returned'}")
    
    except Exception as e:
        print(f"  ❌ EXCEPTION for {test_name}: {e}")

if __name__ == "__main__":
    # Reduce logging to focus on our output
    logging.getLogger().setLevel(logging.WARNING)
    
    test_reduced_scenario()
