#!/usr/bin/env python3
"""
Quick test with fixed depot pickup scenario (no duplicate dropoff locations)
"""

from vrp_scenarios import create_moda_small_scenario
from vrp_optimizer_enhanced_full import VRPOptimizerEnhanced
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_fixed_depot_scenario():
    """Test with fixed depot scenario (no duplicate dropoffs)."""
    
    print("🔧 TESTING FIXED DEPOT SCENARIO")
    print("=" * 60)
    
    # Load original scenario
    scenario = create_moda_small_scenario()
    
    # Fix the duplicate dropoff issue
    print("🔍 Original ride requests:")
    depot_requests = [req for req in scenario.ride_requests if req.pickup_location == "depot_1"]
    field_requests = [req for req in scenario.ride_requests if req.pickup_location != "depot_1"]
    
    print(f"  Depot requests: {len(depot_requests)}")
    print(f"  Field requests: {len(field_requests)}")
    
    # Check for duplicate dropoffs in depot requests
    dropoff_counts = {}
    for req in depot_requests:
        dropoff = req.dropoff_location
        if dropoff in dropoff_counts:
            dropoff_counts[dropoff].append(req.id)
        else:
            dropoff_counts[dropoff] = [req.id]
    
    duplicates = {k: v for k, v in dropoff_counts.items() if len(v) > 1}
    if duplicates:
        print(f"❌ Found duplicate dropoffs in depot requests: {duplicates}")
        
        # Fix by removing duplicate requests
        unique_depot_requests = []
        used_dropoffs = set()
        
        for req in depot_requests:
            if req.dropoff_location not in used_dropoffs:
                unique_depot_requests.append(req)
                used_dropoffs.add(req.dropoff_location)
            else:
                print(f"  Removing duplicate: {req.id} (dropoff: {req.dropoff_location})")
        
        scenario.ride_requests = unique_depot_requests + field_requests
        print(f"✅ Fixed scenario: {len(scenario.ride_requests)} requests ({len(unique_depot_requests)} depot + {len(field_requests)} field)")
    else:
        print("✅ No duplicate dropoffs found")
    
    # Test the fixed scenario
    optimizer = VRPOptimizerEnhanced()
    
    print("\n🧪 Testing Level 5 (pickup-delivery) with fixed scenario...")
    try:
        result = optimizer._solve_with_constraint_level(scenario, "pickup_delivery", 60)
        if result['success']:
            print(f"✅ Level 5 SUCCESS - {result['vehicles_used']} vehicles used")
            
            print("\n🧪 Testing Level 6 (full) with fixed scenario...")
            result = optimizer._solve_with_constraint_level(scenario, "full", 60)
            if result['success']:
                print(f"✅ Level 6 SUCCESS - {result['vehicles_used']} vehicles used")
                print(f"   🎯 Objective: {result['objective_value']}")
                print(f"   ⏱️  Solve time: {result['solve_time']:.2f}s")
            else:
                print(f"❌ Level 6 FAILED - {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ Level 5 FAILED - {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")

if __name__ == "__main__":
    test_fixed_depot_scenario()
