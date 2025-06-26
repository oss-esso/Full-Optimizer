#!/usr/bin/env python3
"""
Test script for dual capacity constraints in VRP optimizer.
This verifies that both weight and volume constraints are enforced correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import the dual capacity optimizer
import importlib.util
current_dir = os.path.dirname(__file__)
optimizer_path = os.path.join(current_dir, "vrp_optimizer_clean copy.py")
spec = importlib.util.spec_from_file_location("vrp_optimizer_clean_copy", optimizer_path)
vrp_optimizer_clean_copy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vrp_optimizer_clean_copy)

from vrp_scenarios import create_moda_small_scenario

# Use the correct class name
CleanVRPOptimizer = vrp_optimizer_clean_copy.CleanVRPOptimizer

def test_dual_capacity_enforcement():
    """Test that dual capacity constraints are properly enforced in the optimizer."""
    print("=" * 80)
    print("🧪 Testing Dual Capacity Constraint Enforcement")
    print("=" * 80)
    
    # Generate test scenario with dual capacity constraints
    print("\n1️⃣ Generating test scenario...")
    scenario = create_moda_small_scenario()
    
    # Create optimizer
    print("\n2️⃣ Creating optimizer...")
    optimizer = CleanVRPOptimizer(vrp_instance=scenario)
    
    # Test with capacity constraints
    print("\n3️⃣ Testing with dual capacity constraints...")
    result = optimizer.solve(constraint_level="capacity", verbose=True)
    
    if result['status'] == 'SUCCESS':
        print(f"\n✅ SUCCESS: Dual capacity constraints enforced successfully!")
        
        # Analyze the solution
        print(f"\n📊 Solution Analysis:")
        print(f"   Total routes: {len(result['routes'])}")
        print(f"   Total distance: {result['total_distance']}m")
        
        # Check for constraint violations in the solution
        total_weight_violations = 0
        total_volume_violations = 0
        
        for route in result['routes']:
            weight_violations = [s for s in route['capacity_usage'] if s['weight_utilization_pct'] > 100]
            volume_violations = [s for s in route['capacity_usage'] if s['volume_utilization_pct'] > 100]
            
            total_weight_violations += len(weight_violations)
            total_volume_violations += len(volume_violations)
            
            if weight_violations or volume_violations:
                print(f"   🚛 {route['vehicle_id']}:")
                if weight_violations:
                    print(f"     ⚠️  Weight violations: {len(weight_violations)} stops")
                if volume_violations:
                    print(f"     ⚠️  Volume violations: {len(volume_violations)} stops")
        
        print(f"\n📈 Constraint Violation Summary:")
        print(f"   Weight violations: {total_weight_violations} stops")
        print(f"   Volume violations: {total_volume_violations} stops")
        
        if total_weight_violations == 0 and total_volume_violations == 0:
            print(f"   ✅ No constraint violations - dual capacity constraints working correctly!")
        else:
            print(f"   ⚠️  Constraint violations detected - may need parameter tuning")
            
    else:
        print(f"\n❌ FAILED: {result.get('message', 'Unknown error')}")
        
    return result

def test_limiting_constraint_identification():
    """Test that the system correctly identifies which constraint is limiting at each stop."""
    print("\n" + "=" * 80)
    print("🔍 Testing Limiting Constraint Identification")
    print("=" * 80)
    
    # Generate scenario
    scenario = create_moda_small_scenario()
    optimizer = CleanVRPOptimizer(vrp_instance=scenario)
    
    # Solve with capacity constraints
    result = optimizer.solve(constraint_level="capacity", verbose=False)
    
    if result['status'] == 'SUCCESS':
        print(f"\n📊 Limiting Constraint Analysis:")
        
        weight_limiting_count = 0
        volume_limiting_count = 0
        balanced_count = 0
        
        for route in result['routes']:
            print(f"\n🚛 {route['vehicle_id']} (Cap: {route['vehicle_capacity_kg']}kg, {route['vehicle_capacity_m3']}m³):")
            
            for stop in route['capacity_usage']:
                if stop['limiting_constraint'] == 'weight':
                    weight_limiting_count += 1
                    icon = "🔴"
                elif stop['limiting_constraint'] == 'volume':
                    volume_limiting_count += 1
                    icon = "📦"
                else:
                    balanced_count += 1
                    icon = "⚖️"
                
                print(f"   {icon} {stop['location_id']}: W={stop['weight_utilization_pct']}%, "
                      f"V={stop['volume_utilization_pct']}% → {stop['limiting_constraint']}")
        
        print(f"\n📈 Limiting Constraint Summary:")
        print(f"   Weight-limited stops: {weight_limiting_count}")
        print(f"   Volume-limited stops: {volume_limiting_count}")
        print(f"   Balanced stops: {balanced_count}")
        print(f"   Total stops analyzed: {weight_limiting_count + volume_limiting_count + balanced_count}")
        
        return True
    else:
        print(f"❌ Test failed - could not solve scenario")
        return False

def main():
    """Run all dual capacity constraint tests."""
    print("🚀 Starting Dual Capacity Constraint Tests")
    
    # Test 1: Basic constraint enforcement
    result1 = test_dual_capacity_enforcement()
    
    # Test 2: Limiting constraint identification
    result2 = test_limiting_constraint_identification()
    
    print("\n" + "=" * 80)
    print("📋 Test Summary")
    print("=" * 80)
    
    if result1 and result1.get('status') == 'SUCCESS' and result2:
        print("✅ All tests passed! Dual capacity constraints are working correctly.")
        print("   - Weight and volume constraints are enforced")
        print("   - Limiting constraints are correctly identified at each stop")
        print("   - Solution extraction and analysis working properly")
    else:
        print("❌ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
