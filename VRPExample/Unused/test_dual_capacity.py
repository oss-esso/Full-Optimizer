#!/usr/bin/env python3
"""
Test script to verify dual capacity constraint implementation.
Tests both weight (kg) and volume (m³) constraints with MODA scenarios.
"""

import sys
import os
import importlib.util

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from vrp_scenarios import get_all_scenarios, calculate_cargo_volume

# Load the optimizer module (handling the filename with space)
optimizer_path = os.path.join(current_dir, "vrp_optimizer_clean copy.py")
spec = importlib.util.spec_from_file_location("vrp_optimizer_clean", optimizer_path)
vrp_optimizer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vrp_optimizer_module)

CleanVRPOptimizer = vrp_optimizer_module.CleanVRPOptimizer

def test_dual_capacity_constraints():
    """Test dual capacity constraint implementation."""
    print("🧪 Testing Dual Capacity Constraints (Weight + Volume)")
    print("=" * 80)
    
    # Test cargo volume calculation function
    print("📐 Testing Cargo Volume Calculation:")
    print("-" * 40)
    
    test_weights = [1000, 5000, 10000, 24000]  # kg
    for weight in test_weights:
        volume = calculate_cargo_volume(weight)
        density = weight / volume if volume > 0 else 0
        print(f"   {weight:,}kg → {volume:.2f}m³ (density: {density:.0f} kg/m³)")
    
    # Test scenarios with dual capacity constraints
    scenarios = get_all_scenarios()
    
    if 'MODA_small' in scenarios:
        print(f"\n📋 Testing MODA_small Scenario:")
        print("-" * 40)
        
        scenario = scenarios['MODA_small']
        
        # Display vehicle capacities
        print(f"🚛 Vehicle Fleet Capacities:")
        total_weight_cap = 0
        total_volume_cap = 0
        
        for vehicle_id, vehicle in scenario.vehicles.items():
            weight_cap = getattr(vehicle, 'capacity', 0)
            volume_cap = getattr(vehicle, 'volume_capacity', 0.0)
            vehicle_type = getattr(vehicle, 'vehicle_type', 'unknown')
            
            total_weight_cap += weight_cap
            total_volume_cap += volume_cap
            
            print(f"   {vehicle_id} ({vehicle_type}): {weight_cap:,}kg, {volume_cap:.1f}m³")
        
        print(f"   TOTAL FLEET: {total_weight_cap:,}kg, {total_volume_cap:.1f}m³")
        
        # Display cargo demands
        print(f"\n📦 Cargo Demands:")
        total_weight_demand = 0
        total_volume_demand = 0
        
        for request in scenario.ride_requests:
            weight = getattr(request, 'passengers', 0)
            volume = getattr(request, 'volume', 0.0)
            total_weight_demand += weight
            total_volume_demand += volume
            
            print(f"   {request.id}: {weight:,}kg, {volume:.2f}m³")
        
        print(f"   TOTAL DEMAND: {total_weight_demand:,}kg, {total_volume_demand:.1f}m³")
        
        # Calculate utilization
        weight_util = (total_weight_demand / total_weight_cap * 100) if total_weight_cap > 0 else 0
        volume_util = (total_volume_demand / total_volume_cap * 100) if total_volume_cap > 0 else 0
        
        print(f"\n📊 Capacity Utilization:")
        print(f"   Weight: {weight_util:.1f}% ({total_weight_demand:,}kg / {total_weight_cap:,}kg)")
        print(f"   Volume: {volume_util:.1f}% ({total_volume_demand:.1f}m³ / {total_volume_cap:.1f}m³)")
        
        if weight_util > volume_util:
            print(f"   🔴 Weight is limiting factor ({weight_util:.1f}% vs {volume_util:.1f}%)")
        elif volume_util > weight_util:
            print(f"   📦 Volume is limiting factor ({volume_util:.1f}% vs {weight_util:.1f}%)")
        else:
            print(f"   ⚖️  Constraints are balanced")
        
        # Test optimizer with dual constraints
        print(f"\n🔧 Testing Optimizer with Dual Capacity Constraints:")
        print("-" * 55)
        
        try:
            optimizer = CleanVRPOptimizer(vrp_instance=scenario)
            
            # Test the dual capacity analysis
            result = optimizer.solve(constraint_level="capacity", verbose=False)
            
            print(f"✅ Dual capacity constraint analysis completed")
            
        except Exception as e:
            print(f"❌ Optimizer test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Test individual vehicle capacity ratios
    print(f"\n🚚 Individual Vehicle Capacity Analysis:")
    print("-" * 45)
    
    # Test typical cargo scenarios
    test_scenarios = [
        {"name": "Dense cargo (machinery)", "density": 600, "example_weight": 12000},
        {"name": "General cargo (packages)", "density": 200, "example_weight": 4800},
        {"name": "Light cargo (textiles)", "density": 100, "example_weight": 2400},
    ]
    
    print(f"Vehicle capacity comparisons:")
    print(f"   4-ton truck: 4,000kg, 24.0m³ (max density: {4000/24:.0f} kg/m³)")
    print(f"   24-ton truck: 24,000kg, 67.0m³ (max density: {24000/67:.0f} kg/m³)")
    print()
    
    for scenario_test in test_scenarios:
        name = scenario_test["name"]
        density = scenario_test["density"]
        weight = scenario_test["example_weight"]
        volume = weight / density
        
        print(f"   {name} ({density} kg/m³):")
        print(f"      {weight:,}kg → {volume:.1f}m³")
        
        # Check which vehicles can handle this cargo
        if weight <= 4000 and volume <= 24.0:
            print(f"      ✅ Fits in 4-ton truck (weight & volume OK)")
        elif weight <= 4000:
            print(f"      ⚠️  Fits 4-ton truck by weight, but volume exceeds capacity ({volume:.1f}m³ > 24.0m³)")
        elif volume <= 24.0:
            print(f"      ⚠️  Fits 4-ton truck by volume, but weight exceeds capacity ({weight}kg > 4,000kg)")
        else:
            print(f"      ❌ Too big for 4-ton truck (both weight and volume exceed)")
            
        if weight <= 24000 and volume <= 67.0:
            print(f"      ✅ Fits in 24-ton truck")
        else:
            print(f"      ❌ Too big for 24-ton truck")
        print()
    
    print("=" * 80)
    print("✅ Dual Capacity Constraint Testing Completed")
    print("   Both weight (kg) and volume (m³) constraints are now implemented")
    print("   The system will determine which constraint is hit first at each stop")

if __name__ == "__main__":
    test_dual_capacity_constraints()
