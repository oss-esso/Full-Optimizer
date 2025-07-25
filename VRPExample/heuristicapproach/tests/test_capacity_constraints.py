#!/usr/bin/env python3
"""
Test for Pallet Capacity Hard Constraint and Weight Soft Constraint

This test verifies that:
1. Pallet capacity is enforced as a hard constraint (routes with violations are infeasible)
2. Weight capacity is now a soft constraint (routes with violations get penalty but remain feasible)
"""

import sys
from pathlib import Path

# Add the algo directory to the path
current_dir = Path(__file__).parent
algo_dir = current_dir.parent / "algo"
sys.path.append(str(algo_dir))

from epdt_data_structures import Route, Task, Vehicle, TaskType
from second_level import is_feasible, calculate_z2_score

def create_test_vehicle(weight_capacity=1000, pallet_capacity=10):
    """Create a test vehicle with specified capacities"""
    return Vehicle(
        id="TEST_VEHICLE",
        depot_id="DEPOT",
        weight_capacity=weight_capacity,
        volume_capacity=50.0,
        pallet_capacity=pallet_capacity,
        cost_per_km=1.0,
        cost_per_hour=50.0
    )

def create_test_task(task_id, demand, pallets, lat=45.0, lon=11.0):
    """Create a test task with specified demand and pallet count"""
    return Task(
        id=task_id,
        location_id=f"LOC_{task_id}",
        task_type=TaskType.PICKUP if demand > 0 else TaskType.DELIVERY,
        order_id=f"ORDER_{task_id}",
        lat=lat,
        lon=lon,
        service_time=15.0,
        demand=demand,
        volume=abs(demand) * 0.001,  # Very small volume to avoid volume constraint
        pallets=pallets,
        priority=1
    )

def test_pallet_capacity_hard_constraint():
    """Test that pallet capacity is enforced as a hard constraint"""
    print("Testing Pallet Capacity Hard Constraint...")
    
    # Create vehicle with 5 pallet capacity
    vehicle = create_test_vehicle(weight_capacity=2000, pallet_capacity=5)
    
    # Create tasks that exceed pallet capacity but not weight capacity
    tasks = [
        create_test_task("PICKUP1", 200, 3),  # 3 pallets
        create_test_task("PICKUP2", 300, 4),  # 4 pallets (total would be 7, exceeds 5)
    ]
    
    route = Route(vehicle=vehicle, tasks=tasks)
    
    # Route should be infeasible due to pallet capacity violation
    feasible = is_feasible(route)
    print(f"Route with {sum(t.pallets for t in tasks)} pallets (capacity: {vehicle.pallet_capacity}): {'FEASIBLE' if feasible else 'INFEASIBLE'}")
    
    assert not feasible, "Route should be infeasible due to pallet capacity violation"
    print("✅ Pallet capacity hard constraint working correctly")

def test_weight_capacity_soft_constraint():
    """Test that weight capacity is now a soft constraint with penalties"""
    print("\nTesting Weight Capacity Soft Constraint...")
    
    # Create vehicle with 1000kg weight capacity
    vehicle = create_test_vehicle(weight_capacity=1000, pallet_capacity=10)
    
    # Create tasks that exceed weight capacity but not pallet capacity
    tasks = [
        create_test_task("PICKUP1", 600, 2),  # 600kg, 2 pallets
        create_test_task("PICKUP2", 500, 3),  # 500kg, 3 pallets (total: 1100kg, exceeds 1000kg)
    ]
    
    route_over_weight = Route(vehicle=vehicle, tasks=tasks)
    
    # Route should be feasible despite weight violation
    feasible = is_feasible(route_over_weight)
    print(f"Route with {sum(t.demand for t in tasks)}kg (capacity: {vehicle.weight_capacity}kg): {'FEASIBLE' if feasible else 'INFEASIBLE'}")
    
    assert feasible, "Route should be feasible despite weight capacity violation"
    
    # Calculate score and check for weight penalty
    score_over = calculate_z2_score(route_over_weight)
    
    # Create a similar route that doesn't exceed weight capacity
    tasks_under = [
        create_test_task("PICKUP1", 400, 2),  # 400kg, 2 pallets
        create_test_task("PICKUP2", 300, 3),  # 300kg, 3 pallets (total: 700kg, under 1000kg)
    ]
    
    route_under_weight = Route(vehicle=vehicle, tasks=tasks_under)
    score_under = calculate_z2_score(route_under_weight)
    
    print(f"Score with weight violation: {score_over:.2f}")
    print(f"Score without weight violation: {score_under:.2f}")
    print(f"Weight violation penalty: {score_over - score_under:.2f}")
    
    assert score_over > score_under, "Route with weight violation should have higher cost"
    print("✅ Weight capacity soft constraint working correctly")

def test_both_constraints_together():
    """Test interaction between pallet hard constraint and weight soft constraint"""
    print("\nTesting Both Constraints Together...")
    
    vehicle = create_test_vehicle(weight_capacity=1000, pallet_capacity=5)
    
    # Test case 1: Weight violation only (should be feasible)
    tasks_weight_only = [
        create_test_task("PICKUP1", 800, 2),  # Under pallet limit, over weight
        create_test_task("PICKUP2", 300, 2),  # Total: 1100kg, 4 pallets
    ]
    route_weight_only = Route(vehicle=vehicle, tasks=tasks_weight_only)
    feasible_weight = is_feasible(route_weight_only)
    print(f"Weight violation only: {'FEASIBLE' if feasible_weight else 'INFEASIBLE'}")
    assert feasible_weight, "Route with weight violation only should be feasible"
    
    # Test case 2: Pallet violation only (should be infeasible)
    tasks_pallet_only = [
        create_test_task("PICKUP1", 200, 3),  # Under weight limit, over pallet
        create_test_task("PICKUP2", 300, 4),  # Total: 500kg, 7 pallets
    ]
    route_pallet_only = Route(vehicle=vehicle, tasks=tasks_pallet_only)
    feasible_pallet = is_feasible(route_pallet_only)
    print(f"Pallet violation only: {'FEASIBLE' if feasible_pallet else 'INFEASIBLE'}")
    assert not feasible_pallet, "Route with pallet violation should be infeasible"
    
    # Test case 3: Both violations (should be infeasible due to pallet)
    tasks_both = [
        create_test_task("PICKUP1", 600, 3),  # Both over weight and pallet
        create_test_task("PICKUP2", 500, 4),  # Total: 1100kg, 7 pallets
    ]
    route_both = Route(vehicle=vehicle, tasks=tasks_both)
    feasible_both = is_feasible(route_both)
    print(f"Both violations: {'FEASIBLE' if feasible_both else 'INFEASIBLE'}")
    assert not feasible_both, "Route with both violations should be infeasible (due to pallet constraint)"
    
    print("✅ Both constraints working correctly together")

def main():
    """Run all constraint tests"""
    print("Testing Enhanced Capacity Constraints\n")
    print("=" * 50)
    
    try:
        test_pallet_capacity_hard_constraint()
        test_weight_capacity_soft_constraint() 
        test_both_constraints_together()
        
        print("\n" + "=" * 50)
        print("🎉 All capacity constraint tests passed!")
        print("\nSummary:")
        print("- ✅ Pallet capacity enforced as hard constraint")
        print("- ✅ Weight capacity enforced as soft constraint with penalties")
        print("- ✅ Both constraints work correctly together")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
