#!/usr/bin/env python3
"""
Debug version of the capacity constraint test to see what's happening
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

def debug_weight_constraint():
    """Debug why weight constraint is still blocking feasibility"""
    print("Debugging Weight Constraint...")
    
    # Create vehicle with 1000kg weight capacity
    vehicle = create_test_vehicle(weight_capacity=1000, pallet_capacity=10)
    
    # Create tasks that exceed weight capacity but not pallet capacity
    tasks = [
        create_test_task("PICKUP1", 600, 2),  # 600kg, 2 pallets
        create_test_task("PICKUP2", 500, 3),  # 500kg, 3 pallets (total: 1100kg, exceeds 1000kg)
    ]
    
    route = Route(vehicle=vehicle, tasks=tasks)
    
    print(f"Vehicle weight capacity: {vehicle.weight_capacity}kg")
    print(f"Vehicle volume capacity: {vehicle.volume_capacity}m³")
    print(f"Vehicle pallet capacity: {vehicle.pallet_capacity}")
    
    total_weight = sum(t.demand for t in tasks)
    total_volume = sum(t.volume for t in tasks)
    total_pallets = sum(getattr(t, 'pallets', 0) for t in tasks)
    
    print(f"Route weight: {total_weight}kg (over capacity: {total_weight > vehicle.weight_capacity})")
    print(f"Route volume: {total_volume}m³ (over capacity: {total_volume > vehicle.volume_capacity})")
    print(f"Route pallets: {total_pallets} (over capacity: {total_pallets > vehicle.pallet_capacity if vehicle.pallet_capacity else False})")
    
    # Test feasibility
    feasible = is_feasible(route)
    print(f"Route feasible: {feasible}")
    
    return feasible

if __name__ == "__main__":
    debug_weight_constraint()
