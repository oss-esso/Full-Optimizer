"""
Multi-Order Packing System

This module implements advanced packing algorithms to assign multiple small orders
to a single vehicle, especially useful for split orders and remaining unassigned orders.

The goal is to achieve 100% assignment by efficiently packing orders together.
"""

from typing import List, Optional, Tuple, TYPE_CHECKING
import copy
from itertools import combinations

if TYPE_CHECKING:
    from epdt_data_structures import Order, Vehicle, Route, Solution

def try_multi_order_packing(unassigned_orders: List['Order'], vehicles: List['Vehicle'], solution: 'Solution') -> int:
    """
    Try to pack multiple unassigned orders into available vehicles.
    
    Args:
        unassigned_orders: List of orders that couldn't be assigned individually
        vehicles: Available vehicles
        solution: Current solution to modify
        
    Returns:
        Number of orders successfully assigned through packing
    """
    if len(unassigned_orders) < 2:
        return 0
        
    print(f"Attempting multi-order packing for {len(unassigned_orders)} unassigned orders...")
    
    orders_assigned = 0
    remaining_orders = unassigned_orders.copy()
    
    # Try packing 2-4 orders together
    for pack_size in [2, 3, 4]:
        if len(remaining_orders) < pack_size:
            continue
            
        print(f"  Trying to pack {pack_size} orders together...")
        
        # Try all combinations of pack_size orders
        combinations_tried = 0
        for order_combo in combinations(remaining_orders, pack_size):
            combinations_tried += 1
            if combinations_tried > 20:  # Limit for performance
                break
                
            # Try to find a vehicle that can fit this combination
            best_vehicle, best_route = find_best_vehicle_for_combo(order_combo, vehicles, solution)
            
            if best_vehicle and best_route:
                # Assign the combination
                solution.routes[best_vehicle.id] = best_route
                
                # Remove assigned orders from remaining
                for order in order_combo:
                    if order in remaining_orders:
                        remaining_orders.remove(order)
                        orders_assigned += 1
                
                print(f"    SUCCESS: Packed {len(order_combo)} orders into vehicle {best_vehicle.id}")
                break  # Move to next pack size
    
    print(f"Multi-order packing assigned {orders_assigned} orders")
    return orders_assigned


def find_best_vehicle_for_combo(order_combo: Tuple['Order', ...], vehicles: List['Vehicle'], solution: 'Solution') -> Tuple[Optional['Vehicle'], Optional['Route']]:
    """
    Find the best vehicle to fit a combination of orders.
    
    Args:
        order_combo: Tuple of orders to pack together
        vehicles: Available vehicles
        solution: Current solution
        
    Returns:
        Tuple of (best_vehicle, best_route) or (None, None) if no fit
    """
    try:
        from second_level import l2_heuristic, calculate_z2_score
    except ImportError:
        try:
            from ..algo.second_level import l2_heuristic, calculate_z2_score
        except ImportError:
            return None, None
    
    best_cost = float('inf')
    best_vehicle = None
    best_route = None
    
    # Calculate total demand for the combination
    total_weight = sum(order.get_total_demand() for order in order_combo)
    total_volume = sum(order.get_total_volume() for order in order_combo)
    total_pallets = sum(order.get_total_pallets() for order in order_combo)
    
    for vehicle in vehicles:
        # Quick capacity check
        if (total_weight > vehicle.weight_capacity * 1.1 or  # Allow 10% overage
            total_volume > vehicle.volume_capacity * 1.1 or
            (vehicle.pallet_capacity and total_pallets > vehicle.pallet_capacity * 1.1)):
            continue
            
        # Get current route for vehicle
        current_route = solution.routes.get(vehicle.id)
        if not current_route:
            try:
                from first_level import _create_base_route
                current_route = _create_base_route(vehicle)
            except ImportError:
                try:
                    from ..algo.first_level import _create_base_route
                    current_route = _create_base_route(vehicle)
                except ImportError:
                    continue
        
        # Try to add all orders to this vehicle's route
        test_route = current_route.copy()
        
        # Add orders one by one using L2 heuristic
        all_orders_fit = True
        for order in order_combo:
            test_route = l2_heuristic(test_route, order, debug_assignment=False)
            if not test_route:
                all_orders_fit = False
                break
        
        if all_orders_fit and test_route:
            cost = calculate_z2_score(test_route)
            if cost < best_cost:
                best_cost = cost
                best_vehicle = vehicle
                best_route = test_route
    
    return best_vehicle, best_route


def try_progressive_relaxation_assignment(unassigned_orders: List['Order'], vehicles: List['Vehicle'], solution: 'Solution') -> int:
    """
    Try to assign orders with progressively relaxed constraints.
    
    Args:
        unassigned_orders: Orders that couldn't be assigned with standard constraints
        vehicles: Available vehicles  
        solution: Current solution to modify
        
    Returns:
        Number of orders successfully assigned
    """
    print(f"Attempting progressive relaxation assignment for {len(unassigned_orders)} orders...")
    
    orders_assigned = 0
    remaining_orders = unassigned_orders.copy()
    
    # Progressive relaxation levels
    relaxation_levels = [
        {"name": "15min time window extension", "time_extension": 15, "capacity_overage": 1.0},
        {"name": "30min extension + 5% capacity", "time_extension": 30, "capacity_overage": 1.05},
        {"name": "60min extension + 10% capacity", "time_extension": 60, "capacity_overage": 1.1},
        {"name": "2hr extension + 15% capacity", "time_extension": 120, "capacity_overage": 1.15},
        {"name": "4hr extension + 20% capacity", "time_extension": 240, "capacity_overage": 1.2},
    ]
    
    for level in relaxation_levels:
        if not remaining_orders:
            break
            
        print(f"  Trying relaxation level: {level['name']}")
        
        newly_assigned = []
        
        for order in remaining_orders:
            if try_assign_order_with_relaxation(order, vehicles, solution, level):
                newly_assigned.append(order)
                orders_assigned += 1
        
        # Remove assigned orders
        for order in newly_assigned:
            remaining_orders.remove(order)
            
        if newly_assigned:
            print(f"    Assigned {len(newly_assigned)} orders with {level['name']}")
    
    print(f"Progressive relaxation assigned {orders_assigned} orders")
    return orders_assigned


def try_assign_order_with_relaxation(order: 'Order', vehicles: List['Vehicle'], solution: 'Solution', relaxation: dict) -> bool:
    """
    Try to assign a single order with relaxed constraints.
    
    Args:
        order: Order to assign
        vehicles: Available vehicles
        solution: Current solution to modify
        relaxation: Relaxation parameters
        
    Returns:
        True if order was assigned, False otherwise
    """
    try:
        from second_level import l2_heuristic
    except ImportError:
        try:
            from ..algo.second_level import l2_heuristic
        except ImportError:
            return False
    
    # Try each vehicle with relaxed constraints
    for vehicle in vehicles:
        # Check relaxed capacity constraints
        capacity_limit_weight = vehicle.weight_capacity * relaxation['capacity_overage']
        capacity_limit_volume = vehicle.volume_capacity * relaxation['capacity_overage']
        capacity_limit_pallets = vehicle.pallet_capacity * relaxation['capacity_overage'] if vehicle.pallet_capacity else None
        
        order_weight = order.get_total_demand()
        order_volume = order.get_total_volume()
        order_pallets = order.get_total_pallets()
        
        if (order_weight > capacity_limit_weight or
            order_volume > capacity_limit_volume or
            (capacity_limit_pallets and order_pallets > capacity_limit_pallets)):
            continue
        
        # Get current route
        current_route = solution.routes.get(vehicle.id)
        if not current_route:
            try:
                from first_level import _create_base_route
                current_route = _create_base_route(vehicle)
            except ImportError:
                try:
                    from ..algo.first_level import _create_base_route
                    current_route = _create_base_route(vehicle)
                except ImportError:
                    continue
        
        # Create a copy of the order with relaxed time windows
        relaxed_order = create_relaxed_order(order, relaxation['time_extension'])
        
        # Try to assign with L2 heuristic
        new_route = l2_heuristic(current_route, relaxed_order, debug_assignment=False)
        
        if new_route:
            solution.routes[vehicle.id] = new_route
            return True
    
    return False


def create_relaxed_order(order: 'Order', time_extension_minutes: int) -> 'Order':
    """
    Create a copy of an order with relaxed time windows.
    
    Args:
        order: Original order
        time_extension_minutes: Minutes to extend time windows
        
    Returns:
        Order with relaxed time windows
    """
    try:
        # Create a deep copy of the order
        relaxed_order = copy.deepcopy(order)
        
        # Relax time windows for all tasks
        for task_list in [relaxed_order.pickup_tasks, relaxed_order.delivery_tasks]:
            for task in task_list:
                if hasattr(task, 'latest_time') and task.latest_time:
                    task.latest_time += time_extension_minutes
                if hasattr(task, 'earliest_time') and task.earliest_time:
                    task.earliest_time = max(0, task.earliest_time - time_extension_minutes)
        
        return relaxed_order
        
    except Exception:
        # If copying fails, return original order
        return order
