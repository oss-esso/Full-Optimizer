"""
Order Splitting Module

This module implements order splitting functionality to handle orders that are too large
to fit into any single vehicle or that have constraints making them difficult to assign.

The goal is to achieve 100% order assignment by intelligently splitting problematic orders
into smaller, more manageable sub-orders that can be assigned to multiple vehicles.
"""

from typing import List, Tuple, Optional, TYPE_CHECKING
import copy

if TYPE_CHECKING:
    from epdt_data_structures import Order, Task, Vehicle

def should_split_order(order: 'Order', vehicles: List['Vehicle']) -> bool:
    """
    Determine if an order should be split based on size and vehicle constraints.
    
    Args:
        order: Order to evaluate for splitting
        vehicles: Available vehicles
        
    Returns:
        True if order should be split, False otherwise
    """
    # Get the largest vehicle capacity
    max_weight = max(v.weight_capacity for v in vehicles)
    max_volume = max(v.volume_capacity for v in vehicles) 
    max_pallets = max(v.pallet_capacity for v in vehicles if v.pallet_capacity)
    
    order_weight = order.get_total_demand()
    order_volume = order.get_total_volume()
    order_pallets = order.get_total_pallets()
    
    # AGGRESSIVE SPLITTING FOR 100% ASSIGNMENT + STRICT PALLET LIMITS
    # Pallets cannot be overloaded, so split any order that could cause issues
    exceeds_weight = order_weight > max_weight * 0.4  # Aggressive: 40% threshold
    exceeds_volume = order_volume > max_volume * 0.4  # Aggressive  
    exceeds_pallets = order_pallets > max_pallets * 0.8 if max_pallets else False  # Conservative for pallets - can't overload
    
    # Split orders with moderately tight time windows too
    has_moderately_tight_window = False
    if order.pickup_tasks:
        for task in order.pickup_tasks:
            if hasattr(task, 'latest_time') and hasattr(task, 'earliest_time'):
                if task.latest_time and task.earliest_time:
                    window_size = task.latest_time - task.earliest_time
                    if window_size < 480:  # Less than 8 hours - split for better assignment
                        has_moderately_tight_window = True
                        break
    
    # SPLIT if ANY difficult characteristics - especially focus on pallet limits
    should_split = (exceeds_weight or exceeds_volume or exceeds_pallets or 
                   has_moderately_tight_window or order_pallets > 8 or order_weight > 1500)
    
    if should_split:
        print(f"AGGRESSIVE SPLIT: Order {order.id} - Weight: {order_weight:.0f}kg, Volume: {order_volume:.0f}m³, Pallets: {order_pallets} (Pallet limit critical: {exceeds_pallets})")
    
    return should_split


def split_order_by_tasks(order: 'Order', max_tasks_per_split: int = 2) -> List['Order']:
    """
    Split an order by dividing its tasks into smaller sub-orders.
    
    Args:
        order: Order to split
        max_tasks_per_split: Maximum pickup/delivery pairs per split order
        
    Returns:
        List of smaller sub-orders
    """
    if not order.pickup_tasks:
        return [order]  # Can't split order with no pickups
        
    # Group pickup and delivery tasks by location/customer
    pickup_delivery_pairs = []
    
    # Simple approach: pair each pickup with its corresponding delivery
    for i, pickup_task in enumerate(order.pickup_tasks):
        corresponding_delivery = None
        
        # Find delivery task for same order at same location
        for delivery_task in order.delivery_tasks:
            if (hasattr(pickup_task, 'order_id') and hasattr(delivery_task, 'order_id') and
                pickup_task.order_id == delivery_task.order_id):
                corresponding_delivery = delivery_task
                break
        
        if corresponding_delivery:
            pickup_delivery_pairs.append((pickup_task, corresponding_delivery))
        else:
            pickup_delivery_pairs.append((pickup_task, None))
    
    # Split into smaller groups
    sub_orders = []
    for i in range(0, len(pickup_delivery_pairs), max_tasks_per_split):
        chunk = pickup_delivery_pairs[i:i + max_tasks_per_split]
        
        # Create new sub-order
        pickup_tasks = []
        delivery_tasks = []
        
        for pickup_task, delivery_task in chunk:
            pickup_tasks.append(copy.deepcopy(pickup_task))
            if delivery_task:
                delivery_tasks.append(copy.deepcopy(delivery_task))
        
        # Create sub-order with modified ID
        sub_order_id = f"{order.id}_split_{len(sub_orders) + 1}"
        
        # Import here to avoid circular imports
        from epdt_data_structures import Order
        sub_order = Order(
            id=sub_order_id,
            pickup_tasks=pickup_tasks,
            delivery_tasks=delivery_tasks,
            priority=order.priority,
            is_urgent=order.is_urgent,
            is_mandatory=order.is_mandatory
        )
        
        sub_orders.append(sub_order)
    
    return sub_orders


def split_order_by_weight(order: 'Order', max_weight_per_split: float) -> List['Order']:
    """
    Split an order by weight, ensuring no sub-order exceeds the weight limit.
    
    Args:
        order: Order to split  
        max_weight_per_split: Maximum weight per split order
        
    Returns:
        List of smaller sub-orders
    """
    if not order.pickup_tasks:
        return [order]
        
    current_weight = 0.0
    current_pickup_tasks = []
    current_delivery_tasks = []
    sub_orders = []
    
    # Sort tasks by weight descending to pack efficiently
    sorted_pickup_tasks = sorted(order.pickup_tasks, key=lambda t: t.demand, reverse=True)
    
    for pickup_task in sorted_pickup_tasks:
        task_weight = pickup_task.demand
        
        # If adding this task would exceed weight limit, create a new sub-order
        if current_weight + task_weight > max_weight_per_split and current_pickup_tasks:
            # Create sub-order with current tasks
            sub_order_id = f"{order.id}_weight_split_{len(sub_orders) + 1}"
            
            from epdt_data_structures import Order
            sub_order = Order(
                id=sub_order_id,
                pickup_tasks=copy.deepcopy(current_pickup_tasks),
                delivery_tasks=copy.deepcopy(current_delivery_tasks),
                priority=order.priority,
                is_urgent=order.is_urgent,
                is_mandatory=order.is_mandatory
            )
            sub_orders.append(sub_order)
            
            # Reset for next sub-order
            current_pickup_tasks = []
            current_delivery_tasks = []
            current_weight = 0.0
        
        # Add task to current sub-order
        current_pickup_tasks.append(pickup_task)
        current_weight += task_weight
        
        # Find corresponding delivery task
        for delivery_task in order.delivery_tasks:
            if (hasattr(pickup_task, 'order_id') and hasattr(delivery_task, 'order_id') and
                pickup_task.order_id == delivery_task.order_id):
                current_delivery_tasks.append(delivery_task)
                break
    
    # Add remaining tasks as final sub-order
    if current_pickup_tasks:
        sub_order_id = f"{order.id}_weight_split_{len(sub_orders) + 1}"
        
        from epdt_data_structures import Order
        sub_order = Order(
            id=sub_order_id,
            pickup_tasks=copy.deepcopy(current_pickup_tasks),
            delivery_tasks=copy.deepcopy(current_delivery_tasks), 
            priority=order.priority,
            is_urgent=order.is_urgent,
            is_mandatory=order.is_mandatory
        )
        sub_orders.append(sub_order)
    
    return sub_orders if sub_orders else [order]


def apply_order_splitting(orders: List['Order'], vehicles: List['Vehicle']) -> List['Order']:
    """
    Apply order splitting to problematic orders in the order list.
    
    Args:
        orders: List of original orders
        vehicles: Available vehicles for capacity reference
        
    Returns:
        Modified order list with split orders
    """
    split_orders = []
    split_count = 0
    
    for order in orders:
        if should_split_order(order, vehicles):
            # Try weight-based splitting first for large orders
            if order.get_total_demand() > 3000:  # Large weight orders
                sub_orders = split_order_by_weight(order, max(v.weight_capacity * 0.6 for v in vehicles))
            else:
                # Split by task count for complex orders
                sub_orders = split_order_by_tasks(order, max_tasks_per_split=1)
            
            split_orders.extend(sub_orders)
            split_count += 1
            
            print(f"Split order {order.id} into {len(sub_orders)} sub-orders")
            
        else:
            split_orders.append(order)
    
    if split_count > 0:
        print(f"Order splitting applied: {split_count} orders split into {len(split_orders)} total orders")
    
    return split_orders
