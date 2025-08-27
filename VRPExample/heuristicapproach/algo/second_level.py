"""
Second-Level Heuristic Implementation for EPDT Algorithm

This module implements the L2 (Second-Level) heuristic which handles intra-route 
optimization within the EPDT (Enhanced Parallel Diversified Tabu) algorithm.

Key Components:
- l2_heuristic: Main entry point for order insertion into routes
- Neighborhood generation functions: Task swap and insertion operations
- Z2 scoring function: Comprehensive route evaluation including HOS compliance
- Local search optimization: First/best improvement strategies

Architecture Notes:
- Separation of Concerns: L2 handles only intra-route optimization
- Modular Design: Each neighborhood operation is a separate generator function
- Performance Optimization: Uses Numba JIT compilation for critical functions
- State Management: Proper deep copying ensures immutable operations
"""

import math
from typing import List, Optional, Iterator, Callable, TYPE_CHECKING, Union, Tuple, Dict
from epdt_data_structures import DriverState, Task
import copy

# Global counter for tracking distance/travel time calculations
_distance_calculation_counter = 0

def get_distance_calculation_count() -> int:
    """Get the current count of distance calculations (equivalent to OSRM calls)."""
    global _distance_calculation_counter
    return _distance_calculation_counter

def reset_distance_calculation_count():
    """Reset the distance calculation counter."""
    global _distance_calculation_counter
    _distance_calculation_counter = 0

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth using the Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of first point in decimal degrees
        lat2, lon2: Latitude and longitude of second point in decimal degrees
        
    Returns:
        Distance in kilometers
    """
    global _distance_calculation_counter
    _distance_calculation_counter += 1
    
    # Convert decimal degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (math.sin(dlat/2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth's radius in kilometers
    earth_radius_km = 6371.0
    
    distance_km = earth_radius_km * c
    return distance_km

def calculate_travel_time_haversine(lat1: float, lon1: float, lat2: float, lon2: float, speed_kmh: float = 60.0) -> float:
    """
    Calculate travel time between two points using Haversine distance and constant speed.
    
    Args:
        lat1, lon1: Starting point coordinates
        lat2, lon2: Ending point coordinates  
        speed_kmh: Travel speed in km/h (default: 100 km/h)
        
    Returns:
        Travel time in minutes
    """
    # Note: Counter is incremented in haversine_distance, not here to avoid double counting
    distance_km = haversine_distance(lat1, lon1, lat2, lon2)
    travel_time_hours = distance_km / speed_kmh
    travel_time_minutes = travel_time_hours * 60.0
    return travel_time_minutes

# Import modular HoS simulation to resolve circular imports
from hos_simulation import (
    simulate_hos_advanced as _simulate_hos_advanced,
    sort_tasks_chronologically as _sort_tasks_chronologically,
    validate_route_hos_feasibility
)
# Import centralized travel time calculation
from route_provider import calculate_travel_time_between_tasks as _calculate_travel_time_between_tasks
from dataclasses import dataclass
import copy
import math

# Numba for JIT compilation of performance-critical functions
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    # Fallback decorator if Numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    NUMBA_AVAILABLE = False

if TYPE_CHECKING:
    try:
        from .epdt_data_structures import Route, Order
    except ImportError:
        from epdt_data_structures import Route, Order
else:
    try:
        from .epdt_data_structures import Route, Order
    except ImportError:
        from epdt_data_structures import Route, Order


def check_hard_constraints(route: 'Route', debug: bool = False) -> Tuple[bool, str]:
    """
    UNIFIED HARD CONSTRAINT CHECKER
    
    This function checks constraints that should NEVER be violated under any circumstances:
    1. Vehicle capabilities (LOADER, LOW_TEMP, HANGERS) - Legal/equipment requirements
    2. Pallet capacity - Physical safety limits
    3. LIFO constraints - Physical loading requirements
    4. Precedence constraints - Pickup before delivery for each order
    
    IMPORTANT: This function validates the existing task order without modifying it.
    
    Args:
        route: Route to validate
        debug: Whether to print debug information
        
    Returns:
        Tuple[bool, str]: (is_valid, reason_if_invalid)
    """
    if not route or not route.tasks:
        return True, "Empty route is valid"
        
    # VALIDATE EXISTING TASK ORDER (don't reorder!)
    tasks = route.tasks
    
    # Check precedence constraints first
    if not _validate_precedence_constraints(tasks):
        reason = "HARD CONSTRAINT VIOLATION: Precedence constraints violated (pickup must come before delivery)"
        if debug:
            print(f"            HARD CONSTRAINT CHECK: {reason}")
        return False, reason
    
    # HARD CONSTRAINT 1: VEHICLE CAPABILITIES
    # Check if vehicle has all required capabilities
    all_required_capabilities = set()
    for task in tasks:
        # Check individual capability flags
        if hasattr(task, 'requires_loader') and task.requires_loader:
            all_required_capabilities.add('LOADER')
        if hasattr(task, 'requires_low_temp') and task.requires_low_temp:
            all_required_capabilities.add('LOW_TEMP')
        if hasattr(task, 'requires_hangers') and task.requires_hangers:
            all_required_capabilities.add('HANGERS')
        
        # Check required_capabilities string/list (comprehensive check)
        if hasattr(task, 'required_capabilities') and task.required_capabilities:
            capabilities_str = str(task.required_capabilities).upper()
            if 'LOADER' in capabilities_str:
                all_required_capabilities.add('LOADER')
            if 'LOW_TEMP' in capabilities_str or 'LOW_TEMPERATURE' in capabilities_str:
                all_required_capabilities.add('LOW_TEMP')
            if 'HANGERS' in capabilities_str:
                all_required_capabilities.add('HANGERS')
    
    if all_required_capabilities:
        # Get vehicle capabilities from the capabilities set
        vehicle_capabilities = set()
        if hasattr(route.vehicle, 'capabilities'):
            if isinstance(route.vehicle.capabilities, (list, set)):
                # Convert capabilities to uppercase and normalize
                for cap in route.vehicle.capabilities:
                    cap_str = str(cap).upper()
                    if cap_str == 'LOW TEMP':
                        vehicle_capabilities.add('LOW_TEMP')  # Normalize to underscore
                    elif cap_str == 'LOADER':
                        vehicle_capabilities.add('LOADER')
                    elif cap_str == 'HANGERS':
                        vehicle_capabilities.add('HANGERS')
            elif isinstance(route.vehicle.capabilities, str):
                caps = [cap.strip().upper() for cap in route.vehicle.capabilities.split(',') if cap.strip()]
                for cap in caps:
                    if cap == 'LOW TEMP':
                        vehicle_capabilities.add('LOW_TEMP')
                    elif cap == 'LOADER':
                        vehicle_capabilities.add('LOADER')
                    elif cap == 'HANGERS':
                        vehicle_capabilities.add('HANGERS')
        
        # REMOVED: The wrong individual capability flag checks that don't exist
        # (vehicles don't have has_loader, has_low_temp, has_hangers attributes)
        
        if not vehicle_capabilities.issuperset(all_required_capabilities):
            missing_caps = all_required_capabilities - vehicle_capabilities
            reason = f"HARD CONSTRAINT VIOLATION: Vehicle {route.vehicle.id} missing capabilities: {', '.join(missing_caps)}"
            if debug:
                print(f"            HARD CONSTRAINT CHECK: {reason}")
            return False, reason
    
    # HARD CONSTRAINT 2: PALLET CAPACITY  
    # This is a physical safety limit that cannot be exceeded
    max_pallets = route.vehicle.pallet_capacity
    if max_pallets is not None:
        current_pallets = 0
        for task in tasks:
            current_pallets += getattr(task, 'pallets', 0)  # pallets can be negative for deliveries
            if current_pallets > max_pallets:
                reason = f"HARD CONSTRAINT VIOLATION: Pallet capacity exceeded: {current_pallets} > {max_pallets} at task {task.id}"
                if debug:
                    print(f"            HARD CONSTRAINT CHECK: {reason}")
                return False, reason
    
    # HARD CONSTRAINT 3: LIFO LOADING CONSTRAINTS
    # Physical loading constraint that cannot be violated
    if hasattr(route.vehicle, 'lifo_required') and route.vehicle.lifo_required:
        lifo_stack = []
        
        # Handle delivery-only orders by pre-loading stack
        delivery_orders = set()
        pickup_orders = set()
        
        for task in tasks:
            if task.is_delivery():
                delivery_orders.add(task.order_id)
            elif task.is_pickup():
                pickup_orders.add(task.order_id)
        
        delivery_only_orders = delivery_orders - pickup_orders
        if delivery_only_orders:
            lifo_stack = list(delivery_only_orders)
        
        # Check LIFO constraint during execution
        for task in tasks:
            if task.is_pickup():
                lifo_stack.append(task.order_id)
            elif task.is_delivery():
                if not lifo_stack:
                    reason = f"HARD CONSTRAINT VIOLATION: LIFO violation - trying to deliver {task.id} when no cargo loaded"
                    if debug:
                        print(f"            HARD CONSTRAINT CHECK: {reason}")
                    return False, reason
                if lifo_stack[-1] != task.order_id:
                    reason = f"HARD CONSTRAINT VIOLATION: LIFO violation - expected {lifo_stack[-1]}, got {task.order_id}"
                    if debug:
                        print(f"            HARD CONSTRAINT CHECK: {reason}")
                    return False, reason
                lifo_stack.pop()
        
        # Final LIFO check
        if lifo_stack:
            reason = f"HARD CONSTRAINT VIOLATION: LIFO constraint - undelivered cargo: {lifo_stack}"
            if debug:
                print(f"            HARD CONSTRAINT CHECK: {reason}")
            return False, reason
    
    if debug:
        print(f"            HARD CONSTRAINT CHECK: All hard constraints satisfied for vehicle {route.vehicle.id}")
    
    return True, "All hard constraints satisfied"


def _validate_precedence_constraints(tasks: List) -> bool:
    """
    Validate that pickup comes before delivery for each order.
    Allows flexible interleaving of multiple orders.
    """
    # Group tasks by order
    orders_tasks = {}
    
    for pos, task in enumerate(tasks):
        order_id = getattr(task, 'order_id', None)
        task_type = getattr(task, 'task_type', None)
        
        if order_id is None or task_type is None:
            continue
        
        # Convert TaskType enum to string if needed
        if hasattr(task_type, 'value'):
            task_type = task_type.value
        
        if task_type not in ['pickup', 'delivery']:
            continue
            
        if order_id not in orders_tasks:
            orders_tasks[order_id] = []
        
        orders_tasks[order_id].append({'task_type': task_type, 'position': pos})
    
    # Check precedence for each order
    for order_id, order_tasks in orders_tasks.items():
        pickups = [t for t in order_tasks if t['task_type'] == 'pickup']
        deliveries = [t for t in order_tasks if t['task_type'] == 'delivery']
        
        # Skip if no pickup/delivery pair
        if not pickups or not deliveries:
            continue
        
        # For simple orders: pickup must come before delivery
        if len(pickups) == 1 and len(deliveries) == 1:
            if pickups[0]['position'] >= deliveries[0]['position']:
                return False
        
        # For complex orders: all pickups must come before all deliveries
        else:
            max_pickup_pos = max(p['position'] for p in pickups)
            min_delivery_pos = min(d['position'] for d in deliveries)
            
            if max_pickup_pos >= min_delivery_pos:
                return False
    
    return True


def calculate_sequence_peak_capacity(task_sequence: List['Task']) -> Dict[str, float]:
    """
    Calculate peak capacity usage (weight, volume, pallets) for a given task sequence.
    
    This is crucial for determining which sequences are feasible for different vehicles.
    For example, Order 7 with PPPDDD sequence needs 21 pallets peak capacity,
    but PDPDPD sequence only needs 8 pallets peak capacity.
    
    Enhanced to track the actual peak point for each dimension separately.
    
    Args:
        task_sequence: Ordered list of tasks to execute
        
    Returns:
        Dict with peak_weight, peak_volume, peak_pallets, peak_details, and sequence_info
    """
    current_weight = 0.0
    current_volume = 0.0
    current_pallets = 0.0
    
    peak_weight = 0.0
    peak_volume = 0.0
    peak_pallets = 0.0
    
    # Track when each peak occurs for debugging
    peak_weight_step = 0
    peak_volume_step = 0
    peak_pallets_step = 0
    peak_weight_details = {}
    peak_volume_details = {}
    peak_pallets_details = {}
    
    sequence_info = []
    
    for i, task in enumerate(task_sequence):
        # Get task demands (positive for pickup, negative for delivery)
        task_weight = getattr(task, 'demand', 0.0) if hasattr(task, 'demand') else 0.0
        task_volume = getattr(task, 'volume', 0.0) if hasattr(task, 'volume') else 0.0
        task_pallets = getattr(task, 'pallets', 0.0) if hasattr(task, 'pallets') else 0.0
        
        # Update current load
        current_weight += task_weight
        current_volume += task_volume
        current_pallets += task_pallets
        
        # Track peaks and when they occur
        if current_weight > peak_weight:
            peak_weight = current_weight
            peak_weight_step = i + 1
            peak_weight_details = {
                'weight': current_weight,
                'volume': current_volume, 
                'pallets': current_pallets,
                'step': i + 1,
                'task_id': getattr(task, 'id', 'unknown')
            }
            
        if current_volume > peak_volume:
            peak_volume = current_volume
            peak_volume_step = i + 1
            peak_volume_details = {
                'weight': current_weight,
                'volume': current_volume, 
                'pallets': current_pallets,
                'step': i + 1,
                'task_id': getattr(task, 'id', 'unknown')
            }
            
        if current_pallets > peak_pallets:
            peak_pallets = current_pallets
            peak_pallets_step = i + 1
            peak_pallets_details = {
                'weight': current_weight,
                'volume': current_volume, 
                'pallets': current_pallets,
                'step': i + 1,
                'task_id': getattr(task, 'id', 'unknown')
            }
        
        # Store sequence info for debugging
        task_type = getattr(task, 'task_type', None)
        task_type_name = task_type.name if hasattr(task_type, 'name') else str(task_type)
        sequence_info.append({
            'step': i + 1,
            'task_id': getattr(task, 'id', 'unknown'),
            'task_type': task_type_name,
            'change': {'weight': task_weight, 'volume': task_volume, 'pallets': task_pallets},
            'total': {'weight': current_weight, 'volume': current_volume, 'pallets': current_pallets}
        })
    
    return {
        'peak_weight': peak_weight,
        'peak_volume': peak_volume,
        'peak_pallets': peak_pallets,
        'peak_details': {
            'weight_peak': peak_weight_details,
            'volume_peak': peak_volume_details,
            'pallets_peak': peak_pallets_details
        },
        'sequence_info': sequence_info
    }


def generate_valid_task_permutations(tasks: List['Task']) -> List[Dict]:
    """
    Generate all valid permutations of tasks respecting precedence constraints.
    Enhanced to detect and optimize sequential pickup-delivery chains.
    NOW INCLUDES PEAK CAPACITY TRACKING for each sequence.
    
    Args:
        tasks: List of Task objects (mix of pickups and deliveries)
        
    Returns:
        List of dictionaries containing:
        - 'sequence': List of tasks
        - 'peak_capacity': Dict with peak_weight, peak_volume, peak_pallets
        - 'is_sequential_chain': Boolean indicating if this is an optimized chain
    """
    from itertools import permutations
    
    valid_sequences = []
    
    # Create mapping from pickup to delivery for precedence checking
    # Use task IDs as keys instead of Task objects to avoid hashability issues
    pickup_to_delivery = {}
    deliveries = [t for t in tasks if hasattr(t, 'task_type') and t.task_type.name == 'DELIVERY']
    pickups = [t for t in tasks if hasattr(t, 'task_type') and t.task_type.name == 'PICKUP']
    
    # Map pickups to deliveries based on location or order logic
    for pickup in pickups:
        for delivery in deliveries:
            # Simple mapping based on location proximity or order ID
            if hasattr(pickup, 'order_id') and hasattr(delivery, 'order_id'):
                if pickup.order_id == delivery.order_id:
                    pickup_id = getattr(pickup, 'id', id(pickup))
                    delivery_id = getattr(delivery, 'id', id(delivery))
                    pickup_to_delivery[pickup_id] = delivery_id
                    break
    
    # ENHANCED: Detect sequential pickup-delivery chains (PDP-DP-DP pattern)
    sequential_chain = detect_sequential_pickup_delivery_chain(pickups, deliveries)
    
    if sequential_chain:
        print(f"    DETECTED SEQUENTIAL CHAIN: Order has connected pickup-delivery sequence")
        # Calculate peak capacity for the sequential chain
        peak_capacity = calculate_sequence_peak_capacity(sequential_chain)
        valid_sequences.append({
            'sequence': sequential_chain,
            'peak_capacity': peak_capacity,
            'is_sequential_chain': True
        })
        print(f"    Sequential chain peak usage: {peak_capacity['peak_pallets']:.0f}pal, {peak_capacity['peak_weight']:.0f}kg, {peak_capacity['peak_volume']:.1f}m³")
        
        # Enhanced: Show details about when each peak occurs
        if 'peak_details' in peak_capacity:
            details = peak_capacity['peak_details']
            print(f"      Peak pallets: {details['pallets_peak']['pallets']:.0f}pal at step {details['pallets_peak']['step']} ({details['pallets_peak']['weight']:.0f}kg, {details['pallets_peak']['volume']:.1f}m³)")
            print(f"      Peak weight: {details['weight_peak']['weight']:.0f}kg at step {details['weight_peak']['step']} ({details['weight_peak']['pallets']:.0f}pal, {details['weight_peak']['volume']:.1f}m³)")
            print(f"      Peak volume: {details['volume_peak']['volume']:.1f}m³ at step {details['volume_peak']['step']} ({details['volume_peak']['pallets']:.0f}pal, {details['volume_peak']['weight']:.0f}kg)")
        
        # IMPORTANT: Also generate alternative sequences to compare capacity usage
        print(f"    Generating alternative sequences for capacity comparison...")
        
    # Generate all permutations and filter valid ones (enhanced with capacity tracking)
    permutation_count = 0
    for perm in permutations(tasks):
        if is_valid_task_sequence(perm, pickup_to_delivery):
            peak_capacity = calculate_sequence_peak_capacity(list(perm))
            valid_sequences.append({
                'sequence': list(perm),
                'peak_capacity': peak_capacity,
                'is_sequential_chain': False
            })
            permutation_count += 1
            
            # Debug output for first few permutations
            if permutation_count <= 3:
                sequence_types = [getattr(t.task_type, 'name', 'UNK')[0] for t in perm]
                sequence_pattern = ''.join(sequence_types)
                print(f"    Sequence {permutation_count} ({sequence_pattern}): {peak_capacity['peak_pallets']:.0f}pal peak")
    
    # Sort sequences by peak capacity usage (lower is better for fitting in vehicles)
    valid_sequences.sort(key=lambda x: (x['peak_capacity']['peak_pallets'], 
                                      x['peak_capacity']['peak_weight'], 
                                      x['peak_capacity']['peak_volume']))
    
    print(f"    Generated {len(valid_sequences)} valid sequences, sorted by peak capacity")
    
    return valid_sequences


def detect_sequential_pickup_delivery_chain(pickups: List['Task'], deliveries: List['Task']) -> Optional[List['Task']]:
    """
    Detect if tasks form a sequential pickup-delivery chain where each delivery location
    becomes the next pickup location (PDP-DP-DP pattern).
    
    Args:
        pickups: List of pickup tasks
        deliveries: List of delivery tasks
        
    Returns:
        Optimized task sequence if sequential chain detected, None otherwise
    """
    if len(pickups) != len(deliveries) or len(pickups) < 2:
        return None
    
    # Check if locations form a sequential chain
    # Format: P1@A → D1@B, P2@B → D2@C, P3@C → D3@D
    
    sequential_tasks = []
    
    # Build location mappings
    pickup_locations = {(task.lat, task.lon): task for task in pickups}
    delivery_locations = {(task.lat, task.lon): task for task in deliveries}
    
    # Find the starting pickup (one that doesn't have a delivery at the same location)
    current_pickup = None
    for pickup in pickups:
        pickup_loc = (pickup.lat, pickup.lon)
        if pickup_loc not in delivery_locations:
            current_pickup = pickup
            break
    
    if not current_pickup:
        # Try alternative: pick any pickup to start the chain
        current_pickup = pickups[0]
    
    # Build the sequential chain
    used_pickup_ids = set()
    used_delivery_ids = set()
    
    while current_pickup and current_pickup.id not in used_pickup_ids:
        sequential_tasks.append(current_pickup)
        used_pickup_ids.add(current_pickup.id)
        
        # Find delivery for this pickup (should be at the next location)
        pickup_loc = (current_pickup.lat, current_pickup.lon)
        corresponding_delivery = None
        
        for delivery in deliveries:
            if delivery.id not in used_delivery_ids and hasattr(delivery, 'order_id') and delivery.order_id == current_pickup.order_id:
                corresponding_delivery = delivery
                break
        
        if corresponding_delivery:
            sequential_tasks.append(corresponding_delivery)
            used_delivery_ids.add(corresponding_delivery.id)
            
            # Find next pickup at the delivery location
            delivery_loc = (corresponding_delivery.lat, corresponding_delivery.lon)
            current_pickup = None
            
            for pickup in pickups:
                if pickup.id not in used_pickup_ids:
                    next_pickup_loc = (pickup.lat, pickup.lon)
                    # Check if next pickup is at or near the delivery location
                    if (abs(next_pickup_loc[0] - delivery_loc[0]) < 0.001 and 
                        abs(next_pickup_loc[1] - delivery_loc[1]) < 0.001):
                        current_pickup = pickup
                        break
        else:
            break
    
    # Verify we used all tasks in a valid sequential pattern
    if len(used_pickup_ids) == len(pickups) and len(used_delivery_ids) == len(deliveries):
        print(f"    Sequential chain validated: {len(sequential_tasks)} tasks in sequence")
        return sequential_tasks
    
    return None


def is_valid_task_sequence(sequence: List['Task'], pickup_to_delivery: dict) -> bool:
    """
    Check if a task sequence respects precedence constraints.
    
    Args:
        sequence: Sequence of tasks to validate
        pickup_to_delivery: Mapping from pickup task IDs to delivery task IDs
        
    Returns:
        True if sequence is valid, False otherwise
    """
    # Create position mapping using task IDs
    task_id_positions = {}
    for i, task in enumerate(sequence):
        task_id = getattr(task, 'id', id(task))
        task_id_positions[task_id] = i
    
    for pickup_id, delivery_id in pickup_to_delivery.items():
        if pickup_id in task_id_positions and delivery_id in task_id_positions:
            if task_id_positions[pickup_id] >= task_id_positions[delivery_id]:
                return False  # Delivery before pickup is invalid
    
    return True


def evaluate_permutation_at_insertion_point(route: 'Route', task_sequence: List['Task'], 
                                           insertion_point: int, debug: bool = False) -> Optional[dict]:
    """
    Evaluate a task sequence at a specific insertion point in the route.
    
    Args:
        route: Current route
        task_sequence: Sequence of tasks to insert
        insertion_point: Position where to insert the sequence
        debug: Whether to print debug information
        
    Returns:
        Dictionary with feasibility, cost, and route info, or None if infeasible
    """
    if debug:
        print(f"      Evaluating sequence at position {insertion_point}")
    
    # Create temporary route with sequence inserted
    temp_route = route.copy()
    
    # Insert tasks at the specified position
    for i, task in enumerate(task_sequence):
        temp_route.insert_task_without_reordering(insertion_point + i, task)
    
    # Check feasibility with real-time load simulation
    if not is_feasible_with_load_tracking(temp_route, insertion_point, task_sequence, debug):
        if debug:
            print(f"      Sequence infeasible at position {insertion_point}")
        return None
    
    # Calculate cost (negative because we want to minimize cost)
    cost = calculate_z2_score(temp_route)
    
    if debug:
        print(f"      Sequence feasible at position {insertion_point}, cost: {cost:.2f}")
    
    return {
        'feasible': True,
        'cost': cost,
        'route': temp_route,
        'insertion_point': insertion_point
    }


def is_feasible_with_load_tracking(route: 'Route', insertion_point: int, 
                                 task_sequence: List['Task'], debug: bool = False) -> bool:
    """
    Check feasibility using real-time load simulation.
    
    Args:
        route: Route with tasks already inserted
        insertion_point: Where the sequence was inserted
        task_sequence: The inserted task sequence
        debug: Whether to print debug information
        
    Returns:
        True if feasible, False otherwise
    """
    if debug:
        print(f"        Load tracking from insertion point {insertion_point}")
    
    # Calculate initial load at insertion point
    current_load = 0
    for i in range(insertion_point):
        if i < len(route.tasks):
            task = route.tasks[i]
            if hasattr(task, 'demand'):
                current_load += task.demand
    
    if debug:
        print(f"        Initial load at insertion: {current_load}")
    
    # Simulate load changes through the inserted sequence
    for i, task in enumerate(task_sequence):
        if hasattr(task, 'demand'):
            current_load += task.demand
        
        if debug:
            print(f"        After task {i+1}: load = {current_load}")
        
        # Check constraints
        if current_load < 0:
            if debug:
                print(f"        VIOLATION: Negative load {current_load}")
            return False
        
        if hasattr(route.vehicle, 'weight_capacity') and current_load > route.vehicle.weight_capacity:
            if debug:
                print(f"        VIOLATION: Exceeds capacity {current_load} > {route.vehicle.weight_capacity}")
            return False
    
    # Check overall route feasibility
    try:
        overall_feasible = route.is_feasible() if hasattr(route, 'is_feasible') else True
        if debug and not overall_feasible:
            print(f"        VIOLATION: Overall route infeasible")
        return overall_feasible
    except Exception as e:
        if debug:
            print(f"        ERROR checking feasibility: {e}")
        return False


def calculate_sequence_profit(route: 'Route', order: 'Order', cost: float) -> float:
    """
    Calculate profit for a sequence insertion.
    
    Args:
        route: Route with order inserted
        order: The order being inserted
        cost: Cost of the route
        
    Returns:
        Profit (revenue - cost)
    """
    # Calculate revenue based on order value or distance
    revenue = 0.0
    
    # Simple revenue calculation based on order demands
    if hasattr(order, 'get_total_demand'):
        total_demand = order.get_total_demand()
        # Use price per unit (example: €1.2 per kg)
        revenue = total_demand * 1.2
    
    # Alternative: calculate based on pickup-delivery distances
    pickups = order.get_pickups()
    deliveries = order.get_deliveries()
    
    if pickups and deliveries:
        # Calculate distance-based revenue
        total_distance = 0.0
        for pickup in pickups:
            for delivery in deliveries:
                if hasattr(pickup, 'latitude') and hasattr(delivery, 'latitude'):
                    distance = haversine_distance(pickup.latitude, pickup.longitude,
                                                delivery.latitude, delivery.longitude)
                    total_distance += distance
        
        # Price per km based on vehicle type
        price_per_km = 0.8  # Default for standard vehicles
        if hasattr(route.vehicle, 'vehicle_type') and 'camion' in str(route.vehicle.vehicle_type).lower():
            price_per_km = 1.25  # Higher rate for heavy vehicles
        
        distance_revenue = total_distance * price_per_km
        revenue = max(revenue, distance_revenue)
    
    profit = revenue - abs(cost)  # Cost is usually negative in Z2 scoring
    return profit


def find_best_sequence_for_complex_order(route: 'Route', order: 'Order', debug: bool = False) -> Optional['Route']:
    """
    Find the best sequence for a complex order using advanced sequencing logic.
    
    This replaces the aggressive splitting approach with intelligent task sequencing
    that maximizes profit while respecting all constraints.
    
    Args:
        route: Current vehicle route
        order: Complex order to be inserted
        debug: Whether to print debug information
        
    Returns:
        Optimized route with order inserted, or None if infeasible
    """
    if debug:
        print(f"    Advanced Sequencing: Processing complex order {order.id}")
    
    # Get all tasks from the order
    all_tasks = order.get_pickups() + order.get_deliveries()
    
    if not all_tasks:
        if debug:
            print(f"    No tasks found in order {order.id}")
        return None
    
    if debug:
        print(f"    Order has {len(order.get_pickups())} pickups and {len(order.get_deliveries())} deliveries")
    
    # Step 2: Generate all valid task permutations
    valid_sequences = generate_valid_task_permutations(all_tasks)
    
    if debug:
        print(f"    Generated {len(valid_sequences)} valid sequences")
    
    if not valid_sequences:
        if debug:
            print(f"    No valid sequences found for order {order.id}")
        return None
    
    # Step 3: Evaluate each valid permutation at all possible insertion points
    best_result = None
    best_profit = float('-inf')
    
    for seq_idx, sequence_data in enumerate(valid_sequences):
        sequence = sequence_data['sequence']
        peak_capacity = sequence_data['peak_capacity']
        is_sequential = sequence_data['is_sequential_chain']
        
        if debug:
            print(f"    Evaluating sequence {seq_idx + 1}/{len(valid_sequences)}")
            sequence_types = [getattr(t.task_type, 'name', 'UNK')[0] for t in sequence]
            sequence_pattern = ''.join(sequence_types)
            print(f"      Pattern: {sequence_pattern}, Peak: {peak_capacity['peak_pallets']:.0f}pal/{peak_capacity['peak_weight']:.0f}kg/{peak_capacity['peak_volume']:.1f}m³")
            if is_sequential:
                print(f"      *** SEQUENTIAL CHAIN DETECTED ***")
        
        # Check if vehicle can handle the peak capacity requirement
        if route.vehicle:
            vehicle_weight = getattr(route.vehicle, 'weight_capacity', float('inf'))
            vehicle_volume = getattr(route.vehicle, 'volume_capacity', float('inf'))
            vehicle_pallets = getattr(route.vehicle, 'pallet_capacity', float('inf'))
            
            if (peak_capacity['peak_weight'] > vehicle_weight or 
                peak_capacity['peak_volume'] > vehicle_volume or 
                peak_capacity['peak_pallets'] > vehicle_pallets):
                if debug:
                    print(f"      CAPACITY EXCEEDED: Vehicle({vehicle_weight:.0f}kg/{vehicle_volume:.1f}m³/{vehicle_pallets:.0f}pal) < Required({peak_capacity['peak_weight']:.0f}kg/{peak_capacity['peak_volume']:.1f}m³/{peak_capacity['peak_pallets']:.0f}pal)")
                continue  # Skip this sequence - vehicle can't handle peak load
        
        # Try all insertion points (after depot start, before depot return)
        for insertion_point in range(1, len(route.tasks)):
            result = evaluate_permutation_at_insertion_point(route, sequence, insertion_point, debug)
            
            if result and result['feasible']:
                # Calculate profit for this sequence
                profit = calculate_sequence_profit(result['route'], order, result['cost'])
                
                if debug:
                    print(f"      Sequence {seq_idx + 1} at position {insertion_point}: profit = {profit:.2f}")
                
                if profit > best_profit:
                    best_profit = profit
                    best_result = result
                    if debug:
                        print(f"      New best sequence found: profit = {profit:.2f}")
    
    # Step 4: Return the best sequence
    if best_result and best_profit > 0:  # Only accept profitable insertions
        if debug:
            print(f"    Best sequence selected with profit: {best_profit:.2f}")
        return best_result['route']
    else:
        if debug:
            print(f"    No profitable sequence found for complex order {order.id}")
        return None


def l2_heuristic(route: 'Route', order: 'Order', debug_assignment: bool = False, enhanced_diagnostics: bool = False, sequencing_strategy: str = 'clustered') -> Optional['Route']:
    """
    Second-Level Heuristic: Finds the best way to insert an order into a route.
    
    This implements the L2 heuristic from the EPDT algorithm, which handles
    intra-route optimization by finding optimal task sequences and performing
    local search improvements.
    
    Args:
        route: The route to insert the order into
        order: The order to be inserted
        debug_assignment: Whether to print debug information
        enhanced_diagnostics: Whether to show detailed capacity and constraint analysis
        sequencing_strategy: Strategy for task sequencing ('clustered' for P-P-D-D or 'interleaved' for P-D-P-D)
        
    Returns:
        Optimized route with the order inserted, or None if infeasible
    """
    
    # Enhanced diagnostic logging for problematic orders
    show_diagnostics = debug_assignment or enhanced_diagnostics
    
    # CRITICAL DEBUG: Track L2 heuristic entry and exit
    if show_diagnostics:
        print(f"      L2_HEURISTIC ENTRY:")
        print(f"         Order: {order.id} ({order.get_total_demand():.0f}kg, {order.get_total_volume():.1f}m³)")
        print(f"         Vehicle: {route.vehicle.id} (Cap: {getattr(route.vehicle, 'weight_capacity', 'unknown')}kg)")
        print(f"         Current route tasks: {len(route.tasks)}")
        print(f"         Strategy: {sequencing_strategy}")
        
    # Add verbose logging (Step 3 from guide)
    verbose = debug_assignment or enhanced_diagnostics
    if verbose:
        print(f"    -- L2 Heuristic: Inserting Order {order.id} into Vehicle {route.vehicle.id} --")
    
    if enhanced_diagnostics:
        print(f"L2 DIAGNOSTIC: Attempting to insert Order {order.id} into vehicle {route.vehicle.id}")
        print(f"   Vehicle capacity: {route.vehicle.weight_capacity}kg, {route.vehicle.volume_capacity}m³")
        
        # Calculate current route load properly
        current_weight = 0
        current_volume = 0
        if hasattr(route, 'tasks') and route.tasks:
            for task in route.tasks:
                if hasattr(task, 'demand'):
                    current_weight += task.demand
                if hasattr(task, 'volume'):
                    current_volume += task.volume
        
        # Get order requirements using proper methods
        order_weight = order.get_total_demand()
        order_volume = order.get_total_volume()
        
        print(f"   Current load: {current_weight}kg, {current_volume:.2f}m³")
        print(f"   Available capacity: {route.vehicle.weight_capacity - current_weight:.1f}kg, {route.vehicle.volume_capacity - current_volume:.2f}m³")
        print(f"   Order requirements: {order_weight:.1f}kg, {order_volume:.2f}m³")
        
        # Basic capacity check
        if (order_weight > (route.vehicle.weight_capacity - current_weight) or
            order_volume > (route.vehicle.volume_capacity - current_volume)):
            print(f"   CAPACITY FAILURE: Order {order.id} exceeds vehicle capacity")
    
    # Step 1: Identify Complex Orders and apply advanced sequencing
    pickups = order.get_pickups()
    deliveries = order.get_deliveries()
    is_complex_order = len(pickups) > 1 or len(deliveries) > 1
    
    # Always print complex order detection for debugging
    if is_complex_order:
        print(f"    *** COMPLEX ORDER DETECTED: Order {order.id} has {len(pickups)} pickups, {len(deliveries)} deliveries ***")
        
        # Execute new advanced sequencing logic for complex orders
        complex_route = find_best_sequence_for_complex_order(route, order, show_diagnostics)
        if complex_route:
            print(f"    *** COMPLEX ORDER SUCCESS: Order {order.id} processed successfully ***")
            return complex_route
        else:
            print(f"    *** COMPLEX ORDER FAILED: Order {order.id} - No feasible sequence found ***")
            return None
    else:
        # Simple order - print for debugging
        if len(pickups) + len(deliveries) > 0:  # Only print if there are tasks
            print(f"    Simple Order: Order {order.id} has {len(pickups)} pickups, {len(deliveries)} deliveries")
    
    # For simple orders, proceed with existing logic
    initial_routes: List['Route'] = _generate_initial_task_sequence(route, order, show_diagnostics, sequencing_strategy)

    if show_diagnostics:
        print(f"      DEBUG L2: Order {order.id} generated {len(initial_routes)} initial routes")

    if not initial_routes:
        if show_diagnostics:
            print(f"      L2_HEURISTIC FAILURE: Order {order.id} - No feasible initial routes found")
            print(f"         This means task sequence generation completely failed")
                    
        if enhanced_diagnostics:
            print(f"   TASK SEQUENCE FAILURE: Could not generate any feasible initial task sequences for Order {order.id}")
            print(f"      Vehicle {route.vehicle.id} with {getattr(route.vehicle, 'weight_capacity', 'unknown')}kg capacity")
            print(f"      Order requires {order.get_total_demand():.0f}kg, {order.get_total_volume():.1f}m³")
            
        return None   # Infeasible insertion
    
    best_initial_route = max(initial_routes, key=calculate_z2_score)

    neighborhoods_to_search = [_task_swap_neighborhood]
    if order.is_fixed:
        neighborhoods_to_search.append(_task_insertion_neighborhood)

    final_route = local_search_l2(best_initial_route, neighborhoods_to_search, order)

    # CRITICAL: Validate final route before returning - reject infeasible solutions
    if final_route:
        # Use strict feasibility check to ensure no constraint violations
        is_route_feasible = is_feasible(final_route, debug_feasibility=False, allow_soft_violations=False)
        
        if not is_route_feasible:
            if show_diagnostics:
                print(f"      DEBUG L2: Order {order.id} - REJECTED: Final route failed strict feasibility check")
            return None  # Reject infeasible routes to prevent violations
    
    if show_diagnostics:
        if final_route:
            print(f"      DEBUG L2: Order {order.id} - Final route feasible: {final_route.is_feasible()}")
        else:
            print(f"      DEBUG L2: Order {order.id} - Local search failed")
            
    if enhanced_diagnostics and not final_route:
        print(f"   LOCAL SEARCH FAILURE: Order {order.id} failed during local search optimization")
    
    # At the end of l2_heuristic, before returning (Step 3 from guide)
    if verbose:
        if final_route:
            print(f"    -- L2 Heuristic Complete: Found best insertion for Order {order.id}. Final Route Score (Z2): {calculate_z2_score(final_route):.2f} --")
        else:
            print(f"    -- L2 Heuristic Complete: No feasible insertion found for Order {order.id}. --")

    return final_route



def _generate_initial_task_sequence(route: 'Route', order: 'Order', debug_assignment: bool = False, sequencing_strategy: str = 'clustered') -> List['Route']:
    """ 
    Generates initial task sequences using a precedence-aware insertion heuristic.
    Ensures pickup tasks are always inserted before delivery tasks for the same order.
    
    Args:
        route: The route to insert tasks into
        order: The order containing tasks to insert
        debug_assignment: Whether to print debug information
        sequencing_strategy: 'clustered' for P-P-D-D pattern or 'interleaved' for P-D-P-D pattern
        
    Returns:
        List of routes with different task sequences attempted
    """
    P = order.get_pickups()
    D = order.get_deliveries()
    
    initial_routes = []
    
    # Handle delivery-only orders (common case for zero-demand deliveries)
    if len(P) == 0 and len(D) > 0:
        if debug_assignment:
            print(f"        DEBUG L2: Handling delivery-only order with {len(D)} deliveries")
        
        current_route = route.copy()
        
        # Insert each delivery at the best feasible position
        for delivery in D:
            best_delivery_cost = float('inf')
            best_delivery_route = None
            best_delivery_pos = None
            
            if debug_assignment:
                print(f"        DEBUG L2: Inserting delivery-only task {getattr(delivery, 'id', 'unknown')}")
            
            # Try all positions for delivery-only orders (between depot tasks)
            # Range ensures we insert after DEPOT_START (index 0) and before DEPOT_RETURN
            for pos in range(1, len(current_route.tasks)):
                test_route = current_route.copy()
                test_route.insert_task_without_reordering(pos, delivery)
                
                if debug_assignment:
                    print(f"        DEBUG L2: Delivery-only position {pos}, feasible: {is_feasible(test_route, debug_feasibility=debug_assignment, allow_soft_violations=False)}")
                
                if is_feasible(test_route, debug_feasibility=debug_assignment, allow_soft_violations=False):
                    cost = calculate_z2_score(test_route)
                    if cost < best_delivery_cost:
                        best_delivery_cost = cost
                        best_delivery_route = test_route
                        best_delivery_pos = pos
            
            if best_delivery_route:
                current_route = best_delivery_route
                if debug_assignment:
                    print(f"        DEBUG L2: Successfully inserted delivery-only at position {best_delivery_pos}")
            else:
                if debug_assignment:
                    print(f"        DEBUG L2: Failed to insert delivery-only task - no feasible positions")
                return []
        
        initial_routes.append(current_route)
        if debug_assignment:
            print(f"        DEBUG L2: Delivery-only order successfully processed with {len(current_route.tasks)} tasks")
        return initial_routes
    
    # Choose strategy based on sequencing_strategy parameter
    if sequencing_strategy == 'clustered':
        return _generate_clustered_sequence(route, order, P, D, debug_assignment)
    elif sequencing_strategy == 'interleaved':
        return _generate_interleaved_sequence(route, order, P, D, debug_assignment)
    else:
        # Default to clustered for unknown strategies
        if debug_assignment:
            print(f"        DEBUG L2: Unknown sequencing strategy '{sequencing_strategy}', defaulting to 'clustered'")
        return _generate_clustered_sequence(route, order, P, D, debug_assignment)


def _generate_clustered_sequence(route: 'Route', order: 'Order', P: List, D: List, debug_assignment: bool = False) -> List['Route']:
    """
    Strategy 1: Cluster-based efficient insertion
    Group pickups first, then deliveries to minimize depot visits
    This creates more efficient pickup->pickup->delivery->delivery patterns
    """
    initial_routes = []
    current_route = route.copy()
    
    if debug_assignment:
        print(f"        DEBUG L2: Starting cluster-based insertion with {len(P)} pickups and {len(D)} deliveries")
    
    # Phase 1: Insert all pickups in an efficient cluster
    for pickup in P:
        best_pickup_cost = float('inf')
        best_pickup_route = None
        best_pickup_pos = None
        
        if debug_assignment:
            print(f"        DEBUG L2: Clustering pickup {pickup.id if hasattr(pickup, 'id') else 'unknown'}")
        
        # Try inserting pickup near other pickups (cluster them together)
        # Ensure we only insert between depot tasks (not at position 0 or at the end)
        max_positions_to_try = min(len(current_route.tasks) - 1, 20)  # -1 to exclude end position
        for pos in range(1, min(1 + max_positions_to_try, len(current_route.tasks))):
            test_route = current_route.copy()
            test_route.insert_task_without_reordering(pos, pickup)
            
            # Add detailed logging from guide (Step 3)
            if debug_assignment:
                print(f"      - Trying to insert {pickup.id if hasattr(pickup, 'id') else 'unknown'} at position {pos}...")
                print(f"        Feasible: {is_feasible(test_route, debug_feasibility=debug_assignment, allow_soft_violations=False)}, New Route Score (Z2): {calculate_z2_score(test_route):.2f}")
            
            if is_feasible(test_route, debug_feasibility=debug_assignment, allow_soft_violations=False):
                cost = calculate_z2_score(test_route)
                if cost < best_pickup_cost:
                    best_pickup_cost = cost
                    best_pickup_route = test_route
                    best_pickup_pos = pos
        
        if best_pickup_route:
            current_route = best_pickup_route
            if debug_assignment:
                print(f"        DEBUG L2: Successfully clustered pickup at position {best_pickup_pos}")
        else:
            if debug_assignment:
                print(f"        DEBUG L2: Failed to cluster pickup - no feasible positions")
            return []
    
    # Phase 2: Insert all deliveries after the pickup cluster
    # This creates the efficient pickup->pickup->delivery->delivery pattern
    pickup_cluster_size = len(P)  # Number of pickups we just inserted
    
    for delivery in D:
        best_delivery_cost = float('inf')
        best_delivery_route = None
        best_delivery_pos = None
        
        if debug_assignment:
            print(f"        DEBUG L2: Adding delivery {delivery.id if hasattr(delivery, 'id') else 'unknown'} after pickup cluster")
        
        # Insert deliveries starting after the pickup cluster
        # This ensures the pattern: depot -> pickup1 -> pickup2 -> pickup3 -> delivery1 -> delivery2 -> delivery3
        # start_pos accounts for DEPOT_START (position 0) + all pickups
        start_pos = 1 + pickup_cluster_size  # Skip DEPOT_START + all pickup tasks
        max_delivery_positions = min(len(current_route.tasks) - start_pos, 15)  # -1 to exclude DEPOT_RETURN position
        
        for pos_offset in range(max_delivery_positions):
            pos = start_pos + pos_offset
            # Ensure we never insert at the very end (where DEPOT_RETURN should be)
            if pos >= len(current_route.tasks):
                break
            test_route = current_route.copy()
            test_route.insert_task_without_reordering(pos, delivery)
            
            if debug_assignment:
                print(f"        DEBUG L2: Delivery cluster position {pos}, feasible: {is_feasible(test_route, debug_feasibility=debug_assignment, allow_soft_violations=False)}")
            
            if is_feasible(test_route, debug_feasibility=debug_assignment, allow_soft_violations=False):
                cost = calculate_z2_score(test_route)
                if cost < best_delivery_cost:
                    best_delivery_cost = cost
                    best_delivery_route = test_route
                    best_delivery_pos = pos
        
        if best_delivery_route:
            current_route = best_delivery_route
            if debug_assignment:
                print(f"        DEBUG L2: Successfully added delivery to cluster at position {best_delivery_pos}")
        else:
            if debug_assignment:
                print(f"        DEBUG L2: Failed to add delivery to cluster - trying flexible insertion")
            
            # Fallback: try any position respecting individual order precedence
            best_fallback_cost = float('inf')
            best_fallback_route = None
            
            # Find corresponding pickup position for precedence
            pickup_pos = None
            for i, task in enumerate(current_route.tasks):
                if (hasattr(task, 'order_id') and hasattr(delivery, 'order_id') and 
                    task.order_id == delivery.order_id and task.is_pickup()):
                    pickup_pos = i
                    break
            
            start_pos = max(1, (pickup_pos + 1) if pickup_pos is not None else 1)
            # Ensure we insert between depot tasks, not at the end where DEPOT_RETURN should be
            for pos in range(start_pos, len(current_route.tasks)):
                test_route = current_route.copy()
                test_route.insert_task_without_reordering(pos, delivery)
                
                if is_feasible(test_route, debug_feasibility=debug_assignment, allow_soft_violations=False):
                    cost = calculate_z2_score(test_route)
                    if cost < best_fallback_cost:
                        best_fallback_cost = cost
                        best_fallback_route = test_route
            
            if best_fallback_route:
                current_route = best_fallback_route
                if debug_assignment:
                    print(f"        DEBUG L2: Successfully inserted delivery via fallback")
            else:
                if debug_assignment:
                    print(f"        DEBUG L2: Failed to insert delivery even with fallback")
                return []
    
    # Result: efficient pickup->pickup->delivery->delivery pattern
    # The route now minimizes depot visits and creates logical task clustering
    
    initial_routes.append(current_route)
    
    if debug_assignment:
        print(f"        DEBUG L2: Successfully created route with {len(current_route.tasks)} tasks")
    
    return initial_routes


def _generate_interleaved_sequence(route: 'Route', order: 'Order', P: List, D: List, debug_assignment: bool = False) -> List['Route']:
    """
    Strategy 2: Interleaved chronological insertion
    Attempts to sequence tasks chronologically based on their time windows,
    while respecting pickup-before-delivery precedence for each pair.
    This naturally creates P-D-P-D patterns.
    """
    initial_routes = []
    current_route = route.copy()
    
    if debug_assignment:
        print(f"        DEBUG L2: Starting interleaved insertion with {len(P)} pickups and {len(D)} deliveries")
    
    # Get all tasks and sort by earliest time
    all_tasks = P + D
    try:
        # Sort tasks by their earliest_time attribute
        all_tasks.sort(key=lambda task: getattr(task, 'earliest_time', 0))
        if debug_assignment:
            print(f"        DEBUG L2: Sorted {len(all_tasks)} tasks by earliest time")
    except:
        # Fallback if no time windows available
        if debug_assignment:
            print(f"        DEBUG L2: No time windows available, using original order")
    
    # Track which pickups have been inserted for precedence checking
    inserted_pickups = set()
    
    # Insert tasks in chronological order, respecting precedence
    for task in all_tasks:
        # If this is a delivery, check that its corresponding pickup has been inserted
        if not task.is_pickup():
            corresponding_pickup_inserted = False
            # Find if any pickup from the same order has been inserted
            for pickup in P:
                if (hasattr(pickup, 'order_id') and hasattr(task, 'order_id') and 
                    pickup.order_id == task.order_id and pickup.id in inserted_pickups):
                    corresponding_pickup_inserted = True
                    break
            
            if not corresponding_pickup_inserted:
                if debug_assignment:
                    print(f"        DEBUG L2: Skipping delivery {getattr(task, 'id', 'unknown')} - no corresponding pickup inserted yet")
                continue
        
        best_task_cost = float('inf')
        best_task_route = None
        best_task_pos = None
        
        if debug_assignment:
            task_type = "pickup" if task.is_pickup() else "delivery"
            print(f"        DEBUG L2: Interleaved insertion of {task_type} {getattr(task, 'id', 'unknown')}")
        
        # Try inserting at all valid positions
        for pos in range(1, len(current_route.tasks)):
            test_route = current_route.copy()
            test_route.insert_task_without_reordering(pos, task)
            
            if debug_assignment:
                print(f"        DEBUG L2: Interleaved position {pos}, feasible: {is_feasible(test_route, debug_feasibility=debug_assignment, allow_soft_violations=False)}")
            
            if is_feasible(test_route, debug_feasibility=debug_assignment, allow_soft_violations=False):
                cost = calculate_z2_score(test_route)
                if cost < best_task_cost:
                    best_task_cost = cost
                    best_task_route = test_route
                    best_task_pos = pos
        
        if best_task_route:
            current_route = best_task_route
            if task.is_pickup():
                inserted_pickups.add(task.id)
            if debug_assignment:
                task_type = "pickup" if task.is_pickup() else "delivery"
                print(f"        DEBUG L2: Successfully inserted {task_type} at position {best_task_pos}")
        else:
            if debug_assignment:
                task_type = "pickup" if task.is_pickup() else "delivery"
                print(f"        DEBUG L2: Failed to insert {task_type} - no feasible positions")
            return []
    
    # Verify all tasks were inserted
    if len(inserted_pickups) == len(P):
        initial_routes.append(current_route)
        if debug_assignment:
            print(f"        DEBUG L2: Successfully created interleaved route with {len(current_route.tasks)} tasks")
    else:
        if debug_assignment:
            print(f"        DEBUG L2: Interleaved insertion incomplete - missing pickups")
        return []
    
    return initial_routes


def _task_insertion_neighborhood(route: 'Route', order: 'Order') -> Iterator['Route']:
    """ 
    Neighborhood Generation Function: Task Insertion Operations
    
    Generates task insertion neighborhoods for the given route and order.
    Uses yield for memory efficiency instead of building a full list.
    
    This is a modular neighborhood function that can be combined with others
    in the local search process for comprehensive route optimization.
    
    Args:
        route: Base route to generate neighbors from
        order: Order whose tasks will be inserted
        
    Yields:
        Route: Feasible neighbor routes with tasks inserted at different positions
    """
    P = order.get_pickups()
    D = order.get_deliveries()
    n = len(route.tasks)
    
    # Insert tasks only between depot tasks (not at position 0 or at the end)
    for i in range(1, n):
        for pickup in P:
            new_route = copy.deepcopy(route)  # Deep copy for proper state management
            new_route.insert_task(i, pickup)
            if new_route.is_feasible():
                yield new_route
        
        for delivery in D:
            new_route = copy.deepcopy(route)  # Deep copy for proper state management
            new_route.insert_task(i, delivery)
            if new_route.is_feasible():
                yield new_route


def _task_swap_neighborhood(route: 'Route', order: 'Order') -> Iterator['Route']:
    """ 
    Neighborhood Generation Function: Task Swap Operations
    
    Generates task swap neighborhoods for the given route and order.
    Uses yield for memory efficiency instead of building a full list.
    
    This is a modular neighborhood function that focuses specifically on
    task swapping operations within the route optimization process.
    
    Args:
        route: Base route to generate neighbors from  
        order: Order context (used for determining applicable tasks)
        
    Yields:
        Route: Feasible neighbor routes with tasks swapped at different positions
    """
    P = order.get_pickups()
    D = order.get_deliveries()
    n = len(route.tasks)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Swap tasks at positions i and j
            new_route = copy.deepcopy(route)  # Deep copy for proper state management
            new_route.swap_tasks(i, j)
            if new_route.is_feasible():
                yield new_route


# TECHNICAL REVIEW FIX: Removed redundant HoS functions
# _calculate_realistic_driver_costs() and _estimate_hos_cost_with_breaks() 
# have been removed as recommended in the technical review.
# All HoS logic is now centralized in HoSEngine.analyze_route()


def _calculate_route_revenue(route: 'Route') -> float:
    """
    Calculate total revenue for a route using the dedicated route model.
    
    CORRECTED Revenue Model:
    - Each order's revenue = price_per_km × distance_of_dedicated_route
    - dedicated_route = depot → order_tasks → depot (as if served independently)
    - Total route revenue = sum of all individual order revenues
    - Price per km: 0.8 for standard vehicles, 1.25 for heavy vehicles
    
    This model correctly incentivizes consolidation:
    - Route cost = actual consolidated route distance × price_per_km
    - Route revenue = sum of dedicated route distances × price_per_km  
    - Profit = revenue - cost (consolidation reduces cost while preserving revenue)
    
    Args:
        route: Route object containing tasks and vehicle
        
    Returns:
        Total revenue in cost units (consistent with cost calculation)
    """
    if not route.tasks:
        return 0.0
    
    # Determine price per km based on vehicle type
    vehicle_type = getattr(route.vehicle, 'vehicle_type', 'standard')
    if vehicle_type == 'heavy':
        price_per_km = 1.25  # Camion
    else:
        price_per_km = 0.8   # Furgone
    
    # Get depot location (default to Asti)
    depot_lat, depot_lon = 44.9009, 8.2057
    
    # Group tasks by order_id (excluding depot tasks)
    orders_map = {}
    for task in route.tasks:
        order_id = getattr(task, 'order_id', None)
        # Skip depot tasks
        is_depot_task = ((hasattr(task, 'is_depot_start') and task.is_depot_start()) or 
                        (hasattr(task, 'is_depot_return') and task.is_depot_return()))
        if order_id and not is_depot_task:
            if order_id not in orders_map:
                orders_map[order_id] = []
            orders_map[order_id].append(task)
    
    # Calculate revenue for each order as if served by dedicated route
    total_revenue = 0.0
    for order_id, order_tasks in orders_map.items():
        if not order_tasks:
            continue
            
        # Calculate dedicated route distance: depot → order_tasks → depot
        dedicated_distance = 0.0
        current_lat, current_lon = depot_lat, depot_lon
        
        # Sort tasks for this order (pickups first, then deliveries)
        pickups = [t for t in order_tasks if t.is_pickup()]
        deliveries = [t for t in order_tasks if t.is_delivery()]
        sorted_tasks = pickups + deliveries
        
        # Calculate distance through all tasks for this order
        for task in sorted_tasks:
            task_lat = getattr(task, 'lat', depot_lat)
            task_lon = getattr(task, 'lon', depot_lon)
            
            # Distance from current position to this task
            distance_km = haversine_distance(current_lat, current_lon, task_lat, task_lon)
            dedicated_distance += distance_km
            
            # Update current position
            current_lat, current_lon = task_lat, task_lon
        
        # Return to depot
        return_distance = haversine_distance(current_lat, current_lon, depot_lat, depot_lon)
        dedicated_distance += return_distance
        
        # Calculate revenue for this order's dedicated route
        order_revenue = dedicated_distance * price_per_km
        total_revenue += order_revenue
    
    return total_revenue


def _calculate_utilization_penalty(route: 'Route', revenue: float) -> float:
    """
    Calculate penalty for underutilized vehicles to encourage consolidation.
    
    This penalty heavily discourages using large vehicles for small loads,
    encouraging the optimizer to consolidate orders into fewer, fuller vehicles.
    
    Penalty Logic:
    - No penalty for vehicles with >70% weight utilization
    - Moderate penalty (10-30% of revenue) for 30-70% utilization  
    - Heavy penalty (50-100% of revenue) for <30% utilization
    - Severe penalty for empty or nearly empty vehicles
    
    Args:
        route: Route object with tasks and vehicle
        revenue: Total revenue for the route (for scaling penalty)
        
    Returns:
        Utilization penalty (positive value that reduces profitability)
    """
    if not route.tasks or not hasattr(route, 'vehicle'):
        return 0.0
    
    # Calculate current utilization
    current_weight = 0.0
    peak_weight = 0.0
    
    for task in route.tasks:
        if hasattr(task, 'demand'):
            current_weight += task.demand
            peak_weight = max(peak_weight, current_weight)
    
    # Get vehicle capacity
    vehicle = route.vehicle
    max_weight = getattr(vehicle, 'weight_capacity', 1.0)
    
    if max_weight <= 0:
        return 0.0
    
    # Calculate weight utilization percentage
    utilization_pct = (peak_weight / max_weight) * 100
    
    # Progressive penalty based on utilization
    if utilization_pct >= 70.0:
        # Good utilization - no penalty
        penalty_factor = 0.0
    elif utilization_pct >= 50.0:
        # Moderate utilization - small penalty
        penalty_factor = 0.2  # 20% of revenue
    elif utilization_pct >= 30.0:
        # Poor utilization - moderate penalty
        penalty_factor = 0.5  # 50% of revenue  
    elif utilization_pct >= 10.0:
        # Very poor utilization - heavy penalty
        penalty_factor = 1.0  # 100% of revenue (makes route unprofitable)
    else:
        # Nearly empty vehicle - severe penalty
        penalty_factor = 2.0  # 200% of revenue (heavily unprofitable)
    
    # Scale penalty by revenue (larger revenue = larger penalty potential)
    base_penalty = revenue * penalty_factor
    
    # Add minimum penalty for very underutilized large vehicles
    if utilization_pct < 20.0 and max_weight > 10000:  # Large vehicles
        base_penalty += 1000.0  # Fixed penalty for underutilizing big trucks
    
    return base_penalty


def calculate_z2_score(route: 'Route') -> float:
    """ 
    Enhanced Z2 score calculation with multi-day support, prospective cost calculation,
    and soft time window penalties.
    
    Components:
    - C(r): Travel cost
    - W(r): Time window penalties (both hard and soft)
    - A(r): Prospective cost for tomorrow tasks
    - D(r): Driver cost (breaks and rest)
    - V(r): Vehicle assignment penalty
    - E(r): End position penalty
    
    Note: Uses the new two-stage HoS system for accurate rest cost calculation.
    """
    # Check if score is already cached
    if hasattr(route, '_z2_score'):
        return route._z2_score
    
    # FIXED: Use only HoSEngine for all driver costs as per technical review
    # Get HoS costs from the cached timeline (single source of truth)
    driver_cost = 0.0
    if hasattr(route, '_cached_rest_costs'):
        driver_cost = route._cached_rest_costs
    else:
        # Fallback: generate timeline and cache it using HoSEngine
        from hos_simulation import build_compliant_timeline
        timeline, rest_costs = build_compliant_timeline(route)
        route._cached_timeline = timeline
        route._cached_rest_costs = rest_costs
        driver_cost = rest_costs
    
    # Still need sorted tasks for other calculations
    sorted_tasks = _enforce_pickup_first_sequencing(route.tasks)
    
    # Initialize cost components
    travel_cost = 0.0  # C(r)
    time_window_penalty = 0.0  # W(r) 
    prospective_cost = 0.0  # A(r)
    vehicle_assignment_penalty = 0.0  # V(r)
    end_position_penalty = 0.0  # E(r)
    soft_time_window_penalty = 0.0  # Additional component for soft violations
    weight_violation_penalty = 0.0  # New component for weight capacity violations
    hos_violation_penalty = 0.0  # NEW: Component for HoS violations allowed through soft constraints

    # Multi-day cost simulation with weight violation tracking
    current_time = 0
    current_day = None
    last_today_task_location = None
    tomorrow_tasks = []
    
    # Track weight load for violation penalties
    load_w = 0.0
    if hasattr(route.vehicle, 'initial_state') and route.vehicle.initial_state:
        load_w = route.vehicle.initial_state.get('load_weight', 0.0)
    max_w = route.vehicle.weight_capacity
    
    # Group tasks by day for prospective cost calculation
    tasks_by_day = {}
    for task in sorted_tasks:
        day = getattr(task, 'day', 0)
        if day not in tasks_by_day:
            tasks_by_day[day] = []
        tasks_by_day[day].append(task)
        
        # Track tomorrow tasks for prospective cost
        if day > 0:  # Tomorrow or later
            tomorrow_tasks.append(task)

    # Simulate route execution
    for i in range(len(sorted_tasks)):
        current_task = sorted_tasks[i]
        task_day = getattr(current_task, 'day', 0)
        
        # Update weight load and check for violations
        if current_task.is_pickup():
            load_w += current_task.demand
        elif current_task.is_delivery():
            load_w += current_task.demand  # demand is negative for deliveries
        
        # Calculate weight capacity violation penalty with progressive scaling
        if load_w > max_w:
            excess_weight = load_w - max_w
            violation_percentage = excess_weight / max_w
            
            # DISABLED: Set weight penalties to 0 for realistic pricing
            weight_violation_penalty += 0.0  # Penalties disabled for cost calculation
        
        # Track day transitions
        if current_day is None:
            current_day = task_day
        elif task_day != current_day:
            # Day boundary crossed - add daily rest time if needed
            if current_day < task_day:
                current_time += 11 * 60  # 11 hours mandatory rest
            current_day = task_day
        
        # Track last task location of "today" (day 0) for prospective cost
        if task_day == 0:
            last_today_task_location = (getattr(current_task, 'lat', 0), 
                                      getattr(current_task, 'lon', 0))
        
        # Add service time
        service_time = getattr(current_task, 'service_time', 0)
        current_time += service_time
        
        # Calculate travel cost to next task
        if i < len(sorted_tasks) - 1:
            next_task = sorted_tasks[i + 1]
            # Use Haversine-based travel time calculation instead of OSRM
            travel_time = calculate_travel_time_haversine(
                current_task.lat, current_task.lon,
                next_task.lat, next_task.lon
            )
            base_travel_cost = travel_time * route.vehicle.cost_per_km * 0.8  # Rough conversion
            
            # Add distance-based inefficiency penalty for mixed pickup/delivery patterns
            # This discourages bouncing between depot and tasks, encouraging efficient clustering
            inefficiency_penalty = _calculate_inefficiency_penalty(current_task, next_task, route.vehicle)
            
            travel_cost += base_travel_cost + inefficiency_penalty
            current_time += travel_time
        
        # Check time window violations - RE-ENABLED with enhanced tracking
        earliest_time = getattr(current_task, 'earliest_time', None)
        latest_time = getattr(current_task, 'latest_time', None)
        is_soft_tw = getattr(current_task, 'soft_time_window', False)
        
        if earliest_time is not None and current_time < earliest_time:
            # Wait until earliest time
            wait_time = earliest_time - current_time
            current_time = earliest_time
            # Waiting time contributes to driver cost
            driver_cost += wait_time * route.vehicle.cost_per_hour
            
        if latest_time is not None and current_time > latest_time:
            delay = current_time - latest_time
            if is_soft_tw:
                # DISABLED: Soft time window penalty set to 0 for realistic pricing
                soft_time_window_penalty += 0.0  # was: delay * penalty_rate
            else:
                # DISABLED: Hard time window penalty set to 0 for realistic pricing
                time_window_penalty += 0.0  # was: delay * 10.0

    # Calculate prospective cost A(r) for tomorrow tasks
    if tomorrow_tasks and last_today_task_location:
        prospective_cost = _calculate_prospective_cost(
            last_today_task_location, tomorrow_tasks, route.vehicle
        )

    # Vehicle assignment penalty
    if hasattr(route, 'preferred_vehicle') and route.vehicle != route.preferred_vehicle:
        vehicle_assignment_penalty = 100.0

    # End position penalty
    if hasattr(route, 'preferred_end_position') and sorted_tasks:
        last_task = sorted_tasks[-1]
        if (hasattr(last_task, 'lat') and hasattr(last_task, 'lon') and
            hasattr(route, 'preferred_end_position')):
            # Calculate distance penalty (simplified)
            end_position_penalty = 50.0
    
    # NEW: Add penalty for HoS violations that were allowed through soft constraints
    if hasattr(route, '_has_hos_violation') and route._has_hos_violation:
        # DISABLED: HoS penalty set to 0 for realistic pricing
        hos_violation_penalty = 0.0  # was: 10000.0

    # --- NEW: Pallet Violation Penalty ---
    pallet_violation_penalty = 0.0
    max_pallets = route.vehicle.pallet_capacity
    if max_pallets is not None:
        current_pallets = 0
        violation_count = 0
        for task in sorted_tasks:
            # Note: task.pallets is negative for deliveries
            current_pallets += getattr(task, 'pallets', 0)
            if current_pallets > max_pallets:
                # Apply a very high penalty for any violation
                excess_pallets = current_pallets - max_pallets
                # Progressive penalty: higher penalty for larger violations
                pallet_violation_penalty += 10000000 + (excess_pallets * 1000000)  # Much larger penalty
                violation_count += 1
        
        # DEBUG: Print penalty calculation
        if pallet_violation_penalty > 0:
            print(f"DEBUG Z2: Vehicle {route.vehicle.id} has {violation_count} pallet violations, penalty: {pallet_violation_penalty}")
            print(f"DEBUG Z2: Max pallets: {max_pallets}, final load: {current_pallets}")

    # Calculate total Z2 score including all violation penalties
    total_cost = (travel_cost + time_window_penalty + prospective_cost + 
                  driver_cost + vehicle_assignment_penalty + end_position_penalty +
                  soft_time_window_penalty + weight_violation_penalty + hos_violation_penalty +
                  pallet_violation_penalty) # Add the new penalty here

    # NEW: Calculate revenue for profit-driven optimization
    total_revenue = _calculate_route_revenue(route)
    
    # NEW: Add utilization penalty to encourage better vehicle consolidation
    utilization_penalty = _calculate_utilization_penalty(route, total_revenue)
    
    # PROFIT-DRIVEN OBJECTIVE: Minimize (Cost - Revenue + Utilization_Penalty)
    # A profitable, well-utilized route will have a negative score (good)
    # An unprofitable or underutilized route will have a positive score (bad)
    profit_driven_score = total_cost - total_revenue + utilization_penalty
    
    # Debug output for profit tracking
    if hasattr(route, 'vehicle') and route.vehicle:
        vehicle_id = getattr(route.vehicle, 'id', 'unknown')
        if total_revenue > 0:  # Only show routes with revenue
            print(f"DEBUG Z2 PROFIT: Vehicle {vehicle_id} - Cost: {total_cost:.2f}, Revenue: {total_revenue:.2f}, Utilization Penalty: {utilization_penalty:.2f}, Profit Score: {profit_driven_score:.2f}")

    # Cache the profit-driven score
    route._z2_score = profit_driven_score
    return profit_driven_score


def _calculate_inefficiency_penalty(current_task, next_task, vehicle) -> float:
    """
    Calculate distance-based inefficiency penalty for mixed pickup/delivery patterns.
    
    This function discourages inefficient routing patterns like:
    - Pickup -> Depot -> Delivery (should be Pickup -> Delivery)
    - Pickup A -> Delivery B -> Pickup C (mixed orders inefficiently)
    - Long distances between related pickup/delivery pairs
    
    Args:
        current_task: Current task
        next_task: Next task in sequence
        vehicle: Vehicle object for cost calculation
        
    Returns:
        Additional penalty cost for inefficient patterns
    """
    penalty = 0.0
    
    # Depot coordinates (Asti location)
    depot_lat, depot_lon = 44.9009, 8.2057
    
    # Pattern 1: Penalty for depot returns in middle of route
    # If current task is at depot and next task is not depot, and we're not at route start/end
    current_is_depot = (hasattr(current_task, 'lat') and hasattr(current_task, 'lon') and
                       abs(current_task.lat - depot_lat) < 0.001 and 
                       abs(current_task.lon - depot_lon) < 0.001)
    next_is_depot = (hasattr(next_task, 'lat') and hasattr(next_task, 'lon') and
                    abs(next_task.lat - depot_lat) < 0.001 and 
                    abs(next_task.lon - depot_lon) < 0.001)
    
    if current_is_depot and not next_is_depot:
        # Leaving depot to go to task - moderate penalty if this seems inefficient
        penalty += 25.0  # Base penalty for depot departure
    
    if not current_is_depot and next_is_depot:
        # Returning to depot from task - high penalty unless it's end of route
        penalty += 50.0  # Higher penalty for depot return
    
    # Pattern 2: Penalty for long distances between pickup and delivery of same order
    current_order_id = getattr(current_task, 'order_id', None)
    next_order_id = getattr(next_task, 'order_id', None)
    
    if (current_order_id and next_order_id and current_order_id == next_order_id and
        current_task.is_pickup() and next_task.is_delivery()):
        # This is an efficient pickup -> delivery pattern, apply discount
        penalty -= 10.0  # Reward efficient patterns
    
    # Pattern 3: Penalty for switching between orders inefficiently
    if (current_order_id and next_order_id and current_order_id != next_order_id):
        # Calculate distance to see if the switch is justified
        distance = haversine_distance(current_task.lat, current_task.lon, 
                                    next_task.lat, next_task.lon)
        if distance > 50:  # More than 50km between different orders
            penalty += distance * 0.5  # Distance-based penalty for far order switches
    
    return penalty


def _calculate_prospective_cost(last_today_location: tuple, tomorrow_tasks: List, vehicle) -> float:
    """
    Calculate the A(r) prospective cost component for routes with tomorrow tasks.
    
    This estimates the travel time/distance from the location of the last "today" task
    to the locations of all "tomorrow" tasks, respecting their internal sequence.
    
    Args:
        last_today_location: (lat, lon) of last task executed today
        tomorrow_tasks: List of tasks scheduled for tomorrow
        vehicle: Vehicle object for cost calculation
        
    Returns:
        Prospective cost for tomorrow tasks
    """
    if not tomorrow_tasks or not last_today_location:
        return 0.0
    
    # Sort tomorrow tasks by their planned sequence
    tomorrow_tasks_sorted = sorted(tomorrow_tasks, key=lambda t: getattr(t, 'sequence_order', 0))
    
    total_prospective_cost = 0.0
    current_location = last_today_location
    
    # Calculate travel cost from last today task to first tomorrow task
    if tomorrow_tasks_sorted:
        first_tomorrow = tomorrow_tasks_sorted[0]
        first_tomorrow_location = (getattr(first_tomorrow, 'lat', 0), 
                                 getattr(first_tomorrow, 'lon', 0))
        
        # Travel cost from end of today to start of tomorrow
        import math
        distance = math.sqrt((first_tomorrow_location[0] - current_location[0])**2 + 
                           (first_tomorrow_location[1] - current_location[1])**2)
        distance_km = distance * 111.32  # Rough conversion to km
        
        # Apply prospective cost factor (higher uncertainty = higher cost)
        prospective_factor = 1.2  # 20% uncertainty premium
        total_prospective_cost = distance_km * vehicle.cost_per_km * prospective_factor
    
    return total_prospective_cost


def local_search_l2(initial_route: 'Route', neighborhoods: List[Callable], 
                   order: 'Order', strategy: str = 'first_improvement') -> 'Route':
    """ 
    Perform local search on the initial route using specified neighborhoods.
    
    Args:
        initial_route: Starting route for local search
        neighborhoods: List of neighborhood generation functions
        order: Order context (used for determining applicable tasks)
        strategy: Local search strategy - 'first_improvement' or 'best_improvement'
    
    Local Search Strategies:
    - 'first_improvement': Takes the first neighbor that improves the current solution.
                          Faster but may not find the best local optimum.
    - 'best_improvement': Evaluates all neighbors and selects the best one.
                         Slower but finds better local optima (steepest descent).
    
    Returns:
        Improved route after local search
    """
    current_route = initial_route
    improved = True
    iterations = 0
    max_iterations = 100  # Safety limit to prevent hanging
    
    if strategy == 'first_improvement':
        # First improvement strategy: take first better neighbor found
        while improved and iterations < max_iterations:
            iterations += 1
            improved = False
            current_score = calculate_z2_score(current_route)
            
            # Modular neighborhood exploration: each function is independent
            for neighborhood in neighborhoods:
                neighbors_checked = 0
                max_neighbors_per_iteration = 50  # Limit neighborhood exploration
                
                for neighbor in neighborhood(current_route, order):  # Pass order for context
                    neighbors_checked += 1
                    if neighbors_checked > max_neighbors_per_iteration:
                        break  # Prevent excessive neighborhood exploration
                        
                    neighbor_score = calculate_z2_score(neighbor)
                    if neighbor_score < current_score:
                        current_route = neighbor
                        improved = True
                        break  # Exit inner loop to restart with new current route
                if improved:
                    break  # Exit outer loop to restart with new current route
    
    elif strategy == 'best_improvement':
        # Best improvement strategy: evaluate all neighbors, choose best
        while improved and iterations < max_iterations:
            iterations += 1
            improved = False
            current_score = calculate_z2_score(current_route)
            best_route = current_route
            best_score = current_score
            
            # Evaluate all neighbors in all neighborhoods
            for neighborhood in neighborhoods:
                neighbors_checked = 0
                max_neighbors_per_iteration = 50  # Limit neighborhood exploration
                
                for neighbor in neighborhood(current_route, order):
                    neighbors_checked += 1
                    if neighbors_checked > max_neighbors_per_iteration:
                        break  # Prevent excessive neighborhood exploration
                        
                    neighbor_score = calculate_z2_score(neighbor)
                    if neighbor_score < best_score:
                        best_route = neighbor
                        best_score = neighbor_score
                        improved = True
            
            # Move to best neighbor if improvement was found
            if improved:
                current_route = best_route
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'first_improvement' or 'best_improvement'")

    if iterations >= max_iterations:
        print(f"Warning: L2 local search reached maximum iterations ({max_iterations}). Stopping to prevent hanging.")
    
    return current_route



def _check_hos_lightweight(route: 'Route', driver_state: 'DriverState', tasks: List) -> bool:
    """
    Lightweight Hours of Service check without expensive OSRM calls.
    
    Uses simple heuristics for driving time estimation:
    - Average speed assumptions for different task types
    - Basic service time estimates
    - Standard HoS limits (11h driving, 14h on-duty per day)
    
    Args:
        route: Route to check
        driver_state: Current driver state
        tasks: Sorted list of tasks
        
    Returns:
        True if route respects HoS constraints, False otherwise
    """
    if not tasks:
        return True
    
    # REMOVED HoS exemption for vehicles without regulations
    # All vehicles must comply with HoS regulations for safety
    # Comment out the exemption that was causing HoS violations:
    # if hasattr(route.vehicle, 'regulations') and not route.vehicle.regulations:
    #     return True  # Vehicles without regulations are exempt from HoS constraints
    
    # HoS limits (hours)
    MAX_DRIVING_TIME = 11.0  # Maximum driving time per day
    MAX_ON_DUTY_TIME = 14.0  # Maximum on-duty time per day
    
    # Speed and service time estimates (conservative)
    AVERAGE_SPEED_KMH = 50.0  # km/h for intercity travel
    SERVICE_TIME_PICKUP = 0.5  # hours per pickup
    SERVICE_TIME_DELIVERY = 0.5  # hours per delivery
    DEPOT_SERVICE_TIME = 0.25  # hours at depot
    
    # Initialize from driver state
    cumulative_driving = getattr(driver_state, 'drive_today', 0.0)
    cumulative_on_duty = getattr(driver_state, 'work_today', 0.0)
    
    # Simple distance estimation for tasks
    for i, task in enumerate(tasks):
        # Service time for this task
        if task.is_pickup():
            service_time = SERVICE_TIME_PICKUP
        elif task.is_delivery():
            service_time = SERVICE_TIME_DELIVERY
        else:
            service_time = DEPOT_SERVICE_TIME
        
        # Calculate proper travel time using Haversine distance
        if i == 0:
            # From depot to first task - use proper calculation if coordinates available
            if hasattr(task, 'lat') and hasattr(task, 'lon'):
                # Depot at Via del Lavoro 38, Asti coordinates
                travel_time = calculate_travel_time_haversine(44.9009, 8.2057, task.lat, task.lon, 60.0) / 60.0  # Convert to hours, use 60 km/h for realistic truck speed
            else:
                travel_time = 1.0  # 1 hour fallback if no coordinates
        else:
            # Between tasks - use proper calculation if coordinates available
            prev_task = tasks[i-1]
            if (hasattr(task, 'lat') and hasattr(task, 'lon') and 
                hasattr(prev_task, 'lat') and hasattr(prev_task, 'lon')):
                travel_time = calculate_travel_time_haversine(prev_task.lat, prev_task.lon, task.lat, task.lon, 60.0) / 60.0  # Convert to hours, use 60 km/h
            else:
                travel_time = 0.5  # 30 minutes fallback if no coordinates
        
        # Update times
        cumulative_driving += travel_time
        cumulative_on_duty += travel_time + service_time
        
        # Check HoS limits
        if cumulative_driving > MAX_DRIVING_TIME:
            return False  # Exceeds daily driving limit
        
        if cumulative_on_duty > MAX_ON_DUTY_TIME:
            return False  # Exceeds daily on-duty limit
    
    return True


def is_feasible_for_insertion(route: 'Route', debug_insertion: bool = False) -> bool:
    """
    Lightweight feasibility check optimized for L2 insertion clustering.
    
    ENHANCED DEBUGGING: This function now provides detailed failure reasons
    to understand why feasible orders are being rejected.
    
    Args:
        route: Route to check feasibility for
        debug_insertion: Whether to print debug information
        
    Returns:
        True if route passes basic feasibility checks for insertion
    """
    
    if debug_insertion:
        print(f"                DEBUG INSERTION: Quick feasibility check for {len(route.tasks)} tasks")
        print(f"                Vehicle: {route.vehicle.id} (Cap: {getattr(route.vehicle, 'weight_capacity', 'unknown')}kg)")
    
    # VALIDATE EXISTING ORDER (don't reorder during insertion!)
    tasks = route.tasks
    
    # Check if we're in initialization phase using call stack inspection
    import inspect
    frame = inspect.currentframe()
    is_initialization = False
    try:
        # Check call stack for initialization functions
        for i in range(10):  # Check up to 10 frames up
            frame = frame.f_back
            if frame and frame.f_code.co_name in ['cluster_aware_initializer', 'build_clustered_route', 'l2_heuristic']:
                # Check if we're specifically in the initialization phase
                if any(keyword in frame.f_code.co_name for keyword in ['initializer', 'build_clustered']):
                    is_initialization = True
                    break
                # Also check if l2_heuristic is called from initializer
                if frame.f_code.co_name == 'l2_heuristic':
                    # Check one more frame up
                    parent_frame = frame.f_back
                    if parent_frame and 'initializer' in parent_frame.f_code.co_name:
                        is_initialization = True
                        break
    except:
        pass
    finally:
        del frame
    
    # H1: Basic capacity check (relaxed for clustering)
    load_w = 0.0
    load_v = 0.0
    load_pallets = 0
    max_w = route.vehicle.weight_capacity
    max_v = route.vehicle.volume_capacity
    max_pallets = route.vehicle.pallet_capacity
    
    # Track peak loads during route execution
    peak_load_w = 0.0
    peak_load_v = 0.0
    peak_load_pallets = 0
    
    for task in tasks:
        if task.is_pickup():
            load_w += task.demand
            load_v += task.volume
            load_pallets += getattr(task, 'pallets', 0)
        elif task.is_delivery():
            load_w += task.demand  # demand is negative for deliveries
            load_v += task.volume  # volume is negative for deliveries
            load_pallets += getattr(task, 'pallets', 0)  # pallets should be negative for delivery tasks
        
        # Track peak loads for soft constraint checking (weight/volume only)
        peak_load_w = max(peak_load_w, load_w)
        peak_load_v = max(peak_load_v, load_v)
        peak_load_pallets = max(peak_load_pallets, load_pallets)
        
        # NOTE: Pallet constraints are now handled by unified hard constraint checker
        # Only soft constraints (weight/volume with tolerances) are checked here
            
        # Weight and volume constraints - with tolerances for flexibility during insertion
        volume_tolerance = 1.5 if is_initialization else 1.1  # 50% tolerance during init, 10% during optimization
        if peak_load_v > max_v * volume_tolerance:
            if debug_insertion:
                print(f"                ❌ FEASIBILITY FAILURE: Volume constraint exceeded")
                print(f"                   Peak volume: {peak_load_v:.2f}m³ > Limit: {max_v:.2f}m³ × {volume_tolerance} = {max_v * volume_tolerance:.2f}m³")
                print(f"                   Current task: {getattr(task, 'id', 'unknown')} adding {getattr(task, 'volume', 0):.2f}m³")
            return False
        
        # Weight constraint - with tolerances for flexibility during insertion
        weight_tolerance = 1.5 if is_initialization else 1.1  # 50% tolerance during init, 10% during optimization
        if peak_load_w > max_w * weight_tolerance:
            if debug_insertion:
                print(f"                ❌ FEASIBILITY FAILURE: Weight constraint exceeded")
                print(f"                   Peak weight: {peak_load_w:.2f}kg > Limit: {max_w:.2f}kg × {weight_tolerance} = {max_w * weight_tolerance:.2f}kg")
                print(f"                   Current task: {getattr(task, 'id', 'unknown')} adding {getattr(task, 'demand', 0):.2f}kg")
            return False
    
    # H2: Basic logical precedence check (detailed LIFO/pallet checks in unified checker)
    # This is just a quick check for obvious precedence violations
    completed_pickups = set()
    
    for task in tasks:
        if task.is_pickup():
            order_id = getattr(task, 'order_id', None)
            if order_id:
                completed_pickups.add(order_id)
        elif task.is_delivery():
            # Check if corresponding pickup was already completed
            order_id = getattr(task, 'order_id', None)
            if order_id and order_id not in completed_pickups:
                if debug_insertion:
                    print(f"                ❌ FEASIBILITY FAILURE: Precedence violation")
                    print(f"                   Delivery {getattr(task, 'id', 'unknown')} for order {order_id} attempted before pickup")
                    print(f"                   Completed pickups: {completed_pickups}")
                return False
    
    # Individual order precedence constraints
    orders = {}
    for i, task in enumerate(tasks):
        order_id = getattr(task, 'order_id', None)
        if order_id is None:
            continue
            
        if order_id not in orders:
            orders[order_id] = {'pickups': [], 'deliveries': []}
        
        if task.is_pickup():
            orders[order_id]['pickups'].append(i)
        elif task.is_delivery():
            orders[order_id]['deliveries'].append(i)

    # Check individual order precedence constraints
    for order_id, tasks in orders.items():
        if tasks['pickups'] and tasks['deliveries']:
            last_pickup = max(tasks['pickups'])
            first_delivery = min(tasks['deliveries'])
            
            if last_pickup >= first_delivery:
                if debug_insertion:
                    print(f"                DEBUG INSERTION: Precedence violated for {order_id}: pickup {last_pickup} >= delivery {first_delivery}")
                return False
    
    if debug_insertion:
        print(f"                DEBUG INSERTION: Route passes basic feasibility checks")
    
    # Enhanced constraint checking: Add basic time window validation during insertion
    # This prevents creating routes with obvious time window violations
    try:
        # Quick time window feasibility check
        current_time = 0
        for i, task in enumerate(tasks):
            # Calculate travel time from previous task
            if i > 0:
                prev_task = tasks[i-1]
                # Use a simplified travel time estimate for insertion checks
                if (hasattr(task, 'lat') and hasattr(task, 'lon') and 
                    hasattr(prev_task, 'lat') and hasattr(prev_task, 'lon')):
                    # Simple distance-based travel time estimate
                    import math
                    lat_diff = task.lat - prev_task.lat
                    lon_diff = task.lon - prev_task.lon
                    distance = math.sqrt(lat_diff*lat_diff + lon_diff*lon_diff)
                    travel_time = distance * 100  # Rough estimate: 100 minutes per degree
                else:
                    travel_time = 30  # Default travel time
                
                current_time += travel_time
            
            # Check basic time window constraints
            if hasattr(task, 'latest_time') and task.latest_time is not None:
                if current_time > task.latest_time:
                    if debug_insertion:
                        print(f"                DEBUG INSERTION: Time window violation predicted at task {task.id}")
                    return False
            
            # Update time with service time and waiting
            if hasattr(task, 'earliest_time') and task.earliest_time is not None:
                if current_time < task.earliest_time:
                    current_time = task.earliest_time  # Wait until earliest time
            
            service_time = getattr(task, 'service_time', 0)
            current_time += service_time
            
    except Exception as e:
        # If time window check fails, be conservative and allow the route
        if debug_insertion:
            print(f"                DEBUG INSERTION: Time window check failed: {e}")
        pass
    
    # H3: UNIFIED HARD CONSTRAINT CHECK (ALWAYS ENFORCED)
    # Use the same unified checker as the main is_feasible function
    hard_constraint_valid, hard_constraint_reason = check_hard_constraints(route, debug_insertion)
    if not hard_constraint_valid:
        if debug_insertion:
            print(f"                DEBUG INSERTION: {hard_constraint_reason}")
        return False
    
    return True


def is_timeline_feasible(timeline: List, route: 'Route') -> Tuple[bool, str]:
    """
    Validate a timeline generated by build_compliant_timeline against time window constraints.
    
    This is the second stage of the two-stage HoS simulation and validation engine.
    It takes a timeline that is already legally compliant (with all mandatory rests inserted)
    and validates it against business constraints, primarily customer time windows.
    
    Args:
        timeline: List of SimulatedEvent objects from build_compliant_timeline
        route: The route being validated (for task information)
        
    Returns:
        Tuple of (is_feasible: bool, failure_reason: str)
        - is_feasible: True if timeline satisfies all time windows, False otherwise
        - failure_reason: Detailed description of the first violation found, or "Feasible" if valid
    """
    if not timeline or not route.tasks:
        return True, "Feasible"
    
    # Create a mapping from task IDs to their time window constraints
    task_time_windows = {}
    for task in route.tasks:
        if hasattr(task, 'id'):
            task_time_windows[task.id] = {
                'earliest_start_time': getattr(task, 'earliest_start_time', None),
                'latest_start_time': getattr(task, 'latest_start_time', None),
                'earliest_time': getattr(task, 'earliest_time', None),
                'latest_time': getattr(task, 'latest_time', None)
            }
    
    # Iterate through timeline events to check time window violations
    for event in timeline:
        # Only check events that correspond to customer tasks (WORK events)
        if event.event_type == 'WORK' and event.task_id:
            task_id = event.task_id
            
            if task_id in task_time_windows:
                time_window = task_time_windows[task_id]
                
                # Check earliest start time constraint
                earliest_start = time_window.get('earliest_start_time') or time_window.get('earliest_time')
                if earliest_start is not None and event.start_time < earliest_start:
                    return False, f"Time window violation at task {task_id}: service starts at {event.start_time:.1f} min, but earliest allowed is {earliest_start:.1f} min"
                
                # Check latest start time constraint
                latest_start = time_window.get('latest_start_time') or time_window.get('latest_time')
                if latest_start is not None and event.start_time > latest_start:
                    return False, f"Time window violation at task {task_id}: service starts at {event.start_time:.1f} min, but latest allowed is {latest_start:.1f} min (violation due to mandatory rest)"
        
        # Also check DRIVE events that arrive at tasks
        elif event.event_type == 'DRIVE' and event.task_id and '->' in event.task_id:
            # Extract destination task ID from drive event
            parts = event.task_id.split('->')
            if len(parts) == 2:
                dest_task_id = parts[1]
                
                if dest_task_id in task_time_windows:
                    time_window = task_time_windows[dest_task_id]
                    
                    # Check if arrival time violates latest time window
                    latest_arrival = time_window.get('latest_start_time') or time_window.get('latest_time')
                    if latest_arrival is not None and event.end_time > latest_arrival:
                        return False, f"Time window violation at task {dest_task_id}: arrival at {event.end_time:.1f} min, but latest allowed is {latest_arrival:.1f} min (violation due to mandatory rest)"
    
    return True, "Feasible"


def is_feasible(route: 'Route', debug_feasibility: bool = False, return_reason: bool = False, allow_soft_violations: bool = False) -> Union[bool, Tuple[bool, str]]:
    """ 
    Check if the route is feasible according to all constraints.
    Enhanced to support multi-day planning and LIFO loading constraints.
    
    NEW: Implements soft constraint system - allows routes with soft violations 
    to pass but ensures they get penalized in Z2 scoring.
    
    Args:
        route: Route to check feasibility for
        debug_feasibility: Whether to print debug information
        return_reason: Whether to return detailed failure reason
        allow_soft_violations: Whether to allow soft constraint violations (NEW)
    
    Returns:
        If return_reason=False: bool (feasible or not)
        If return_reason=True: Tuple[bool, str] (feasible, reason)
    
    Constraint Categories:
    - HARD: Vehicle type, license, depot structure, logical precedence, safety-critical HoS
    - SOFT: Time windows, capacity (with safety buffers), non-critical HoS violations
    """
    
    # Enhanced debugging for assignment failures
    if debug_feasibility:
        #print(f"            DEBUG FEASIBILITY: Checking route feasibility for vehicle {route.vehicle.id}")
        #print(f"            DEBUG FEASIBILITY: Route has {len(route.tasks)} tasks")
        pass
    
    # H0: DEPOT START/END VALIDATION - Must be first check
    # Every route must start with a depot start task and end with a depot return task
    if route.tasks:
        # Check first task is depot start
        first_task = route.tasks[0]
        if not (hasattr(first_task, 'is_depot_start') and first_task.is_depot_start()):
            reason = f"Route validation failed: First task ({first_task.id if hasattr(first_task, 'id') else 'unknown'}) is not a depot start task"
            if debug_feasibility:
                #print(f"            DEBUG FEASIBILITY: {reason}")
                pass
            if return_reason:
                return False, reason
            return False
        
        # Check last task is depot return
        last_task = route.tasks[-1]
        if not (hasattr(last_task, 'is_depot_return') and last_task.is_depot_return()):
            reason = f"Route validation failed: Last task ({last_task.id if hasattr(last_task, 'id') else 'unknown'}) is not a depot return task"
            if debug_feasibility:
                #print(f"            DEBUG FEASIBILITY: {reason}")
                pass
            if return_reason:
                return False, reason
            return False
        
        if debug_feasibility:
            #print(f"            DEBUG FEASIBILITY: OK Depot validation passed - route starts and ends at depot")
            pass
    
    # H1: Multi-day chronological simulation setup
    # Check original task order first for pickup-before-delivery precedence constraint
    original_tasks = route.tasks
    
    # H2: Enhanced Logical Precedence Constraint Check
    # NEW RULE: Allow deliveries before pickups as long as:
    # 1. Each delivery has its corresponding pickup already completed
    # 2. Pallet capacity is never exceeded (physical constraint)
    # 3. Distance penalty will discourage inefficient patterns
    
    # Track completed pickups for each order to validate deliveries
    completed_pickups = set()
    load_pallets_check = 0  # Track pallet load for physical constraint
    
    # Iterate through tasks in chronological sequence
    for task in original_tasks:
        if task.is_pickup():
            # Record this pickup as completed
            order_id = getattr(task, 'order_id', None)
            if order_id:
                completed_pickups.add(order_id)
            # Add pallets to load
            load_pallets_check += getattr(task, 'pallets', 0)
            
        elif task.is_delivery():
            # Check if corresponding pickup was already completed
            order_id = getattr(task, 'order_id', None)
            if order_id and order_id not in completed_pickups:
                reason = f"Enhanced logical precedence violated: Delivery {task.id} attempted before its pickup was completed"
                if debug_feasibility:
                    #print(f"            DEBUG FEASIBILITY: {reason}")
                    pass
                if return_reason:
                    return False, reason
                return False
            # Remove pallets from load
            load_pallets_check += getattr(task, 'pallets', 0)  # pallets should be negative for delivery tasks
            
        # Physical constraint: never exceed pallet capacity during route execution
        max_pallets = route.vehicle.pallet_capacity
        if max_pallets is not None and load_pallets_check > max_pallets:
            reason = f"Pallet capacity exceeded during route execution: {load_pallets_check} > {max_pallets}"
            if debug_feasibility:
                #print(f"            DEBUG FEASIBILITY: {reason}")
                pass
            if return_reason:
                return False, reason
            return False
    
    # Sort tasks with pickup-first sequencing for proper HoS simulation
    # Apply LIFO sequencing if required by vehicle
    if hasattr(route.vehicle, 'lifo_required') and route.vehicle.lifo_required:
        sorted_tasks = _enforce_pickup_first_sequencing_with_lifo(route.tasks, route.vehicle)
    else:
        sorted_tasks = _enforce_pickup_first_sequencing_basic(route.tasks)
    
    # Initialize vehicle state from previous day if available
    initial_state = getattr(route.vehicle, 'initial_state', None)
    if initial_state:
        current_position = initial_state.get('position', None)
        previous_day_load_w = initial_state.get('load_weight', 0.0)
        previous_day_load_v = initial_state.get('load_volume', 0.0)
        
        # Use route's assigned driver state if available, otherwise create new state
        if route.driver and route.driver.hos_state:
            driver_state = copy.deepcopy(route.driver.hos_state)
        else:
            driver_state = DriverState()
            # Initialize driver state from previous day
            previous_driver_state = initial_state.get('driver_state', {})
            driver_state.drive_today = previous_driver_state.get('drive_today', 0.0)
            driver_state.work_today = previous_driver_state.get('work_today', 0.0)
            driver_state.drive_this_week = previous_driver_state.get('drive_this_week', 0.0)
    else:
        current_position = None
        previous_day_load_w = 0.0
        previous_day_load_v = 0.0
        
        # Use route's assigned driver state if available, otherwise create new state
        if route.driver and route.driver.hos_state:
            driver_state = copy.deepcopy(route.driver.hos_state)
        else:
            driver_state = DriverState()
    
    # H3: Capacity check with multi-day simulation
    load_w = previous_day_load_w
    load_v = previous_day_load_v
    load_pallets = initial_state.get('load_pallets', 0) if initial_state else 0
    max_w = route.vehicle.weight_capacity
    max_v = route.vehicle.volume_capacity
    max_pallets = route.vehicle.pallet_capacity  # Hard constraint on pallets
    
    # H4: LIFO Loading Constraint check
    lifo_stack = []
    if route.vehicle.lifo_required:
        # Initialize stack with any cargo from previous day
        if initial_state and 'cargo_stack' in initial_state:
            lifo_stack = initial_state['cargo_stack'].copy()
        
        # Special handling for delivery-only orders:
        # Pre-initialize LIFO stack with orders that need to be delivered
        # This simulates that the vehicle was pre-loaded with the required cargo
        delivery_orders = set()
        pickup_orders = set()
        
        for task in sorted_tasks:
            if task.is_delivery():
                delivery_orders.add(task.order_id)
            elif task.is_pickup():
                pickup_orders.add(task.order_id)
        
        # Orders that have deliveries but no pickups are delivery-only
        delivery_only_orders = delivery_orders - pickup_orders
        
        if delivery_only_orders and not lifo_stack:
            # Pre-load the LIFO stack with delivery-only orders
            # Order doesn't matter for delivery-only since they'll all be delivered
            lifo_stack = list(delivery_only_orders)
            if debug_feasibility:
                #print(f"            DEBUG FEASIBILITY: Pre-loaded LIFO stack for delivery-only orders: {lifo_stack}")
                pass

    for task in sorted_tasks:
        if task.is_pickup():
            load_w += task.demand
            load_v += task.volume
            load_pallets += getattr(task, 'pallets', 0)  # Add pallets for pickup
            
            # LIFO constraint: push order_id onto stack
            if route.vehicle.lifo_required:
                lifo_stack.append(task.order_id)
                
        elif task.is_delivery():
            load_w += task.demand  # demand is negative for deliveries
            load_v += task.volume  # volume is negative for deliveries
            load_pallets += getattr(task, 'pallets', 0)  # pallets should be negative for delivery tasks
            
            # LIFO constraint: check if this delivery matches top of stack
            if route.vehicle.lifo_required:
                if not lifo_stack:
                    reason = f"LIFO violation: trying to deliver {task.id} when no cargo loaded"
                    if debug_feasibility:
                        #print(f"            DEBUG FEASIBILITY: {reason}")
                        pass
                    if return_reason:
                        return False, reason
                    return False
                if lifo_stack[-1] != task.order_id:
                    reason = f"LIFO violation: expected {lifo_stack[-1]}, got {task.order_id}"
                    if debug_feasibility:
                        #print(f"            DEBUG FEASIBILITY: {reason}")
                        pass
                    if return_reason:
                        return False, reason
                    return False
                lifo_stack.pop()  # Remove delivered order from stack
        
        # Check capacity constraints
        # UPDATED: Hard constraints (capabilities, pallets, LIFO) are now handled by unified checker
        # Only soft constraints (volume, some weight limits) are handled here
        
        # Volume constraint - can be soft for optimization flexibility
        if allow_soft_violations:
            if load_v > max_v * 2.0:  # Only fail on extreme volume violations
                reason = f"Extreme volume constraint violated: {load_v:.2f} > {max_v * 2.0:.2f} (200% capacity) for task {task.id}"
                if debug_feasibility:
                    print(f"            DEBUG FEASIBILITY: {reason}")
                if return_reason:
                    return False, reason
                return False
            # Moderate volume violations (100%-200%) are allowed and will be penalized in Z2
        else:
            # Original strict volume check if soft violations not allowed
            if load_v > max_v:
                reason = f"Volume constraint violated: {load_v:.2f} > {max_v:.2f} for task {task.id}"
                if debug_feasibility:
                    print(f"            DEBUG FEASIBILITY: {reason}")
                if return_reason:
                    return False, reason
                return False
            
        # NOTE: Pallet capacity is now checked by unified hard constraint checker

    # LIFO final check: all cargo must be delivered
    if route.vehicle.lifo_required and lifo_stack:
        reason = f"LIFO constraint violated: undelivered cargo: {lifo_stack}"
        if debug_feasibility:
            #print(f"            DEBUG FEASIBILITY: {reason}")
            pass
        if return_reason:
            return False, reason
        return False

    # H5: Per-order precedence constraints check with multi-day consideration
    orders = {}
    for i, task in enumerate(sorted_tasks):
        order_id = getattr(task, 'order_id', None)
        if order_id is None:
            continue
            
        if order_id not in orders:
            orders[order_id] = {'pickups': [], 'deliveries': []}
        
        # Store both index and day for proper precedence checking
        task_info = (i, getattr(task, 'day', 0))
        if task.is_pickup():
            orders[order_id]['pickups'].append(task_info)
        elif task.is_delivery():
            orders[order_id]['deliveries'].append(task_info)

    # Check precedence constraints across days
    for order_id, tasks in orders.items():
        if tasks['pickups'] and tasks['deliveries']:
            try:
                # Find last pickup (considering day and position)
                last_pickup = max(tasks['pickups'], key=lambda x: (x[1], x[0]))  # Sort by day, then position
                # Find first delivery
                first_delivery = min(tasks['deliveries'], key=lambda x: (x[1], x[0]))
                
                # Check if last pickup happens before first delivery
                if (last_pickup[1] > first_delivery[1] or 
                    (last_pickup[1] == first_delivery[1] and last_pickup[0] >= first_delivery[0])):
                    reason = f"Precedence constraint violated for {order_id}: last pickup (day={last_pickup[1]}, pos={last_pickup[0]}) >= first delivery (day={first_delivery[1]}, pos={first_delivery[0]})"
                    if debug_feasibility:
                        #print(f"            DEBUG FEASIBILITY: {reason}")
                        pass
                    if return_reason:
                        return False, reason
                    return False
            except (ValueError, TypeError, KeyError) as e:
                if debug_feasibility:
                    #print(f"            DEBUG FEASIBILITY: Error in precedence check for {order_id}: {e}")
                    pass
                # Skip this order's precedence check if there's an error
                continue

    # H6: Multi-day Hours of Service check - DISABLED during initialization
    # Check if we're in initialization phase using call stack inspection
    import inspect
    frame = inspect.currentframe()
    is_initialization = False
    try:
        # Check call stack for initialization functions
        for i in range(10):  # Check up to 10 frames up
            frame = frame.f_back
            if frame and frame.f_code.co_name in ['cluster_aware_initializer', 'build_clustered_route', 'l2_heuristic']:
                # Check if we're specifically in the initialization phase
                if any(keyword in frame.f_code.co_name for keyword in ['initializer', 'build_clustered']):
                    is_initialization = True
                    break
                # Also check if l2_heuristic is called from initializer
                if frame.f_code.co_name == 'l2_heuristic':
                    # Check one more frame up
                    parent_frame = frame.f_back
                    if parent_frame and 'initializer' in parent_frame.f_code.co_name:
                        is_initialization = True
                        break
    except:
        pass
    finally:
        del frame
    
    # Apply HoS validation with rest-aware time window checking
    # UNIFIED APPROACH: Single source of truth using HoSEngine.analyze_route()
    try:
        # B license drivers are exempt from HoS regulations
        if route.driver and hasattr(route.driver, 'license') and route.driver.license == 'B':
            if debug_feasibility:
                print(f"            DEBUG FEASIBILITY: B license driver exemption - skipping HoS check")
            pass  # B license drivers are exempt - skip HoS check
        else:
            if debug_feasibility:
                print(f"            DEBUG FEASIBILITY: Performing unified HoS validation for route with {len(route.tasks)} tasks")
            
            # UNIFIED VALIDATION: Single call to HoSEngine for authoritative feasibility
            from hos_simulation import HoSEngine
            hos_engine = HoSEngine()
            hos_result = hos_engine.analyze_route(route)
            
            if debug_feasibility:
                print(f"            DEBUG FEASIBILITY: HoS Engine analysis complete - feasible: {hos_result.is_feasible}")
                if hos_result.violations:
                    print(f"            DEBUG FEASIBILITY: Violations found: {hos_result.violations}")
            
            # Cache the timeline and rest costs on the route object for use by calculate_z2_score
            route._cached_timeline = hos_result.timeline
            route._cached_rest_costs = hos_result.rest_cost
            
            # Check the unified feasibility result
            if not hos_result.is_feasible:
                if not allow_soft_violations:
                    reason = f"HoS validation failed: {'; '.join(hos_result.violations)}"
                    if debug_feasibility:
                        print(f"            DEBUG FEASIBILITY: {reason}")
                    if return_reason:
                        return False, reason
                    return False
                else:
                    # Allow soft violations but log them
                    if debug_feasibility:
                        print(f"            DEBUG FEASIBILITY: HoS violations found but allowing soft violations: {hos_result.violations}")
            
            # HoS validation passed - timeline is both HoS-compliant and respects time windows
            if debug_feasibility:
                print(f"            DEBUG FEASIBILITY: Unified HoS and time window validation passed")
                
    except Exception as e:
        # HoS analysis failed - this indicates either HoS violations or time window conflicts
        error_msg = str(e)
        if allow_soft_violations:
            # More lenient handling of HoS/time window integration issues
            # Only fail for severe violations
            severe_keywords = ["safety-critical", "mandatory rest limit exceeded", "weekly limit"]
            is_severe = any(keyword in error_msg.lower() for keyword in severe_keywords)
            
            if is_severe:
                reason = f"Severe HoS/time window violation: {error_msg}"
                if debug_feasibility:
                    print(f"            DEBUG FEASIBILITY: {reason}")
                if return_reason:
                    return False, reason
                return False
            else:
                # Allow moderate violations but mark for penalties
                if debug_feasibility:
                    print(f"            DEBUG FEASIBILITY: Moderate HoS/time window issue allowed: {error_msg}")
                route._has_hos_violation = True
                route._hos_violation_reason = error_msg
        else:
            # Original strict behavior
            reason = f"HoS/time window validation failed: {error_msg}"
            if debug_feasibility:
                print(f"            DEBUG FEASIBILITY: {reason}")
            if return_reason:
                return False, reason
            return False
    
    # H7: Hard time windows check - Enhanced with arrival time simulation
    try:
        from route_provider import calculate_travel_time_between_tasks
    except ImportError:
        # Fallback if route_provider is not available
        def calculate_travel_time_between_tasks(task1, task2, vehicle):
            return 0  # Simplified fallback
    
    current_time = 0  # Start at planning time 0
    
    for i, task in enumerate(sorted_tasks):
        # Calculate travel time from previous task (if any)
        if i > 0:
            prev_task = sorted_tasks[i-1]
            travel_time = calculate_travel_time_between_tasks(prev_task, task, route.vehicle)
            current_time += travel_time
        
        # Current time is now the arrival time at this task
        arrival_time = current_time
        
        # NEW SOFT TIME WINDOW CONSTRAINT SYSTEM
        if hasattr(task, 'earliest_time') and hasattr(task, 'latest_time'):
            if not getattr(task, 'soft_time_window', False):  # Only check hard time windows
                # NEW: Make time windows more lenient with grace periods
                # Check for severe lateness - only fail on extreme violations
                late_by = arrival_time - task.latest_time if task.latest_time is not None else 0
                
                if allow_soft_violations:
                    # Allow only minor lateness (up to 10 minutes) - strict enforcement
                    GRACE_PERIOD_MINUTES = 10.0
                    if task.latest_time is not None and late_by > GRACE_PERIOD_MINUTES:
                        reason = f"Time window violation at task {task.id}: arrived at {arrival_time:.1f}, latest allowed {task.latest_time}, late by {late_by:.1f} minutes (exceeds {GRACE_PERIOD_MINUTES} min grace period)"
                        if debug_feasibility:
                            print(f"            DEBUG FEASIBILITY: {reason}")
                        if return_reason:
                            return False, reason
                        return False
                    # Only minor lateness (0-10 minutes) is allowed
                else:
                    # Original strict time window check
                    if task.latest_time is not None and arrival_time > task.latest_time:
                        reason = f"Late arrival at task {task.id}: arrived at {arrival_time:.1f}, latest allowed {task.latest_time}"
                        if debug_feasibility:
                            print(f"            DEBUG FEASIBILITY: {reason}")
                        if return_reason:
                            return False, reason
                        return False
                    elif debug_feasibility and task.latest_time is not None:
                        print(f"            DEBUG FEASIBILITY: Task {task.id} time check OK: arrival {arrival_time:.1f} <= latest {task.latest_time}")
                
                # Check for early arrival - vehicle must wait
                if task.earliest_time is not None and arrival_time < task.earliest_time:
                    wait_time = task.earliest_time - arrival_time
                    current_time = task.earliest_time  # Update current time to earliest allowed time
                    if debug_feasibility:
                        #print(f"            DEBUG FEASIBILITY: Early arrival at task {task.id}, waiting {wait_time:.1f} minutes")
                        pass
        
        # Add service time to current time
        service_time = getattr(task, 'service_time', 0)
        current_time += service_time
            
    # H8: Driver regulations check (already done in _check_hos_multiday)
    # H9: Route ending position check
    if hasattr(route, 'preferred_end_position') and sorted_tasks:
        last_task = sorted_tasks[-1]
        if hasattr(last_task, 'location') and last_task.location != route.preferred_end_position:
            reason = f"Route ending position mismatch: expected {route.preferred_end_position}, got {last_task.location}"
            if return_reason:
                return False, reason
            return False

    # H10: UNIFIED HARD CONSTRAINT CHECK (ALWAYS ENFORCED)
    # Check all hard constraints that should never be violated
    hard_constraint_valid, hard_constraint_reason = check_hard_constraints(route, debug_feasibility)
    if not hard_constraint_valid:
        if debug_feasibility:
            print(f"            DEBUG FEASIBILITY: {hard_constraint_reason}")
        if return_reason:
            return False, hard_constraint_reason
        return False
    
    # H11: ROUTE DURATION CHECK - Prevent routes that exceed 24 hours with time-sensitive tasks
    # Calculate total route duration from depot start to depot return
    if sorted_tasks and len(sorted_tasks) >= 2:
        first_task = sorted_tasks[0]
        last_task = sorted_tasks[-1]
        
        # Get departure time from first task
        departure_time = getattr(first_task, 'departure_time', getattr(first_task, 'arrival_time', 0))
        
        # Get arrival time at last task (depot return)
        final_arrival_time = getattr(last_task, 'arrival_time', departure_time)
        
        # Calculate total route duration
        total_duration = final_arrival_time - departure_time
        
        # CRITICAL: Reject routes longer than 24 hours that contain time-sensitive tasks
        MAX_ROUTE_DURATION = 1440  # 24 hours in minutes
        if total_duration > MAX_ROUTE_DURATION:
            # Check if route contains tasks with specific time windows (not depot tasks)
            has_time_sensitive_tasks = False
            for task in sorted_tasks:
                if hasattr(task, 'earliest_time') and hasattr(task, 'latest_time'):
                    if task.earliest_time is not None or task.latest_time is not None:
                        has_time_sensitive_tasks = True
                        break
            
            if has_time_sensitive_tasks:
                reason = f"Route duration {total_duration:.1f} minutes exceeds {MAX_ROUTE_DURATION} minute limit for routes with time-sensitive tasks. This prevents day-late deliveries."
                if debug_feasibility:
                    print(f"            DEBUG FEASIBILITY: {reason}")
                if return_reason:
                    return False, reason
                return False
    
    # H12: ENHANCED SEQUENTIAL TIME WINDOW VALIDATION WITH HoS CONSIDERATION
    # Use HoS timeline to get accurate arrival times, then validate time windows
    if sorted_tasks and not allow_soft_violations:
        if debug_feasibility:
            print(f"            DEBUG FEASIBILITY: HoS-AWARE SEQUENTIAL VALIDATION - Checking {len(sorted_tasks)} tasks against HoS timeline")
        
        # Try to get cached timeline from HoS validation above
        timeline_violations = []
        if hasattr(route, '_cached_timeline') and route._cached_timeline:
            timeline = route._cached_timeline
            if debug_feasibility:
                print(f"            DEBUG FEASIBILITY: Using cached HoS timeline with {len(timeline)} events")
            
            # Extract tasks from timeline and validate their arrival times against time windows
            for i, event in enumerate(timeline):
                if hasattr(event, 'task') and event.task and hasattr(event, 'start_time'):
                    task = event.task
                    arrival_time = event.start_time
                    
                    # Check time window violations using HoS timeline data
                    if hasattr(task, 'latest_time') and task.latest_time is not None:
                        if arrival_time > task.latest_time:
                            late_by = arrival_time - task.latest_time
                            timeline_violations.append({
                                'task_id': getattr(task, 'id', f'event_{i}'),
                                'arrival_time': arrival_time,
                                'latest_time': task.latest_time,
                                'late_by': late_by
                            })
                            if debug_feasibility:
                                print(f"            DEBUG FEASIBILITY: HoS TIMELINE VIOLATION - Task {getattr(task, 'id', f'event_{i}')}: arrival {arrival_time:.1f} > latest {task.latest_time:.1f} (late by {late_by:.1f} min)")
            
            if timeline_violations:
                reason = f"HoS timeline validation found {len(timeline_violations)} time window violations: {[v['task_id'] for v in timeline_violations]}"
                if debug_feasibility:
                    print(f"            DEBUG FEASIBILITY: {reason}")
                    for violation in timeline_violations:
                        print(f"            DEBUG FEASIBILITY: HoS VIOLATION DETAIL - Task {violation['task_id']}: arrival {violation['arrival_time']:.1f} > latest {violation['latest_time']:.1f} (late by {violation['late_by']:.1f} min)")
                if return_reason:
                    return False, reason
                return False
            elif debug_feasibility:
                tasks_checked = len([e for e in timeline if hasattr(e, 'task') and e.task and hasattr(e, 'start_time')])
                print(f"            DEBUG FEASIBILITY: HoS timeline validation - All {tasks_checked} tasks passed time window validation")
        else:
            # Fallback to sequential validation if no HoS timeline available
            if debug_feasibility:
                print(f"            DEBUG FEASIBILITY: No HoS timeline available, using fallback sequential validation")
            
            # Start with proper timeline calculation (mimicking display logic)
            completion_time = 0  # Start at depot at time 0
            simulation_violations = []
            
            for i, task in enumerate(sorted_tasks):
                # Skip depot tasks (they don't have time windows)
                if (hasattr(task, 'is_depot_start') and task.is_depot_start()) or \
                   (hasattr(task, 'is_depot_return') and task.is_depot_return()):
                    service_time = getattr(task, 'service_time', 5.0)
                    completion_time += service_time
                    continue
                
                # Calculate travel time to this task from previous location
                travel_time = 0
                wait_time = 0
                if i > 0:
                    prev_task = sorted_tasks[i-1]
                    try:
                        travel_time = calculate_travel_time_between_tasks(prev_task, task, route.vehicle)
                    except:
                        travel_time = 60  # Fallback as in display logic
                    
                    # Check if we need to wait for earliest time (as display does)
                    try:
                        if hasattr(task, 'earliest_time') and task.earliest_time is not None:
                            required_departure_time = task.earliest_time - travel_time
                            if completion_time < required_departure_time:
                                wait_time = required_departure_time - completion_time
                    except:
                        pass
                
                # Calculate arrival time (same logic as display)
                departure_time = completion_time + wait_time
                arrival_time = departure_time + travel_time
                
                # ENHANCED DEBUG: Show detailed time calculations
                if debug_feasibility:
                    print(f"            DEBUG FEASIBILITY: FALLBACK SEQUENTIAL - Task {getattr(task, 'id', f'task_{i}')}: completion_time={completion_time:.1f}, wait_time={wait_time:.1f}, travel_time={travel_time:.1f}, arrival_time={arrival_time:.1f}, latest_time={getattr(task, 'latest_time', 'None')}")
                
                # Check time window constraints (same logic as display)
                if hasattr(task, 'latest_time') and task.latest_time is not None:
                    if arrival_time > task.latest_time:
                        late_by = arrival_time - task.latest_time
                        simulation_violations.append({
                            'task_id': getattr(task, 'id', f'task_{i}'),
                            'arrival_time': arrival_time,
                            'latest_time': task.latest_time,
                            'late_by': late_by
                        })
                        if debug_feasibility:
                            print(f"            DEBUG FEASIBILITY: FALLBACK SEQUENTIAL VIOLATION - Task {getattr(task, 'id', f'task_{i}')}: arrival {arrival_time:.1f} > latest {task.latest_time:.1f} (late by {late_by:.1f} min)")
                
                # Update completion time for next iteration
                service_time = getattr(task, 'service_time', 5.0)
                completion_time = arrival_time + service_time
            
            if simulation_violations:
                reason = f"Fallback sequential validation found {len(simulation_violations)} time window violations: {[v['task_id'] for v in simulation_violations]}"
                if debug_feasibility:
                    print(f"            DEBUG FEASIBILITY: {reason}")
                    for violation in simulation_violations:
                        print(f"            DEBUG FEASIBILITY: FALLBACK VIOLATION DETAIL - Task {violation['task_id']}: arrival {violation['arrival_time']:.1f} > latest {violation['latest_time']:.1f} (late by {violation['late_by']:.1f} min)")
                if return_reason:
                    return False, reason
                return False
            elif debug_feasibility:
                non_depot_tasks = len([t for t in sorted_tasks if not (hasattr(t, 'is_depot_start') and t.is_depot_start()) and not (hasattr(t, 'is_depot_return') and t.is_depot_return())])
                print(f"            DEBUG FEASIBILITY: Fallback sequential validation - All {non_depot_tasks} non-depot tasks passed time window validation")
    
    # All checks passed - route is feasible
    if return_reason:
        return True, "All feasibility checks passed"
    return True
        
# Remove the old _check_hos function as it's replaced by _check_hos_multiday

def format_absolute_minutes(minutes):
    """Format absolute minutes from planning start into readable Day X, HH:MM format."""
    if minutes is None:
        return "No window"
    day = int(minutes / 1440) + 1
    remaining_minutes = int(minutes % 1440)
    hour = remaining_minutes // 60
    minute = remaining_minutes % 60
    return f"Day {day}, {hour:02d}:{minute:02d}"

def _enforce_pickup_first_sequencing_basic(tasks: List) -> List:
    """
    Basic pickup-first sequencing without LIFO constraints.
    """
    if not tasks:
        return []
    
    pickups = []
    deliveries = []
    others = []
    
    for task in tasks:
        if task.is_pickup():
            pickups.append(task)
        elif task.is_delivery():
            deliveries.append(task)
        else:
            others.append(task)
    
    # Sort by order_id for consistency
    pickups.sort(key=lambda t: getattr(t, 'order_id', ''))
    deliveries.sort(key=lambda t: getattr(t, 'order_id', ''))
    
    return others + pickups + deliveries


def _enforce_pickup_first_sequencing_with_lifo(tasks: List, vehicle) -> List:
    """
    Pickup-first sequencing with LIFO constraints for deliveries.
    """
    if not tasks:
        return []
    
    pickups = []
    deliveries = []
    others = []
    
    for task in tasks:
        if task.is_pickup():
            pickups.append(task)
        elif task.is_delivery():
            deliveries.append(task)
        else:
            others.append(task)
    
    # Sort pickups by order_id for consistency
    pickups.sort(key=lambda t: getattr(t, 'order_id', ''))
    
    # For LIFO: deliveries must be in reverse order of pickups
    # Create a mapping of order_id to pickup position
    pickup_order = {getattr(p, 'order_id', ''): i for i, p in enumerate(pickups)}
    
    # Sort deliveries in reverse pickup order (LIFO)
    deliveries.sort(key=lambda t: pickup_order.get(getattr(t, 'order_id', ''), 999), reverse=True)
    
    return others + pickups + deliveries


def _enforce_pickup_first_sequencing(tasks: List) -> List:
    """
    Enforce pickup-first sequencing as required by EPDT algorithm.
    
    Args:
        tasks: List of tasks to reorder
        
    Returns:
        List of tasks reordered to ensure all pickups come before all deliveries
    """
    if not tasks:
        return []
    
    pickups = []
    deliveries = []
    others = []
    
    for task in tasks:
        if task.is_pickup():
            pickups.append(task)
        elif task.is_delivery():
            deliveries.append(task)
        else:
            others.append(task)
    
    # Combine: others first (like depot returns), then pickups, then deliveries
    return others + pickups + deliveries


def _fix_global_pickup_delivery_constraint(route: 'Route') -> 'Route':
    """
    Fix global pickup-before-delivery constraint violations by reordering tasks.
    
    This function ensures that ALL pickup tasks are completed before ANY delivery tasks begin,
    which is the core requirement of the EPDT algorithm.
    
    Args:
        route: Route with potentially violating task sequence
        
    Returns:
        Route with properly ordered tasks (all pickups before all deliveries)
    """
    if not route.tasks:
        return route
    
    # Create a copy to avoid modifying the original
    from copy import deepcopy
    fixed_route = deepcopy(route)
    
    # Reorder tasks to enforce global pickup-before-delivery constraint
    fixed_route.tasks = _enforce_pickup_first_sequencing(route.tasks)
    
    # Clear any cached scores since task order changed
    if hasattr(fixed_route, '_z2_score'):
        delattr(fixed_route, '_z2_score')
    
    return fixed_route


def _enforce_pickup_first_sequencing(tasks: List) -> List:
    """
    Reorder tasks to ensure pickup-first sequencing within each day.
    For LIFO-required vehicles, also ensures deliveries are in reverse pickup order.
    
    Args:
        tasks: List of tasks to reorder
        
    Returns:
        List of tasks with pickups before deliveries, with LIFO ordering if required
    """
    if not tasks:
        return []
    
    # Check if this route requires LIFO constraints
    lifo_required = False
    if tasks:
        # Try to find the vehicle through the task's route
        for task in tasks:
            if hasattr(task, 'route') and task.route and hasattr(task.route, 'vehicle'):
                lifo_required = getattr(task.route.vehicle, 'lifo_required', False)
                break
    
    # Group tasks by day
    tasks_by_day = {}
    for task in tasks:
        day = getattr(task, 'day', 0)  # Default to today (0) if no day attribute
        if day not in tasks_by_day:
            tasks_by_day[day] = []
        tasks_by_day[day].append(task)
    
    # For each day, separate pickups and deliveries
    reordered_tasks = []
    for day in sorted(tasks_by_day.keys()):
        day_tasks = tasks_by_day[day]
        
        depot_starts = []
        pickups = []
        deliveries = []
        depot_returns = []
        
        for task in day_tasks:
            if task.is_pickup():
                pickups.append(task)
            elif task.is_delivery():
                deliveries.append(task)
            elif getattr(task, 'id', '').startswith('DEPOT_START'):
                depot_starts.append(task)
            elif getattr(task, 'id', '').startswith('DEPOT_RETURN'):
                depot_returns.append(task)
            else:
                depot_starts.append(task)  # Default to depot start category
        
        # Sort pickups and deliveries by order_id for consistency
        pickups.sort(key=lambda t: getattr(t, 'order_id', ''))
        deliveries.sort(key=lambda t: getattr(t, 'order_id', ''))
        
        if lifo_required and pickups and deliveries:
            # For LIFO constraint: deliveries must be in reverse order of pickups
            # Create a mapping of order_id to pickup position
            pickup_order = {getattr(p, 'order_id', ''): i for i, p in enumerate(pickups)}
            
            # Sort deliveries in reverse pickup order (LIFO)
            deliveries.sort(key=lambda t: pickup_order.get(getattr(t, 'order_id', ''), 999), reverse=True)
        
        # Combine this day's tasks: depot starts, pickups, deliveries, depot returns
        day_reordered = depot_starts + pickups + deliveries + depot_returns
        reordered_tasks.extend(day_reordered)
    
    return reordered_tasks


def _enforce_lifo_sequencing(route: 'Route') -> 'Route':
    """
    Reorder tasks in route to satisfy LIFO constraints.
    
    LIFO (Last In, First Out) means:
    - The last order picked up must be the first order delivered
    - This simulates physical loading where you can only access the top of the stack
    
    Args:
        route: Route to reorder
        
    Returns:
        Route with LIFO-compliant task sequence
    """
    if not hasattr(route.vehicle, 'lifo_required') or not route.vehicle.lifo_required:
        return route  # No LIFO constraint, return as-is
    
    # Create new route with LIFO-compliant sequence
    new_route = route.copy()
    new_route.tasks = _enforce_pickup_first_sequencing(route.tasks)
    
    return new_route


def _sort_tasks_chronologically(tasks: List) -> List:
    """
    Sort tasks in strict chronological order: yesterday's tasks, then today's, then tomorrow's.
    Within each day, maintain the original sequence.
    
    Args:
        tasks: List of tasks to sort
        
    Returns:
        List of tasks sorted chronologically by day
    """
    if not tasks:
        return []
    
    # Group tasks by day
    tasks_by_day = {}
    for task in tasks:
        day = getattr(task, 'day', 0)  # Default to today (0) if no day attribute
        if day not in tasks_by_day:
            tasks_by_day[day] = []
        tasks_by_day[day].append(task)
    
    # Sort days and concatenate tasks
    sorted_tasks = []
    for day in sorted(tasks_by_day.keys()):
        sorted_tasks.extend(tasks_by_day[day])
    
    return sorted_tasks


def _check_hos_multiday(route: 'Route', driver_state: 'DriverState', sorted_tasks: List) -> tuple[bool, float]:
    """
    Enhanced Hours of Service check with multi-day support and detailed European regulations.
    
    European HoS Regulations (only apply to CE license drivers):
    - After 4.5 hours of driving, a 45-minute break is mandatory (can be split into 15 + 30 mins)
    - Maximum 9 hours of driving per day (extendable to 10 hours twice a week)
    - Maximum 13 hours of work per day (extendable to 14 hours twice a week)
    - Minimum 11 hours of daily rest (can be reduced to 9 hours under certain conditions)
    
    Note: Drivers with B licenses are exempt from HoS regulations and can drive unlimited hours.
    
    Args:
        route: The route being checked
        driver_state: Current driver state (potentially from previous day)
        sorted_tasks: Tasks sorted chronologically
        
    Returns:
        Tuple of (is_feasible: bool, hos_cost: float)
    """
    if not sorted_tasks or len(sorted_tasks) < 2:
        return True, 0.0

    total_rest_cost = 0.0
    driver_cost_per_minute = (route.vehicle.cost_per_hour / 60.0) if route.vehicle else (25.0 / 60.0)
    sim_state = driver_state.copy()
    
    # B license drivers are exempt from HoS regulations
    if route.driver and hasattr(route.driver, 'license') and route.driver.license == 'B':
        return True, 0.0
    
    current_time = 0
    current_day = getattr(sorted_tasks[0], 'day', 0)
    extensions_used_this_week = {'driving': 0, 'work': 0}  # Track extensions
    
    # Process each task in chronological order
    for i in range(len(sorted_tasks) - 1):
        # Placed at the start of the for loop
        start_task = sorted_tasks[i]
        end_task = sorted_tasks[i+1]

        # 1. Simulate Service Time at the start_task
        service_time = start_task.service_time
        if service_time > 0:
            if sim_state.work_today + service_time > sim_state.MAX_WORK_PER_DAY:
                return False, total_rest_cost # Infeasible
            sim_state.work_today += service_time
            sim_state.work_this_week += service_time
        
        # Check for day transition
        task_day = getattr(start_task, 'day', 0)
        if task_day != current_day:
            # New day: check if sufficient daily rest was taken
            if sim_state.work_today > 0:  # If work was done previous day
                # Minimum 11 hours daily rest required
                rest_time = 11 * 60  # 11 hours in minutes
                total_rest_cost += rest_time * driver_cost_per_minute
                current_time += rest_time
                sim_state.reset_daily()  # Reset daily counters
            current_day = task_day
        
        # Calculate travel time using proper route provider
        # Use consistent travel time calculation (OSRM-based)
        try:
            from route_provider import calculate_travel_time_between_tasks
            travel_time = calculate_travel_time_between_tasks(start_task, end_task, route.vehicle)
            # travel_time is already in minutes - no conversion needed
        except ImportError:
            # Fallback to haversine if route provider not available
            travel_time = calculate_travel_time_haversine(
                start_task.lat, start_task.lon,
                end_task.lat, end_task.lon
            )
            # calculate_travel_time_haversine returns minutes, so no conversion needed
        
        travel_time_remaining = travel_time
        while travel_time_remaining > 0:
            # This block replaces the old state update logic inside the new while loop
            max_drive_before_break = sim_state.MAX_DRIVE_WITHOUT_BREAK - sim_state.drive_since_break
            max_drive_before_daily_limit = sim_state.MAX_DRIVE_PER_DAY - sim_state.drive_today
            max_work_before_daily_limit = sim_state.MAX_WORK_PER_DAY - sim_state.work_today

            drivable_time = min(max_drive_before_break, max_drive_before_daily_limit, max_work_before_daily_limit, travel_time_remaining)

            # Simulate driving for the calculated time
            sim_state.drive_since_break += drivable_time
            sim_state.drive_today += drivable_time
            sim_state.work_today += drivable_time
            sim_state.drive_this_week += drivable_time
            sim_state.work_this_week += drivable_time
            travel_time_remaining -= drivable_time

            # If travel is not complete, a rest would be required - this makes the route infeasible
            if travel_time_remaining > 0:
                # A) 4.5-hour driving break would be required
                if sim_state.drive_since_break >= sim_state.MAX_DRIVE_WITHOUT_BREAK:
                    # Route is infeasible - cannot complete travel without mandatory break
                    return False, total_rest_cost

                # B) Daily or Weekly limit would be reached
                if sim_state.drive_today >= sim_state.MAX_DRIVE_PER_DAY or sim_state.work_today >= sim_state.MAX_WORK_PER_DAY:
                    # Route is infeasible - cannot complete travel without mandatory rest
                    return False, total_rest_cost
                
                if sim_state.work_this_week >= sim_state.MAX_WORK_PER_WEEK:
                    # Route is infeasible - weekly limit exceeded
                    return False, total_rest_cost
        
        # Check for soft time window compliance (if applicable) - using start_task
        if hasattr(start_task, 'soft_time_window') and start_task.soft_time_window:
            if (hasattr(start_task, 'latest_time') and 
                start_task.latest_time is not None and 
                current_time > start_task.latest_time):
                # Soft time window violation - continue but will be penalized in scoring
                pass
        elif (hasattr(start_task, 'latest_time') and 
              start_task.latest_time is not None and 
              current_time > start_task.latest_time):
            # Hard time window violation
            return False, total_rest_cost
        
        # Handle waiting for earliest time with correct HoS accounting
        if (hasattr(start_task, 'earliest_time') and 
            start_task.earliest_time is not None and 
            current_time < start_task.earliest_time):
            wait_time = start_task.earliest_time - current_time
            current_time = start_task.earliest_time
            
            # Determine if this is depot waiting or customer waiting
            is_depot_waiting = (hasattr(start_task, 'is_depot_start') and start_task.is_depot_start()) or \
                              (hasattr(start_task, 'task_type') and 'depot' in str(start_task.task_type).lower())
            
            if not is_depot_waiting:
                # Customer waiting counts as work time per European regulations
                sim_state.work_today += wait_time
                sim_state.work_this_week += wait_time
                
                # Check if waiting time would cause HoS violation
                if sim_state.work_today > sim_state.MAX_WORK_PER_DAY:
                    return False, total_rest_cost
            # Depot waiting does not count toward work time - driver shift hasn't started
    
    return True, total_rest_cost


# HoS simulation functions have been moved to hos_simulation.py module
# to resolve circular imports and improve modularity.
# The functions below are kept for backward compatibility.

def _get_max_drive_per_day(extensions_used: dict) -> float:
    """Helper function for backward compatibility."""
    driving_extensions = extensions_used.get('driving', 0)
    return 10 * 60 if driving_extensions < 2 else 9 * 60

def _get_max_work_per_day(extensions_used: dict) -> float:
    """Helper function for backward compatibility.""" 
    work_extensions = extensions_used.get('work', 0)
    return 14 * 60 if work_extensions < 2 else 13 * 60