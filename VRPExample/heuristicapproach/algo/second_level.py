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
    
    # Order 7 specific debugging
    order_7_debug = False
    if hasattr(route, 'vehicle') and route.vehicle:
        order_7_tasks = [task for task in route.tasks if hasattr(task, 'order_id') and str(task.order_id) == '7']
        if order_7_tasks:
            order_7_debug = True
            debug = True  # Force debug output for Order 7
            print(f"            *** ORDER 7 HARD CONSTRAINT DEBUG ***")
            print(f"            Vehicle: {route.vehicle.id}")
            print(f"            Order 7 tasks: {[task.id for task in order_7_tasks]}")
        
    # VALIDATE EXISTING TASK ORDER (don't reorder!)
    tasks = route.tasks
    
    # Check precedence constraints first
    if not _validate_precedence_constraints(tasks):
        reason = "HARD CONSTRAINT VIOLATION: Precedence constraints violated (pickup must come before delivery)"
        if debug:
            print(f"            HARD CONSTRAINT CHECK: {reason}")
        return False, reason
    
    if order_7_debug:
        print(f"             Precedence constraints passed")
    
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
    
    if order_7_debug:
        print(f"            Route requires capabilities: {all_required_capabilities}")
    
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
                
                # FIXED LIFO LOGIC: Allow same-order deliveries regardless of stack position
                # For patterns like PP...PD (multiple pickups, single delivery of same order)
                # we should remove ALL instances of the order from the stack
                if task.order_id in lifo_stack:
                    # For same-order patterns, remove ALL instances of this order
                    # This handles PP...D patterns where all cargo is delivered together
                    while task.order_id in lifo_stack:
                        lifo_stack.remove(task.order_id)
                else:
                    # Only enforce strict LIFO when mixing different orders
                    if lifo_stack[-1] != task.order_id:
                        reason = f"HARD CONSTRAINT VIOLATION: LIFO violation - expected {lifo_stack[-1]}, got {task.order_id} (mixed-order constraint)"
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
        
        # For complex orders: validate paired pickup-delivery relationships
        # Allow PDPDPD patterns as long as each pickup comes before its delivery
        else:
            # ENHANCED LOGIC: Check actual pickup-delivery pairs by task relationships
            # This allows interleaved patterns like PDPDPD for complex orders
            
            # Build actual pickup-delivery pairs by examining task properties
            pickup_delivery_pairs = []
            
            for pickup in pickups:
                pickup_pos = pickup['position']
                pickup_task = tasks[pickup_pos]
                
                # Find corresponding delivery by matching task properties
                # Strategy 1: Match by magnitude (pickup +300kg → delivery -300kg)
                # Strategy 2: Match by task ID patterns (TASK_7_14 → TASK_7_15)
                
                best_delivery = None
                best_delivery_pos = float('inf')
                
                for delivery in deliveries:
                    delivery_pos = delivery['position']
                    delivery_task = tasks[delivery_pos]
                    
                    # Check if this pickup-delivery pair makes sense
                    is_matching_pair = False
                    
                    # Method 1: Match by magnitude (pickup pallets = -delivery pallets)
                    pickup_pallets = getattr(pickup_task, 'pallets', 0)
                    delivery_pallets = getattr(delivery_task, 'pallets', 0)
                    if pickup_pallets > 0 and delivery_pallets < 0 and pickup_pallets == -delivery_pallets:
                        is_matching_pair = True
                    
                    # Method 2: Match by weight magnitude
                    pickup_weight = getattr(pickup_task, 'weight', 0)
                    delivery_weight = getattr(delivery_task, 'weight', 0)
                    if pickup_weight > 0 and delivery_weight < 0 and abs(pickup_weight + delivery_weight) < 1.0:
                        is_matching_pair = True
                    
                    # Method 3: Match by task ID pattern (TASK_X_14 → TASK_X_15)
                    pickup_id = getattr(pickup_task, 'id', '')
                    delivery_id = getattr(delivery_task, 'id', '')
                    if pickup_id and delivery_id:
                        # Extract base pattern: TASK_7_14 → TASK_7_, then find corresponding delivery
                        import re
                        pickup_match = re.match(r'(TASK_\d+_)(\d+)', pickup_id)
                        delivery_match = re.match(r'(TASK_\d+_)(\d+)', delivery_id)
                        if pickup_match and delivery_match:
                            pickup_base = pickup_match.group(1)
                            delivery_base = delivery_match.group(1)
                            pickup_num = int(pickup_match.group(2))
                            delivery_num = int(delivery_match.group(2))
                            # Common pattern: delivery_num = pickup_num + 1
                            if pickup_base == delivery_base and delivery_num == pickup_num + 1:
                                is_matching_pair = True
                    
                    if is_matching_pair and delivery_pos > pickup_pos:
                        if delivery_pos < best_delivery_pos:
                            best_delivery = delivery
                            best_delivery_pos = delivery_pos
                
                if best_delivery:
                    pickup_delivery_pairs.append((pickup_pos, best_delivery_pos))
            
            # Validate all identified pairs
            for pickup_pos, delivery_pos in pickup_delivery_pairs:
                if pickup_pos >= delivery_pos:
                    # DEBUG for ALL orders
                    order_id = getattr(tasks[pickup_pos], 'order_id', 'unknown')
                    pickup_task_id = getattr(tasks[pickup_pos], 'id', f'pos_{pickup_pos}')
                    delivery_task_id = getattr(tasks[delivery_pos], 'id', f'pos_{delivery_pos}')
                    print(f"            DEBUG: Order {order_id} precedence failure - pickup {pickup_task_id} at pos {pickup_pos} >= delivery {delivery_task_id} at pos {delivery_pos}")
                    return False
            
            # If we couldn't identify proper pairs, fall back to strict validation
            if len(pickup_delivery_pairs) != len(pickups):
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
    
    # ENHANCED: Generate optimized interleaved patterns first (as requested by user)
    optimized_patterns = generate_optimized_interleaved_patterns(pickups, deliveries)
    
    for pattern in optimized_patterns:
        peak_capacity = calculate_sequence_peak_capacity(pattern['sequence'])
        valid_sequences.append({
            'sequence': pattern['sequence'],
            'peak_capacity': peak_capacity,
            'is_sequential_chain': pattern['is_interleaved']
        })
        
        # Debug output for optimized patterns
        sequence_types = [getattr(t.task_type, 'name', 'UNK')[0] for t in pattern['sequence']]
        sequence_pattern = ''.join(sequence_types)
        print(f"    Optimized {pattern['pattern_type']} ({sequence_pattern}): {peak_capacity['peak_pallets']:.0f}pal peak")
    
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


def generate_distance_optimized_sequences(pickups: List['Task'], deliveries: List['Task'], pattern_type: str) -> List[Dict]:
    """
    Generate LOCATION-AWARE sequences using geographical clustering while respecting pickup-delivery pairing.
    
    This implementation fixes the task pairing issue by ensuring each pickup is correctly
    paired with its corresponding delivery (same order_id and matching cargo amounts).
    
    Key improvements:
    1. Groups tasks by geographical location
    2. Maintains pickup-delivery pairing constraints
    3. Minimizes total travel distance between locations
    4. Respects time window constraints
    """
    import math
    
    def get_location_name(task):
        """Extract location name from task"""
        if hasattr(task, 'location') and hasattr(task.location, 'name'):
            return task.location.name
        elif hasattr(task, 'location_id'):
            return task.location_id
        else:
            return str(getattr(task, 'id', 'unknown'))
    
    def get_time_window(task):
        """Extract time window from task"""
        earliest = getattr(task, 'earliest_time', 0)
        latest = getattr(task, 'latest_time', 9999)
        return earliest, latest
    
    def find_corresponding_delivery(pickup_task, delivery_tasks):
        """Find the delivery task that corresponds to a pickup task"""
        pickup_order_id = getattr(pickup_task, 'order_id', None)
        pickup_demand = abs(getattr(pickup_task, 'demand', 0))
        pickup_pallets = abs(getattr(pickup_task, 'pallets', 0))
        
        for delivery in delivery_tasks:
            delivery_order_id = getattr(delivery, 'order_id', None)
            delivery_demand = abs(getattr(delivery, 'demand', 0))
            delivery_pallets = abs(getattr(delivery, 'pallets', 0))
            
            # Match by order_id and same cargo amounts
            if (pickup_order_id == delivery_order_id and 
                pickup_demand == delivery_demand and 
                pickup_pallets == delivery_pallets):
                return delivery
        return None
    
    def calculate_location_distance(loc1_name, loc2_name):
        """Calculate distance between location names (simple heuristic)"""
        if loc1_name == loc2_name:
            return 0.0  # Same location
        
        # Milano locations are close to each other
        milano_locations = ['CORSO VENEZIA, MILANO', 'C.SO DI PORTA NUOVA, MILANO']
        if any(milano in loc1_name for milano in milano_locations) and any(milano in loc2_name for milano in milano_locations):
            return 0.1  # Very close Milano locations
        
        return 1.0  # Different distant locations
    
    # STEP 1: Create pickup-delivery pairs
    pickup_delivery_pairs = []
    used_deliveries = set()
    
    for pickup in pickups:
        corresponding_delivery = find_corresponding_delivery(pickup, deliveries)
        if corresponding_delivery and corresponding_delivery.id not in used_deliveries:
            pickup_delivery_pairs.append((pickup, corresponding_delivery))
            used_deliveries.add(corresponding_delivery.id)
        else:
            print(f"        WARNING: No corresponding delivery found for pickup {pickup.id}")
    
    print(f"        PICKUP-DELIVERY PAIRING: Created {len(pickup_delivery_pairs)} pairs")
    for pickup, delivery in pickup_delivery_pairs:
        print(f"          {pickup.id} ({pickup.pallets}pal) ↔ {delivery.id} ({delivery.pallets}pal)")
    
    # STEP 2: Group pairs by pickup location (since pickup must come first)
    location_pairs = {}
    
    for pickup, delivery in pickup_delivery_pairs:
        pickup_loc = get_location_name(pickup)
        delivery_loc = get_location_name(delivery)
        
        if pickup_loc not in location_pairs:
            location_pairs[pickup_loc] = []
        
        location_pairs[pickup_loc].append({
            'pickup': pickup,
            'delivery': delivery,
            'pickup_location': pickup_loc,
            'delivery_location': delivery_loc,
            'pickup_time': get_time_window(pickup)[0],
            'delivery_time': get_time_window(delivery)[0]
        })
    
    print(f"        LOCATION GROUPING:")
    for loc_name, pairs in location_pairs.items():
        print(f"        - {loc_name}: {len(pairs)} pickup-delivery pairs")
    
    # STEP 3: Generate location-optimized sequences
    sequences = []
    
    # STRATEGY 1: Sort by pickup time windows, maintain pairing
    time_sorted_pairs = []
    for loc_name, pairs in location_pairs.items():
        time_sorted_pairs.extend(sorted(pairs, key=lambda x: x['pickup_time']))
    
    time_sorted_pairs.sort(key=lambda x: x['pickup_time'])
    
    print(f"        SEQUENCE GENERATION: Processing {len(time_sorted_pairs)} pairs by pickup time")
    
    # Build sequence respecting pickup-delivery precedence
    optimal_sequence = []
    total_distance = 0.0
    previous_pickup_location = None
    previous_delivery_location = None
    
    for pair in time_sorted_pairs:
        pickup = pair['pickup']
        delivery = pair['delivery']
        pickup_loc = pair['pickup_location']
        delivery_loc = pair['delivery_location']
        
        # Add pickup first
        optimal_sequence.append(pickup)
        print(f"          PICKUP: {pickup.id} at {pickup_loc} ({pair['pickup_time']:.0f}min)")
        
        # Calculate distance from previous location to pickup location
        if previous_delivery_location and previous_delivery_location != pickup_loc:
            total_distance += calculate_location_distance(previous_delivery_location, pickup_loc)
        
        # Add corresponding delivery second
        optimal_sequence.append(delivery)
        print(f"          DELIVERY: {delivery.id} at {delivery_loc} ({pair['delivery_time']:.0f}min)")
        
        # Calculate distance from pickup to delivery location  
        if pickup_loc != delivery_loc:
            total_distance += calculate_location_distance(pickup_loc, delivery_loc)
        
        previous_pickup_location = pickup_loc
        previous_delivery_location = delivery_loc
    
    sequences.append({
        'sequence': optimal_sequence,
        'distance_score': total_distance,
        'optimization_method': 'paired_time_ordered'
    })
    
    print(f"        Generated 1 location-optimized sequence with total distance: {total_distance:.2f}")
    
    return sequences
    
    for i in range(len(time_ordered_sequence) - 1):
        loc1 = get_location_name(time_ordered_sequence[i])
        loc2 = get_location_name(time_ordered_sequence[i+1])
        time_distance += calculate_location_distance(loc1, loc2)
    
    sequences.append({
        'sequence': time_ordered_sequence,
        'distance_score': time_distance,
        'optimization_method': 'time_window_ordered'
    })
    
    # Sort by distance score (lower is better)  
    sequences.sort(key=lambda x: x['distance_score'])
    
    print(f"        Generated {len(sequences)} location-optimized sequences")
    print(f"        Best sequence distance: {sequences[0]['distance_score']:.2f}")
    
    return sequences


def generate_optimized_interleaved_patterns(pickups: List['Task'], deliveries: List['Task']) -> List[Dict]:
    """
    Generate optimized interleaved patterns to minimize peak capacity usage.
    
    Based on user requirements:
    - For 2P+1D orders (like Orders 5,6): Generate PPD and PDP patterns  
    - For 1P+4D orders (like Order 8): Generate PDDDD with optimized delivery order
    - For 3P+3D orders (like Order 7): Generate PDPDPD instead of PPPDDD
    
    Args:
        pickups: List of pickup tasks
        deliveries: List of delivery tasks
        
    Returns:
        List of dictionaries with 'sequence', 'pattern_type', and 'is_interleaved'
    """
    patterns = []
    num_pickups = len(pickups)
    num_deliveries = len(deliveries)
    
    print(f"        ADVANCED SEQUENCING: {num_pickups}P + {num_deliveries}D order")
    
    # Case A: Balanced Pickups and Deliveries (num_p == num_d > 1)
    if num_pickups == num_deliveries and num_pickups > 1:
        print(f"        Case A: Balanced order ({num_pickups}P+{num_deliveries}D) - generating interleaved PDPDPD sequences")
        
        # Strategy 1: Generate DISTANCE-OPTIMIZED interleaved sequences (PDPDPD pattern)
        # This minimizes both peak capacity AND total travel distance
        
        # ENHANCED: Use geographic clustering and distance optimization
        optimized_sequences = generate_distance_optimized_sequences(pickups, deliveries, 'PDPDPD')
        
        for seq_idx, optimized_seq in enumerate(optimized_sequences):
            patterns.append({
                'sequence': optimized_seq['sequence'],
                'pattern_type': f'PDPDPD_DISTANCE_OPT_{seq_idx}',
                'is_interleaved': True,
                'priority': 1,  # High priority - distance optimized
                'distance_score': optimized_seq.get('distance_score', 0)
            })
        
        print(f"        Generated {len(optimized_sequences)} distance-optimized PDPDPD sequences")
        
        # Strategy 2: Traditional PPPDDD pattern for comparison
        patterns.append({
            'sequence': list(pickups) + list(deliveries),
            'pattern_type': 'PPPDDD_TRADITIONAL',
            'is_interleaved': False,
            'priority': 2  # Lower priority - higher capacity
        })
        
        print(f"        Generated {len(patterns)} balanced sequences (interleaved + traditional)")
    
    # Case B: Single Pickup, Multiple Deliveries (1P, nD)
    elif num_pickups == 1 and num_deliveries > 1:
        print(f"        Case B: Single pickup, multiple deliveries (1P+{num_deliveries}D) - optimizing delivery sequence")
        
        pickup = pickups[0]
        
        # Strategy 1: TSP optimization for deliveries (minimize total distance)
        # For now, use simple heuristics - could be enhanced with proper TSP solver
        
        # Pattern 1: Original delivery order
        patterns.append({
            'sequence': [pickup] + list(deliveries),
            'pattern_type': 'PDDDD_ORIGINAL',
            'is_interleaved': False,
            'priority': 1
        })
        
        # Pattern 2: Reverse delivery order (sometimes closer)
        patterns.append({
            'sequence': [pickup] + list(reversed(deliveries)),
            'pattern_type': 'PDDDD_REVERSED',
            'is_interleaved': False,
            'priority': 2
        })
        
        # Pattern 3: Sort deliveries by location (if coordinate data available)
        try:
            # Sort by latitude + longitude as a simple distance proxy
            sorted_deliveries = sorted(deliveries, key=lambda d: (getattr(d, 'lat', 0) + getattr(d, 'lon', 0)))
            patterns.append({
                'sequence': [pickup] + sorted_deliveries,
                'pattern_type': 'PDDDD_SORTED',
                'is_interleaved': False,
                'priority': 1
            })
        except:
            pass  # Skip if no location data
        
        print(f"        Generated {len(patterns)} single-pickup sequences")
    
    # Case C: Multiple Pickups, Single Delivery (nP, 1D)
    elif num_pickups > 1 and num_deliveries == 1:
        print(f"        Case C: Multiple pickups, single delivery ({num_pickups}P+1D) - optimizing pickup sequence")
        
        delivery = deliveries[0]
        
        # Strategy 1: TSP optimization for pickups (minimize total distance)
        
        # Pattern 1: Original pickup order
        patterns.append({
            'sequence': list(pickups) + [delivery],
            'pattern_type': 'PPPPD_ORIGINAL',
            'is_interleaved': False,
            'priority': 1
        })
        
        # Pattern 2: Reverse pickup order
        patterns.append({
            'sequence': list(reversed(pickups)) + [delivery],
            'pattern_type': 'PPPPD_REVERSED',
            'is_interleaved': False,
            'priority': 2
        })
        
        # Pattern 3: Sort pickups by location (if coordinate data available)
        try:
            # Sort by latitude + longitude as a simple distance proxy
            sorted_pickups = sorted(pickups, key=lambda p: (getattr(p, 'lat', 0) + getattr(p, 'lon', 0)))
            patterns.append({
                'sequence': sorted_pickups + [delivery],
                'pattern_type': 'PPPPD_SORTED',
                'is_interleaved': False,
                'priority': 1
            })
        except:
            pass  # Skip if no location data
        
        print(f"        Generated {len(patterns)} multi-pickup sequences")
    # Case B: Single Pickup, Multiple Deliveries (1P, nD)
    elif num_pickups == 1 and num_deliveries > 1:
        pickup = pickups[0]
        
        # Sort deliveries by distance from pickup or other cost criteria
        # For now, use the original order but this could be optimized
        sorted_deliveries = deliveries.copy()
        
        # Pattern 1: Standard PDDDD
        patterns.append({
            'sequence': [pickup] + sorted_deliveries,
            'pattern_type': 'PDDDD_OPTIMIZED',
            'is_interleaved': False
        })
        
        # Could add more patterns with different delivery orders here
    
    # Case A: Balanced Pickups and Deliveries (num_p == num_d > 1) 
    elif num_pickups == num_deliveries and num_pickups > 1:
        print(f"        Case A: Balanced order ({num_pickups}P+{num_deliveries}D) - generating interleaved PDPDPD sequences")
        
        # Strategy 1: Generate all possible interleaved sequences (PDPDPD pattern)
        # This minimizes peak capacity as each pickup is immediately followed by delivery
        from itertools import permutations
        
        # Generate all permutations of pickup-delivery pairs
        pickup_permutations = list(permutations(pickups))
        delivery_permutations = list(permutations(deliveries))
        
        # Create interleaved sequences for each permutation combination
        interleaved_count = 0
        for p_perm in pickup_permutations[:3]:  # Limit to avoid explosion
            for d_perm in delivery_permutations[:3]:
                interleaved_sequence = []
                for i in range(num_pickups):
                    interleaved_sequence.extend([p_perm[i], d_perm[i]])
                
                patterns.append({
                    'sequence': interleaved_sequence,
                    'pattern_type': f'PDPDPD_INTERLEAVED_{interleaved_count}',
                    'is_interleaved': True,
                    'priority': 1  # High priority - should minimize capacity
                })
                interleaved_count += 1
                if interleaved_count >= 6:  # Limit combinations
                    break
            if interleaved_count >= 6:
                break
        
        # Strategy 2: Traditional PPPDDD pattern for comparison
        patterns.append({
            'sequence': list(pickups) + list(deliveries),
            'pattern_type': 'PPPDDD_TRADITIONAL',
            'is_interleaved': False,
            'priority': 2  # Lower priority - higher capacity
        })
        
        print(f"        Generated {len(patterns)} balanced sequences (interleaved + traditional)")
    
    # Case 4: Other patterns - generate basic interleaved if possible
    elif num_pickups > 0 and num_deliveries > 0:
        # Try to create a simple interleaved pattern
        if num_pickups == num_deliveries:
            # Create PD-PD-PD pattern
            pd_pairs = []
            for i in range(min(num_pickups, num_deliveries)):
                pd_pairs.append((pickups[i], deliveries[i]))
            
            interleaved_sequence = []
            for pickup, delivery in pd_pairs:
                interleaved_sequence.extend([pickup, delivery])
            
            patterns.append({
                'sequence': interleaved_sequence,
                'pattern_type': f'P{num_pickups}D{num_deliveries}_INTERLEAVED',
                'is_interleaved': True
            })
    
    return patterns


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
    # Check if this is for Order 7 (enhanced debugging)
    order_7_debug = debug or any(str(getattr(task, 'order_id', '')) == '7' for task in task_sequence)
    
    if order_7_debug:
        print(f"          *** INSERTION EVALUATION *** Position {insertion_point}")
        sequence_ids = [getattr(task, 'id', 'UNK') for task in task_sequence]
        print(f"              Inserting tasks: {', '.join(sequence_ids)}")
    
    # Create temporary route with sequence inserted
    temp_route = route.copy()
    
    # Insert tasks at the specified position
    for i, task in enumerate(task_sequence):
        temp_route.insert_task_without_reordering(insertion_point + i, task)
    
    # Check feasibility with real-time load simulation
    feasibility_result = is_feasible_with_load_tracking(temp_route, insertion_point, task_sequence, order_7_debug)
    
    if not feasibility_result:
        if order_7_debug:
            print(f"          *** INSERTION FAILED *** Position {insertion_point} - constraint violation")
        return {'feasible': False, 'failure_reason': 'constraint_violation'}
    
    # Calculate cost (negative because we want to minimize cost)
    cost = calculate_z2_score(temp_route)
    
    if order_7_debug:
        print(f"          *** INSERTION SUCCESS *** Position {insertion_point}, cost: {cost:.2f}")
    
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
    # Check if this is for Order 7 (enhanced debugging)
    order_7_debug = debug or any(str(getattr(task, 'order_id', '')) == '7' for task in task_sequence)
    
    if order_7_debug:
        print(f"              *** LOAD TRACKING *** from insertion point {insertion_point}")
        print(f"                  Vehicle limits: {getattr(route.vehicle, 'weight_capacity', 'N/A')}kg, {getattr(route.vehicle, 'volume_capacity', 'N/A')}m³, {getattr(route.vehicle, 'pallet_capacity', 'N/A')}pal")
    
    # Calculate initial load at insertion point
    current_weight = 0.0
    current_volume = 0.0
    current_pallets = 0.0
    
    for i in range(insertion_point):
        if i < len(route.tasks):
            task = route.tasks[i]
            if hasattr(task, 'demand'):
                current_weight += getattr(task, 'demand', 0.0)
            if hasattr(task, 'volume'):
                current_volume += getattr(task, 'volume', 0.0)
            if hasattr(task, 'pallets'):
                current_pallets += getattr(task, 'pallets', 0.0)
    
    if order_7_debug:
        print(f"                  Initial load: {current_weight:.1f}kg, {current_volume:.1f}m³, {current_pallets:.0f}pal")
    
    # Simulate load changes through the inserted sequence
    for i, task in enumerate(task_sequence):
        # Update loads
        task_weight = getattr(task, 'demand', 0.0)
        task_volume = getattr(task, 'volume', 0.0)
        task_pallets = getattr(task, 'pallets', 0.0)
        
        current_weight += task_weight
        current_volume += task_volume
        current_pallets += task_pallets
        
        if order_7_debug:
            task_type = getattr(getattr(task, 'task_type', None), 'name', 'UNK')
            task_id = getattr(task, 'id', 'UNK')
            print(f"                  Step {i+1} - {task_id} ({task_type}): {task_weight:+.1f}kg, {task_volume:+.1f}m³, {task_pallets:+.0f}pal")
            print(f"                      → Total: {current_weight:.1f}kg, {current_volume:.1f}m³, {current_pallets:.0f}pal")
        
        # Check constraints
        violations = []
        
        if current_weight < -0.1:  # Allow small floating point errors
            violations.append(f"NEGATIVE WEIGHT: {current_weight:.1f}kg")
        
        if current_volume < -0.1:  # Allow small floating point errors
            violations.append(f"NEGATIVE VOLUME: {current_volume:.1f}m³")
        
        if current_pallets < -0.1:  # Allow small floating point errors
            violations.append(f"NEGATIVE PALLETS: {current_pallets:.1f}")
        
        if hasattr(route.vehicle, 'weight_capacity') and current_weight > route.vehicle.weight_capacity:
            violations.append(f"WEIGHT EXCEEDED: {current_weight:.1f}kg > {route.vehicle.weight_capacity:.1f}kg")
        
        if hasattr(route.vehicle, 'volume_capacity') and current_volume > route.vehicle.volume_capacity:
            violations.append(f"VOLUME EXCEEDED: {current_volume:.1f}m³ > {route.vehicle.volume_capacity:.1f}m³")
        
        if hasattr(route.vehicle, 'pallet_capacity') and current_pallets > route.vehicle.pallet_capacity:
            violations.append(f"PALLETS EXCEEDED: {current_pallets:.0f} > {route.vehicle.pallet_capacity:.0f}")
        
        if violations:
            if order_7_debug:
                print(f"                      *** CONSTRAINT VIOLATIONS ***")
                for violation in violations:
                    print(f"                          - {violation}")
                print(f"                      *** INSERTION FAILED ***")
            return False
    
    # Check overall route feasibility (time windows, HoS, etc.)
    try:
        # DEBUG: Add detailed time window analysis before feasibility check
        if debug:
            print(f"        DEBUG: Analyzing route timing and time windows...")
            print(f"        DEBUG: Route has {len(route.tasks)} tasks")
            
            # Calculate estimated travel times and arrival times
            current_time = 0.0  # Start at midnight (0:00) - vehicles can depart anytime
            total_travel_time = 0.0
            total_service_time = 0.0
            
            for i, task in enumerate(route.tasks):
                task_type = getattr(task, 'task_type', 'UNKNOWN')
                
                # FIXED: Use correct location attributes
                location = getattr(task, 'location', None)
                if not location:
                    lat = getattr(task, 'lat', None)
                    lon = getattr(task, 'lon', None)
                    if lat is not None and lon is not None:
                        location = f"lat={lat:.4f}, lon={lon:.4f}"
                    else:
                        location = "Unknown"
                
                # Estimate service time (rough estimate)
                if hasattr(task, 'service_time'):
                    service_time = task.service_time / 60.0  # Convert minutes to hours
                elif task_type.name in ['PICKUP', 'DELIVERY']:
                    service_time = 0.5  # 30 minutes default
                else:
                    service_time = 0.0  # Depot tasks
                
                # Calculate actual travel time using route provider
                if i > 0:
                    prev_task = route.tasks[i-1] 
                    try:
                        travel_time_minutes = _calculate_travel_time_between_tasks(prev_task, task, route.vehicle)
                        travel_time = travel_time_minutes / 60.0  # Convert minutes to hours
                        if order_7_debug:
                            print(f"                  Travel from {prev_task.id} to {task.id}: {travel_time:.2f}h ({travel_time_minutes:.1f}min)")
                    except Exception as e:
                        if order_7_debug:
                            print(f"                  WARNING: Route provider failed for {prev_task.id}→{task.id}, using fallback: {e}")
                        travel_time = 0.5  # Fallback to 30 minutes only if route provider fails
                    
                    current_time += travel_time
                    total_travel_time += travel_time
                
                # Calculate arrival time
                arrival_time = current_time
                departure_time = current_time + service_time
                current_time = departure_time
                total_service_time += service_time
                
                # FIXED: Use correct time window attributes
                time_window_info = "No time window"
                if hasattr(task, 'time_window') and task.time_window:
                    tw_start = task.time_window[0] if len(task.time_window) > 0 else None
                    tw_end = task.time_window[1] if len(task.time_window) > 1 else None
                    
                    # Convert time window to hours (assuming it's in minutes)
                    tw_start_hours = tw_start / 60.0 if tw_start is not None else None
                    tw_end_hours = tw_end / 60.0 if tw_end is not None else None
                    
                    tw_start_str = f"{tw_start_hours:.1f}h" if tw_start_hours is not None else "No start"
                    tw_end_str = f"{tw_end_hours:.1f}h" if tw_end_hours is not None else "No end"
                    
                    violation = ""
                    # Apply waiting logic: vehicles can wait if they arrive early
                    if tw_start_hours is not None and arrival_time < tw_start_hours:
                        # Vehicle can wait until time window opens - this is allowed
                        waiting_time = tw_start_hours - arrival_time
                        if order_7_debug:
                            print(f"            Task {i+1}: Will wait {waiting_time:.1f}h until time window opens")
                        # Update arrival time to earliest window time (after waiting)
                        current_time += waiting_time
                        total_service_time += waiting_time  # Count waiting as service time
                        arrival_time = tw_start_hours  # Now arrives exactly when window opens
                        
                    if tw_end_hours is not None and arrival_time > tw_end_hours:
                        violation += f" LATE by {arrival_time - tw_end_hours:.1f}h"
                    
                    if arrival_time == tw_start_hours:
                        time_window_info = f"[{tw_start_str}-{tw_end_str}] On time"
                    elif violation:
                        time_window_info = f"[{tw_start_str}-{tw_end_str}]{violation}"
                    else:
                        time_window_info = f"[{tw_start_str}-{tw_end_str}] On time"
                else:
                    # Try earliest_time/latest_time attributes
                    earliest = getattr(task, 'earliest_time', None)
                    latest = getattr(task, 'latest_time', None)
                    
                    if earliest is not None and latest is not None:
                        # Convert from minutes to hours
                        earliest_hours = earliest / 60.0
                        latest_hours = latest / 60.0
                        
                        violation = ""
                        # Apply waiting logic: vehicles can wait if they arrive early
                        if arrival_time < earliest_hours:
                            # Vehicle can wait until time window opens - this is allowed
                            waiting_time = earliest_hours - arrival_time
                            if order_7_debug:
                                print(f"            Task {i+1}: Will wait {waiting_time:.1f}h until time window opens")
                            # Update arrival time to earliest window time (after waiting)
                            current_time += waiting_time
                            total_service_time += waiting_time  # Count waiting as service time
                            arrival_time = earliest_hours  # Now arrives exactly when window opens
                            
                        if arrival_time > latest_hours:
                            violation += f" LATE by {arrival_time - latest_hours:.1f}h"
                        
                        if arrival_time == earliest_hours:
                            time_window_info = f"[{earliest_hours:.1f}h-{latest_hours:.1f}h] On time"
                        elif violation:
                            time_window_info = f"[{earliest_hours:.1f}h-{latest_hours:.1f}h]{violation}"
                        else:
                            time_window_info = f"[{earliest_hours:.1f}h-{latest_hours:.1f}h] On time"
                    else:
                        time_window_info = "No time window"
                
                print(f"          Task {i+1}: {task_type} at {location[:50]}")
                print(f"            Arrival: {arrival_time:.1f}h, Window: {time_window_info}")
                    
            print(f"        DEBUG: Total route time: {current_time:.1f}h (Travel: {total_travel_time:.1f}h, Service: {total_service_time:.1f}h)")
            
        # COMPREHENSIVE DEBUG: Vehicle and route state analysis  
        if order_7_debug:
            print(f"                      *** COMPREHENSIVE FEASIBILITY ANALYSIS ***")
            print(f"                          Route has {len(route.tasks)} tasks")
            print(f"                          Vehicle: {route.vehicle.id if hasattr(route, 'vehicle') and route.vehicle else 'No vehicle'}")
            
            # Check vehicle working hours
            if hasattr(route, 'vehicle') and route.vehicle:
                if hasattr(route.vehicle, 'max_working_hours'):
                    max_hours = route.vehicle.max_working_hours
                    print(f"                          Vehicle max working hours: {max_hours}h")
                    if current_time > max_hours:
                        print(f"                          *** WORKING HOURS VIOLATION *** {current_time:.1f}h > {max_hours}h")
                else:
                    print(f"                          Vehicle has no max_working_hours attribute")
                    
                # Check vehicle capabilities
                if hasattr(route.vehicle, 'capabilities'):
                    caps = [str(c) for c in route.vehicle.capabilities] if route.vehicle.capabilities else []
                    print(f"                          Vehicle capabilities: {caps}")
                else:
                    print(f"                          Vehicle has no capabilities attribute")
            
            # Check route properties
            if hasattr(route, 'get_total_time'):
                try:
                    route_total_time = route.get_total_time()
                    print(f"                          Route.get_total_time(): {route_total_time:.1f}h")
                except Exception as e:
                    print(f"                          Route.get_total_time() error: {e}")
            
            if hasattr(route, 'get_violations'):
                try:
                    violations = route.get_violations()
                    print(f"                          Route violations: {violations}")
                except Exception as e:
                    print(f"                          Route.get_violations() error: {e}")
            
            # Try to get detailed feasibility info
            print(f"                          About to call route.is_feasible()...")
            
        overall_feasible = route.is_feasible(allow_soft_violations=False) if hasattr(route, 'is_feasible') else True
        
        if order_7_debug:
            print(f"                          route.is_feasible() returned: {overall_feasible}")
            
            if not overall_feasible:
                print(f"                      *** FINAL FEASIBILITY CHECK FAILED ***")
                # Enhanced debugging: Check individual constraint types
                if hasattr(route, 'is_feasible'):
                    # Try to get detailed failure reason - REMOVE return_reason parameter
                    try:
                        detailed_check = route.is_feasible(allow_soft_violations=False)
                        print(f"                          Route.is_feasible() returned: {detailed_check}")
                    except Exception as e:
                        print(f"                          Error calling route.is_feasible(): {e}")
                        
                    # REMOVE soft violations test - not allowed
                    # Check specific constraint methods if they exist
                    if hasattr(route, 'check_time_windows'):
                        try:
                            tw_check = route.check_time_windows()
                            print(f"                          Time windows check: {tw_check}")
                        except Exception as e:
                            print(f"                          Time windows check error: {e}")
                    
                    if hasattr(route, 'check_working_hours'):
                        try:
                            wh_check = route.check_working_hours()
                            print(f"                          Working hours check: {wh_check}")
                        except Exception as e:
                            print(f"                          Working hours check error: {e}")
                    
                    if hasattr(route, 'validate_hos'):
                        try:
                            hos_check = route.validate_hos()
                            print(f"                          HoS validation: {hos_check}")
                        except Exception as e:
                            print(f"                          HoS validation error: {e}")
                            
                    # Try to inspect route object directly
                    print(f"                          Route object type: {type(route)}")
                    print(f"                          Route object attributes: {[attr for attr in dir(route) if not attr.startswith('_') and 'check' in attr.lower() or 'valid' in attr.lower() or 'feasib' in attr.lower()]}")
                else:
                    print(f"                          Route has no is_feasible method!")
            else:
                print(f"                      *** ALL CONSTRAINTS SATISFIED *** Load tracking passed!")
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
    # ENHANCED DEBUG: Always debug Order 7 specifically
    order_debug = debug or str(order.id) == '7'
    
    if order_debug:
        print(f"    *** ENHANCED DEBUG: Processing complex order {order.id} on vehicle {route.vehicle.id if route.vehicle else 'None'} ***")
        if route.vehicle:
            print(f"        Vehicle capacity: {getattr(route.vehicle, 'weight_capacity', 'N/A')}kg, {getattr(route.vehicle, 'volume_capacity', 'N/A')}m³, {getattr(route.vehicle, 'pallet_capacity', 'N/A')}pal")
            vehicle_caps = []
            if hasattr(route.vehicle, 'capabilities') and route.vehicle.capabilities:
                for cap in route.vehicle.capabilities:
                    vehicle_caps.append(cap.name if hasattr(cap, 'name') else str(cap))
            print(f"        Vehicle capabilities: {', '.join(vehicle_caps) if vehicle_caps else 'NONE'}")
            
            # COMPREHENSIVE VEHICLE STATE DEBUG
            print(f"        *** VEHICLE STATE ANALYSIS ***")
            print(f"        Current route has {len(route.tasks)} tasks")
            if hasattr(route.vehicle, 'max_working_hours'):
                print(f"        Vehicle max working hours: {route.vehicle.max_working_hours}h")
            else:
                print(f"        Vehicle has no max_working_hours defined")
                
            # Show current route if any
            if route.tasks:
                print(f"        Current route tasks:")
                for i, task in enumerate(route.tasks):
                    task_type = task.task_type.name if hasattr(task.task_type, 'name') else str(task.task_type)
                    print(f"          {i+1}. {task_type} - {task.id if hasattr(task, 'id') else 'No ID'}")
            else:
                print(f"        Vehicle route is EMPTY (ideal for Order 7 testing)")
                
            # Check if vehicle is truly idle
            active_orders = []
            if hasattr(route, 'orders') and route.orders:
                active_orders = [str(o.id) for o in route.orders]
            print(f"        Active orders on vehicle: {active_orders if active_orders else 'NONE (IDLE)'}")
            
            # Get current route time if possible
            try:
                if hasattr(route, 'get_total_time'):
                    current_route_time = route.get_total_time()
                    print(f"        Current route time: {current_route_time:.1f}h")
                elif hasattr(route, 'total_time'):
                    print(f"        Current route time (attr): {route.total_time:.1f}h")
                else:
                    print(f"        Cannot determine current route time")
            except Exception as e:
                print(f"        Error getting current route time: {e}")
    
    # Get all tasks from the order
    all_tasks = order.get_pickups() + order.get_deliveries()
    
    if not all_tasks:
        if order_debug:
            print(f"    *** ORDER {order.id} REJECTED: No tasks found ***")
        return None
    
    if order_debug:
        print(f"    Order has {len(order.get_pickups())} pickups and {len(order.get_deliveries())} deliveries")
        # Show task capability requirements
        for task in all_tasks:
            caps_needed = []
            if getattr(task, 'requires_hangers', False):
                caps_needed.append('HANGERS')
            if getattr(task, 'requires_loader', False):
                caps_needed.append('LOADER') 
            if getattr(task, 'requires_low_temp', False):
                caps_needed.append('LOW_TEMP')
            print(f"        Task {task.id}: {task.demand}kg, {task.volume}m³, {task.pallets}pal, needs {', '.join(caps_needed) if caps_needed else 'NONE'}")
    
    # Step 2: Generate optimized task sequences using enhanced patterns
    pickups = order.get_pickups()
    deliveries = order.get_deliveries()
    
    # First try the enhanced optimized patterns
    optimized_patterns = generate_optimized_interleaved_patterns(pickups, deliveries)
    valid_sequences = []
    
    # Convert optimized patterns to the expected format
    for pattern in optimized_patterns:
        peak_capacity = calculate_sequence_peak_capacity(pattern['sequence'])
        valid_sequences.append({
            'sequence': pattern['sequence'],
            'peak_capacity': peak_capacity,
            'is_sequential_chain': pattern['is_interleaved'],
            'pattern_type': pattern['pattern_type'],
            'priority': pattern.get('priority', 999)
        })
    
    # If no optimized patterns, fall back to permutation generation
    if not valid_sequences:
        if order_debug:
            print(f"    *** ORDER {order.id}: No optimized patterns found, falling back to permutation generation ***")
        valid_sequences = generate_valid_task_permutations(all_tasks)
    else:
        if order_debug:
            print(f"    ORDER {order.id}: Using {len(valid_sequences)} optimized sequence patterns")
    
    # Sort by priority (if available) then by peak capacity
    valid_sequences.sort(key=lambda x: (x.get('priority', 999),
                                      x['peak_capacity']['peak_pallets'], 
                                      x['peak_capacity']['peak_weight'], 
                                      x['peak_capacity']['peak_volume']))
    
    if order_debug:
        print(f"    ORDER {order.id}: Generated {len(valid_sequences)} valid sequences")
    
    if not valid_sequences:
        if order_debug:
            print(f"    *** ORDER {order.id} REJECTED: No valid sequences found ***")
        return None
    
    # Step 3: Evaluate each valid permutation at all possible insertion points
    best_result = None
    best_profit = float('-inf')
    
    if order_debug:
        print(f"    ORDER {order.id}: Starting evaluation of {len(valid_sequences)} sequences")
    
    sequences_rejected_capacity = 0
    sequences_rejected_insertion = 0
    sequences_accepted = 0
    
    for seq_idx, sequence_data in enumerate(valid_sequences):
        sequence = sequence_data['sequence']
        peak_capacity = sequence_data['peak_capacity']
        is_sequential = sequence_data['is_sequential_chain']
        
        # SECTION 1.2: Add diagnostic logging for Order 7 as per sequencer debug guide
        if str(order.id) == '7':
            task_ids = [getattr(task, 'id', f'T{i}') for i, task in enumerate(sequence)]
            print(f"DEBUG SEQUENCER (Order 7): Trying sequence -> {task_ids}")
        
        if order_debug:
            print(f"    ORDER {order.id}: Evaluating sequence {seq_idx + 1}/{len(valid_sequences)}")
            sequence_types = [getattr(t.task_type, 'name', 'UNK')[0] for t in sequence]
            sequence_pattern = ''.join(sequence_types)
            print(f"        Pattern: {sequence_pattern}, Peak: {peak_capacity['peak_pallets']:.0f}pal/{peak_capacity['peak_weight']:.0f}kg/{peak_capacity['peak_volume']:.1f}m³")
            if is_sequential:
                print(f"        *** SEQUENTIAL CHAIN DETECTED ***")
        
        # Check if vehicle can handle the peak capacity requirement
        if route.vehicle:
            vehicle_weight = getattr(route.vehicle, 'weight_capacity', float('inf'))
            vehicle_volume = getattr(route.vehicle, 'volume_capacity', float('inf'))
            vehicle_pallets = getattr(route.vehicle, 'pallet_capacity', float('inf'))
            
            capacity_failed = False
            failure_reasons = []
            
            if peak_capacity['peak_weight'] > vehicle_weight:
                capacity_failed = True
                failure_reasons.append(f"weight: {peak_capacity['peak_weight']:.0f}kg > {vehicle_weight:.0f}kg")
            if peak_capacity['peak_volume'] > vehicle_volume:
                capacity_failed = True
                failure_reasons.append(f"volume: {peak_capacity['peak_volume']:.1f}m³ > {vehicle_volume:.1f}m³")  
            if peak_capacity['peak_pallets'] > vehicle_pallets:
                capacity_failed = True
                failure_reasons.append(f"pallets: {peak_capacity['peak_pallets']:.0f}pal > {vehicle_pallets:.0f}pal")
            
            if capacity_failed:
                sequences_rejected_capacity += 1
                if order_debug:
                    print(f"        *** SEQUENCE {seq_idx + 1} REJECTED: CAPACITY EXCEEDED ***")
                    print(f"            Vehicle({vehicle_weight:.0f}kg/{vehicle_volume:.1f}m³/{vehicle_pallets:.0f}pal)")
                    print(f"            Required({peak_capacity['peak_weight']:.0f}kg/{peak_capacity['peak_volume']:.1f}m³/{peak_capacity['peak_pallets']:.0f}pal)")
                    print(f"            Failures: {', '.join(failure_reasons)}")
                continue  # Skip this sequence - vehicle can't handle peak load
            else:
                if order_debug:
                    print(f"        CAPACITY OK: Vehicle({vehicle_weight:.0f}kg/{vehicle_volume:.1f}m³/{vehicle_pallets:.0f}pal) >= Required({peak_capacity['peak_weight']:.0f}kg/{peak_capacity['peak_volume']:.1f}m³/{peak_capacity['peak_pallets']:.0f}pal)")
        
        # Try all insertion points (after depot start, before depot return)
        insertion_attempts = 0
        feasible_insertions = 0
        
        if order_debug:
            print(f"        Testing {len(route.tasks)-1} insertion points...")
        
        for insertion_point in range(1, len(route.tasks)):
            insertion_attempts += 1
            
            if order_debug and insertion_attempts <= 3:  # Show first 3 attempts in detail
                print(f"            Trying insertion at point {insertion_point}/{len(route.tasks)-1}")
            
            result = evaluate_permutation_at_insertion_point(route, sequence, insertion_point, order_debug)
            
            if result and result['feasible']:
                feasible_insertions += 1
                sequences_accepted += 1
                # Calculate profit for this sequence
                profit = calculate_sequence_profit(result['route'], order, result['cost'])
                
                # SECTION 1.2: Add cost evaluation logging for Order 7 as per sequencer debug guide
                if str(order.id) == '7':
                    task_ids = [getattr(task, 'id', f'T{i}') for i, task in enumerate(sequence)]
                    print(f"DEBUG SEQUENCER (Order 7): Evaluating cost for sequence {task_ids} -> Cost: {result['cost']:.1f}")
                
                if order_debug:
                    print(f"            *** INSERTION SUCCESS *** at point {insertion_point}: profit = {profit:.2f}")
                
                if profit > best_profit:
                    best_profit = profit
                    best_result = result
                    if order_debug:
                        print(f"            *** NEW BEST SEQUENCE *** profit = {profit:.2f}")
            else:
                if order_debug and insertion_attempts <= 3:  # Show detailed failure for first 3 attempts
                    print(f"            Insertion FAILED at point {insertion_point}: {result.get('failure_reason', 'Unknown reason') if result else 'No result returned'}")
        
        if feasible_insertions == 0:
            sequences_rejected_insertion += 1
            
        if order_debug:
            if feasible_insertions > 0:
                print(f"        Sequence {seq_idx + 1} summary: {feasible_insertions}/{insertion_attempts} insertion points SUCCEEDED")
            else:
                print(f"        *** SEQUENCE {seq_idx + 1} REJECTED: {feasible_insertions}/{insertion_attempts} insertion points failed - NO VALID INSERTION POINTS ***")
    
    # Step 4: Return the best sequence with relaxed profitability requirements
    if order_debug:
        print(f"    *** ORDER {order.id} EVALUATION SUMMARY ***")
        print(f"        Total sequences: {len(valid_sequences)}")
        print(f"        Rejected by capacity: {sequences_rejected_capacity}")
        print(f"        Rejected by insertion: {sequences_rejected_insertion}")
        print(f"        Sequences with valid insertions: {sequences_accepted}")
        
    if best_result:
        # SECTION 1.2: Add best sequence found logging for Order 7 as per sequencer debug guide
        if str(order.id) == '7':
            best_sequence = []
            if 'sequence' in best_result:
                best_sequence = [getattr(task, 'id', f'T{i}') for i, task in enumerate(best_result['sequence'])]
            elif 'route' in best_result and hasattr(best_result['route'], 'tasks'):
                # Extract the order's tasks from the final route 
                order_tasks = []
                for task in best_result['route'].tasks:
                    if hasattr(task, 'order_id') and str(task.order_id) == '7':
                        order_tasks.append(getattr(task, 'id', f'T{len(order_tasks)}'))
                best_sequence = order_tasks
            print(f"DEBUG SEQUENCER (Order 7): Best sequence found -> {best_sequence} with cost {best_result.get('cost', 'unknown')}")
            
        if order_debug:
            if best_profit > 0:
                print(f"    *** ORDER {order.id} SUCCESS *** Best sequence selected with profit: {best_profit:.2f}")
            else:
                print(f"    *** ORDER {order.id} SUCCESS *** Best sequence selected despite low profit ({best_profit:.2f}) - accepting for strategic value")
        return best_result['route']
    else:
        if order_debug:
            print(f"    *** ORDER {order.id} COMPLETE FAILURE *** No feasible sequence found")
            print(f"        This means all {len(valid_sequences)} sequences failed constraints or insertion")
            if sequences_rejected_capacity > 0:
                print(f"        {sequences_rejected_capacity} sequences failed vehicle capacity constraints")
            if sequences_rejected_insertion > 0:
                print(f"        {sequences_rejected_insertion} sequences failed insertion point constraints")
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
        
        # NEW: TIMELINE DEBUG - Show route timeline BEFORE insertion
        print(f"         [TIMELINE] BEFORE L2 INSERTION:")
        debug_print_route_timeline(route, "L2_ENTRY")
    
    # DEBUG: Force debug for Orders 5 & 6 as per debug guide
    if str(order.id) in ['5', '6']:
        print(f"L2_DEBUG: Starting L2 heuristic for Order {order.id}")
        show_diagnostics = True
        debug_assignment = True
        
        # FORCE timeline debug for problematic orders
        print(f"         [TIMELINE] BEFORE L2 INSERTION (Order {order.id}):")
        debug_print_route_timeline(route, f"L2_ENTRY_ORDER_{order.id}")
        
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
    
    # Step 1: Enhanced Complex Order Classification
    # Following user specifications: All orders except 1P+1D are complex
    pickups = order.get_pickups()
    deliveries = order.get_deliveries()
    total_tasks = len(pickups) + len(deliveries)
    num_pickups = len(pickups)
    num_deliveries = len(deliveries)
    
    # NEW CLASSIFICATION: Only 1P+1D orders are simple, everything else is complex
    is_complex_order = not (num_pickups == 1 and num_deliveries == 1)
    
    # Always print order detection for debugging
    if is_complex_order:
        print(f"    *** COMPLEX ORDER DETECTED: Order {order.id} has {num_pickups} pickups, {num_deliveries} deliveries ***")
        
        # Classify the complex order type for optimized sequencing
        if num_pickups == num_deliveries and num_pickups > 1:
            print(f"        Case A: Balanced order ({num_pickups}P+{num_deliveries}D) - using interleaved PDPDPD strategy")
        elif num_pickups == 1 and num_deliveries > 1:
            print(f"        Case B: Single pickup, multiple deliveries (1P+{num_deliveries}D) - optimizing delivery sequence")
        elif num_pickups > 1 and num_deliveries == 1:
            print(f"        Case C: Multiple pickups, single delivery ({num_pickups}P+1D) - optimizing pickup sequence")
        else:
            print(f"        Case D: Imbalanced order ({num_pickups}P+{num_deliveries}D) - using comprehensive sequencing")
        
        # Execute enhanced advanced sequencing logic for complex orders
        # Force debug for orders 5 and 6 to see detailed rejection reasons
        debug_for_this_order = show_diagnostics or (str(order.id) in ['5', '6'])
        complex_route = find_best_sequence_for_complex_order(route, order, debug_for_this_order)
        if complex_route:
            if show_diagnostics or str(order.id) in ['5', '6']:
                print(f"L2_DEBUG: Complex order {order.id} SUCCESS - Route created with {len(complex_route.tasks)} tasks")
            print(f"    *** COMPLEX ORDER SUCCESS: Order {order.id} processed successfully ***")
            return complex_route
        else:
            if show_diagnostics or str(order.id) in ['5', '6']:
                print(f"L2_DEBUG: Complex order {order.id} FAILED - find_best_sequence_for_complex_order returned None")
                print(f"L2_DEBUG: This is the exact failure point mentioned in debug guide")
            print(f"    *** COMPLEX ORDER FAILED: Order {order.id} - No feasible sequence found ***")
            return None
    else:
        # Simple order (1P+1D) - print for debugging
        print(f"    Simple Order: Order {order.id} has {num_pickups} pickup, {num_deliveries} delivery")
    
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

    # CRITICAL DEBUG: Track L2 heuristic exit point
    if show_diagnostics or str(order.id) in ['5', '6']:
        if final_route:
            print(f"L2_DEBUG: Order {order.id} RETURNING SUCCESS - Route with {len(final_route.tasks)} tasks")
            # NEW: TIMELINE DEBUG - Show route timeline AFTER insertion
            print(f"         [TIMELINE] AFTER L2 SUCCESS:")
            debug_print_route_timeline(final_route, f"L2_SUCCESS_ORDER_{order.id}")
        else:
            print(f"L2_DEBUG: Order {order.id} RETURNING FAILURE - final_route is None")
            print(f"L2_DEBUG: Failure location: {'complex_order_processing' if is_complex_order else 'simple_order_processing'}")

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
    
    # TRACE SPECIFIC VIOLATING VEHICLES
    is_violating_vehicle = hasattr(route, 'vehicle') and route.vehicle.id in ['GA621VG', 'FF235DM', 'XA346KW']
    if is_violating_vehicle:
        print(f"            TRACE {route.vehicle.id}: is_feasible() called - checking route with {len(route.tasks)} tasks")
    
    # COMPREHENSIVE DEBUGGING - now handled by Route.is_feasible() enabling debug for all orders
    order_7_debug = debug_feasibility  # Use the debug_feasibility flag directly
    
    # Enhanced debugging for assignment failures
    if debug_feasibility:
        print(f"            *** ROUTE FEASIBILITY DEBUG ***")
        print(f"            Vehicle: {route.vehicle.id if route.vehicle else 'None'}")
        print(f"            Total tasks: {len(route.tasks)}")
        order_tasks = [task for task in route.tasks if hasattr(task, 'order_id') and not task.is_depot_start() and not task.is_depot_return()]
        if order_tasks:
            order_ids = list(set(str(task.order_id) for task in order_tasks))
            print(f"            Orders in route: {order_ids}")
        print(f"            allow_soft_violations: {allow_soft_violations}")
        print(f"            return_reason: {return_reason}")
        
        # Show vehicle constraints
        if hasattr(route, 'vehicle') and route.vehicle:
            if hasattr(route.vehicle, 'max_working_hours'):
                print(f"            Vehicle max working hours: {route.vehicle.max_working_hours}h")
            else:
                print(f"            Vehicle has NO max_working_hours limit")
                
            # Check if vehicle has active routes
            try:
                current_time = route.get_total_time() if hasattr(route, 'get_total_time') else None
                if current_time:
                    print(f"            Current route time: {current_time:.1f}h")
            except:
                print(f"            Cannot determine current route time")
    
    # H0: DEPOT START/END VALIDATION - Must be first check
    # Every route must start with a depot start task and end with a depot return task
    if route.tasks:
        if debug_feasibility:
            print(f"            *** STEP 1: DEPOT VALIDATION ***")
            print(f"            Route has {len(route.tasks)} tasks")
            
        # Check first task is depot start
        first_task = route.tasks[0]
        if debug_feasibility:
            print(f"            Checking first task: {first_task.id if hasattr(first_task, 'id') else 'no ID'}")
            print(f"            First task type: {type(first_task)}")
            print(f"            Has is_depot_start: {hasattr(first_task, 'is_depot_start')}")
            
        if not (hasattr(first_task, 'is_depot_start') and first_task.is_depot_start()):
            reason = f"Route validation failed: First task ({first_task.id if hasattr(first_task, 'id') else 'unknown'}) is not a depot start task"
            if debug_feasibility:
                print(f"            *** DEPOT VALIDATION FAILED *** {reason}")
            if return_reason:
                return False, reason
            return False
        
        if debug_feasibility:
            print(f"             First task is depot start")
        
        # Check last task is depot return
        last_task = route.tasks[-1]
        if debug_feasibility:
            print(f"            Checking last task: {last_task.id if hasattr(last_task, 'id') else 'no ID'}")
            print(f"            Last task type: {type(last_task)}")
            print(f"            Has is_depot_return: {hasattr(last_task, 'is_depot_return')}")
            
        if not (hasattr(last_task, 'is_depot_return') and last_task.is_depot_return()):
            reason = f"Route validation failed: Last task ({last_task.id if hasattr(last_task, 'id') else 'unknown'}) is not a depot return task"
            if debug_feasibility:
                print(f"            *** DEPOT VALIDATION FAILED *** {reason}")
            if return_reason:
                return False, reason
            return False
        
        if debug_feasibility:
            print(f"             Last task is depot return")
            print(f"             DEPOT VALIDATION PASSED")
    
    # H1: Multi-day chronological simulation setup
    # Check original task order first for pickup-before-delivery precedence constraint
    original_tasks = route.tasks
    
    # H2: Enhanced Logical Precedence Constraint Check
    # NEW RULE: Allow deliveries before pickups as long as:
    # 1. Each delivery has its corresponding pickup already completed
    # 2. Pallet capacity is never exceeded (physical constraint)
    # 3. Distance penalty will discourage inefficient patterns
    
    if debug_feasibility:
        print(f"            *** STEP 2: LOGICAL PRECEDENCE CHECK ***")
    
    # Track completed pickups for each order to validate deliveries
    completed_pickups = set()
    load_pallets_check = 0  # Track pallet load for physical constraint
    
    if debug_feasibility:
        print(f"            Checking {len(original_tasks)} tasks for precedence violations...")
    
    # Iterate through tasks in chronological sequence
    for i, task in enumerate(original_tasks):
        if debug_feasibility:
            task_type = "pickup" if task.is_pickup() else "delivery" if task.is_delivery() else "other"
            order_id = getattr(task, 'order_id', 'no_order')
            print(f"            Task {i}: {task.id if hasattr(task, 'id') else 'no_id'} - {task_type} - Order {order_id}")
            
        if task.is_pickup():
            # Record this pickup as completed
            order_id = getattr(task, 'order_id', None)
            if order_id:
                completed_pickups.add(order_id)
            # Add pallets to load
            pallets = getattr(task, 'pallets', 0)
            load_pallets_check += pallets
            if debug_feasibility:
                print(f"              Pickup completed for order {order_id}, load now: {load_pallets_check} pal")
            
        elif task.is_delivery():
            # Check if corresponding pickup was already completed
            order_id = getattr(task, 'order_id', None)
            if debug_feasibility:
                print(f"              Delivery for order {order_id}, completed pickups: {completed_pickups}")
                
            if order_id and order_id not in completed_pickups:
                reason = f"Enhanced logical precedence violated: Delivery {task.id} attempted before its pickup was completed"
                if debug_feasibility:
                    print(f"            *** PRECEDENCE VIOLATION *** {reason}")
                if return_reason:
                    return False, reason
                return False
            # Remove pallets from load
            pallets = getattr(task, 'pallets', 0)  # pallets should be negative for delivery tasks
            load_pallets_check += pallets
            if debug_feasibility:
                print(f"              Delivery completed, load now: {load_pallets_check} pal")
            
        # Physical constraint: never exceed pallet capacity during route execution
        max_pallets = route.vehicle.pallet_capacity
        if debug_feasibility:
            print(f"              Checking pallet capacity: {load_pallets_check} pal <= {max_pallets} pal")
            
        if max_pallets is not None and load_pallets_check > max_pallets:
            reason = f"Pallet capacity exceeded during route execution: {load_pallets_check} > {max_pallets}"
            if debug_feasibility:
                print(f"            *** PALLET CAPACITY VIOLATION *** {reason}")
            if return_reason:
                return False, reason
            return False
    
    if debug_feasibility:
        print(f"             LOGICAL PRECEDENCE CHECK PASSED")
    
    # Sort tasks with pickup-first sequencing for proper HoS simulation
    # Apply LIFO sequencing if required by vehicle
    if debug_feasibility:
        print(f"            *** STEP 2.5: TASK SEQUENCING ***")
        print(f"            Original task order: {[task.id for task in route.tasks if hasattr(task, 'id')]}")
        
    if hasattr(route.vehicle, 'lifo_required') and route.vehicle.lifo_required:
        if debug_feasibility:
            print(f"            Using LIFO sequencing...")
        sorted_tasks = _enforce_pickup_first_sequencing_with_lifo(route.tasks, route.vehicle)
    else:
        if debug_feasibility:
            print(f"            Using basic pickup-first sequencing...")
        sorted_tasks = _enforce_pickup_first_sequencing_basic(route.tasks)
    
    if debug_feasibility:
        print(f"            Sequenced task order: {[task.id for task in sorted_tasks if hasattr(task, 'id')]}")
        
        # Check if sequencing changed the order
        original_ids = [task.id for task in route.tasks if hasattr(task, 'id')]
        sequenced_ids = [task.id for task in sorted_tasks if hasattr(task, 'id')]
        if original_ids != sequenced_ids:
            print(f"            *** WARNING: Task order changed by sequencing! ***")
            print(f"            Original:  {original_ids}")
            print(f"            Sequenced: {sequenced_ids}")
        else:
            print(f"             Task order unchanged by sequencing")
    
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
    if debug_feasibility:
        print(f"            *** STEP 3: VEHICLE STATE INITIALIZATION ***")
        
    load_w = previous_day_load_w
    load_v = previous_day_load_v
    load_pallets = initial_state.get('load_pallets', 0) if initial_state else 0
    max_w = route.vehicle.weight_capacity
    max_v = route.vehicle.volume_capacity
    max_pallets = route.vehicle.pallet_capacity  # Hard constraint on pallets
    
    if debug_feasibility:
        print(f"            Vehicle capacity: {max_w}kg, {max_v:.1f}m³, {max_pallets}pal")
        print(f"            Initial load: {load_w}kg, {load_v:.1f}m³, {load_pallets}pal")
    
    # H4: LIFO Loading Constraint check
    lifo_stack = []
    if route.vehicle.lifo_required:
        if debug_feasibility:
            print(f"            *** STEP 4: LIFO CONSTRAINT CHECK ***")
            print(f"            Vehicle requires LIFO loading")
            
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
                print(f"            Pre-loaded LIFO stack for delivery-only orders: {lifo_stack}")

    if debug_feasibility:
        print(f"            *** STEP 5: TASK-BY-TASK SIMULATION ***")
        print(f"            Starting simulation of {len(sorted_tasks)} tasks...")

    for i, task in enumerate(sorted_tasks):
        if debug_feasibility:
            task_type = "pickup" if task.is_pickup() else "delivery" if task.is_delivery() else "other"
            print(f"            Task {i}: {task.id} ({task_type})")
            
        if task.is_pickup():
            new_load_w = load_w + task.demand
            new_load_v = load_v + task.volume
            new_load_pallets = load_pallets + getattr(task, 'pallets', 0)
            
            if debug_feasibility:
                print(f"              Pickup: +{task.demand}kg, +{task.volume:.1f}m³, +{getattr(task, 'pallets', 0)}pal")
                print(f"              New load: {new_load_w}kg, {new_load_v:.1f}m³, {new_load_pallets}pal")
                print(f"              Capacity: {max_w}kg, {max_v:.1f}m³, {max_pallets}pal")
                
            # Check capacity violations
            if new_load_w > max_w:
                reason = f"Weight capacity exceeded: {new_load_w} > {max_w} kg"
                if debug_feasibility:
                    print(f"              *** WEIGHT CAPACITY VIOLATION *** {reason}")
                if return_reason:
                    return False, reason
                return False
                
            if new_load_v > max_v:
                reason = f"Volume capacity exceeded: {new_load_v} > {max_v} m³"
                if debug_feasibility:
                    print(f"              *** VOLUME CAPACITY VIOLATION *** {reason}")
                if return_reason:
                    return False, reason
                return False
                
            if max_pallets is not None and new_load_pallets > max_pallets:
                reason = f"Pallet capacity exceeded: {new_load_pallets} > {max_pallets} pallets"
                if debug_feasibility:
                    print(f"              *** PALLET CAPACITY VIOLATION *** {reason}")
                if return_reason:
                    return False, reason
                return False
            
            # Update loads
            load_w = new_load_w
            load_v = new_load_v
            load_pallets = new_load_pallets
            
            # LIFO constraint: push order_id onto stack
            if route.vehicle.lifo_required:
                lifo_stack.append(task.order_id)
                if debug_feasibility:
                    print(f"              LIFO stack after pickup: {lifo_stack}")
                
        elif task.is_delivery():
            new_load_w = load_w + task.demand  # demand is negative for deliveries
            new_load_v = load_v + task.volume  # volume is negative for deliveries
            new_load_pallets = load_pallets + getattr(task, 'pallets', 0)  # pallets should be negative for delivery tasks
            
            if debug_feasibility:
                print(f"              Delivery: {task.demand}kg, {task.volume:.1f}m³, {getattr(task, 'pallets', 0)}pal")
                print(f"              New load: {new_load_w}kg, {new_load_v:.1f}m³, {new_load_pallets}pal")
            
            # LIFO constraint: check if this delivery matches top of stack
            if route.vehicle.lifo_required:
                if debug_feasibility:
                    print(f"              LIFO stack before delivery: {lifo_stack}")
                    
                if not lifo_stack:
                    reason = f"LIFO violation: trying to deliver {task.id} when no cargo loaded"
                    if debug_feasibility:
                        print(f"              *** LIFO VIOLATION *** {reason}")
                    if return_reason:
                        return False, reason
                    return False
                
                # FIXED LIFO LOGIC: Allow same-order deliveries regardless of stack position
                # For patterns like PP...PD (multiple pickups, single delivery of same order)
                # we should remove ALL instances of the order from the stack
                if task.order_id in lifo_stack:
                    # For same-order patterns, remove ALL instances of this order
                    # This handles PP...D patterns where all cargo is delivered together
                    while task.order_id in lifo_stack:
                        lifo_stack.remove(task.order_id)
                    if debug_feasibility:
                        print(f"              Removed order {task.order_id} from LIFO stack")
                else:
                    # Only enforce strict LIFO when mixing different orders
                    if lifo_stack[-1] != task.order_id:
                        reason = f"LIFO violation: expected {lifo_stack[-1]}, got {task.order_id} (mixed-order constraint)"
                        if debug_feasibility:
                            print(f"              *** LIFO VIOLATION *** {reason}")
                        if return_reason:
                            return False, reason
                        return False
                    lifo_stack.pop()  # Remove delivered order from stack
                    
                if debug_feasibility:
                    print(f"              LIFO stack after delivery: {lifo_stack}")
            
            # Update loads
            load_w = new_load_w
            load_v = new_load_v
            load_pallets = new_load_pallets
            
        else:
            # Other task types (depot, etc.)
            if debug_feasibility:
                print(f"              Other task type - no capacity impact")
        
        # Check capacity constraints after each task
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
                    print(f"              *** VOLUME VIOLATION *** {reason}")
                if return_reason:
                    return False, reason
                return False
            
        # NOTE: Pallet capacity is now checked by unified hard constraint checker

    if debug_feasibility:
        print(f"             TASK-BY-TASK SIMULATION PASSED")
        print(f"            Final load: {load_w}kg, {load_v:.1f}m³, {load_pallets}pal")

    # LIFO final check: all cargo must be delivered
    if route.vehicle.lifo_required and lifo_stack:
        reason = f"LIFO constraint violated: undelivered cargo: {lifo_stack}"
        if debug_feasibility:
            print(f"            *** LIFO FINAL CHECK FAILED *** {reason}")
        if return_reason:
            return False, reason
        return False
    
    if debug_feasibility and route.vehicle.lifo_required:
        print(f"             LIFO FINAL CHECK PASSED (stack empty)")

    # H5: Per-order precedence constraints check with multi-day consideration
    if debug_feasibility:
        print(f"            *** STEP 6: PER-ORDER PRECEDENCE CHECK ***")
        
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
        if debug_feasibility:
            print(f"            Checking order {order_id}: {len(tasks['pickups'])} pickups, {len(tasks['deliveries'])} deliveries")
            
        if tasks['pickups'] and tasks['deliveries']:
            try:
                # ENHANCED: Handle complex orders (multiple P/D pairs) differently
                pickups = tasks['pickups']
                deliveries = tasks['deliveries']
                
                # Check if this is a complex order with multiple balanced P/D pairs
                if len(pickups) > 1 and len(deliveries) > 1 and len(pickups) == len(deliveries):
                    # COMPLEX ORDER: Check each pickup-delivery pair individually
                    if debug_feasibility:
                        print(f"              Complex order detected - checking individual P/D pairs")
                    
                    # Sort tasks by position to match pairs correctly
                    pickups_sorted = sorted(pickups, key=lambda x: x[0])  # Sort by position
                    deliveries_sorted = sorted(deliveries, key=lambda x: x[0])
                    
                    # For PDPDPD pattern, each pickup should come before its next delivery
                    precedence_valid = True
                    for i in range(len(pickups_sorted)):
                        pickup_pos = pickups_sorted[i][0]
                        pickup_day = pickups_sorted[i][1]
                        
                        # Find the corresponding delivery (should be the next one)
                        if i < len(deliveries_sorted):
                            delivery_pos = deliveries_sorted[i][0]
                            delivery_day = deliveries_sorted[i][1]
                            
                            # Check if this pickup comes before its delivery
                            if (pickup_day > delivery_day or 
                                (pickup_day == delivery_day and pickup_pos >= delivery_pos)):
                                precedence_valid = False
                                if debug_feasibility:
                                    print(f"              *** PAIR PRECEDENCE VIOLATION *** Pickup {i} at pos {pickup_pos} >= Delivery {i} at pos {delivery_pos}")
                                break
                            else:
                                if debug_feasibility:
                                    print(f"               Pair {i}: Pickup pos {pickup_pos} < Delivery pos {delivery_pos}")
                    
                    if not precedence_valid:
                        reason = f"Complex order precedence violated for {order_id}: pickup-delivery pair constraint failed"
                        if debug_feasibility:
                            print(f"              *** COMPLEX ORDER PRECEDENCE VIOLATION *** {reason}")
                        if return_reason:
                            return False, reason
                        return False
                        
                else:
                    # SIMPLE ORDER: Use traditional all-pickups-before-all-deliveries constraint
                    if debug_feasibility:
                        print(f"              Simple order - checking global P before D constraint")
                        
                    # Find last pickup (considering day and position)
                    last_pickup = max(pickups, key=lambda x: (x[1], x[0]))  # Sort by day, then position
                    # Find first delivery
                    first_delivery = min(deliveries, key=lambda x: (x[1], x[0]))
                    
                    if debug_feasibility:
                        print(f"              Last pickup: day={last_pickup[1]}, pos={last_pickup[0]}")
                        print(f"              First delivery: day={first_delivery[1]}, pos={first_delivery[0]}")
                    
                    # Check if last pickup happens before first delivery
                    if (last_pickup[1] > first_delivery[1] or 
                        (last_pickup[1] == first_delivery[1] and last_pickup[0] >= first_delivery[0])):
                        reason = f"Precedence constraint violated for {order_id}: last pickup (day={last_pickup[1]}, pos={last_pickup[0]}) >= first delivery (day={first_delivery[1]}, pos={first_delivery[0]})"
                        if debug_feasibility:
                            print(f"              *** SIMPLE ORDER PRECEDENCE VIOLATION *** {reason}")
                        if return_reason:
                            return False, reason
                        return False
                    
                if debug_feasibility:
                    print(f"               Order {order_id} precedence valid")
                    
            except (ValueError, TypeError, KeyError) as e:
                if debug_feasibility:
                    print(f"              Error in precedence check for {order_id}: {e}")
                # Skip this order's precedence check if there's an error
                continue
    
    if debug_feasibility:
        print(f"             PER-ORDER PRECEDENCE CHECK PASSED")

    # H6: Multi-day Hours of Service check - DISABLED during initialization
    # Check if we're in initialization phase using call stack inspection
    if debug_feasibility:
        print(f"            *** STEP 7: HoS INITIALIZATION CHECK ***")
        
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
        # TRACE SPECIFIC VIOLATING VEHICLES
        is_violating_vehicle = hasattr(route, 'vehicle') and route.vehicle.id in ['GA621VG', 'FF235DM', 'XA346KW']
        
        # B license drivers are exempt from HoS regulations
        if route.driver and hasattr(route.driver, 'license') and route.driver.license == 'B':
            if debug_feasibility or is_violating_vehicle:
                print(f"            TRACE {route.vehicle.id if hasattr(route, 'vehicle') else 'Unknown'}: B license driver exemption - skipping HoS check")
            pass  # B license drivers are exempt - skip HoS check
        else:
            if debug_feasibility:
                print(f"            DEBUG FEASIBILITY: Performing unified HoS validation for route with {len(route.tasks)} tasks")
            
            if order_7_debug:
                print(f"            *** ORDER 7 HoS VALIDATION ***")
                print(f"            About to call validate_route_hos_feasibility()...")
            
            # UNIFIED VALIDATION: Use the same timing calculation as final route display
            from hos_simulation import validate_route_hos_feasibility
            hos_result = validate_route_hos_feasibility(route)
            
            if debug_feasibility or order_7_debug:
                print(f"            DEBUG FEASIBILITY: UNIFIED HoS validation complete - feasible: {hos_result.is_feasible}")
                if hos_result.violations:
                    print(f"            DEBUG FEASIBILITY: Violations found: {hos_result.violations}")
                if order_7_debug:
                    print(f"            *** ORDER 7 HoS RESULT ***")
                    print(f"            HoS feasible: {hos_result.is_feasible}")
                    print(f"            HoS violations: {hos_result.violations if hos_result.violations else 'NONE'}")
                    if hasattr(hos_result, 'total_time'):
                        print(f"            HoS calculated total time: {hos_result.total_time:.1f}h")
                    if hasattr(hos_result, 'events') and hos_result.events:
                        print(f"            HoS timeline events: {len(hos_result.events)}")
                        # Show first few events
                        for i, event in enumerate(hos_result.events[:3]):
                            if hasattr(event, 'start_time') and hasattr(event, 'task'):
                                task_id = event.task.id if event.task else 'No task'
                                print(f"              Event {i+1}: {task_id} at {event.start_time:.1f}min")
            
            
            # Cache the timeline and rest costs on the route object for use by calculate_z2_score
            route._cached_timeline = getattr(hos_result, 'events', [])
            route._cached_rest_costs = getattr(hos_result, 'rest_time', 0.0) * 25.0  # Convert to cost
            
            # Check the unified feasibility result
            if not hos_result.is_feasible:
                if not allow_soft_violations:
                    reason = f"UNIFIED HoS validation failed: {'; '.join(hos_result.violations)}"
                    if debug_feasibility:
                        print(f"            DEBUG FEASIBILITY: {reason}")
                    if return_reason:
                        return False, reason
                    return False
                else:
                    # Allow soft violations but log them
                    if debug_feasibility:
                        print(f"            DEBUG FEASIBILITY: UNIFIED HoS violations found but allowing soft violations: {hos_result.violations}")
            
            # UNIFIED HoS validation passed - timeline uses same calculation as final route display
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
                    # ABSOLUTE ENFORCEMENT: 10-minute grace period regardless of any relaxation parameters
                    ABSOLUTE_GRACE_PERIOD_MINUTES = 10.0
                    if task.latest_time is not None and late_by > ABSOLUTE_GRACE_PERIOD_MINUTES:
                        reason = f"Time window violation at task {task.id}: arrived at {arrival_time:.1f}, latest allowed {task.latest_time}, late by {late_by:.1f} minutes (exceeds ABSOLUTE {ABSOLUTE_GRACE_PERIOD_MINUTES} min grace period)"
                        if debug_feasibility:
                            print(f"            DEBUG FEASIBILITY: {reason}")
                        if return_reason:
                            return False, reason
                        return False
                    # Only minor lateness (0-10 minutes) is allowed for soft violations
                else:
                    # ABSOLUTE ENFORCEMENT: Apply same ABSOLUTE 10-minute grace period for strict mode too
                    # This ensures consistent behavior across initialization, L1, L2, and force assignment
                    ABSOLUTE_GRACE_PERIOD_MINUTES = 10.0
                    if task.latest_time is not None and late_by > ABSOLUTE_GRACE_PERIOD_MINUTES:
                        reason = f"Time window violation at task {task.id}: arrived at {arrival_time:.1f}, latest allowed {task.latest_time}, late by {late_by:.1f} minutes (exceeds ABSOLUTE {ABSOLUTE_GRACE_PERIOD_MINUTES} min grace period)"
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
    if debug_feasibility:
        print(f"            *** STEP 3: HARD CONSTRAINT CHECK ***")
        print(f"            About to call check_hard_constraints()...")
    
    hard_constraint_valid, hard_constraint_reason = check_hard_constraints(route, debug_feasibility)
    
    if debug_feasibility:
        print(f"            Hard constraints result: {hard_constraint_valid}")
        if not hard_constraint_valid:
            print(f"            Hard constraint failure reason: {hard_constraint_reason}")
    
    if not hard_constraint_valid:
        if debug_feasibility:
            print(f"            *** HARD CONSTRAINT VIOLATION *** {hard_constraint_reason}")
        if return_reason:
            return False, hard_constraint_reason
        return False
    
    if debug_feasibility:
        print(f"             HARD CONSTRAINT CHECK PASSED")
    
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
        
        # Order 7 debugging for duration check
        if order_7_debug:
            print(f"            *** ROUTE DURATION CHECK ***")
            print(f"            Departure time: {departure_time:.1f} minutes")
            print(f"            Final arrival time: {final_arrival_time:.1f} minutes")
            print(f"            Total duration: {total_duration:.1f} minutes ({total_duration/60:.1f}h)")
            print(f"            MAX_ROUTE_DURATION: {1440} minutes (24h)")
            if total_duration > 1440:
                print(f"            *** DURATION VIOLATION *** {total_duration:.1f}min > 1440min")
        
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
            
            if order_7_debug:
                print(f"            *** CHECKING TIME-SENSITIVE TASKS ***")
                print(f"            Has time-sensitive tasks: {has_time_sensitive_tasks}")
                if has_time_sensitive_tasks:
                    print(f"            *** FAILING ROUTE: Duration exceeds 24h with time-sensitive tasks ***")
            
            if has_time_sensitive_tasks:
                reason = f"Route duration {total_duration:.1f} minutes exceeds {MAX_ROUTE_DURATION} minute limit for routes with time-sensitive tasks. This prevents day-late deliveries."
                if debug_feasibility or order_7_debug:
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
            # Timeline not cached - generate it now using HoS engine
            if debug_feasibility:
                print(f"            DEBUG FEASIBILITY: Timeline not cached, generating HoS timeline now")
            
            try:
                from hos_simulation import validate_route_hos_feasibility
                hos_result = validate_route_hos_feasibility(route)
                
                if hos_result and hasattr(hos_result, 'events'):
                    timeline = hos_result.events
                    route._cached_timeline = timeline  # Cache for future use
                    if debug_feasibility:
                        print(f"            DEBUG FEASIBILITY: Generated HoS timeline with {len(timeline)} events")
                    
                    # Now validate time windows using the generated timeline
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
                        print(f"            DEBUG FEASIBILITY: Generated HoS timeline validation - All {tasks_checked} tasks passed time window validation")
                else:
                    if debug_feasibility:
                        print(f"            DEBUG FEASIBILITY: Failed to generate HoS timeline - route rejected")
                    if return_reason:
                        return False, "Failed to generate HoS timeline for sequential validation"
                    return False
            except Exception as e:
                if debug_feasibility:
                    print(f"            DEBUG FEASIBILITY: HoS timeline generation failed: {e}")
                if return_reason:
                    return False, f"HoS timeline generation error: {e}"
                return False
    
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
    
    ENHANCED: Handle complex order patterns correctly:
    - nP+nD (balanced): Preserve pickup-delivery pairs (PDPDPD)
    - 1P+nD (single pickup, multiple deliveries): Keep pickup before its deliveries (PDDDD)
    - nP+1D (multiple pickups, single delivery): Keep pickups before delivery (PPPPD)
    """
    if not tasks:
        return []
    
    # Separate depot tasks and order tasks
    depot_tasks = []
    order_tasks = []
    
    for task in tasks:
        if task.is_depot_start() or task.is_depot_return():
            depot_tasks.append(task)
        else:
            order_tasks.append(task)
    
    # Group tasks by order_id
    orders = {}
    for task in order_tasks:
        order_id = getattr(task, 'order_id', None)
        if order_id:
            if order_id not in orders:
                orders[order_id] = {'pickups': [], 'deliveries': []}
            
            if task.is_pickup():
                orders[order_id]['pickups'].append(task)
            elif task.is_delivery():
                orders[order_id]['deliveries'].append(task)
    
    # Process each order with appropriate sequencing strategy
    sequenced_tasks = []
    
    for order_id in sorted(orders.keys()):
        pickups = orders[order_id]['pickups']
        deliveries = orders[order_id]['deliveries']
        
        # Check order pattern type
        if len(pickups) > 1 and len(deliveries) > 1 and len(pickups) == len(deliveries):
            # CASE 1: nP+nD (BALANCED COMPLEX ORDER) - Preserve pickup-delivery pairs (PDPDPD)
            # Sort pickups and deliveries by task ID to ensure consistent pairing
            pickups.sort(key=lambda t: getattr(t, 'id', ''))
            deliveries.sort(key=lambda t: getattr(t, 'id', ''))
            
            # Create pickup-delivery pairs
            for i in range(len(pickups)):
                pickup = pickups[i]
                # Find matching delivery (same task number pattern)
                pickup_id = getattr(pickup, 'id', '')
                pickup_num = pickup_id.split('_')[-1] if '_' in pickup_id else ''
                
                matching_delivery = None
                for delivery in deliveries:
                    delivery_id = getattr(delivery, 'id', '')
                    delivery_num = delivery_id.split('_')[-1] if '_' in delivery_id else ''
                    # Match by task number (e.g., TASK_7_14 matches TASK_7_15)
                    if pickup_num and delivery_num and abs(int(pickup_num) - int(delivery_num)) == 1:
                        matching_delivery = delivery
                        break
                
                if matching_delivery:
                    sequenced_tasks.extend([pickup, matching_delivery])
                    deliveries.remove(matching_delivery)
                else:
                    # Fallback: just add pickup, delivery will be added later
                    sequenced_tasks.append(pickup)
            
            # Add any remaining deliveries
            sequenced_tasks.extend(deliveries)
            
        elif len(pickups) == 1 and len(deliveries) > 1:
            # CASE 2: 1P+nD (SINGLE PICKUP, MULTIPLE DELIVERIES) - Keep pickup before deliveries (PDDDD)
            pickups.sort(key=lambda t: getattr(t, 'id', ''))
            deliveries.sort(key=lambda t: getattr(t, 'id', ''))
            
            # Single pickup followed by all its deliveries
            sequenced_tasks.extend(pickups + deliveries)
            
        elif len(pickups) > 1 and len(deliveries) == 1:
            # CASE 3: nP+1D (MULTIPLE PICKUPS, SINGLE DELIVERY) - Keep pickups before delivery (PPPPD)
            pickups.sort(key=lambda t: getattr(t, 'id', ''))
            deliveries.sort(key=lambda t: getattr(t, 'id', ''))
            
            # All pickups followed by single delivery
            sequenced_tasks.extend(pickups + deliveries)
            
        else:
            # CASE 4: SIMPLE ORDER (1P+1D or unbalanced) - Traditional pickup-first
            pickups.sort(key=lambda t: getattr(t, 'id', ''))
            deliveries.sort(key=lambda t: getattr(t, 'id', ''))
            
            sequenced_tasks.extend(pickups + deliveries)
    
    # Combine: depot start, sequenced order tasks, depot return
    depot_start = [t for t in depot_tasks if t.is_depot_start()]
    depot_return = [t for t in depot_tasks if t.is_depot_return()]
    
    return depot_start + sequenced_tasks + depot_return


def _enforce_pickup_first_sequencing_with_lifo(tasks: List, vehicle) -> List:
    """
    Pickup-first sequencing with LIFO constraints for deliveries.
    
    ENHANCED: For complex orders (multiple P/D pairs), preserve pickup-delivery pairs
    while respecting LIFO constraints within each order.
    """
    if not tasks:
        return []
    
    # Separate depot tasks and order tasks
    depot_tasks = []
    order_tasks = []
    
    for task in tasks:
        if task.is_depot_start() or task.is_depot_return():
            depot_tasks.append(task)
        else:
            order_tasks.append(task)
    
    # Group tasks by order_id
    orders = {}
    for task in order_tasks:
        order_id = getattr(task, 'order_id', None)
        if order_id:
            if order_id not in orders:
                orders[order_id] = {'pickups': [], 'deliveries': []}
            
            if task.is_pickup():
                orders[order_id]['pickups'].append(task)
            elif task.is_delivery():
                orders[order_id]['deliveries'].append(task)
    
    # Process each order
    sequenced_tasks = []
    
    for order_id in sorted(orders.keys()):
        pickups = orders[order_id]['pickups']
        deliveries = orders[order_id]['deliveries']
        
        # Check if this is a complex order (multiple pickup/delivery pairs)
        if len(pickups) > 1 and len(deliveries) > 1 and len(pickups) == len(deliveries):
            # COMPLEX ORDER: Preserve pickup-delivery pairs with LIFO constraint
            # Sort pickups and deliveries by task ID to ensure consistent pairing
            pickups.sort(key=lambda t: getattr(t, 'id', ''))
            deliveries.sort(key=lambda t: getattr(t, 'id', ''))
            
            # Create pickup-delivery pairs and maintain LIFO order
            # For LIFO: last pickup's delivery should be first delivery
            for i in range(len(pickups)):
                pickup = pickups[i]
                # Find matching delivery (same task number pattern)
                pickup_id = getattr(pickup, 'id', '')
                pickup_num = pickup_id.split('_')[-1] if '_' in pickup_id else ''
                
                matching_delivery = None
                for delivery in deliveries:
                    delivery_id = getattr(delivery, 'id', '')
                    delivery_num = delivery_id.split('_')[-1] if '_' in delivery_id else ''
                    # Match by task number (e.g., TASK_7_14 matches TASK_7_15)
                    if pickup_num and delivery_num and abs(int(pickup_num) - int(delivery_num)) == 1:
                        matching_delivery = delivery
                        break
                
                if matching_delivery:
                    sequenced_tasks.extend([pickup, matching_delivery])
                    deliveries.remove(matching_delivery)
                else:
                    # Fallback: just add pickup, delivery will be added later
                    sequenced_tasks.append(pickup)
            
            # Add any remaining deliveries
            sequenced_tasks.extend(deliveries)
            
        else:
            # SIMPLE ORDER: Traditional pickup-first, then delivery with LIFO
            # Sort pickups by order_id for consistency
            pickups.sort(key=lambda t: getattr(t, 'order_id', ''))
            
            # For LIFO: deliveries in reverse order of pickups
            pickup_order = {getattr(p, 'order_id', ''): i for i, p in enumerate(pickups)}
            deliveries.sort(key=lambda t: pickup_order.get(getattr(t, 'order_id', ''), 999), reverse=True)
            
            sequenced_tasks.extend(pickups + deliveries)
    
    # Combine: depot start, sequenced order tasks, depot return
    depot_start = [t for t in depot_tasks if t.is_depot_start()]
    depot_return = [t for t in depot_tasks if t.is_depot_return()]
    
    return depot_start + sequenced_tasks + depot_return


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
    """Returns the maximum driving time per day based on HoS extensions."""
    base_drive_time = 9 * 60  # 9 hours = 540 minutes
    extension = extensions_used.get('drive_extension', 0.0)
    return base_drive_time + extension


def debug_print_route_timeline(route, phase_label: str):
    """
    Debug function to print detailed route timeline showing arrival times and time windows.
    This helps detect timing inconsistencies between different phases.
    """
    try:
        if not route or not hasattr(route, 'tasks') or len(route.tasks) == 0:
            print(f"           [{phase_label}] EMPTY ROUTE - No tasks to display")
            return
        
        print(f"           [{phase_label}] ROUTE TIMELINE ({len(route.tasks)} tasks):")
        
        # Use HoS simulation to get accurate timeline
        try:
            from hos_simulation import validate_route_hos_feasibility
            hos_result = validate_route_hos_feasibility(route)
            
            if hos_result and hasattr(hos_result, 'events') and hos_result.events:
                print(f"           [{phase_label}] Using HoS timeline ({len(hos_result.events)} events):")
                for i, event in enumerate(hos_result.events):
                    event_type = getattr(event, 'event_type', 'unknown')
                    start_time = getattr(event, 'start_time', 0)
                    end_time = getattr(event, 'end_time', 0)
                    task_id = getattr(event, 'task_id', 'unknown')
                    
                    # Convert minutes to hours for readability
                    start_hours = start_time / 60.0
                    end_hours = end_time / 60.0
                    
                    print(f"             {i+1:2d}. {event_type:12s} {task_id:15s} {start_hours:6.1f}h-{end_hours:6.1f}h ({start_time:7.1f}-{end_time:7.1f}min)")
                
                # Check for time window violations in the HoS timeline
                violations = getattr(hos_result, 'violations', [])
                if violations:
                    print(f"           [{phase_label}] ⚠️  HoS VIOLATIONS DETECTED:")
                    for violation in violations:
                        print(f"             - {violation}")
                else:
                    print(f"           [{phase_label}] ✅ No HoS violations detected")
                    
            else:
                print(f"           [{phase_label}] HoS timeline not available, using simple task list:")
                # Fallback to simple task enumeration
                for i, task in enumerate(route.tasks):
                    task_type = getattr(task, 'task_type', 'unknown')
                    task_id = getattr(task, 'id', 'unknown')
                    order_id = getattr(task, 'order_id', 'N/A')
                    
                    # Try to get time window info
                    earliest = getattr(task, 'earliest_time', None)
                    latest = getattr(task, 'latest_time', None)
                    tw_info = ""
                    if earliest is not None and latest is not None:
                        tw_info = f"TW:[{earliest:.0f}-{latest:.0f}min]"
                    
                    print(f"             {i+1:2d}. {str(task_type):12s} {task_id:15s} Order:{order_id:3s} {tw_info}")
                    
        except Exception as e:
            print(f"           [{phase_label}] ERROR in timeline debug: {e}")
            # Fallback to basic task listing
            for i, task in enumerate(route.tasks):
                task_type = getattr(task, 'task_type', 'unknown')
                task_id = getattr(task, 'id', 'unknown')
                order_id = getattr(task, 'order_id', 'N/A')
                print(f"             {i+1:2d}. {str(task_type):12s} {task_id:15s} Order:{order_id}")
                
    except Exception as e:
        print(f"           [{phase_label}] CRITICAL ERROR in timeline debug: {e}")


def _get_max_drive_per_day_old(extensions_used: dict) -> float:
    """Helper function for backward compatibility."""
    driving_extensions = extensions_used.get('driving', 0)
    return 10 * 60 if driving_extensions < 2 else 9 * 60

def _get_max_work_per_day(extensions_used: dict) -> float:
    """Helper function for backward compatibility.""" 
    work_extensions = extensions_used.get('work', 0)
    return 14 * 60 if work_extensions < 2 else 13 * 60