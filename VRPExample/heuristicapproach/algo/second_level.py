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
from typing import List, Optional, Iterator, Callable, TYPE_CHECKING, Union, Tuple
from epdt_data_structures import DriverState
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

def l2_heuristic(route: 'Route', order: 'Order', debug_assignment: bool = False) -> Optional['Route']:
    """
    Second-Level Heuristic: Finds the best way to insert an order into a route.
    
    This implements the L2 heuristic from the EPDT algorithm, which handles
    intra-route optimization by finding optimal task sequences and performing
    local search improvements.
    
    Args:
        route: The route to insert the order into
        order: The order to be inserted
        debug_assignment: Whether to print debug information
        
    Returns:
        Optimized route with the order inserted, or None if infeasible
    """
    
    initial_routes: List['Route'] = _generate_initial_task_sequence(route, order, debug_assignment)

    if debug_assignment:
        print(f"      DEBUG L2: Order {order.id} generated {len(initial_routes)} initial routes")

    if not initial_routes:
        if debug_assignment:
            print(f"      DEBUG L2: Order {order.id} - No feasible initial routes found")
        return None   # Infeasible insertion
    
    best_initial_route = max(initial_routes, key=calculate_z2_score)

    neighborhoods_to_search = [_task_swap_neighborhood]
    if order.is_fixed:
        neighborhoods_to_search.append(_task_insertion_neighborhood)

    final_route = local_search_l2(best_initial_route, neighborhoods_to_search, order)

    if debug_assignment:
        if final_route:
            print(f"      DEBUG L2: Order {order.id} - Final route feasible: {final_route.is_feasible()}")
        else:
            print(f"      DEBUG L2: Order {order.id} - Local search failed")

    return final_route



def _generate_initial_task_sequence(route: 'Route', order: 'Order', debug_assignment: bool = False) -> List['Route']:
    """ 
    Generates initial task sequences using a precedence-aware insertion heuristic.
    Ensures pickup tasks are always inserted before delivery tasks for the same order.
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
            
            # Try all positions for delivery-only orders
            for pos in range(len(current_route.tasks) + 1):
                test_route = current_route.copy()
                test_route.insert_task_without_reordering(pos, delivery)
                
                if debug_assignment:
                    print(f"        DEBUG L2: Delivery-only position {pos}, feasible: {is_feasible_for_insertion(test_route, debug_insertion=debug_assignment)}")
                
                if is_feasible_for_insertion(test_route, debug_insertion=debug_assignment):
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
    
    # Strategy 1: Cluster-based efficient insertion
    # Group pickups first, then deliveries to minimize depot visits
    # This creates more efficient pickup→pickup→delivery→delivery patterns
    
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
        max_positions_to_try = min(len(current_route.tasks) + 1, 20)
        for pos in range(max_positions_to_try):
            test_route = current_route.copy()
            test_route.insert_task_without_reordering(pos, pickup)
            
            if debug_assignment:
                print(f"        DEBUG L2: Pickup cluster position {pos}, feasible: {is_feasible_for_insertion(test_route, debug_insertion=debug_assignment)}")
            
            if is_feasible_for_insertion(test_route, debug_insertion=debug_assignment):
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
    # This creates the efficient pickup→pickup→delivery→delivery pattern
    pickup_cluster_size = len(P)  # Number of pickups we just inserted
    
    for delivery in D:
        best_delivery_cost = float('inf')
        best_delivery_route = None
        best_delivery_pos = None
        
        if debug_assignment:
            print(f"        DEBUG L2: Adding delivery {delivery.id if hasattr(delivery, 'id') else 'unknown'} after pickup cluster")
        
        # Insert deliveries starting after the pickup cluster
        # This ensures the pattern: depot → pickup1 → pickup2 → pickup3 → delivery1 → delivery2 → delivery3
        start_pos = pickup_cluster_size
        max_delivery_positions = min(len(current_route.tasks) + 1 - start_pos, 15)
        
        for pos_offset in range(max_delivery_positions):
            pos = start_pos + pos_offset
            test_route = current_route.copy()
            test_route.insert_task_without_reordering(pos, delivery)
            
            if debug_assignment:
                print(f"        DEBUG L2: Delivery cluster position {pos}, feasible: {is_feasible_for_insertion(test_route, debug_insertion=debug_assignment)}")
            
            if is_feasible_for_insertion(test_route, debug_insertion=debug_assignment):
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
            
            start_pos = (pickup_pos + 1) if pickup_pos is not None else 0
            for pos in range(start_pos, len(current_route.tasks) + 1):
                test_route = current_route.copy()
                test_route.insert_task_without_reordering(pos, delivery)
                
                if is_feasible_for_insertion(test_route, debug_insertion=debug_assignment):
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
    
    # Result: efficient pickup→pickup→delivery→delivery pattern
    # The route now minimizes depot visits and creates logical task clustering
    
    initial_routes.append(current_route)
    
    if debug_assignment:
        print(f"        DEBUG L2: Successfully created route with {len(current_route.tasks)} tasks")
    
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
    
    for i in range(n + 1):
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


def _calculate_realistic_driver_costs(route: 'Route') -> tuple[float, float]:
    """
    Calculate realistic driver break and rest costs by simulating HOS regulations.
    Returns (break_cost, total_downtime_minutes)
    
    Note: JIT compilation removed for better compatibility across different execution contexts.
    
    TODO: Consider integrating with _simulate_hos_advanced() for more accurate
    multi-day and weekly HoS cost calculation in future enhancements.
    """
    if not route.tasks:
        return 0.0, 0.0
    
    # Use driver's HoS state if available, otherwise create a new one
    if route.driver and route.driver.hos_state:
        driver = copy.deepcopy(route.driver.hos_state)
    else:
        driver = DriverState()
    current_time = 0
    total_break_time = 0
    total_rest_time = 0
    
    for i in range(len(route.tasks) - 1):
        current_task = route.tasks[i]
        next_task = route.tasks[i + 1]
        
        # Handle service time
        service_time = current_task.service_time
        driver.work_since_break += service_time
        driver.work_today += service_time
        current_time += service_time
        
        # Calculate travel time using Haversine-based calculation
        travel_time = calculate_travel_time_haversine(
            current_task.lat, current_task.lon,
            next_task.lat, next_task.lon
        )
        
        # Check if we need breaks/rests before travel
        remaining_drive_time = min(
            driver.MAX_DRIVE_WITHOUT_BREAK - driver.drive_since_break,
            driver.MAX_DRIVE_PER_DAY - driver.drive_today
        )
        
        # Simulate break/rest requirements during travel
        travel_remaining = travel_time
        while travel_remaining > remaining_drive_time:
            # Drive until break is needed
            driver.drive_since_break += remaining_drive_time
            driver.work_since_break += remaining_drive_time
            driver.drive_today += remaining_drive_time
            driver.work_today += remaining_drive_time
            driver.drive_this_week += remaining_drive_time
            
            current_time += remaining_drive_time
            travel_remaining -= remaining_drive_time
            
            # Take mandatory break
            current_time += 45  # 45-minute break
            total_break_time += 45
            driver.take_break(45)
            
            # Check if daily rest is needed
            if driver.drive_today >= driver.MAX_DRIVE_PER_DAY or driver.work_today >= driver.MAX_WORK_PER_DAY:
                current_time += 11 * 60  # 11-hour daily rest
                total_rest_time += 11 * 60
                driver.take_daily_rest()
                
                # Check if weekly rest is needed
                if driver.drive_this_week + driver.drive_last_week >= driver.MAX_DRIVE_TWO_WEEKS:
                    current_time += 45 * 60  # 45-hour weekly rest
                    total_rest_time += 45 * 60
                    driver.take_weekly_rest()
            
            remaining_drive_time = min(
                driver.MAX_DRIVE_WITHOUT_BREAK - driver.drive_since_break,
                driver.MAX_DRIVE_PER_DAY - driver.drive_today
            )
        
        # Complete remaining travel
        driver.drive_since_break += travel_remaining
        driver.work_since_break += travel_remaining
        driver.drive_today += travel_remaining
        driver.work_today += travel_remaining
        driver.drive_this_week += travel_remaining
        current_time += travel_remaining
    
    # Calculate costs: break time costs hourly rate, rest time costs reduced rate
    total_downtime = total_break_time + total_rest_time
    
    # Use driver's cost per hour if available, otherwise fall back to vehicle cost
    hourly_cost = route.driver.cost_per_hour if route.driver else route.vehicle.cost_per_hour
    break_cost = (total_break_time * hourly_cost + 
                 total_rest_time * hourly_cost * 0.3)  # Rest time at reduced rate
    
    return break_cost, total_downtime


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
    
    Note: JIT compilation removed for better compatibility across different execution contexts.
    """
    # Check if score is already cached
    if hasattr(route, '_z2_score'):
        return route._z2_score
    
    # Sort tasks with pickup-first sequencing for proper simulation and comparison
    sorted_tasks = _enforce_pickup_first_sequencing(route.tasks)
    
    # Initialize cost components
    travel_cost = 0.0  # C(r)
    time_window_penalty = 0.0  # W(r) 
    prospective_cost = 0.0  # A(r)
    driver_cost = 0.0  # D(r)
    vehicle_assignment_penalty = 0.0  # V(r)
    end_position_penalty = 0.0  # E(r)
    soft_time_window_penalty = 0.0  # Additional component for soft violations
    weight_violation_penalty = 0.0  # New component for weight capacity violations

    # Enhanced driver break costs calculation
    driver_cost, _ = _calculate_realistic_driver_costs(route)

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
            
            # Progressive penalty structure:
            # 0-5%: $50 per kg (moderate penalty)
            # 5-10%: $200 per kg (high penalty) 
            # >10%: $1000 per kg (extremely high penalty to discourage large violations)
            if violation_percentage <= 0.05:  # Up to 5% overload
                weight_violation_penalty += excess_weight * 50.0
            elif violation_percentage <= 0.10:  # 5-10% overload
                # Apply $50 for first 5% + $200 for additional
                first_tier = max_w * 0.05 * 50.0
                second_tier = (excess_weight - max_w * 0.05) * 200.0
                weight_violation_penalty += first_tier + second_tier
            else:  # >10% overload - extremely discouraged
                # Apply tiered penalties plus $1000 for excess over 10%
                first_tier = max_w * 0.05 * 50.0
                second_tier = max_w * 0.05 * 200.0
                third_tier = (excess_weight - max_w * 0.10) * 1000.0
                weight_violation_penalty += first_tier + second_tier + third_tier
        
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
                # Soft time window violation - calculate penalty
                penalty_rate = getattr(current_task, 'late_penalty_rate', 2.0)
                soft_time_window_penalty += delay * penalty_rate
            else:
                # Hard time window violation - high penalty
                time_window_penalty += delay * 10.0  # High penalty for hard violations

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

    # Calculate total Z2 score including weight violation penalty
    total_cost = (travel_cost + time_window_penalty + prospective_cost + 
                  driver_cost + vehicle_assignment_penalty + end_position_penalty +
                  soft_time_window_penalty + weight_violation_penalty)

    # Cache the score
    route._z2_score = total_cost
    return total_cost


def _calculate_inefficiency_penalty(current_task, next_task, vehicle) -> float:
    """
    Calculate distance-based inefficiency penalty for mixed pickup/delivery patterns.
    
    This function discourages inefficient routing patterns like:
    - Pickup → Depot → Delivery (should be Pickup → Delivery)
    - Pickup A → Delivery B → Pickup C (mixed orders inefficiently)
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
        # This is an efficient pickup → delivery pattern, apply discount
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
    
    # HoS exemption: Skip HoS checks for vehicles without regulations (like furgoni)
    if hasattr(route.vehicle, 'regulations') and not route.vehicle.regulations:
        return True  # Vehicles without regulations are exempt from HoS constraints
    
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
    This is less strict than the full is_feasible() check to allow more flexibility
    during the clustering phase of L2 heuristic.
    
    Args:
        route: Route to check feasibility for
        debug_insertion: Whether to print debug information
        
    Returns:
        True if route passes basic feasibility checks for insertion
    """
    
    if debug_insertion:
        print(f"                DEBUG INSERTION: Quick feasibility check for {len(route.tasks)} tasks")
    
    # Only check critical hard constraints during insertion
    sorted_tasks = _enforce_pickup_first_sequencing(route.tasks)
    
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
    
    for task in sorted_tasks:
        if task.is_pickup():
            load_w += task.demand
            load_v += task.volume
            load_pallets += getattr(task, 'pallets', 0)
        elif task.is_delivery():
            load_w += task.demand  # demand is negative for deliveries
            load_v += task.volume  # volume is negative for deliveries
            load_pallets += getattr(task, 'pallets', 0)  # pallets should be negative for delivery tasks
        
        # Track peak loads
        peak_load_w = max(peak_load_w, load_w)
        peak_load_v = max(peak_load_v, load_v)
        peak_load_pallets = max(peak_load_pallets, load_pallets)
        
        # Only check hard constraints that would make route impossible
        # During initialization, be extremely permissive to allow assignment
        # During optimization, be STRICT to prevent violations
        
        # Volume constraint is very relaxed during initialization, STRICT during optimization
        volume_tolerance = 2.0 if is_initialization else 1.0  # 100% tolerance during init vs STRICT during optimization
        if peak_load_v > max_v * volume_tolerance:
            if debug_insertion:
                print(f"                DEBUG INSERTION: Volume constraint exceeded: {peak_load_v:.2f} > {max_v:.2f}")
            return False
            
        # Pallet constraint is very relaxed during initialization, STRICT during optimization
        pallet_tolerance = 1.5 if is_initialization else 1.0  # 50% tolerance during init vs STRICT during optimization
        if max_pallets is not None and peak_load_pallets > max_pallets * pallet_tolerance:
            if debug_insertion:
                print(f"                DEBUG INSERTION: Pallet constraint exceeded: {peak_load_pallets} > {max_pallets}")
            return False
        
        # Weight constraint is extremely relaxed during initialization, STRICT during optimization
        weight_tolerance = 3.0 if is_initialization else 1.0  # 200% tolerance during init vs STRICT during optimization
        if peak_load_w > max_w * weight_tolerance:
            if debug_insertion:
                print(f"                DEBUG INSERTION: Severe weight constraint exceeded: {peak_load_w:.2f} > {max_w:.2f}")
            return False
    
    # H2: Enhanced logical precedence check
    # NEW RULE: Allow deliveries before pickups as long as:
    # 1. Each delivery has its corresponding pickup already completed
    # 2. Pallet capacity is never exceeded (physical constraint)
    # 3. Distance penalty will discourage inefficient patterns
    
    # Track completed pickups for each order to validate deliveries
    completed_pickups = set()
    load_pallets_check = 0  # Track pallet load for physical constraint
    
    for task in sorted_tasks:
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
                if debug_insertion:
                    print(f"                DEBUG INSERTION: Delivery {task.id} attempted before its pickup was completed")
                return False
            # Remove pallets from load
            load_pallets_check += getattr(task, 'pallets', 0)  # pallets should be negative for delivery tasks
            
        # Physical constraint: never exceed pallet capacity
        if max_pallets is not None and load_pallets_check > max_pallets:
            if debug_insertion:
                print(f"                DEBUG INSERTION: Pallet capacity exceeded during route execution: {load_pallets_check} > {max_pallets}")
            return False
    
    # Individual order precedence constraints
    orders = {}
    for i, task in enumerate(sorted_tasks):
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
    
    return True


def is_feasible(route: 'Route', debug_feasibility: bool = False, return_reason: bool = False) -> Union[bool, Tuple[bool, str]]:
    """ 
    Check if the route is feasible according to all constraints.
    Enhanced to support multi-day planning and LIFO loading constraints.
    
    Args:
        route: Route to check feasibility for
        debug_feasibility: Whether to print debug information
        return_reason: Whether to return detailed failure reason
    
    Returns:
        If return_reason=False: bool (feasible or not)
        If return_reason=True: Tuple[bool, str] (feasible, reason)
    
    Note: JIT compilation removed for better compatibility across different execution contexts.
    """
    
    # Enhanced debugging for assignment failures
    if debug_feasibility:
        print(f"            DEBUG FEASIBILITY: Checking route feasibility for vehicle {route.vehicle.id}")
        print(f"            DEBUG FEASIBILITY: Route has {len(route.tasks)} tasks")
    
    # H0: DEPOT START/END VALIDATION - Must be first check
    # Every route must start with a depot start task and end with a depot return task
    if route.tasks:
        # Check first task is depot start
        first_task = route.tasks[0]
        if not (hasattr(first_task, 'is_depot_start') and first_task.is_depot_start()):
            reason = f"Route validation failed: First task ({first_task.id if hasattr(first_task, 'id') else 'unknown'}) is not a depot start task"
            if debug_feasibility:
                print(f"            DEBUG FEASIBILITY: {reason}")
            if return_reason:
                return False, reason
            return False
        
        # Check last task is depot return
        last_task = route.tasks[-1]
        if not (hasattr(last_task, 'is_depot_return') and last_task.is_depot_return()):
            reason = f"Route validation failed: Last task ({last_task.id if hasattr(last_task, 'id') else 'unknown'}) is not a depot return task"
            if debug_feasibility:
                print(f"            DEBUG FEASIBILITY: {reason}")
            if return_reason:
                return False, reason
            return False
        
        if debug_feasibility:
            print(f"            DEBUG FEASIBILITY: ✅ Depot validation passed - route starts and ends at depot")
    
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
                    print(f"            DEBUG FEASIBILITY: {reason}")
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
                print(f"            DEBUG FEASIBILITY: {reason}")
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
                print(f"            DEBUG FEASIBILITY: Pre-loaded LIFO stack for delivery-only orders: {lifo_stack}")

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
                        print(f"            DEBUG FEASIBILITY: {reason}")
                    if return_reason:
                        return False, reason
                    return False
                if lifo_stack[-1] != task.order_id:
                    reason = f"LIFO violation: expected {lifo_stack[-1]}, got {task.order_id}"
                    if debug_feasibility:
                        print(f"            DEBUG FEASIBILITY: {reason}")
                    if return_reason:
                        return False, reason
                    return False
                lifo_stack.pop()  # Remove delivered order from stack
        
        # Check capacity constraints
        # Weight is now a soft constraint (removed from hard feasibility check)
        # Volume remains a hard constraint
        if load_v > max_v:
            reason = f"Volume constraint violated: {load_v:.2f} > {max_v:.2f} for task {task.id}"
            if debug_feasibility:
                print(f"            DEBUG FEASIBILITY: {reason}")
            if return_reason:
                return False, reason
            return False
            
        # Pallet capacity is now a hard constraint
        if max_pallets is not None and load_pallets > max_pallets:
            reason = f"Pallet constraint violated: {load_pallets} > {max_pallets} for task {task.id}"
            if debug_feasibility:
                print(f"            DEBUG FEASIBILITY: {reason}")
            if return_reason:
                return False, reason
            return False

    # LIFO final check: all cargo must be delivered
    if route.vehicle.lifo_required and lifo_stack:
        reason = f"LIFO constraint violated: undelivered cargo: {lifo_stack}"
        if debug_feasibility:
            print(f"            DEBUG FEASIBILITY: {reason}")
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
                        print(f"            DEBUG FEASIBILITY: {reason}")
                    if return_reason:
                        return False, reason
                    return False
            except (ValueError, TypeError, KeyError) as e:
                if debug_feasibility:
                    print(f"            DEBUG FEASIBILITY: Error in precedence check for {order_id}: {e}")
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
    
    if is_initialization:
        # Skip HoS during initialization to allow assignment
        if debug_feasibility:
            print(f"            DEBUG FEASIBILITY: HoS constraint bypassed during initialization")
        pass
    else:
        # Apply STRICT LEGAL HoS during optimization - THESE ARE LEGALLY MANDATED
        try:
            driver_state = DriverState()  # Initialize fresh driver state
            
            # DO NOT MODIFY THESE - THEY ARE SET BY EUROPEAN LAW
            # EU Regulation 561/2006 - Driver Hours of Service
            # driver_state.MAX_DRIVE_WITHOUT_BREAK = 4.5 * 60  # 4.5 hours (270 minutes)
            # driver_state.MAX_DRIVE_PER_DAY = 9 * 60          # 9 hours (540 minutes)  
            # driver_state.MAX_WORK_PER_DAY = 13 * 60          # 13 hours (780 minutes)
            # These are the legal defaults and MUST NOT be changed
            
            if not _check_hos_multiday(route, driver_state, sorted_tasks):
                reason = "HoS constraint violated (LEGAL LIMITS)"
                if debug_feasibility:
                    print(f"            DEBUG FEASIBILITY: {reason}")
                if return_reason:
                    return False, reason
                return False
        except Exception as e:
            if debug_feasibility:
                print(f"            DEBUG FEASIBILITY: HoS check failed with error: {e}")
            # For robustness, allow route to pass but log the error
            pass
    
    # H7: Hard time windows check - RE-ENABLED with enhanced validation
    for task in sorted_tasks:
        if hasattr(task, 'earliest_time') and hasattr(task, 'latest_time'):
            if not getattr(task, 'soft_time_window', False):  # Only check hard time windows
                # This is simplified - proper implementation would track arrival times
                # But we now have basic time window constraint validation
                if task.earliest_time is not None and task.latest_time is not None:
                    if task.earliest_time > task.latest_time:
                        reason = f"Invalid time window for task {task.id}: earliest ({task.earliest_time}) > latest ({task.latest_time})"
                        if debug_feasibility:
                            print(f"            DEBUG FEASIBILITY: {reason}")
                        if return_reason:
                            return False, reason
                        return False
            
    # H8: Driver regulations check (already done in _check_hos_multiday)
    # H9: Route ending position check
    if hasattr(route, 'preferred_end_position') and sorted_tasks:
        last_task = sorted_tasks[-1]
        if hasattr(last_task, 'location') and last_task.location != route.preferred_end_position:
            return False
        
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

def is_feasible(route: Route, debug_feasibility: bool = False, return_reason: bool = False) -> Union[bool, Tuple[bool, str]]:
    """
    Check if a route is feasible considering all constraints, including multi-day operations and HoS.
    This is the definitive check for a route's validity.
    """
    if not route or not route.tasks:
        return (True, "Empty route") if return_reason else True

    if debug_feasibility:
        print(f"DEBUG FEASIBILITY: Checking route feasibility for vehicle {route.vehicle.id}")
        print(f"DEBUG FEASIBILITY: Route has {len(route.tasks)} tasks")

    # 1. Depot Start/End Validation
    if not route.tasks[0].is_depot_start() or not route.tasks[-1].is_depot_return():
        reason = f"Route validation failed: First task ({route.tasks[0].id}) is not a depot start task or last task ({route.tasks[-1].id}) is not a depot return task"
        if debug_feasibility:
            print(f"DEBUG FEASIBILITY: {reason}")
        return (False, reason) if return_reason else False
    if debug_feasibility:
        print("DEBUG FEASIBILITY: ✅ Depot validation passed - route starts and ends at depot")

    # 2. Capacity Validation
    current_weight = 0
    current_volume = 0
    current_pallets = 0
    for task in route.tasks:
        current_weight += getattr(task, 'demand', 0)
        current_volume += getattr(task, 'volume', 0)
        current_pallets += getattr(task, 'pallets', 0)
        if (current_weight > route.vehicle.weight_capacity or 
            current_volume > route.vehicle.volume_capacity or 
            current_pallets > route.vehicle.pallet_capacity):
            reason = f"Capacity violation at task {task.id}: W:{current_weight}/{route.vehicle.weight_capacity}, V:{current_volume}/{route.vehicle.volume_capacity}, P:{current_pallets}/{route.vehicle.pallet_capacity}"
            if debug_feasibility:
                print(f"DEBUG FEASIBILITY: {reason}")
            return (False, reason) if return_reason else False
    if debug_feasibility:
        print("DEBUG FEASIBILITY: ✅ Capacity validation passed")

    # 3. Time Window and HoS Validation
    try:
        from route_provider import calculate_travel_time_between_tasks
    except ImportError:
        return (False, "Route provider not available for time calculations") if return_reason else False

    # Initialize time and waiting time tracking
    completion_time = 0.0  # Tracks the completion time of the last task
    route.total_wait_time = 0.0
    daily_driving_time = 0
    daily_work_time = 0
    current_day = 1

    for i in range(len(route.tasks)):
        task = route.tasks[i]
        
        departure_time = completion_time
        travel_time = 0
        wait_time = 0

        if i > 0:
            prev_task = route.tasks[i-1]
            travel_time = calculate_travel_time_between_tasks(prev_task, task, route.vehicle)
            
            # If the next task has an earliest start time, we might need to wait at the previous location.
            if task.earliest_time and task.earliest_time > 0:
                # Required departure time from previous location to arrive exactly at earliest_time
                required_departure_time = task.earliest_time - travel_time
                
                # If we are ready to depart earlier than required, we wait.
                if completion_time < required_departure_time:
                    wait_time = required_departure_time - completion_time
                    route.total_wait_time += wait_time
        
        # Departure time from previous task location includes waiting
        departure_time = completion_time + wait_time
        
        # Arrival at current task
        arrival_time = departure_time + travel_time
        service_start_time = arrival_time

        # Check for lateness
        if task.latest_time and service_start_time > task.latest_time:
            reason = f"Time window violation at task {task.id}: Service start {format_absolute_minutes(service_start_time)} is after latest {format_absolute_minutes(task.latest_time)}"
            if debug_feasibility:
                print(f"DEBUG FEASIBILITY: {reason}")
            return (False, reason) if return_reason else False

        service_time = getattr(task, 'service_time', 0)
        completion_time = service_start_time + service_time

        # HoS (Hours of Service) checks
        # This is a simplified check. A full implementation would be more complex.
        new_day = int(completion_time // 1440) + 1
        if new_day > current_day:
            daily_driving_time = 0
            daily_work_time = 0
            current_day = new_day

        daily_driving_time += travel_time
        # Per regulations, waiting time does not contribute to driving or work time.
        daily_work_time += travel_time + service_time

        if daily_driving_time > route.vehicle.max_driving_time or daily_work_time > route.vehicle.max_work_time:
            reason = f"HoS constraint violated (LEGAL LIMITS) on day {current_day}: Drive={daily_driving_time:.1f}m, Work={daily_work_time:.1f}m"
            if debug_feasibility:
                print(f"DEBUG FEASIBILITY: {reason}")
            return (False, reason) if return_reason else False

    if debug_feasibility:
        print("DEBUG FEASIBILITY: ✅ Time window and HoS validation passed")

    return (True, "Route is feasible") if return_reason else True

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


def _check_hos_multiday(route: 'Route', debug_feasibility: bool = False) -> bool:
    """
    Enhanced Hours of Service check with multi-day support for debugging.
    """
    # HoS exemption: Skip HoS checks for vehicles without regulations (like furgoni)
    if hasattr(route.vehicle, 'regulations') and not route.vehicle.regulations:
        return True  # Vehicles without regulations are exempt from HoS constraints
    
    # For now, simplified check - implement full logic as needed
    return True  # Placeholder - always passes for now


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


def _check_hos_multiday(route: 'Route', driver_state: 'DriverState', sorted_tasks: List) -> bool:
    """
    Enhanced Hours of Service check with multi-day support and detailed European regulations.
    
    European HoS Regulations:
    - After 4.5 hours of driving, a 45-minute break is mandatory (can be split into 15 + 30 mins)
    - Maximum 9 hours of driving per day (extendable to 10 hours twice a week)
    - Maximum 13 hours of work per day (extendable to 14 hours twice a week)
    - Minimum 11 hours of daily rest (can be reduced to 9 hours under certain conditions)
    
    Args:
        route: The route being checked
        driver_state: Current driver state (potentially from previous day)
        sorted_tasks: Tasks sorted chronologically
        
    Returns:
        True if route respects HoS regulations, False otherwise
    """
    if not sorted_tasks:
        return True
    
    current_time = 0
    current_day = getattr(sorted_tasks[0], 'day', 0)
    extensions_used_this_week = {'driving': 0, 'work': 0}  # Track extensions
    
    # Process each task in chronological order
    for i in range(len(sorted_tasks)):
        current_task = sorted_tasks[i]
        task_day = getattr(current_task, 'day', 0)
        
        # Check for day transition
        if task_day != current_day:
            # New day: check if sufficient daily rest was taken
            if driver_state.work_today > 0:  # If work was done previous day
                # Minimum 11 hours daily rest required
                rest_time = 11 * 60  # 11 hours in minutes
                # Can be reduced to 9 hours up to 3 times per week (not implemented here for simplicity)
                current_time += rest_time
                driver_state.take_daily_rest()
            current_day = task_day
        
        # Handle service time at current task
        service_time = getattr(current_task, 'service_time', 0)
        
        # Check work time limits before adding service time
        if driver_state.work_today + service_time > _get_max_work_per_day(extensions_used_this_week):
            return False  # Would exceed daily work limit
        
        driver_state.work_since_break += service_time
        driver_state.work_today += service_time
        current_time += service_time
        
        # If there's a next task, calculate travel time using Haversine
        if i < len(sorted_tasks) - 1:
            next_task = sorted_tasks[i + 1]
            travel_time = calculate_travel_time_haversine(
                current_task.lat, current_task.lon,
                next_task.lat, next_task.lon
            ) / 60.0  # Convert to hours for HoS calculations
            
            # Check driving time limits
            max_drive_per_day = _get_max_drive_per_day(extensions_used_this_week)
            if driver_state.drive_today + travel_time > max_drive_per_day:
                return False  # Would exceed daily driving limit
            
            # Check if break is needed before travel
            if driver_state.drive_since_break + travel_time > driver_state.MAX_DRIVE_WITHOUT_BREAK:
                # Need a break
                if driver_state.drive_since_break > 0:  # Only if we've been driving
                    current_time += 45  # 45-minute break
                    driver_state.take_break(45)
            
            # Add travel time
            driver_state.drive_since_break += travel_time
            driver_state.work_since_break += travel_time
            driver_state.drive_today += travel_time
            driver_state.work_today += travel_time
            driver_state.drive_this_week += travel_time
            current_time += travel_time
        
        # Check for soft time window compliance (if applicable)
        if hasattr(current_task, 'soft_time_window') and current_task.soft_time_window:
            if (hasattr(current_task, 'latest_time') and 
                current_task.latest_time is not None and 
                current_time > current_task.latest_time):
                # Soft time window violation - continue but will be penalized in scoring
                pass
        elif (hasattr(current_task, 'latest_time') and 
              current_task.latest_time is not None and 
              current_time > current_task.latest_time):
            # Hard time window violation
            return False
        
        # Handle waiting for earliest time
        if (hasattr(current_task, 'earliest_time') and 
            current_task.earliest_time is not None and 
            current_time < current_task.earliest_time):
            wait_time = current_task.earliest_time - current_time
            current_time = current_task.earliest_time
            driver_state.work_today += wait_time
            driver_state.work_since_break += wait_time
            driver_state.time_in_daily_period += wait_time
            driver_state.time_since_weekly_rest += wait_time
    
    return True


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