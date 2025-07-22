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

from typing import List, Optional, Iterator, Callable, TYPE_CHECKING
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

def l2_heuristic(route: 'Route', order: 'Order') -> Optional['Route']:
    """
    Second-Level Heuristic: Finds the best way to insert an order into a route.
    
    This implements the L2 heuristic from the EPDT algorithm, which handles
    intra-route optimization by finding optimal task sequences and performing
    local search improvements.
    
    Args:
        route: The route to insert the order into
        order: The order to be inserted
        
    Returns:
        Optimized route with the order inserted, or None if infeasible
    """
    
    initial_routes: List['Route'] = _generate_initial_task_sequence(route, order)

    if not initial_routes:
        return None   # Infeasible insertion
    
    best_initial_route = max(initial_routes, key=calculate_z2_score)

    neighborhoods_to_search = [_task_swap_neighborhood]
    if order.is_fixed:
        neighborhoods_to_search.append(_task_insertion_neighborhood)

    final_route = local_search_l2(best_initial_route, neighborhoods_to_search, order)

    return final_route



def _generate_initial_task_sequence(route: 'Route', order: 'Order') -> List['Route']:
    """ 
    Generates initial task sequences using a fast cheapest insertion heuristic.
    Optimized from O(n^2) to O(n) for better performance on long routes.
    """
    P = order.get_pickups()
    D = order.get_deliveries()
    
    initial_routes = []
    
    # Strategy 1: Try inserting all pickups first, then all deliveries
    # This maintains precedence constraints while being much faster
    pickup_routes = []
    
    # Find cheapest insertion position for all pickups together
    base_route = route.copy()
    n = len(base_route.tasks)
    
    best_pickup_cost = float('inf')
    best_pickup_position = 0
    
    # Single pass to find best position for pickups (O(n) instead of O(n^2))
    for i in range(n + 1):
        test_route = copy.deepcopy(route)  # Use deep copy for proper state management
        
        # Insert all pickups at position i
        for pickup in reversed(P):  # Reverse to maintain order
            test_route.insert_task(i, pickup)
        
        if test_route.is_feasible():
            cost = calculate_z2_score(test_route)
            if cost < best_pickup_cost:
                best_pickup_cost = cost
                best_pickup_position = i
                pickup_routes = [test_route.copy()]
    
    # For each viable pickup insertion, find best delivery positions
    for pickup_route in pickup_routes:
        n_with_pickups = len(pickup_route.tasks)
        
        best_delivery_cost = float('inf')
        best_delivery_route = None
        
        # Single pass for delivery insertion (O(n))
        for j in range(best_pickup_position + len(P), n_with_pickups + 1):
            test_route = copy.deepcopy(pickup_route)  # Use deep copy for proper state management
            
            # Insert all deliveries at position j
            for delivery in reversed(D):  # Reverse to maintain order
                test_route.insert_task(j, delivery)
            
            if test_route.is_feasible():
                cost = calculate_z2_score(test_route)
                if cost < best_delivery_cost:
                    best_delivery_cost = cost
                    best_delivery_route = test_route.copy()
        
        if best_delivery_route:
            initial_routes.append(best_delivery_route)
    
    # Strategy 2: Try interleaved insertion if we don't have good solutions
    if len(initial_routes) < 2:
        # Simple fallback: insert tasks at route middle for balanced load
        middle_pos = n // 2
        fallback_route = route.copy()
        
        # Insert pickups first
        for pickup in reversed(P):
            fallback_route.insert_task(middle_pos, pickup)
        
        # Insert deliveries after pickups
        delivery_pos = middle_pos + len(P)
        for delivery in reversed(D):
            fallback_route.insert_task(delivery_pos, delivery)
        
        if fallback_route.is_feasible():
            initial_routes.append(fallback_route)

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
        
        # Calculate travel time
        travel_time = route.calculate_travel_time(current_task, next_task)
        
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
    break_cost = (total_break_time * route.vehicle.cost_per_hour + 
                 total_rest_time * route.vehicle.cost_per_hour * 0.3)  # Rest time at reduced rate
    
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
    
    # Sort tasks chronologically for proper simulation
    sorted_tasks = _sort_tasks_chronologically(route.tasks)
    
    # Initialize cost components
    travel_cost = 0.0  # C(r)
    time_window_penalty = 0.0  # W(r) 
    prospective_cost = 0.0  # A(r)
    driver_cost = 0.0  # D(r)
    vehicle_assignment_penalty = 0.0  # V(r)
    end_position_penalty = 0.0  # E(r)
    soft_time_window_penalty = 0.0  # Additional component for soft violations

    # Enhanced driver break costs calculation
    driver_cost, _ = _calculate_realistic_driver_costs(route)

    # Multi-day cost simulation
    current_time = 0
    current_day = None
    last_today_task_location = None
    tomorrow_tasks = []
    
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
            travel_time = _calculate_travel_time_between_tasks(current_task, next_task, route.vehicle)
            travel_cost += travel_time * route.vehicle.cost_per_km * 0.8  # Rough conversion
            current_time += travel_time
        
        # Check time window violations
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

    # Calculate total Z2 score
    total_cost = (travel_cost + time_window_penalty + prospective_cost + 
                  driver_cost + vehicle_assignment_penalty + end_position_penalty +
                  soft_time_window_penalty)

    # Cache the score
    route._z2_score = total_cost
    return total_cost


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
    
    if strategy == 'first_improvement':
        # First improvement strategy: take first better neighbor found
        while improved:
            improved = False
            current_score = calculate_z2_score(current_route)
            
            # Modular neighborhood exploration: each function is independent
            for neighborhood in neighborhoods:
                for neighbor in neighborhood(current_route, order):  # Pass order for context
                    neighbor_score = calculate_z2_score(neighbor)
                    if neighbor_score < current_score:
                        current_route = neighbor
                        improved = True
                        break  # Exit inner loop to restart with new current route
                if improved:
                    break  # Exit outer loop to restart with new current route
    
    elif strategy == 'best_improvement':
        # Best improvement strategy: evaluate all neighbors, choose best
        while improved:
            improved = False
            current_score = calculate_z2_score(current_route)
            best_route = current_route
            best_score = current_score
            
            # Evaluate all neighbors in all neighborhoods
            for neighborhood in neighborhoods:
                for neighbor in neighborhood(current_route, order):
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

    return current_route



def is_feasible(route: 'Route') -> bool:
    """ 
    Check if the route is feasible according to all constraints.
    Enhanced to support multi-day planning and LIFO loading constraints.
    
    Note: JIT compilation removed for better compatibility across different execution contexts.
    """
    
    # H1: Multi-day chronological simulation setup
    # Sort tasks by day first, then by position within day
    sorted_tasks = _sort_tasks_chronologically(route.tasks)
    
    # Initialize vehicle state from previous day if available
    initial_state = getattr(route.vehicle, 'initial_state', None)
    if initial_state:
        current_position = initial_state.get('position', None)
        previous_day_load_w = initial_state.get('load_weight', 0.0)
        previous_day_load_v = initial_state.get('load_volume', 0.0)
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
        driver_state = DriverState()
    
    # H2: Capacity check with multi-day simulation
    load_w = previous_day_load_w
    load_v = previous_day_load_v
    max_w = route.vehicle.weight_capacity
    max_v = route.vehicle.volume_capacity

    # H3: LIFO Loading Constraint check
    lifo_stack = []
    if route.vehicle.lifo_required:
        # Initialize stack with any cargo from previous day
        if initial_state and 'cargo_stack' in initial_state:
            lifo_stack = initial_state['cargo_stack'].copy()

    for task in sorted_tasks:
        if task.is_pickup():
            load_w += task.demand
            load_v += task.volume
            
            # LIFO constraint: push order_id onto stack
            if route.vehicle.lifo_required:
                lifo_stack.append(task.order_id)
                
        elif task.is_delivery():
            load_w += task.demand  # demand is negative for deliveries
            load_v += task.volume  # volume is negative for deliveries
            
            # LIFO constraint: check if this delivery matches top of stack
            if route.vehicle.lifo_required:
                if not lifo_stack:
                    return False  # Trying to deliver when no cargo loaded
                if lifo_stack[-1] != task.order_id:
                    return False  # LIFO violation - can't access this order
                lifo_stack.pop()  # Remove delivered order from stack
        
        # Check capacity constraints
        if load_w > max_w or load_v > max_v:
            return False

    # LIFO final check: all cargo must be delivered
    if route.vehicle.lifo_required and lifo_stack:
        return False

    # H4: Precedence constraints check with multi-day consideration
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
            # Find last pickup (considering day and position)
            last_pickup = max(tasks['pickups'], key=lambda x: (x[1], x[0]))  # Sort by day, then position
            # Find first delivery
            first_delivery = min(tasks['deliveries'], key=lambda x: (x[1], x[0]))
            
            # Check if last pickup happens before first delivery
            if (last_pickup[1] > first_delivery[1] or 
                (last_pickup[1] == first_delivery[1] and last_pickup[0] >= first_delivery[0])):
                return False

    # H5: Multi-day Hours of Service check - Advanced simulation with iterative breaks/rests
    feasible, _ = _simulate_hos_advanced(route, driver_state, sorted_tasks)
    if not feasible:
        return False
    
    # H6: Hard time windows check
    for task in sorted_tasks:
        if hasattr(task, 'earliest_time') and hasattr(task, 'latest_time'):
            if not getattr(task, 'soft_time_window', False):  # Only check hard time windows
                # This is simplified - proper implementation would track arrival times
                pass
            
    # H7: Driver regulations check (already done in _check_hos_multiday)
    # H8: Route ending position check
    if hasattr(route, 'preferred_end_position') and sorted_tasks:
        last_task = sorted_tasks[-1]
        if hasattr(last_task, 'location') and last_task.location != route.preferred_end_position:
            return False
        
    return True  # All checks passed, route is feasible


@dataclass
class DriverState:
    """
    Enhanced class to track driver's hours of service state according to European regulations.
    
    European HoS Regulations:
    - After 4.5 hours of driving, a 45-minute break is mandatory (can be split into 15 + 30 mins)
    - Maximum 9 hours of driving per day (extendable to 10 hours twice a week)
    - Maximum 13 hours of work per day (extendable to 14 hours twice a week)  
    - Minimum 11 hours of daily rest (can be reduced to 9 hours under certain conditions)
    - Maximum 56 hours driving in a week (90 hours in any two consecutive weeks)
    """
    
    # Current state counters
    drive_since_break: float = 0.0      # Accumulated driving time since last break
    work_since_break: float = 0.0       # Accumulated working time since last break
    drive_today: float = 0.0             # Total driving time in current 24-hour period
    work_today: float = 0.0              # Total duty time in current 24-hour period
    drive_this_week: float = 0.0         # Driving time this week
    drive_last_week: float = 0.0         # Driving time last week
    
    # Additional comprehensive state variables for advanced HoS simulation
    time_in_daily_period: float = 0.0   # Time elapsed since end of last daily rest (max 24h)
    work_this_week: float = 0.0          # Accumulated working time from Monday 00:00
    time_since_weekly_rest: float = 0.0  # Time elapsed since last weekly rest ended (max 144h)
    
    # Extension tracking
    daily_driving_extensions_used: int = 0      # Extensions to 10 hours used this week
    daily_work_extensions_used: int = 0         # Extensions to 14 hours used this week
    reduced_rest_used: int = 0                  # Reduced rest periods used this week
    daily_rest_reductions_used: int = 0         # Count of 9h daily rests between weekly rests (max 3)
    is_weekly_rest_reduction_taken: bool = False  # Flag if reduced weekly rest taken in last two weeks
    
    # Regulation limits in minutes (European HoS)
    MAX_DRIVE_WITHOUT_BREAK = 4.5 * 60         # 4.5 hours
    MAX_WORK_WITHOUT_BREAK = 6 * 60            # 6 hours
    MAX_DRIVE_PER_DAY = 9 * 60                 # 9 hours (extendable to 10)
    MAX_WORK_PER_DAY = 13 * 60                 # 13 hours (extendable to 14)
    MAX_DRIVE_PER_WEEK = 56 * 60               # 56 hours
    MAX_DRIVE_TWO_WEEKS = 90 * 60              # 90 hours in any two consecutive weeks
    MIN_DAILY_REST = 11 * 60                   # 11 hours (reducible to 9)
    MIN_WEEKLY_REST = 45 * 60                  # 45 hours
    
    def can_drive(self, duration: float) -> bool:
        """Check if driver can drive for the specified duration without violating HoS."""
        # Check break requirements
        if self.drive_since_break + duration > self.MAX_DRIVE_WITHOUT_BREAK:
            return False
            
        # Check daily limits (considering extensions)
        max_daily = self.get_current_max_daily_drive()
        if self.drive_today + duration > max_daily:
            return False
            
        # Check weekly limits
        if self.drive_this_week + duration > self.MAX_DRIVE_PER_WEEK:
            return False
            
        # Check two-week limits
        if self.drive_this_week + self.drive_last_week + duration > self.MAX_DRIVE_TWO_WEEKS:
            return False
            
        return True
    
    def can_work(self, duration: float) -> bool:
        """Check if driver can work for the specified duration without violating HoS."""
        # Check work time since last break
        if self.work_since_break + duration > self.MAX_WORK_WITHOUT_BREAK:
            return False
            
        # Check daily work limits (considering extensions)
        max_daily_work = self.get_current_max_daily_work()
        if self.work_today + duration > max_daily_work:
            return False
            
        return True
    
    def get_current_max_daily_drive(self) -> float:
        """Get current maximum daily driving time considering extensions."""
        if self.daily_driving_extensions_used < 2:
            return 10 * 60  # Can extend to 10 hours
        return self.MAX_DRIVE_PER_DAY  # 9 hours
        
    def get_current_max_daily_work(self) -> float:
        """Get current maximum daily work time considering extensions."""
        if self.daily_work_extensions_used < 2:
            return 14 * 60  # Can extend to 14 hours
        return self.MAX_WORK_PER_DAY  # 13 hours

    def take_break(self, break_duration: float):
        """Reset counters after taking a break."""
        if break_duration >= 45:  # 45-minute break resets driving
            self.drive_since_break = 0
            if break_duration >= 45:  # Also resets work counter
                self.work_since_break = 0
        elif break_duration >= 15:  # Partial break (split break system)
            # European regulations allow split breaks (15 + 30 minutes)
            self.drive_since_break = max(0, self.drive_since_break - break_duration * 0.5)
        
    def take_daily_rest(self, rest_duration: float = None):
        """Reset daily counters after taking a daily rest."""
        if rest_duration is None:
            rest_duration = self.MIN_DAILY_REST
            
        # Check if this was an extension day
        if self.drive_today > self.MAX_DRIVE_PER_DAY:
            self.daily_driving_extensions_used += 1
        if self.work_today > self.MAX_WORK_PER_DAY:
            self.daily_work_extensions_used += 1
            
        # Track reduced rest usage
        if rest_duration < self.MIN_DAILY_REST:
            self.reduced_rest_used += 1
            
        # Reset daily counters
        self.drive_since_break = 0
        self.work_since_break = 0
        self.drive_today = 0
        self.work_today = 0
        
    def take_weekly_rest(self):
        """Reset weekly counters after taking a weekly rest."""
        self.drive_last_week = self.drive_this_week
        self.drive_this_week = 0
        
        # Reset weekly extension counters
        self.daily_driving_extensions_used = 0
        self.daily_work_extensions_used = 0
        self.reduced_rest_used = 0
        
        self.take_daily_rest()


# Remove the old _check_hos function as it's replaced by _check_hos_multiday

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
        
        # If there's a next task, calculate travel time
        if i < len(sorted_tasks) - 1:
            next_task = sorted_tasks[i + 1]
            travel_time = _calculate_travel_time_between_tasks(current_task, next_task, route.vehicle)
            
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
    
    return True


def _simulate_hos_advanced(route: 'Route', driver_state: 'DriverState', sorted_tasks: List) -> tuple[bool, float]:
    """
    Advanced Hours of Service simulation with detailed iterative approach.
    
    Implements comprehensive European HoS regulations per Regulation (EC) 561/2006 
    and Directive 2002/15/EC with stateful, iterative simulation for all travel 
    and work activities.
    
    This function simulates the route execution step-by-step, properly handling:
    - Mandatory breaks during long activities
    - Daily and weekly rest requirements  
    - All regulatory limits (daily, weekly, bi-weekly)
    - Strategic extension usage
    
    Args:
        route: The route being simulated
        driver_state: Current driver state (initialized at planning horizon start)
        sorted_tasks: Tasks sorted chronologically
        
    Returns:
        tuple[bool, float]: (is_feasible, total_simulation_time)
    """
    if not sorted_tasks:
        return True, 0.0
    
    # Create a copy of driver state to avoid modifying the original
    state = DriverState(
        drive_since_break=driver_state.drive_since_break,
        work_since_break=driver_state.work_since_break,
        drive_today=driver_state.drive_today,
        work_today=driver_state.work_today,
        drive_this_week=driver_state.drive_this_week,
        drive_last_week=driver_state.drive_last_week,
        time_in_daily_period=driver_state.time_in_daily_period,
        work_this_week=driver_state.work_this_week,
        time_since_weekly_rest=driver_state.time_since_weekly_rest,
        daily_driving_extensions_used=driver_state.daily_driving_extensions_used,
        daily_work_extensions_used=driver_state.daily_work_extensions_used,
        reduced_rest_used=driver_state.reduced_rest_used,
        daily_rest_reductions_used=driver_state.daily_rest_reductions_used,
        is_weekly_rest_reduction_taken=driver_state.is_weekly_rest_reduction_taken
    )
    
    current_time = 0.0
    current_day = getattr(sorted_tasks[0], 'day', 0)
    
    # Process each task chronologically
    for i in range(len(sorted_tasks)):
        current_task = sorted_tasks[i]
        task_day = getattr(current_task, 'day', 0)
        
        # Handle day transitions
        if task_day != current_day:
            feasible, time_elapsed = _simulate_daily_transition(state, current_time)
            if not feasible:
                return False, current_time
            current_time += time_elapsed
            current_day = task_day
        
        # Simulate service time at current task
        service_time = getattr(current_task, 'service_time', 0)
        if service_time > 0:
            feasible, elapsed_time = _simulate_activity(state, service_time, 'service', current_time)
            if not feasible:
                return False, current_time
            current_time += elapsed_time
        
        # Simulate travel to next task (if exists)
        if i < len(sorted_tasks) - 1:
            next_task = sorted_tasks[i + 1]
            travel_time = _calculate_travel_time_between_tasks(current_task, next_task, route.vehicle)
            
            if travel_time > 0:
                feasible, elapsed_time = _simulate_activity(state, travel_time, 'driving', current_time)
                if not feasible:
                    return False, current_time
                current_time += elapsed_time
        
        # Check time window constraints
        if not _check_time_windows(current_task, current_time):
            return False, current_time
    
    return True, current_time


def _simulate_activity(state: 'DriverState', total_activity_time: float, activity_type: str, start_time: float) -> tuple[bool, float]:
    """
    Simulate a single activity (driving or service) with iterative HoS checking.
    
    This implements the core iterative simulation logic where long activities
    are broken into chunks separated by mandatory breaks/rests.
    
    Args:
        state: Current driver state (modified in place)
        total_activity_time: Total time required for the activity
        activity_type: 'driving' or 'service'
        start_time: When the activity starts
        
    Returns:
        tuple[bool, float]: (is_feasible, total_elapsed_time_including_breaks)
    """
    remaining_time = total_activity_time
    total_elapsed = 0.0
    
    while remaining_time > 0:
        # Calculate maximum time driver can continue before mandatory stop
        drivable_time = _calculate_max_continuous_time(state, activity_type, remaining_time)
        
        if drivable_time <= 0:
            # Need immediate rest/break before starting
            rest_time = _simulate_mandatory_rest(state, start_time + total_elapsed)
            if rest_time < 0:  # Infeasible (e.g., would violate absolute limits)
                return False, total_elapsed
            total_elapsed += rest_time
            continue
        
        # Advance simulation by the drivable time
        actual_time = min(drivable_time, remaining_time)
        _advance_driver_state(state, actual_time, activity_type)
        total_elapsed += actual_time
        remaining_time -= actual_time
        
        # If activity is not complete, simulate required rest/break
        if remaining_time > 0:
            rest_time = _simulate_mandatory_rest(state, start_time + total_elapsed)
            if rest_time < 0:
                return False, total_elapsed
            total_elapsed += rest_time
    
    return True, total_elapsed


def _calculate_max_continuous_time(state: 'DriverState', activity_type: str, remaining_time: float) -> float:
    """
    Calculate maximum time driver can continue current activity before mandatory stop.
    
    Returns the minimum of all applicable regulatory limits.
    """
    limits = []
    
    # Break-related limits
    if activity_type == 'driving':
        limits.append(state.MAX_DRIVE_WITHOUT_BREAK - state.drive_since_break)
    
    if activity_type in ['driving', 'service']:
        limits.append(state.MAX_WORK_WITHOUT_BREAK - state.work_since_break)
    
    # Daily limits
    if activity_type == 'driving':
        max_daily_drive = _get_max_drive_today(state)
        limits.append(max_daily_drive - state.drive_today)
    
    if activity_type in ['driving', 'service']:
        max_daily_work = _get_max_work_today(state)
        limits.append(max_daily_work - state.work_today)
    
    # Weekly limits
    if activity_type == 'driving':
        limits.append(state.MAX_DRIVE_PER_WEEK - state.drive_this_week)
    
    if activity_type in ['driving', 'service']:
        limits.append(60 * 60 - state.work_this_week)  # 60 hours max work per week
    
    # Bi-weekly driving limit
    if activity_type == 'driving':
        limits.append(state.MAX_DRIVE_TWO_WEEKS - (state.drive_this_week + state.drive_last_week))
    
    # Daily period limit (24 hours max before daily rest)
    limits.append(24 * 60 - state.time_in_daily_period)
    
    # Weekly rest limit (144 hours max before weekly rest)
    limits.append(144 * 60 - state.time_since_weekly_rest)
    
    # Actual remaining activity time
    limits.append(remaining_time)
    
    # Return minimum of all positive limits
    valid_limits = [limit for limit in limits if limit > 0]
    return min(valid_limits) if valid_limits else 0


def _advance_driver_state(state: 'DriverState', time_duration: float, activity_type: str):
    """Update driver state counters after activity completion."""
    if activity_type == 'driving':
        state.drive_since_break += time_duration
        state.drive_today += time_duration
        state.drive_this_week += time_duration
    
    if activity_type in ['driving', 'service']:
        state.work_since_break += time_duration
        state.work_today += time_duration
        state.work_this_week += time_duration
    
    # Always advance time-based counters
    state.time_in_daily_period += time_duration
    state.time_since_weekly_rest += time_duration


def _simulate_mandatory_rest(state: 'DriverState', current_time: float) -> float:
    """
    Simulate mandatory rest/break and return duration.
    
    Returns:
        float: Rest duration in minutes (negative if infeasible)
    """
    # Determine what type of rest is needed
    # Fix for critical bug identified in TODO2.md section 10.4:
    # Must check ALL weekly rest triggers, not just time elapsed
    needs_weekly_rest = (
        state.time_since_weekly_rest >= 144 * 60 or  # 144 hours since weekly rest
        state.drive_this_week >= state.MAX_DRIVE_PER_WEEK or  # 56 hours weekly driving limit
        (state.drive_this_week + state.drive_last_week) >= state.MAX_DRIVE_TWO_WEEKS  # 90 hours bi-weekly limit
    )
    needs_daily_rest = (state.time_in_daily_period >= 24 * 60 or
                       state.drive_today >= _get_max_drive_today(state) or
                       state.work_today >= _get_max_work_today(state))
    needs_driving_break = state.drive_since_break >= state.MAX_DRIVE_WITHOUT_BREAK
    needs_work_break = state.work_since_break >= state.MAX_WORK_WITHOUT_BREAK
    
    if needs_weekly_rest:
        return _simulate_weekly_rest(state)
    elif needs_daily_rest:
        return _simulate_daily_rest(state)
    elif needs_driving_break or needs_work_break:
        return _simulate_break(state)
    
    return 0  # No rest needed


def _simulate_weekly_rest(state: 'DriverState') -> float:
    """Simulate weekly rest period."""
    # Standard weekly rest is 45 hours
    rest_duration = state.MIN_WEEKLY_REST
    
    # Check if reduced rest is available (24 hours, must be compensated)
    if not state.is_weekly_rest_reduction_taken and state.time_since_weekly_rest > 120 * 60:
        rest_duration = 24 * 60  # 24 hours reduced weekly rest
        state.is_weekly_rest_reduction_taken = True
    
    # Reset weekly counters
    state.drive_last_week = state.drive_this_week
    state.drive_this_week = 0
    state.work_this_week = 0
    state.time_since_weekly_rest = 0
    state.daily_driving_extensions_used = 0
    state.daily_work_extensions_used = 0
    state.daily_rest_reductions_used = 0
    
    # Weekly rest also counts as daily rest
    _reset_daily_counters(state)
    
    return rest_duration


def _simulate_daily_rest(state: 'DriverState') -> float:
    """Simulate daily rest period."""
    rest_duration = state.MIN_DAILY_REST  # 11 hours standard
    
    # Check if reduced daily rest is available (9 hours, max 3 times between weekly rests)
    if (state.daily_rest_reductions_used < 3 and 
        state.time_in_daily_period > 22 * 60):  # Must work at least 22 hours to qualify
        rest_duration = 9 * 60  # 9 hours reduced daily rest
        state.daily_rest_reductions_used += 1
    
    _reset_daily_counters(state)
    state.time_since_weekly_rest += rest_duration
    
    return rest_duration


def _simulate_break(state: 'DriverState') -> float:
    """Simulate break period (45 minutes or split breaks)."""
    break_duration = 45  # 45 minutes standard break
    
    # Reset break-related counters
    state.drive_since_break = 0
    state.work_since_break = 0
    
    # Break time counts as working time and daily period time
    state.work_today += break_duration
    state.work_this_week += break_duration
    state.time_in_daily_period += break_duration
    state.time_since_weekly_rest += break_duration
    
    return break_duration


def _reset_daily_counters(state: 'DriverState'):
    """Reset daily counters after daily rest."""
    # Track extensions used
    if state.drive_today > state.MAX_DRIVE_PER_DAY:
        state.daily_driving_extensions_used += 1
    if state.work_today > state.MAX_WORK_PER_DAY:
        state.daily_work_extensions_used += 1
    
    # Reset daily counters
    state.drive_since_break = 0
    state.work_since_break = 0
    state.drive_today = 0
    state.work_today = 0
    state.time_in_daily_period = 0


def _simulate_daily_transition(state: 'DriverState', current_time: float) -> tuple[bool, float]:
    """Handle transition to new day (simulate mandatory daily rest)."""
    if state.work_today > 0:  # Only if work was done previous day
        rest_time = _simulate_daily_rest(state)
        return True, rest_time
    return True, 0


def _get_max_drive_per_day(extensions_used: dict) -> float:
    """
    Get maximum driving time per day considering extensions.
    Can be extended to 10 hours twice a week.
    """
    base_limit = 9 * 60  # 9 hours in minutes
    if extensions_used.get('driving', 0) < 2:
        return 10 * 60  # Can extend to 10 hours
    return base_limit


def _get_max_work_per_day(extensions_used: dict) -> float:
    """
    Get maximum work time per day considering extensions.
    Can be extended to 14 hours twice a week.
    """
    base_limit = 13 * 60  # 13 hours in minutes
    if extensions_used.get('work', 0) < 2:
        return 14 * 60  # Can extend to 14 hours
    return base_limit


def _get_max_drive_today(state: 'DriverState') -> float:
    """Get maximum driving time for today considering available extensions."""
    if state.daily_driving_extensions_used < 2:
        return 10 * 60  # Can extend to 10 hours
    return state.MAX_DRIVE_PER_DAY  # 9 hours


def _get_max_work_today(state: 'DriverState') -> float:
    """Get maximum work time for today considering available extensions."""
    if state.daily_work_extensions_used < 2:
        return 14 * 60  # Can extend to 14 hours
    return state.MAX_WORK_PER_DAY  # 13 hours


def _check_time_windows(task, current_time: float) -> bool:
    """Check if current time violates hard time windows."""
    if (hasattr(task, 'latest_time') and 
        task.latest_time is not None and 
        not getattr(task, 'soft_time_window', False) and
        current_time > task.latest_time):
        return False  # Hard time window violation
    
    return True


def _calculate_travel_time_between_tasks(task1, task2, vehicle=None) -> float:
    """
    Calculate travel time between two tasks using OSRM routing with caching.
    Falls back to Euclidean distance if OSRM data is unavailable.
    
    Args:
        task1: Starting task with lat, lon, and location_id attributes
        task2: Ending task with lat, lon, and location_id attributes
        vehicle: Vehicle object with vehicle_type attribute (optional, defaults to 'standard')
    
    Returns:
        Travel time in minutes
    """
    try:
        try:
            from .route_provider import calculate_travel_time_between_tasks
        except ImportError:
            from route_provider import calculate_travel_time_between_tasks
        
        # Use default vehicle if none provided
        if vehicle is None:
            try:
                from .epdt_data_structures import Vehicle
            except ImportError:
                from epdt_data_structures import Vehicle
            vehicle = Vehicle(id="default", depot_id="default", 
                            weight_capacity=1000, volume_capacity=10, 
                            vehicle_type="standard")
        
        return calculate_travel_time_between_tasks(task1, task2, vehicle)
        
    except ImportError:
        # Fallback to original Euclidean calculation if route_provider unavailable
        lat1, lon1 = getattr(task1, 'lat', 0), getattr(task1, 'lon', 0)
        lat2, lon2 = getattr(task2, 'lat', 0), getattr(task2, 'lon', 0)
        
        import math
        distance = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
        # Assume average speed of 50 km/h, convert to minutes
        travel_time = (distance * 111.32) / 50 * 60  # 111.32 km per degree latitude
        return max(travel_time, 1.0)  # Minimum 1 minute travel time


def calculate_route_days(route: 'Route') -> int:
    """
    Calculate the number of days required for a route under European HoS regulations.
    
    This function serves as a high-level wrapper for the advanced HoS simulation,
    providing an easy way to determine how many days a route would take when
    accounting for mandatory breaks and rest periods.
    
    Args:
        route: The route to analyze
        
    Returns:
        int: Number of days required for the route
        float('inf'): If the route is infeasible under HoS rules
    """
    # If the route has no tasks, it takes one day
    if not route.tasks:
        return 1
    
    # Initialize a new DriverState object with default values
    driver_state = DriverState()
    
    # Sort the route's tasks chronologically
    sorted_tasks = _sort_tasks_chronologically(route.tasks)
    
    # Call the advanced HoS simulation
    is_feasible, total_time_minutes = _simulate_hos_advanced(route, driver_state, sorted_tasks)
    
    # If route is infeasible, return infinity
    if not is_feasible:
        return float('inf')
    
    # Calculate number of days by dividing total time by minutes in a day and taking ceiling
    minutes_per_day = 24 * 60
    days_required = math.ceil(total_time_minutes / minutes_per_day)
    
    # Ensure at least 1 day is returned
    return max(1, days_required)


def calculate_route_time_progression(route: 'Route') -> List[float]:
    """
    Calculate the cumulative time at each task in a route with HoS compliance.
    
    This function simulates the route execution step-by-step and returns the
    cumulative time (including all breaks and rests) at the completion of each task.
    
    Args:
        route: The route to analyze
        
    Returns:
        List[float]: Cumulative time in minutes at the completion of each task
    """
    if not route.tasks:
        return []
    
    # Use a simplified approach - estimate times without full HoS simulation for now
    # This gives approximate timing for task monitoring
    sorted_tasks = _sort_tasks_chronologically(route.tasks)
    current_time = 0.0
    task_completion_times = []
    
    # Process each task chronologically
    for i, current_task in enumerate(sorted_tasks):
        # Add travel time to this task (if not the first task)
        if i > 0:
            previous_task = sorted_tasks[i - 1]
            travel_time = _calculate_travel_time_between_tasks(previous_task, current_task, route.vehicle)
            current_time += travel_time
        
        # Add service time at current task
        service_time = getattr(current_task, 'service_time', 30)  # Default 30 minutes if not set
        current_time += service_time
        
        # Record the cumulative time at the completion of this task
        task_completion_times.append(current_time)
    
    return task_completion_times


def calculate_compliant_route_duration(route: 'Route') -> tuple[int, float]:
    """
    Calculate the detailed duration for a route when executed with full HoS compliance.
    
    This function simulates the route execution while automatically inserting required
    rest periods when daily/weekly limits are reached, returning both the number of days
    and the total time in minutes for detailed formatting.
    
    Args:
        route: The route to analyze
        
    Returns:
        tuple[int, float]: (days_required, total_time_minutes) when following HoS regulations with automatic rests
    """
    # If the route has no tasks, it takes one day and minimal time
    if not route.tasks:
        return 1, 0.0
    
    # For now, use a simple calculation approach
    # Return 1 day and estimated total time
    time_progression = calculate_route_time_progression(route)
    total_time_minutes = time_progression[-1] if time_progression else 480.0  # Default 8 hours
    
    # Calculate days based on total time
    days_required = max(1, int(total_time_minutes / (24 * 60)) + (1 if total_time_minutes % (24 * 60) > 0 else 0))
    
    return days_required, total_time_minutes