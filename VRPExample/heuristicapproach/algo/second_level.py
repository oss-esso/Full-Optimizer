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
    from route import Route
    from order import Order

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


@jit(nopython=False, forceobj=True)  # Performance optimization
def _calculate_realistic_driver_costs(route: 'Route') -> tuple[float, float]:
    """
    Calculate realistic driver break and rest costs by simulating HOS regulations.
    Returns (break_cost, total_downtime_minutes)
    
    This function is performance-critical and optimized with Numba JIT compilation.
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


@jit(nopython=False, forceobj=True)  # Use object mode for complex Route objects
def calculate_z2_score(route: 'Route') -> float:
    """ 
    Calculates the Z2 score for a given route.
    Uses realistic driver break/rest costs based on HOS regulations simulation.
    
    This is a performance-critical function optimized with Numba JIT compilation.
    """
    # Check if score is already cached
    if hasattr(route, '_z2_score'):
        return route._z2_score
        
    # Initialize cost components
    travel_cost = 0
    time_window_penalty = 0
    tomorrow_task_cost = 0
    vehicle_assignment_penalty = 0
    end_position_penalty = 0

    # Calculate realistic driver break costs
    driver_break_cost, _ = _calculate_realistic_driver_costs(route)

    # Calculate travel time and costs (simplified - detailed HOS handled above)
    current_time = 0
    for i in range(len(route.tasks) - 1):
        current_task = route.tasks[i]
        next_task = route.tasks[i + 1]
        
        # Add service time for current task
        current_time += current_task.service_time
        
        # Calculate travel time to next task
        travel_time = route.calculate_travel_time(current_task, next_task)
        travel_cost += travel_time * route.vehicle.cost_per_hour
        
        # Move to next task's time
        current_time += travel_time
        
        # Check for time window violations
        if hasattr(next_task, 'time_window'):
            if current_time > next_task.time_window.latest:
                time_window_penalty += (current_time - next_task.time_window.latest) * 2
            if current_time < next_task.time_window.earliest:
                current_time = next_task.time_window.earliest

    # Add tomorrow task costs
    tomorrow_tasks = [t for t in route.tasks if hasattr(t, 'is_tomorrow') and t.is_tomorrow]
    tomorrow_task_cost = len(tomorrow_tasks) * 50  # Penalty per tomorrow task

    # Add vehicle assignment penalty if applicable
    if hasattr(route, 'preferred_vehicle') and route.vehicle != route.preferred_vehicle:
        vehicle_assignment_penalty = 100

    # Add end position penalty
    if hasattr(route, 'preferred_end_position'):
        last_task = route.tasks[-1] if route.tasks else None
        if last_task and last_task.location != route.preferred_end_position:
            end_position_penalty = 75

    # Calculate total cost
    total_cost = (travel_cost + time_window_penalty + driver_break_cost + 
                  tomorrow_task_cost + vehicle_assignment_penalty + end_position_penalty)

    # Cache the score
    route._z2_score = total_cost

    return total_cost


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



@jit(nopython=False, forceobj=True)  # Performance optimization for HOS checking
def is_feasible(route: 'Route') -> bool:
    """ 
    Check if the route is feasible according to all constraints.
    
    This function is performance-critical and optimized with Numba JIT compilation.
    """
    
    #H1 Capacity check
    load_w = 0
    load_v = 0

    max_w = route.vehicle.max_weight
    max_v = route.vehicle.max_volume

    for task in route.tasks:
        if task.is_pickup():
            load_w += task.weight
            load_v += task.volume
        elif task.is_delivery():
            load_w -= task.weight
            load_v -= task.volume
        
        if load_w > max_w or load_v > max_v:
            return False

    # H2 Precedence constraints check
    orders = {}
    for i, task in enumerate(route.tasks):
        # Get order identifier (try different possible attribute names)
        order_id = None
        for attr in ['order_id', 'order']:
            if hasattr(task, attr):
                order_id = getattr(task, attr)
                break
        
        if order_id is None:
            continue  # Skip tasks without order information
            
        if order_id not in orders:
            orders[order_id] = {'pickups': [], 'deliveries': []}
        
        if task.is_pickup():
            orders[order_id]['pickups'].append(i)
        elif task.is_delivery():
            orders[order_id]['deliveries'].append(i)

    # Check precedence constraints
    for order_id, tasks in orders.items():
        if tasks['pickups'] and tasks['deliveries']:  # Only check if order has both pickups and deliveries
            last_pickup_idx = max(tasks['pickups'])
            first_delivery_idx = min(tasks['deliveries'])
            
            if last_pickup_idx >= first_delivery_idx:
                return False  # Precedence constraint violated
    
    #H3 Hard time windows check
    for task in route.tasks:
        if hasattr(task, 'hard_time_window'):
            if task.hard_time_window.latest < task.arrival_time:
                return False  # Time window violation
            
    #H6 Driver regulations check
    if not route._check_hos():
        return False
    
    #H7 Route ending position check
    if hasattr(route, 'preferred_end_position'):
        last_task = route.tasks[-1]
        if last_task and last_task.location != route.preferred_end_position:
            return False
        
    return True  # All checks passed, route is feasible

@dataclass
class DriverState:
    """Class to track driver's hours of service state"""
    drive_since_break: float = 0.0  # Accumulated driving time since last break (limit: 4.5 hours)
    work_since_break: float = 0.0   # Accumulated working time since last break (limit: 6 hours)
    drive_today: float = 0.0        # Total driving time in current 24-hour period (limit: 9 hours)
    work_today: float = 0.0         # Total duty time in current 24-hour period (limit: 13 hours)
    drive_this_week: float = 0.0    # Driving time this week
    drive_last_week: float = 0.0    # Driving time last week
    
    # Regulation limits in minutes
    MAX_DRIVE_WITHOUT_BREAK = 4.5 * 60  # 4.5 hours in minutes
    MAX_WORK_WITHOUT_BREAK = 6 * 60     # 6 hours in minutes
    MAX_DRIVE_PER_DAY = 9 * 60          # 9 hours in minutes
    MAX_WORK_PER_DAY = 13 * 60          # 13 hours in minutes
    MAX_DRIVE_TWO_WEEKS = 90 * 60       # 90 hours in minutes

    def take_break(self, break_duration: float):
        """Reset counters after taking a break"""
        if break_duration >= 45:  # 45-minute break
            self.drive_since_break = 0
            self.work_since_break = 0
        
    def take_daily_rest(self):
        """Reset daily counters after taking a daily rest"""
        self.drive_since_break = 0
        self.work_since_break = 0
        self.drive_today = 0
        self.work_today = 0
        
    def take_weekly_rest(self):
        """Reset weekly counters after taking a weekly rest"""
        self.drive_last_week = self.drive_this_week
        self.drive_this_week = 0
        self.take_daily_rest()

@jit(nopython=False, forceobj=True)  # Performance optimization for HOS checking
def _check_hos(route: 'Route') -> bool:
    """
    Check if the route respects Hours of Service (HOS) regulations.
    Returns True if the route is feasible, False otherwise.
    
    This function is performance-critical and optimized with Numba JIT compilation.
    """
    if not route.tasks:
        return True  # Empty route is always feasible
    
    # Initialize driver state and current time
    driver = DriverState()
    current_time = 0
    
    # Process each task in the route
    for i in range(len(route.tasks) - 1):
        current_task = route.tasks[i]
        next_task = route.tasks[i + 1]
        
        # Handle service time at current task
        service_time = current_task.service_time
        driver.work_since_break += service_time
        driver.work_today += service_time
        current_time += service_time
        
        # Calculate travel time to next task
        travel_time = route.calculate_travel_time(current_task, next_task)
        
        # Check if this segment would exceed weekly driving limits
        if driver.drive_this_week + driver.drive_last_week + travel_time > driver.MAX_DRIVE_TWO_WEEKS:
            return False
        
        # Check if we need a break before starting travel
        remaining_drive_time = min(
            driver.MAX_DRIVE_WITHOUT_BREAK - driver.drive_since_break,
            driver.MAX_DRIVE_PER_DAY - driver.drive_today
        )
        
        # If we can't complete the travel segment without a break
        while travel_time > remaining_drive_time:
            # Add driving time until break is needed
            driver.drive_since_break += remaining_drive_time
            driver.work_since_break += remaining_drive_time
            driver.drive_today += remaining_drive_time
            driver.work_today += remaining_drive_time
            driver.drive_this_week += remaining_drive_time
            
            current_time += remaining_drive_time
            travel_time -= remaining_drive_time
            
            # Take a 45-minute break
            current_time += 45
            driver.take_break(45)
            
            # Check if we need a daily rest
            if driver.drive_today >= driver.MAX_DRIVE_PER_DAY or driver.work_today >= driver.MAX_WORK_PER_DAY:
                current_time += 11 * 60  # 11 hours daily rest
                driver.take_daily_rest()
                
                # Check if we need a weekly rest
                if driver.drive_this_week + driver.drive_last_week >= driver.MAX_DRIVE_TWO_WEEKS:
                    current_time += 45 * 60  # 45 hours weekly rest
                    driver.take_weekly_rest()
            
            # Recalculate remaining drive time
            remaining_drive_time = min(
                driver.MAX_DRIVE_WITHOUT_BREAK - driver.drive_since_break,
                driver.MAX_DRIVE_PER_DAY - driver.drive_today
            )
        
        # Complete the remaining travel
        driver.drive_since_break += travel_time
        driver.work_since_break += travel_time
        driver.drive_today += travel_time
        driver.work_today += travel_time
        driver.drive_this_week += travel_time
        current_time += travel_time
        
        # Check time windows
        if hasattr(next_task, 'time_window'):
            if current_time > next_task.time_window.latest:
                return False
            if current_time < next_task.time_window.earliest:
                wait_time = next_task.time_window.earliest - current_time
                current_time = next_task.time_window.earliest
                driver.work_today += wait_time
    
    # Check final task
    last_task = route.tasks[-1]
    service_time = last_task.service_time
    
    if (driver.work_since_break + service_time > driver.MAX_WORK_WITHOUT_BREAK or 
        driver.work_today + service_time > driver.MAX_WORK_PER_DAY):
        return False
    
    return True
