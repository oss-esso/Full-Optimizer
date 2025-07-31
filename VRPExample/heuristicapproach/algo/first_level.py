"""
First-Level Heuristic Implementation for EPDT Algorithm

This module implements the L1 (First-Level) heuristic which handles inter-route 
optimization within the EPDT (Enhanced Parallel Diversified Tabu) algorithm.

Key Components:
- l1_heuristic: Main Tabu Search VND algorithm for solution optimization
- Initial solution construction methods: Best insertion and round-robin approaches  
- Z1 scoring function: Solution-wide evaluation including route costs and penalties
- Neighborhood exploration: Placeholder functions for order relocation and swapping

Architecture Notes:
- Separation of Concerns: L1 handles inter-route optimization, calls L2 for intra-route
- Modular Design: Different initialization strategies and neighborhood functions
- Performance Optimization: Uses Numba JIT compilation for critical scoring functions
- State Management: Proper solution copying and tabu list management
"""

from typing import List, Optional, Iterator, Callable, TYPE_CHECKING
from dataclasses import dataclass
import copy
import collections

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
    from .epdt_data_structures import Route, Order, Vehicle, Solution
else:
    try:
        from .epdt_data_structures import Route, Order, Vehicle, Solution
    except ImportError:
        from epdt_data_structures import Route, Order, Vehicle, Solution

try:
    from second_level import l2_heuristic
except ImportError:
    try:
        from .second_level import l2_heuristic
    except ImportError:
        print("Warning: Could not import l2_heuristic")
        l2_heuristic = None

# Removed problematic circular imports - these modules will be imported locally where needed

def l1_heuristic(orders: List['Order'], vehicles: List['Vehicle'], params: dict) -> 'Solution':
    """
    Main Tabu Search VND algorithm.
    
    This implements the L1 (First-Level) heuristic from the EPDT algorithm,
    which handles inter-route optimization through Tabu Search with Variable
    Neighborhood Descent (VND).
    
    Args:
        orders: List of orders to be assigned
        vehicles: List of available vehicles  
        params: Algorithm parameters including tabu_tenure, M1, M2, exploration_strategy, initialization_method
        
    Returns:
        Optimized solution after Tabu Search
    """
    
    # 1. Create initial solution using selected initialization method
    initialization_method = params.get('initialization_method', 'cluster_aware')
    
    if initialization_method == 'regret_k':
        print("🧠 Using Regret-k initialization strategy")
        initial_solution = regret_k_initializer(orders, vehicles, params)
    else:
        print("🏗️  Using cluster-aware initialization strategy") 
        initial_solution = cluster_aware_initializer(orders, vehicles, params)
    
    # Check for destroy and repair if enabled and there are unassigned orders
    if params.get('enable_destroy_and_repair', False):
        unassigned_count = len(getattr(initial_solution, 'unassigned_orders', set()))
        if unassigned_count > 0:
            print(f"🔧 Applying destroy and repair for {unassigned_count} unassigned orders")
            try:
                from destroy_and_repair import destroy_and_repair_large_orders
                initial_solution = destroy_and_repair_large_orders(initial_solution, orders, vehicles, params)
            except ImportError:
                print("⚠️  Warning: destroy_and_repair module not available")
    
    # Do NOT enforce pickup-first ordering - let the initialization patterns stand
    
    # 2. Initialize state
    best_solution = copy.deepcopy(initial_solution)
    center_solution = copy.deepcopy(initial_solution)
    tabu_list = collections.deque(maxlen=params['tabu_tenure'])
    non_improving_iters = 0
    total_iters = 0

    # Initial tabu list entry to prevent immediate reversal
    if initial_solution.routes and any(route.tasks for route in initial_solution.routes.values()):
        # Add a representation of the initial move to the tabu list
        initial_move = ('initial', total_iters)
        tabu_list.append(initial_move)

    # 3. Main loop
    print(f"🔄 Starting L1 main optimization loop with M1={params['M1']}, M2={params['M2']}")
    while non_improving_iters < params['M1'] and total_iters < params['M2']:
        total_iters += 1
        improvement_found = False
        best_neighbors_pool = []
        
        if total_iters % 10 == 1:  # Print every 10 iterations
            print(f"🔄 L1 Iteration {total_iters}: non_improving={non_improving_iters}, score={calculate_z1_score(center_solution, params, orders):.2f}")
        
        # Intelligent termination with convergence analysis
        import time
        if not hasattr(l1_heuristic, '_start_time'):
            l1_heuristic._start_time = time.time()
            l1_heuristic._score_history = []
            l1_heuristic._last_improvement_time = time.time()
        
        current_time = time.time()
        elapsed_time = current_time - l1_heuristic._start_time
        current_score = calculate_z1_score(center_solution, params, orders)
        
        # Track score progression for convergence analysis
        l1_heuristic._score_history.append(current_score)
        
        # Check for improvement
        if len(l1_heuristic._score_history) >= 2:
            improvement = l1_heuristic._score_history[-2] - current_score
            if improvement > 0.1:  # Meaningful improvement threshold
                l1_heuristic._last_improvement_time = current_time
        
        # Intelligent termination conditions:
        # 1. Convergence: No improvement for 15+ seconds
        time_since_improvement = current_time - l1_heuristic._last_improvement_time
        if time_since_improvement > 15:
            print(f"🎯 Convergence reached: No improvement for {time_since_improvement:.1f}s")
            break
            
        # 2. Quality threshold: Stop if we have good assignment rate and reasonable runtime
        if elapsed_time > 10 and len(l1_heuristic._score_history) >= 5:
            recent_scores = l1_heuristic._score_history[-5:]
            score_variance = max(recent_scores) - min(recent_scores)
            if score_variance < 1.0:  # Very stable scores
                print(f"✅ Solution stabilized: variance={score_variance:.2f}, time={elapsed_time:.1f}s")
                break
        
        # 3. Maximum time limit (60s as safety net, not hard production requirement)
        if elapsed_time > 60:
            print(f"⏰ Maximum time limit reached: {elapsed_time:.1f} seconds")
            break
            
        # Safety check to prevent infinite loops
        if total_iters > 20:  # Further reduced from 100 for production target
            print(f"⚠️  L1 reached 20 iterations, stopping for production performance")
            break

        # 4. VND Loop - Variable Neighborhood Descent
        # Include unassigned order insertion as the first neighborhood (highest priority)
        neighborhoods = [unassigned_order_insertion_neighborhood, single_order_relocation_neighborhood, two_orders_swap_neighborhood]
        if params.get('enable_advanced_neighborhoods', False):
            neighborhoods.extend([multiple_order_relocation_neighborhood, two_opt_routes_neighborhood])
        
        # Import granular search locally to avoid circular imports
        if params.get('enable_granular_search', False):
            try:
                from .granular_tabu_search import granular_multiple_order_relocation_neighborhood, NETWORKX_AVAILABLE
                if NETWORKX_AVAILABLE:
                    neighborhoods.append(granular_multiple_order_relocation_neighborhood)
            except ImportError:
                try:
                    from granular_tabu_search import granular_multiple_order_relocation_neighborhood, NETWORKX_AVAILABLE
                    if NETWORKX_AVAILABLE:
                        neighborhoods.append(granular_multiple_order_relocation_neighborhood)
                except ImportError:
                    if total_iters % 100 == 1:  # Only warn occasionally
                        print("Warning: Could not import granular_tabu_search, skipping granular search")
        
        if total_iters % 50 == 1:  # Less frequent debug for VND
            print(f"🔍 Starting VND with {len(neighborhoods)} neighborhoods")
        
        # Track total neighbors evaluated across all neighborhoods in this VND iteration
        total_neighbors_this_vnd = 0
        max_neighbors_per_iteration = params.get('max_neighbors_per_iteration', 20)
        
        for neighborhood_idx, neighborhood_func in enumerate(neighborhoods):
            if total_iters % 50 == 1:
                print(f"🔍 Exploring neighborhood {neighborhood_idx+1}/{len(neighborhoods)}: {neighborhood_func.__name__}")
            
            # Check if we've already evaluated too many neighbors in this VND iteration
            if total_neighbors_this_vnd >= max_neighbors_per_iteration:
                if total_iters % 50 == 1:
                    print(f"⚠️  VND iteration reached {max_neighbors_per_iteration} neighbors limit, skipping remaining neighborhoods")
                break
            
            # Explore neighborhood, find best valid neighbor
            best_neighbor_in_N = None
            best_neighbor_score = float('-inf')
            neighbors_evaluated = 0
            
            for neighbor in neighborhood_func(center_solution, orders, vehicles):
                neighbors_evaluated += 1
                total_neighbors_this_vnd += 1
                
                # Use configurable neighbor limit for performance optimization
                max_neighbors = params.get('max_neighbors_to_evaluate', 15)
                if neighbors_evaluated > max_neighbors:
                    if total_iters % 50 == 1:  # Only print occasionally to reduce log noise
                        print(f"⚠️  Neighborhood {neighborhood_func.__name__} evaluated {max_neighbors}+ neighbors, breaking for sub-30s target")
                    break
                
                # Also check global VND iteration limit
                if total_neighbors_this_vnd >= max_neighbors_per_iteration:
                    if total_iters % 50 == 1:
                        print(f"⚠️  VND iteration reached {max_neighbors_per_iteration} total neighbors, breaking")
                    break
                
                neighbor_score = calculate_z1_score(neighbor, params, orders)
                
                # Check if move is tabu
                move_attrs = get_move_attributes(center_solution, neighbor)
                is_tabu = move_attrs in tabu_list
                
                # Apply aspiration criteria - allow tabu moves if they lead to best solution so far
                aspiration = neighbor_score > calculate_z1_score(best_solution, params, orders)
                
                if (not is_tabu or aspiration) and (best_neighbor_in_N is None or neighbor_score > best_neighbor_score):
                    best_neighbor_in_N = neighbor
                    best_neighbor_score = neighbor_score
                    
                    # First improvement strategy: if we found an improving move, stop searching this neighborhood
                    current_score = calculate_z1_score(center_solution, params, orders)
                    if params.get('local_search_strategy', 'best_improvement') == 'first_improvement' and neighbor_score > current_score:
                        if total_iters % 50 == 1:
                            print(f"⚠️  First improvement found, stopping neighborhood search early")
                        break
                        
                elif not is_tabu:
                    # Add to pool for diversification if not tabu but not improving
                    best_neighbors_pool.append((neighbor, neighbor_score))
            
            # Check for improvement
            if best_neighbor_in_N and best_neighbor_score > calculate_z1_score(center_solution, params, orders):
                # Update tabu list with move attributes
                move_attrs = get_move_attributes(center_solution, best_neighbor_in_N)
                tabu_list.append(move_attrs)
                
                center_solution = best_neighbor_in_N
                non_improving_iters = 0
                improvement_found = True
                
                # Update global best if needed
                if best_neighbor_score > calculate_z1_score(best_solution, params, orders):
                    best_solution = copy.deepcopy(center_solution)
                    # Allow flexible task ordering - do not enforce pickup-first ordering here
                    # for vehicle_id, route in best_solution.routes.items():
                    #     route.ensure_pickup_first_ordering()
                
                break # Go back to the first neighborhood (VND restart)
        
        if total_iters % 50 == 1:
            print(f"🔍 VND completed. improvement_found={improvement_found}, neighbors_pool_size={len(best_neighbors_pool)}")
        
        # 5. Diversification / Non-improving move
        if not improvement_found:
            non_improving_iters += 1
            
            # Select a non-improving move from the pool
            if best_neighbors_pool:
                # Store the current center solution before making the move
                previous_center = center_solution
                
                # Sort by score for deterministic selection or use weighted random
                if params.get('exploration_strategy', 'deterministic') == 'deterministic':
                    best_neighbors_pool.sort(key=lambda x: x[1], reverse=True)
                    selected_neighbor = best_neighbors_pool[0][0]
                else:  # Random selection
                    import random
                    # Weight by score - higher scores have higher probability
                    weights = [score - min(s for _, s in best_neighbors_pool) + 1 
                              for _, score in best_neighbors_pool]
                    total_weight = sum(weights)
                    weights = [w/total_weight for w in weights]
                    selected_idx = random.choices(range(len(best_neighbors_pool)), weights=weights, k=1)[0]
                    selected_neighbor = best_neighbors_pool[selected_idx][0]
                
                # Update center solution
                center_solution = selected_neighbor
                
                # Allow flexible task ordering during search
                # for vehicle_id, route in center_solution.routes.items():
                #     route.ensure_pickup_first_ordering()
                
                # Add move attributes to tabu list (from previous center to new center)
                move_attrs = get_move_attributes(previous_center, center_solution)
                tabu_list.append(move_attrs)

    print(f"🏁 L1 optimization completed after {total_iters} iterations")
    print(f"🏁 Final score: {calculate_z1_score(best_solution, params, orders):.2f}")
    
    # Final enforcement of pickup-first ordering only if explicitly requested
    # Allow flexible task ordering by default for better efficiency
    enforce_final_ordering = params.get('enforce_pickup_first_ordering', False) if params else False
    if enforce_final_ordering:
        for vehicle_id, route in best_solution.routes.items():
            route.ensure_pickup_first_ordering()
    
    # Final check: Ensure all active routes have depot start/end tasks
    depot_location_id = "DEPOT-ASTI"
    depot_lat, depot_lon = 44.9009, 8.2057
    
    from epdt_data_structures import Task, TaskType
    
    for vehicle_id, route in best_solution.routes.items():
        if route.tasks:  # Only for routes that are not empty
            # Check for depot start
            if not route.tasks[0].is_depot_start():
                start_task = Task(
                    id=f"depot_start_order_{vehicle_id}",
                    location_id=depot_location_id,
                    task_type=TaskType.DEPOT_START,
                    order_id=f"depot_start_order_{vehicle_id}",
                    lat=depot_lat,
                    lon=depot_lon,
                    service_time=5.0,
                    demand=0.0,
                    volume=0.0
                )
                route.tasks.insert(0, start_task)
            
            # Check for depot return
            if not route.tasks[-1].is_depot_return():
                return_task = Task(
                    id=f"depot_return_order_{vehicle_id}",
                    location_id=depot_location_id,
                    task_type=TaskType.DEPOT_RETURN,
                    order_id=f"depot_return_order_{vehicle_id}",
                    lat=depot_lat,
                    lon=depot_lon,
                    service_time=5.0,
                    demand=0.0,
                    volume=0.0
                )
                route.tasks.append(return_task)
    
    # FINAL VALIDATION: Strictly enforce HoS compliance on all routes
    print("🔍 Performing final HoS validation on all routes...")
    final_solution = _validate_and_filter_solution(best_solution)
    
    return final_solution


def _validate_and_filter_solution(solution: 'Solution') -> 'Solution':
    """
    Perform final strict validation on all routes and remove any that violate HoS constraints.
    
    This ensures that the final solution only contains routes that pass strict HoS validation,
    regardless of whether they were created during initialization (when checks are bypassed).
    """
    try:
        from second_level import is_feasible
    except ImportError:
        try:
            from .second_level import is_feasible
        except ImportError:
            print("Warning: Could not import is_feasible function for final validation")
            return solution
    
    validated_routes = []
    removed_routes = 0
    
    print(f"Validating {len(solution.routes)} routes...")
    
    for i, route in enumerate(solution.routes):
        print(f"Route {i}: type={type(route)}, value={route}")
        if hasattr(route, 'tasks') and route.tasks:  # Only validate non-empty routes with tasks
            # Force strict HoS validation (no initialization bypass)
            if is_feasible(route, debug_feasibility=False, return_reason=False):
                validated_routes.append(route)
            else:
                removed_routes += 1
                print(f"⚠️  Route for vehicle {getattr(route.vehicle, 'id', 'unknown')} removed due to HoS violation")
                
                # Add orders from removed route back to unassigned
                for task in route.tasks:
                    if hasattr(task, 'order_id') and task.order_id:
                        if not hasattr(solution, 'unassigned_orders'):
                            solution.unassigned_orders = set()
                        solution.unassigned_orders.add(task.order_id)
        else:
            print(f"Skipping route {i} - no tasks or invalid type")
            validated_routes.append(route)  # Keep empty routes
    
    # Update solution with validated routes
    solution.routes = validated_routes
    
    if removed_routes > 0:
        print(f"🔧 Final validation complete: {removed_routes} routes removed due to HoS violations")
    else:
        print("✅ Final validation complete: All routes pass HoS constraints")
    
    return solution


def get_move_attributes(from_solution: 'Solution', to_solution: 'Solution') -> tuple:
    """
    Extract the attributes of a move between two solutions for tabu list tracking.
    
    For a relocation (1R): returns (order_id, source_vehicle_id, destination_vehicle_id)
    For a swap (2S): returns (order1_id, route1_id, order2_id, route2_id)
    
    Args:
        from_solution: Source solution
        to_solution: Destination solution
        
    Returns:
        Tuple representing the move attributes for tabu list management
    """
    if not from_solution or not to_solution or not from_solution.routes or not to_solution.routes:
        return ('invalid_move',)
    
    # Analyze differences between solutions to determine move type and attributes
    from_assignments = _get_order_vehicle_assignments(from_solution)
    to_assignments = _get_order_vehicle_assignments(to_solution)
    
    # Find orders that changed vehicles
    changed_orders = []
    for order_id in set(from_assignments.keys()) | set(to_assignments.keys()):
        from_vehicle = from_assignments.get(order_id, None)
        to_vehicle = to_assignments.get(order_id, None)
        if from_vehicle != to_vehicle:
            changed_orders.append((order_id, from_vehicle, to_vehicle))
    
    if len(changed_orders) == 1:
        # Single order relocation (1R)
        order_id, from_vehicle, to_vehicle = changed_orders[0]
        return ('relocation', order_id, from_vehicle, to_vehicle)
    elif len(changed_orders) == 2:
        # Two orders swap (2S)
        order1_id, from_vehicle1, to_vehicle1 = changed_orders[0]
        order2_id, from_vehicle2, to_vehicle2 = changed_orders[1]
        # Normalize the order for consistent tabu representation
        if order1_id < order2_id:
            return ('swap', order1_id, from_vehicle1, order2_id, from_vehicle2)
        else:
            return ('swap', order2_id, from_vehicle2, order1_id, from_vehicle1)
    elif len(changed_orders) > 2:
        # Multiple order relocation (mR) or complex move
        order_ids = [order_id for order_id, _, _ in changed_orders]
        source_vehicles = [from_vehicle for _, from_vehicle, _ in changed_orders]
        return ('multi_relocation', tuple(sorted(order_ids)), tuple(sorted(source_vehicles)))
    else:
        # No changes detected (shouldn't happen in normal operation)
        return ('no_change',)


def _get_order_vehicle_assignments(solution: 'Solution') -> dict:
    """
    Extract order-to-vehicle assignments from a solution.
    
    Args:
        solution: Solution to analyze
        
    Returns:
        Dictionary mapping order_id to vehicle_id
    """
    assignments = {}
    
    for vehicle_id, route in solution.routes.items():
        if route and route.tasks:
            for task in route.tasks:
                order_id = getattr(task, 'order_id', None)
                if order_id is not None:
                    assignments[order_id] = vehicle_id
    
    return assignments


def best_insertion_initializer(orders: List['Order'], vehicles: List['Vehicle'], params: dict = None) -> 'Solution':
    """
    Initial solution using best insertion heuristic with support for open routes.
    
    This method implements a greedy best insertion approach where for each unassigned 
    order, we try inserting it into each vehicle's route using the L2 heuristic and 
    select the move that results in the highest Z1 score improvement.
    
    Enhanced to handle vehicle initial states and yesterday's tasks.
    
    Args:
        orders: List of orders to be assigned
        vehicles: List of available vehicles
        params: Algorithm parameters
        
    Returns:
        Initial solution with all orders assigned to vehicles
    """
    print(f"🏗️  Starting best insertion initializer with {len(orders)} orders and {len(vehicles)} vehicles")
    
    try:
        from .epdt_data_structures import Solution, Route  # Import here to avoid circular imports
    except ImportError:
        from epdt_data_structures import Solution, Route  # Fallback import
    
    # Initialize empty solution
    solution = Solution()
    
    # Get debug setting from params
    debug_assignment = params.get('debug_assignment', False) if params else False
    
    # Initialize routes with vehicle initial states and yesterday's tasks
    for vehicle in vehicles:
        # Create initial route for this vehicle
        initial_route = Route(vehicle=vehicle)
        
        # Handle vehicle initial state and open routes
        if hasattr(vehicle, 'initial_state') and vehicle.initial_state:
            initial_state = vehicle.initial_state
            
            # Extract pending tasks from previous day
            pending_tasks = initial_state.get('pending_tasks', [])
            yesterday_tasks = initial_state.get('yesterday_tasks', [])
            
            # Add yesterday's tasks first (these are fixed)
            for task in yesterday_tasks:
                # Mark task as from yesterday
                task.day = -1
                task.is_fixed = True
                initial_route.add_task(task)
            
            # Add any pending tasks
            for task in pending_tasks:
                # These might be today's tasks that were already planned
                if not hasattr(task, 'day'):
                    task.day = 0
                task.is_fixed = True
                initial_route.add_task(task)
        
        solution.add_route(vehicle.id, initial_route)
    
    # Filter orders to only include new orders (not already in routes)
    new_orders = []
    existing_order_ids = set()
    
    # Collect order IDs that are already in routes (from initial states)
    for route in solution.routes.values():
        for task in route.tasks:
            if hasattr(task, 'order_id'):
                existing_order_ids.add(task.order_id)
    
    # Only process orders that aren't already assigned
    for order in orders:
        if order.id not in existing_order_ids:
            new_orders.append(order)
    
    # Keep track of unassigned new orders
    unassigned_orders = new_orders.copy()
    
    # Greedy insertion loop for new orders only
    assignment_attempts = 0
    max_assignment_attempts = len(new_orders) * len(vehicles) * 2  # Safety limit
    
    while unassigned_orders and assignment_attempts < max_assignment_attempts:
        assignment_attempts += 1
        best_move = None
        best_score_improvement = float('-inf')
        best_new_route = None
        best_order = None
        best_vehicle_idx = None
        
        # Try inserting each unassigned order into each vehicle
        for order in unassigned_orders:
            if debug_assignment:
                print(f"  DEBUG L1: Trying to assign order {order.id}")
            
            # Performance optimization: use "best k insertions" strategy
            best_k = params.get('best_k_insertions', 5) if params else 5
            
            # Calculate Euclidean distances to all vehicles as a cheap proxy for insertion cost
            vehicle_distances = []
            for vehicle_idx, vehicle in enumerate(vehicles):
                # Get order's first pickup location for distance calculation
                pickup_tasks = order.get_pickups()
                if pickup_tasks:
                    pickup_lat = pickup_tasks[0].lat
                    pickup_lon = pickup_tasks[0].lon
                    
                    # Calculate distance to vehicle's depot or current location
                    vehicle_lat = getattr(vehicle, 'depot_lat', 44.9009)  # Default Asti coordinates (Via del Lavoro 38)
                    vehicle_lon = getattr(vehicle, 'depot_lon', 8.2057)
                    
                    # Simple Euclidean distance calculation
                    import math
                    distance = math.sqrt((pickup_lat - vehicle_lat)**2 + (pickup_lon - vehicle_lon)**2)
                    vehicle_distances.append((distance, vehicle_idx, vehicle))
                else:
                    # If no pickup tasks, add with maximum distance (low priority)
                    vehicle_distances.append((float('inf'), vehicle_idx, vehicle))
            
            # Sort by distance and only evaluate the best k vehicles
            vehicle_distances.sort(key=lambda x: x[0])
            candidate_vehicles = vehicle_distances[:best_k]
            
            if debug_assignment and len(candidate_vehicles) < len(vehicles):
                print(f"    DEBUG L1: Using best-{best_k} strategy, evaluating {len(candidate_vehicles)}/{len(vehicles)} vehicles")
            
            for distance, vehicle_idx, vehicle in candidate_vehicles:
                current_route = solution.routes.get(vehicle.id)
                
                # Create empty route if it doesn't exist
                if current_route is None:
                    current_route = Route(vehicle=vehicle)
                
                if debug_assignment:
                    print(f"    DEBUG L1: Trying vehicle {vehicle.id} (current route has {len(current_route.tasks)} tasks)")
                
                # Use L2 heuristic to find best way to insert order into route
                new_route = l2_heuristic(current_route, order, debug_assignment)
                
                if debug_assignment:
                    if new_route is not None:
                        print(f"    DEBUG L1: L2 succeeded for {vehicle.id}, new route has {len(new_route.tasks)} tasks")
                    else:
                        print(f"    DEBUG L1: L2 failed for {vehicle.id}")
                
                if new_route is not None:  # Feasible insertion
                    # Calculate Z1 score improvement
                    temp_solution = solution.copy()
                    temp_solution.add_route(vehicle.id, new_route)
                    
                    score_improvement = calculate_z1_score(temp_solution, params, orders) - calculate_z1_score(solution, params, orders)
                    
                    if debug_assignment:
                        print(f"    DEBUG L1: Score improvement for {vehicle.id}: {score_improvement}")
                    
                    if score_improvement > best_score_improvement:
                        best_score_improvement = score_improvement
                        best_move = (order, vehicle.id)
                        best_new_route = new_route
                        best_order = order
                        best_vehicle_idx = vehicle.id
        
        # Perform the best move found
        if best_move is not None:
            solution.add_route(best_vehicle_idx, best_new_route)
            unassigned_orders.remove(best_order)
        else:
            # No feasible insertion found - add to unassigned orders
            if hasattr(solution, 'unassigned_orders'):
                solution.unassigned_orders.add(unassigned_orders[0].id)
            else:
                solution.unassigned_orders = {unassigned_orders[0].id}
            print(f"Warning: Could not assign order {unassigned_orders[0].id} to any vehicle")
            unassigned_orders.pop(0)  # Remove the problematic order
    
    # Safety check for hanging prevention
    if assignment_attempts >= max_assignment_attempts:
        print(f"Warning: Assignment loop reached maximum attempts ({max_assignment_attempts}). Stopping to prevent hanging.")
        for remaining_order in unassigned_orders:
            if hasattr(solution, 'unassigned_orders'):
                solution.unassigned_orders.add(remaining_order.id)
            else:
                solution.unassigned_orders = {remaining_order.id}
    
    return solution


def round_robin_insertion_with_priority_initializer(orders: List['Order'], vehicles: List['Vehicle'], params: dict = None) -> 'Solution':
    """
    Initial solution using round-robin insertion with vehicle priority classes and open route support.
    
    This method groups vehicles into three priority classes as defined in the paper,
    sorts orders based on proximity criterion, and assigns orders in round-robin
    fashion within each vehicle class.
    
    Enhanced to handle vehicle initial states and yesterday's tasks.
    
    Args:
        orders: List of orders to be assigned  
        vehicles: List of available vehicles
        params: Algorithm parameters
        
    Returns:
        Initial solution with orders assigned using round-robin approach
    """
    try:
        from .epdt_data_structures import Solution, Route  # Import here to avoid circular imports
    except ImportError:
        from epdt_data_structures import Solution, Route  # Fallback import
    
    # Initialize solution with vehicle initial states
    solution = Solution()
    
    # Initialize routes with vehicle initial states and yesterday's tasks
    for vehicle in vehicles:
        # Create initial route for this vehicle
        initial_route = Route(vehicle=vehicle)
        
        # Handle vehicle initial state and open routes
        if hasattr(vehicle, 'initial_state') and vehicle.initial_state:
            initial_state = vehicle.initial_state
            
            # Extract pending tasks from previous day
            pending_tasks = initial_state.get('pending_tasks', [])
            yesterday_tasks = initial_state.get('yesterday_tasks', [])
            
            # Add yesterday's tasks first (these are fixed)
            for task in yesterday_tasks:
                # Mark task as from yesterday
                task.day = -1
                task.is_fixed = True
                initial_route.add_task(task)
            
            # Add any pending tasks
            for task in pending_tasks:
                # These might be today's tasks that were already planned
                if not hasattr(task, 'day'):
                    task.day = 0
                task.is_fixed = True
                initial_route.add_task(task)
        
        solution.add_route(vehicle.id, initial_route)
    
    # Filter orders to only include new orders (not already in routes)
    new_orders = []
    existing_order_ids = set()
    
    # Collect order IDs that are already in routes (from initial states)
    for route in solution.routes.values():
        for task in route.tasks:
            if hasattr(task, 'order_id'):
                existing_order_ids.add(task.order_id)
    
    # Only process orders that aren't already assigned
    for order in orders:
        if order.id not in existing_order_ids:
            new_orders.append(order)
    
    # Group vehicles into classes (assuming vehicles have a 'priority_class' attribute)
    # Class 1: High priority vehicles (e.g., dedicated/specialized vehicles)
    # Class 2: Medium priority vehicles (e.g., regular fleet)
    # Class 3: Low priority vehicles (e.g., backup/overflow vehicles)
    vehicle_classes = {1: [], 2: [], 3: []}
    
    for idx, vehicle in enumerate(vehicles):
        vehicle_class = getattr(vehicle, 'priority_class', 2)  # Default to class 2
        vehicle_classes[vehicle_class].append((vehicle.id, vehicle))
    
    # Sort new orders based on proximity criterion
    sorted_orders = _sort_orders_by_proximity(new_orders)
    
    # Round-robin assignment within each vehicle class for new orders only
    unassigned_orders = sorted_orders.copy()
    
    # Process vehicle classes in priority order (1, 2, 3)
    for class_num in [1, 2, 3]:
        if not vehicle_classes[class_num] or not unassigned_orders:
            continue
            
        class_vehicles = vehicle_classes[class_num]
        vehicle_idx = 0  # Round-robin index
        
        # Assign orders to this vehicle class until we run out of orders or capacity
        orders_assigned_to_class = 0
        max_orders_per_class = len(unassigned_orders) // len([c for c in vehicle_classes.values() if c])
        
        while unassigned_orders and orders_assigned_to_class < max_orders_per_class:
            order = unassigned_orders[0]
            current_vehicle_id, current_vehicle = class_vehicles[vehicle_idx % len(class_vehicles)]
            
            # Get current route for this vehicle
            current_route = solution.routes.get(current_vehicle_id)
            
            # Create empty route if it doesn't exist
            if current_route is None:
                current_route = Route(vehicle=current_vehicle)
            
            # Try to insert order using L2 heuristic
            new_route = l2_heuristic(current_route, order)
            
            if new_route is not None:  # Feasible insertion
                solution.add_route(current_vehicle_id, new_route)
                unassigned_orders.remove(order)
                orders_assigned_to_class += 1
            else:
                # If insertion failed, add to unassigned and try next vehicle
                if hasattr(solution, 'unassigned_orders'):
                    solution.unassigned_orders.add(order.id)
                else:
                    solution.unassigned_orders = {order.id}
                unassigned_orders.remove(order)
            
            vehicle_idx += 1  # Move to next vehicle in round-robin
    
    # Handle any remaining unassigned orders with best insertion
    if unassigned_orders:
        print(f"Warning: {len(unassigned_orders)} orders remain unassigned after round-robin")
        for order in unassigned_orders:
            if hasattr(solution, 'unassigned_orders'):
                solution.unassigned_orders.add(order.id)
            else:
                solution.unassigned_orders = {order.id}
    
    return solution


def _sort_orders_by_proximity(orders: List['Order']) -> List['Order']:
    """
    Sort orders based on proximity criterion.
    
    This can be based on various factors:
    - Distance from depot
    - Time window urgency
    - Order priority
    - Geographic clustering
    
    Args:
        orders: List of orders to sort
        
    Returns:
        Sorted list of orders
    """
    # Simple implementation: sort by earliest time window if available,
    # otherwise by order priority or ID
    def proximity_key(order):
        if hasattr(order, 'time_window') and order.time_window:
            return order.time_window.earliest
        elif hasattr(order, 'priority'):
            return -order.priority  # Higher priority first (negative for reverse)
        else:
            return getattr(order, 'id', 0)
    
    return sorted(orders, key=proximity_key)


def _merge_routes(route1: 'Route', route2: 'Route') -> 'Route':
    """
    Merge two routes into a single route using sophisticated optimization.
    
    This improved merge strategy:
    1. Extracts all orders from both routes
    2. Creates an empty route with the vehicle from route1
    3. Re-inserts all orders using the L2 heuristic for optimal placement
    
    Args:
        route1: First route (will provide the vehicle for the merged route)
        route2: Second route (orders will be merged into route1's vehicle)
        
    Returns:
        Optimized merged route
    """
    if not route1 and not route2:
        return None
    
    if not route1:
        return route2.copy()
    
    if not route2:
        return route1.copy()
    
    # Extract all orders from both routes
    orders_to_merge = []
    
    # Get orders from route1
    route1_orders = set()
    for task in route1.tasks:
        order_id = getattr(task, 'order_id', None)
        if order_id is not None:
            route1_orders.add(order_id)
    
    # Get orders from route2
    route2_orders = set()
    for task in route2.tasks:
        order_id = getattr(task, 'order_id', None)
        if order_id is not None:
            route2_orders.add(order_id)
    
    all_order_ids = route1_orders | route2_orders
    
    # Retrieve order objects (assuming we have access to the solution's orders)
    # In practice, this would need to be passed as a parameter or retrieved from context
    orders_to_merge = []
    
    # Fallback strategy 1: Use best insertion with L2 optimization
    # Start with route1 as base and progressively add orders from route2
    merged_route = route1.copy()
    
    # Try to insert each order from route2 into the merged route
    for task in route2.tasks:
        order_id = getattr(task, 'order_id', None)
        if order_id is not None and order_id not in route1_orders:
            # This is a simplified approach - in a full implementation,
            # we would reconstruct the order object and use L2 heuristic
            
            # For now, use a simple insertion strategy
            best_position = _find_best_insertion_position(merged_route, task)
            if best_position is not None:
                merged_route.insert_task(best_position, task)
    
    # Fallback strategy 2: If simple insertion doesn't work well,
    # apply local 2-opt optimization to the merged route
    optimized_routes = _apply_2opt_to_route(merged_route)
    if optimized_routes:
        # Return the best optimized route
        from second_level import calculate_z2_score
        best_route = max(optimized_routes, key=lambda r: calculate_z2_score(r))
        return best_route
    
    return merged_route


def _find_best_insertion_position(route: 'Route', task_to_insert) -> Optional[int]:
    """
    Find the best position to insert a task into a route.
    
    This uses a simple cost-based heuristic to find the position
    that minimizes the increase in route cost.
    
    Args:
        route: Route to insert task into
        task_to_insert: Task to find insertion position for
        
    Returns:
        Best insertion position (index), or None if no valid position found
    """
    if not route or not route.tasks:
        return 0  # Insert at beginning if route is empty
    
    best_position = None
    best_cost_increase = float('inf')
    
    # Try inserting at each possible position
    for position in range(len(route.tasks) + 1):
        # Calculate cost increase for inserting at this position
        cost_increase = _calculate_insertion_cost(route, task_to_insert, position)
        
        # Check if insertion at this position is feasible
        if cost_increase < float('inf') and cost_increase < best_cost_increase:
            best_cost_increase = cost_increase
            best_position = position
    
    return best_position


def _calculate_insertion_cost(route: 'Route', task_to_insert, position: int) -> float:
    """
    Calculate the cost increase of inserting a task at a specific position.
    
    This is a simplified cost calculation - in practice this would
    consider distance, time windows, capacity constraints, etc.
    
    Args:
        route: Route to insert into
        task_to_insert: Task to insert
        position: Position to insert at
        
    Returns:
        Cost increase (lower is better), float('inf') if infeasible
    """
    # Simplified cost calculation
    # In practice, this would be much more sophisticated
    
    if not route.tasks:
        return 0.0  # No cost to insert into empty route
    
    # Basic feasibility check - task type and order constraints
    if not _is_feasible_insertion(route, task_to_insert, position):
        return float('inf')
    
    # Simple distance-based cost (placeholder)
    # In reality, this would calculate actual travel time/distance changes
    base_cost = 10.0  # Base insertion cost
    
    # Add penalty for position relative to optimal sequence
    if hasattr(task_to_insert, 'preferred_position'):
        position_penalty = abs(position - task_to_insert.preferred_position) * 2.0
        base_cost += position_penalty
    
    return base_cost


def _is_feasible_insertion(route: 'Route', task_to_insert, position: int) -> bool:
    """
    Check if inserting a task at a position maintains route feasibility.
    
    This checks pickup/delivery order constraints and other basic feasibility.
    
    Args:
        route: Route to insert into
        task_to_insert: Task to insert
        position: Position to insert at
        
    Returns:
        True if insertion is feasible, False otherwise
    """
    # Create a temporary route with the task inserted
    temp_tasks = route.tasks.copy()
    temp_tasks.insert(position, task_to_insert)
    
    # Create temporary route object
    temp_route = route.copy()
    temp_route.tasks = temp_tasks
    # Allow flexible ordering - do not enforce pickup-first automatically
    # temp_route.ensure_pickup_first_ordering()  # Commented out for efficiency
    
    # Check if the new task order is valid
    return _is_valid_route_order(temp_route)


# Placeholder implementations for neighborhood functions and utilities
# These will be implemented in future tasks

def unassigned_order_insertion_neighborhood(solution: 'Solution', orders: List['Order'], vehicles: List['Vehicle']) -> Iterator['Solution']:
    """
    Generate neighborhood by attempting to insert unassigned orders into existing routes or idle vehicles.
    
    This is crucial for the L1 heuristic to recover from initialization failures.
    It tries to assign orders that failed during the cluster-aware initializer phase.
    
    Args:
        solution: Current solution to generate neighbors from
        orders: All orders (including unassigned ones)
        
    Yields:
        New solutions with previously unassigned orders inserted
    """
    if not solution:
        return
    
    # Find unassigned orders
    assigned_order_ids = set()
    for route in solution.routes.values():
        if route and route.tasks:
            for task in route.tasks:
                order_id = getattr(task, 'order_id', None)
                if order_id:
                    assigned_order_ids.add(order_id)
    
    unassigned_orders = [order for order in orders if order.id not in assigned_order_ids]
    
    if not unassigned_orders:
        return  # No unassigned orders to insert
    
    # Strategy 1: Try inserting into existing routes
    for unassigned_order in unassigned_orders:
        for vehicle_id, route in solution.routes.items():
            if not route:
                continue
                
            # Try L2 insertion into existing route
            from second_level import l2_heuristic
            new_route = l2_heuristic(route, unassigned_order, debug_assignment=False)
            
            if new_route is not None:
                # Create new solution with the updated route
                new_solution = solution.copy()
                new_solution.routes[vehicle_id] = new_route
                
                # Remove from unassigned if solution tracks them
                if hasattr(new_solution, 'unassigned_orders') and unassigned_order.id in new_solution.unassigned_orders:
                    new_solution.unassigned_orders.remove(unassigned_order.id)
                
                yield new_solution
    
    # Strategy 2: Try assigning to idle vehicles
    used_vehicle_ids = set(solution.routes.keys())
    idle_vehicles = [v for v in vehicles if v.id not in used_vehicle_ids]
    
    print(f"🔍 Found {len(idle_vehicles)} idle vehicles for unassigned order insertion")
    
    for unassigned_order in unassigned_orders:
        for idle_vehicle in idle_vehicles:
            # Create a new empty route for the idle vehicle
            from epdt_data_structures import Route
            empty_route = Route(vehicle=idle_vehicle)
            
            # Try L2 insertion into the empty route
            from second_level import l2_heuristic
            new_route = l2_heuristic(empty_route, unassigned_order, debug_assignment=False)
            
            if new_route is not None and new_route.tasks:
                # Create new solution with the new route for the idle vehicle
                new_solution = solution.copy()
                new_solution.routes[idle_vehicle.id] = new_route
                
                # Remove from unassigned if solution tracks them
                if hasattr(new_solution, 'unassigned_orders') and unassigned_order.id in new_solution.unassigned_orders:
                    new_solution.unassigned_orders.remove(unassigned_order.id)
                
                print(f"✅ Successfully assigned unassigned order {unassigned_order.id} to idle vehicle {idle_vehicle.id}")
                yield new_solution

    return
def single_order_relocation_neighborhood(solution: 'Solution', orders: List['Order'], vehicles: List['Vehicle']) -> Iterator['Solution']:
    """
    Generate neighborhood by relocating a single order from one route to another.
    
    This implements the 1R (single order relocation) neighborhood operation from the EPDT algorithm.
    For each order in each route, try moving it to every other route.
    
    Args:
        solution: Current solution to generate neighbors from
        
    Yields:
        New solutions with orders relocated between routes
    """
    if not solution or not solution.routes:
        return
    
    # For each source vehicle route
    for from_vehicle_id, from_route in solution.routes.items():
        if not from_route or not from_route.tasks:
            continue
            
        # Group tasks by order
        orders_in_route = {}
        for task in from_route.tasks:
            # Get order identifier (assuming tasks have an order_id attribute)
            order_id = getattr(task, 'order_id', None)
            if order_id is None:
                continue
                
            if order_id not in orders_in_route:
                orders_in_route[order_id] = []
            orders_in_route[order_id].append(task)
        
        # For each order in the route
        for order_id, tasks in orders_in_route.items():
            # For each destination vehicle route (different from source)
            for to_vehicle_id, to_route in solution.routes.items():
                if from_vehicle_id == to_vehicle_id:
                    continue  # Skip same route
                
                # Create a new solution with the order relocated
                new_solution = copy.deepcopy(solution)
                
                # Remove tasks from source route
                source_route = new_solution.routes[from_vehicle_id]
                for task in tasks:
                    # Find the task position and remove it
                    try:
                        position = source_route.tasks.index(task)
                        source_route.remove_task(position)
                    except (ValueError, IndexError):
                        # Fallback: direct removal from list
                        if task in source_route.tasks:
                            source_route.tasks.remove(task)
                
                # Add tasks to destination route using L2 heuristic
                dest_route = new_solution.routes[to_vehicle_id]
                # Get the order object from the original orders list
                order = next((o for o in orders if o.id == order_id), None)
                if order:
                    # Use L2 heuristic to optimally insert the order
                    optimized_route = l2_heuristic(dest_route, order)
                    if optimized_route:
                        new_solution.routes[to_vehicle_id] = optimized_route
                        yield new_solution
        
        # Strategy 2: Try relocating orders to idle vehicles  
        used_vehicle_ids = set(solution.routes.keys())
        idle_vehicles = [v for v in vehicles if v.id not in used_vehicle_ids]
        
        if idle_vehicles:
            print(f"🔍 Found {len(idle_vehicles)} idle vehicles for order relocation")
            
            # For each order in the current route
            for order_id, tasks in orders_in_route.items():
                for idle_vehicle in idle_vehicles:
                    # Create a new solution with order moved to idle vehicle
                    new_solution = copy.deepcopy(solution)
                    
                    # Remove tasks from source route
                    source_route = new_solution.routes[from_vehicle_id]
                    for task in tasks:
                        try:
                            position = source_route.tasks.index(task)
                            source_route.remove_task(position)
                        except (ValueError, IndexError):
                            if task in source_route.tasks:
                                source_route.tasks.remove(task)
                    
                    # Create new route for idle vehicle
                    from epdt_data_structures import Route
                    empty_route = Route(vehicle=idle_vehicle)
                    
                    # Get the order object and use L2 heuristic to insert
                    order = next((o for o in orders if o.id == order_id), None)
                    if order:
                        optimized_route = l2_heuristic(empty_route, order)
                        if optimized_route and optimized_route.tasks:
                            new_solution.routes[idle_vehicle.id] = optimized_route
                            print(f"✅ Successfully relocated order {order_id} to idle vehicle {idle_vehicle.id}")
                            yield new_solution


def two_orders_swap_neighborhood(solution: 'Solution', orders: List['Order'], vehicles: List['Vehicle']) -> Iterator['Solution']:
    """
    Generate neighborhood by swapping two orders between different routes.
    
    This implements the 2S (two orders swap) neighborhood operation from the EPDT algorithm.
    For each pair of orders in different routes, try swapping them.
    
    Args:
        solution: Current solution to generate neighbors from
        
    Yields:
        New solutions with orders swapped between routes
    """
    if not solution or not solution.routes:
        return
    
    # For each source vehicle route
    for route1_vehicle_id, route1 in solution.routes.items():
        if not route1 or not route1.tasks:
            continue
            
        # Group tasks by order for route1
        orders_in_route1 = {}
        for task in route1.tasks:
            order_id = getattr(task, 'order_id', None)
            if order_id is None:
                continue
                
            if order_id not in orders_in_route1:
                orders_in_route1[order_id] = []
            orders_in_route1[order_id].append(task)
        
        # For each destination vehicle route
        for route2_vehicle_id, route2 in solution.routes.items():
            if route1_vehicle_id == route2_vehicle_id or not route2 or not route2.tasks:
                continue
                
            # Group tasks by order for route2
            orders_in_route2 = {}
            for task in route2.tasks:
                order_id = getattr(task, 'order_id', None)
                if order_id is None:
                    continue
                    
                if order_id not in orders_in_route2:
                    orders_in_route2[order_id] = []
                orders_in_route2[order_id].append(task)
            
            # For each pair of orders to swap
            for order1_id in orders_in_route1:
                for order2_id in orders_in_route2:
                    # Create a new solution with the orders swapped
                    new_solution = copy.deepcopy(solution)
                    
                    # Get the order objects
                    order1 = next((o for o in orders if o.id == order1_id), None)
                    order2 = next((o for o in orders if o.id == order2_id), None)
                    
                    if order1 and order2:
                        # Remove order1 from route1
                        new_route1 = new_solution.routes[route1_vehicle_id]
                        for task in orders_in_route1[order1_id]:
                            try:
                                position = new_route1.tasks.index(task)
                                new_route1.remove_task(position)
                            except (ValueError, IndexError):
                                if task in new_route1.tasks:
                                    new_route1.tasks.remove(task)
                        
                        # Remove order2 from route2
                        new_route2 = new_solution.routes[route2_vehicle_id]
                        for task in orders_in_route2[order2_id]:
                            try:
                                position = new_route2.tasks.index(task)
                                new_route2.remove_task(position)
                            except (ValueError, IndexError):
                                if task in new_route2.tasks:
                                    new_route2.tasks.remove(task)
                        
                        # Insert order2 into route1
                        optimized_route1 = l2_heuristic(new_route1, order2)
                        
                        # Insert order1 into route2
                        optimized_route2 = l2_heuristic(new_route2, order1)
                        
                        if optimized_route1 and optimized_route2:
                            new_solution.routes[route1_vehicle_id] = optimized_route1
                            new_solution.routes[route2_vehicle_id] = optimized_route2
                            yield new_solution


def multiple_order_relocation_neighborhood(solution: 'Solution', orders: List['Order'], vehicles: List['Vehicle'], max_orders: int = 3) -> Iterator['Solution']:
    """
    Generate neighborhood by relocating multiple orders from one route to another (mR).
    
    This implements the mR neighborhood operation from the EPDT algorithm.
    For each route, try moving subsets of orders (up to max_orders) to other routes.
    
    Args:
        solution: Current solution to generate neighbors from
        max_orders: Maximum number of orders to relocate in a single move
        
    Yields:
        New solutions with multiple orders relocated between routes
    """
    if not solution or not solution.routes:
        return
    
    from itertools import combinations
    
    # For each source vehicle route
    for from_vehicle_id, from_route in solution.routes.items():
        if not from_route or not from_route.tasks:
            continue
            
        # Group tasks by order
        orders_in_route = {}
        for task in from_route.tasks:
            order_id = getattr(task, 'order_id', None)
            if order_id is None:
                continue
                
            if order_id not in orders_in_route:
                orders_in_route[order_id] = []
            orders_in_route[order_id].append(task)
        
        # Generate all possible combinations of orders to move (up to max_orders)
        order_ids = list(orders_in_route.keys())
        for num_orders in range(1, min(max_orders + 1, len(order_ids) + 1)):
            for order_combination in combinations(order_ids, num_orders):
                # For each destination vehicle route (different from source)
                for to_vehicle_id, to_route in solution.routes.items():
                    if from_vehicle_id == to_vehicle_id:
                        continue  # Skip same route
                    
                    # Create a new solution with the orders relocated
                    new_solution = copy.deepcopy(solution)
                    
                    # Remove tasks from source route
                    source_route = new_solution.routes[from_vehicle_id]
                    for order_id in order_combination:
                        for task in orders_in_route[order_id]:
                            try:
                                position = source_route.tasks.index(task)
                                source_route.remove_task(position)
                            except (ValueError, IndexError):
                                if task in source_route.tasks:
                                    source_route.tasks.remove(task)
                    
                    # Add all orders to destination route using L2 heuristic
                    dest_route = new_solution.routes[to_vehicle_id]
                    all_inserted = True
                    
                    for order_id in order_combination:
                        # Get the order object
                        order = next((o for o in orders if o.id == order_id), None)
                        if order:
                            # Use L2 heuristic to optimally insert the order
                            optimized_route = l2_heuristic(dest_route, order)
                            if optimized_route:
                                dest_route = optimized_route
                            else:
                                all_inserted = False
                                break
                    
                    if all_inserted:
                        new_solution.routes[to_vehicle_id] = dest_route
                        yield new_solution


def two_opt_routes_neighborhood(solution: 'Solution', orders: List['Order'], vehicles: List['Vehicle']) -> Iterator['Solution']:
    """
    Generate neighborhood by applying 2-opt optimization within routes (2C).
    
    This implements the 2C neighborhood operation from the EPDT algorithm.
    For each route, try all possible 2-opt swaps to improve route efficiency.
    
    Args:
        solution: Current solution to generate neighbors from
        
    Yields:
        New solutions with routes optimized using 2-opt
    """
    if not solution or not solution.routes:
        return
    
    # For each vehicle route
    for vehicle_id, route in solution.routes.items():
        if not route or not route.tasks or len(route.tasks) < 4:
            continue  # Need at least 4 tasks for meaningful 2-opt
        
        # Apply 2-opt optimization to this route
        optimized_routes = _apply_2opt_to_route(route)
        
        for optimized_route in optimized_routes:
            # Create new solution with the optimized route
            new_solution = copy.deepcopy(solution)
            new_solution.routes[vehicle_id] = optimized_route
            yield new_solution


def _apply_2opt_to_route(route: 'Route') -> List['Route']:
    """
    Apply 2-opt optimization to a single route.
    
    2-opt works by removing two edges from the route and reconnecting
    the route in a different way. This can eliminate crossovers and
    improve route efficiency.
    
    Args:
        route: Route to optimize
        
    Returns:
        List of improved routes (may be empty if no improvements found)
    """
    if not route or not route.tasks or len(route.tasks) < 4:
        return []
    
    improved_routes = []
    tasks = route.tasks.copy()
    
    # Try all possible 2-opt swaps
    for i in range(1, len(tasks) - 2):
        for j in range(i + 1, len(tasks)):
            # Create new task sequence by reversing the segment between i and j
            new_tasks = tasks[:i] + tasks[i:j+1][::-1] + tasks[j+1:]
            
            # Create new route with the reordered tasks
            new_route = route.copy()
            new_route.tasks = new_tasks
            # Allow flexible ordering - only enforce individual order precedence
            # new_route.ensure_pickup_first_ordering()  # Commented out for efficiency
            
            # Validate the new route (check constraints, feasibility)
            if _is_valid_route_order(new_route):
                # Calculate improvement - only add if it's actually better
                from second_level import calculate_z2_score
                if calculate_z2_score(new_route) > calculate_z2_score(route):
                    improved_routes.append(new_route)
    
    return improved_routes


def _is_valid_route_order(route: 'Route') -> bool:
    """
    Check if a route's task order is valid (respects pickup/delivery constraints).
    
    For each order, pickup tasks must come before delivery tasks.
    
    Args:
        route: Route to validate
        
    Returns:
        True if route order is valid, False otherwise
    """
    if not route or not route.tasks:
        return True
    
    # Track task positions by order
    order_positions = {}
    
    for pos, task in enumerate(route.tasks):
        order_id = getattr(task, 'order_id', None)
        task_type = getattr(task, 'task_type', None)  # 'pickup' or 'delivery'
        
        if order_id is None or task_type is None:
            continue
        
        # Convert TaskType enum to string if needed
        if hasattr(task_type, 'value'):
            task_type = task_type.value
        
        if order_id not in order_positions:
            order_positions[order_id] = {'pickup': [], 'delivery': []}
        
        # Only process recognized task types
        if task_type in ['pickup', 'delivery']:
            order_positions[order_id][task_type].append(pos)
    
    # Check that all pickups come before all deliveries for each order
    for order_id, positions in order_positions.items():
        pickup_positions = positions.get('pickup', [])
        delivery_positions = positions.get('delivery', [])
        
        # All pickup positions must be less than all delivery positions
        if pickup_positions and delivery_positions:
            max_pickup_pos = max(pickup_positions)
            min_delivery_pos = min(delivery_positions)
            
            if max_pickup_pos >= min_delivery_pos:
                return False
    
    return True


#@jit(nopython=False, forceobj=True)  # Performance optimization
def calculate_z1_score(solution: 'Solution', params: dict = None, orders: List['Order'] = None) -> float:
    """
    Calculate the Z1 score for a complete solution with enhanced priority-based penalties.
    
    This function evaluates a solution by summing the Z2 scores of all routes
    and applying solution-wide penalties for constraint violations, with special
    handling for order priorities (mandatory, urgent, normal).
    
    Args:
        solution: Complete solution to evaluate
        params: Algorithm parameters containing penalty values
        orders: List of all orders for priority lookup
        
    Returns:
        Z1 score (higher is better)
    """
    if not solution or not solution.routes:
        return float('-inf')  # Invalid solution
    
    # Default penalty parameters if not provided
    if params is None:
        params = {}
    
    # Extract penalty parameters with defaults
    vehicle_penalty_per_vehicle = params.get('vehicle_penalty_per_vehicle', 100)
    unassigned_order_base_penalty = params.get('unassigned_order_base_penalty', 1000)
    depot_capacity_penalty = params.get('depot_capacity_penalty', 500)
    mandatory_order_penalty = params.get('mandatory_order_penalty', 800)
    vehicle_assignment_penalty = params.get('vehicle_assignment_penalty', 150)
    
    total_score = 0.0
    
    # Sum Z2 scores from all routes
    for route in solution.routes.values():
        if route and route.tasks:
            from second_level import calculate_z2_score
            # Use the function with just the route parameter
            total_score += calculate_z2_score(route)
    
    # Revenue for assigned orders
    if hasattr(solution, 'orders'):
        assigned_orders = [o for o in solution.orders if getattr(o, 'is_assigned', True)]
        total_score += sum(getattr(o, 'revenue', 0) for o in assigned_orders)
    
    # Apply solution-wide penalties
    
    # Penalty for number of vehicles used (encourage using fewer vehicles)
    vehicles_used = sum(1 for route in solution.routes.values() if route and route.tasks)
    vehicle_penalty = vehicles_used * vehicle_penalty_per_vehicle
    
    # Penalty for unassigned orders with enhanced priority-based logic
    unassigned_penalty = 0
    if hasattr(solution, 'unassigned_orders'):
        for order_id in solution.unassigned_orders:
            # Get the actual order object to check its priority
            order = None
            # Find the order object from the order_id
            if orders:
                for o in orders:
                    if getattr(o, 'id', None) == order_id:
                        order = o
                        break
            
            if order is not None:
                # Enhanced priority-based penalty logic
                priority = getattr(order, 'priority', 1)
                is_urgent = getattr(order, 'is_urgent', False)
                is_mandatory = getattr(order, 'is_mandatory', True)
                
                if is_mandatory or priority == 'mandatory':
                    # Very high penalty for unassigned mandatory orders
                    unassigned_penalty += unassigned_order_base_penalty * 10.0
                elif is_urgent or priority == 'urgent':
                    # Apply Lo penalty for unassigned urgent orders
                    lo_penalty = params.get('Lo', unassigned_order_base_penalty * 2.0)
                    unassigned_penalty += lo_penalty
                elif priority == 'normal' or priority == 1:
                    # Apply no penalty for unassigned normal orders
                    pass
                else:
                    # Fallback for numeric priority levels
                    if isinstance(priority, (int, float)) and priority > 1:
                        unassigned_penalty += unassigned_order_base_penalty * priority
                    else:
                        unassigned_penalty += unassigned_order_base_penalty
            else:
                # Fallback if order object not found
                unassigned_penalty += unassigned_order_base_penalty
    
    # Penalty for depot capacity violations (if applicable)
    depot_penalty = 0
    if hasattr(solution, 'check_depot_capacity'):
        if not solution.check_depot_capacity():
            depot_penalty = depot_capacity_penalty
    
    # Penalty for mandatory orders not satisfied (if applicable)
    mandatory_penalty = 0
    if hasattr(solution, 'get_unsatisfied_mandatory_orders'):
        unsatisfied_mandatory = solution.get_unsatisfied_mandatory_orders()
        mandatory_penalty = len(unsatisfied_mandatory) * mandatory_order_penalty
    
    # Penalty for preferred vehicle assignments (ID soft constraint)
    vehicle_assignment_penalty_total = 0
    if hasattr(solution, 'get_vehicle_assignment_violations'):
        violations = solution.get_vehicle_assignment_violations()
        vehicle_assignment_penalty_total = len(violations) * vehicle_assignment_penalty
    
    # Penalty for preferred order groupings (IE soft constraint)
    order_grouping_penalty = 0
    if hasattr(solution, 'get_order_grouping_violations'):
        violations = solution.get_order_grouping_violations()
        order_grouping_penalty = sum(v[1] for v in violations)  # Weighted penalty based on violation severity
    
    # Load balancing penalty to discourage extreme utilization imbalances
    load_balancing_penalty = 0
    vehicle_utilizations = []
    
    for route in solution.routes.values():
        if route and hasattr(route, 'vehicle') and route.vehicle:
            # Calculate weight utilization for this route
            total_weight = sum(getattr(task.order, 'weight', 0) 
                             for task in getattr(route, 'tasks', []) 
                             if hasattr(task, 'order') and task.order)
            capacity = getattr(route.vehicle, 'weight_capacity', 1)
            utilization = total_weight / capacity if capacity > 0 else 0
            vehicle_utilizations.append(utilization)
    
    if len(vehicle_utilizations) > 1:
        # Calculate standard deviation of utilizations
        mean_util = sum(vehicle_utilizations) / len(vehicle_utilizations)
        variance = sum((u - mean_util) ** 2 for u in vehicle_utilizations) / len(vehicle_utilizations)
        std_dev = variance ** 0.5
        
        # Penalty increases with utilization imbalance
        # Especially penalize when some vehicles are extremely overloaded while others are empty
        load_balancing_penalty = std_dev * params.get('load_balancing_penalty_factor', 200.0)
        
        # Extra penalty for extreme cases (>400% utilization differences)
        max_util = max(vehicle_utilizations) if vehicle_utilizations else 0
        min_util = min(vehicle_utilizations) if vehicle_utilizations else 0
        if max_util - min_util > 4.0:  # >400% difference
            load_balancing_penalty += 1000.0  # Heavy penalty for extreme imbalance

    # Calculate final Z1 score (note: we subtract penalties since Z1 is to be maximized)
    z1_score = (total_score - vehicle_penalty - unassigned_penalty - depot_penalty - 
                mandatory_penalty - vehicle_assignment_penalty_total - order_grouping_penalty - 
                load_balancing_penalty)
    
    return z1_score


def cluster_aware_initializer(orders: List['Order'], vehicles: List['Vehicle'], params: dict = None) -> 'Solution':
    """
    Enhanced initializer that creates efficient pickup→pickup→delivery→delivery patterns.
    
    This initializer addresses the bouncing problem by:
    1. Grouping multiple orders per vehicle based on capacity
    2. Building complete routes with efficient task clustering  
    3. Creating pickup→pickup→delivery→delivery patterns instead of pickup→delivery→pickup→delivery
    """
    print(f"🏗️  Starting cluster-aware initializer with {len(orders)} orders and {len(vehicles)} vehicles")
    
    solution = Solution()
    unassigned_orders = orders.copy()
    
    # Get debug setting from params
    debug_assignment = params.get('debug_assignment', False) if params else False
    
    # Phase 1: Smart Order-to-Vehicle Assignment
    # Group multiple orders per vehicle to enable efficient clustering
    vehicle_assignments = {}  # vehicle_id -> list of orders
    
    # Initialize assignment tracking
    for vehicle in vehicles:
        vehicle_assignments[vehicle.id] = []
    
    # Enhanced assignment strategy: Capacity-aware order grouping
    ordered_vehicles = sorted(vehicles, key=lambda v: v.weight_capacity, reverse=True)  # Start with largest vehicles
    
    while unassigned_orders:
        assignment_made = False
        
        for vehicle in ordered_vehicles:
            if not unassigned_orders:
                break
                
            # Find the best-fitting order for this vehicle considering current load
            current_weight = sum(order.get_total_demand() for order in vehicle_assignments[vehicle.id])
            current_volume = sum(order.get_total_volume() for order in vehicle_assignments[vehicle.id])
            
            # Select orders that fit within vehicle capacity
            compatible_orders = []
            for order in unassigned_orders:
                if (current_weight + order.get_total_demand() <= vehicle.weight_capacity and 
                    current_volume + order.get_total_volume() <= vehicle.volume_capacity):
                    compatible_orders.append(order)
            
            # Assign the largest compatible order to maximize vehicle utilization
            if compatible_orders:
                # Sort by weight descending to fill vehicles efficiently
                best_order = max(compatible_orders, key=lambda o: o.get_total_demand())
                vehicle_assignments[vehicle.id].append(best_order)
                unassigned_orders.remove(best_order)
                assignment_made = True
                
                if debug_assignment:
                    print(f"  DEBUG L1: Assigned order {best_order.id} to {vehicle.id} "
                          f"(total load: {current_weight + best_order.get_total_demand()}kg)")
        
        # If no assignments were made, assign remaining orders to least loaded vehicles
        if not assignment_made and unassigned_orders:
            # Find vehicle with lowest current load
            lightest_vehicle = min(ordered_vehicles, 
                                 key=lambda v: sum(order.get_total_demand() for order in vehicle_assignments[v.id]))
            
            # Assign the first remaining order
            order_to_assign = unassigned_orders[0]
            vehicle_assignments[lightest_vehicle.id].append(order_to_assign)
            unassigned_orders.remove(order_to_assign)
            
            if debug_assignment:
                print(f"  DEBUG L1: Force-assigned order {order_to_assign.id} to {lightest_vehicle.id}")
    
    # Phase 2: Build Efficient Routes Using Enhanced Strategy
    for vehicle in vehicles:
        assigned_orders = vehicle_assignments[vehicle.id]
        
        if not assigned_orders:
            continue  # Skip vehicles with no orders
        
        if debug_assignment:
            print(f"  DEBUG L1: Building route for {vehicle.id} with {len(assigned_orders)} orders")
        
        # Create base route
        current_route = Route(vehicle=vehicle)
        
        # Use enhanced multi-order insertion strategy
        final_route = build_clustered_route(current_route, assigned_orders, debug_assignment)
        
        if final_route is not None:
            solution.add_route(vehicle.id, final_route)
            if debug_assignment:
                print(f"  DEBUG L1: Successfully built route for {vehicle.id} with {len(final_route.tasks)} tasks")
        else:
            # Fallback: try inserting orders one by one using original L2
            if debug_assignment:
                print(f"  DEBUG L1: Clustered routing failed for {vehicle.id}, falling back to sequential insertion")
            
            for order in assigned_orders:
                new_route = l2_heuristic(current_route, order, debug_assignment)
                if new_route is not None:
                    current_route = new_route
                else:
                    # Add to unassigned if even fallback fails
                    if hasattr(solution, 'unassigned_orders'):
                        solution.unassigned_orders.add(order.id)
                    else:
                        solution.unassigned_orders = {order.id}
                    print(f"Warning: Could not assign order {order.id} to any vehicle")
            
            if len(current_route.tasks) > 0:
                solution.add_route(vehicle.id, current_route)
    
    return solution


def build_clustered_route(route: 'Route', orders: List, debug_assignment: bool = False) -> Optional['Route']:
    """
    Build an efficient route with multiple orders using cluster-based insertion.
    
    This creates depot_start → pickup→pickup→delivery→delivery → depot_return patterns.
    """
    if debug_assignment:
        print(f"    DEBUG L1: Building clustered route with {len(orders)} orders")
    
    # Extract all pickup and delivery tasks from orders
    all_pickups = []
    all_deliveries = []
    
    for order in orders:
        pickup_tasks = order.get_pickups()
        delivery_tasks = order.get_deliveries()
        all_pickups.extend(pickup_tasks)
        all_deliveries.extend(delivery_tasks)
    
    # Start with an empty route and add depot start task
    current_route = route.copy()
    
    # Add depot start task if we have any actual cargo tasks
    if all_pickups or all_deliveries:
        # Get depot information from vehicle
        depot_location_id = getattr(route.vehicle, 'depot_id', 'main_depot')
        depot_lat = getattr(route.vehicle, 'depot_lat', 44.9009)  # Default Asti coordinates (Via del Lavoro 38)
        depot_lon = getattr(route.vehicle, 'depot_lon', 8.2057)
        
        # Import Task class for creating depot tasks
        from epdt_data_structures import Task
        
        # Create and add depot start task at the beginning
        depot_start_task = Task.create_depot_start_task(
            vehicle_id=route.vehicle.id,
            depot_location_id=depot_location_id,
            depot_lat=depot_lat,
            depot_lon=depot_lon
        )
        current_route.insert_task_without_reordering(0, depot_start_task)
    
    # Phase 1: Insert all pickups in a cluster
    for pickup in all_pickups:
        best_cost = float('inf')
        best_route = None
        
        # Try inserting at each position using relaxed feasibility check
        for pos in range(len(current_route.tasks) + 1):
            test_route = current_route.copy()
            test_route.insert_task_without_reordering(pos, pickup)
            
            # Use relaxed feasibility check during initialization to allow more exploration
            from second_level import is_feasible_for_insertion
            if is_feasible_for_insertion(test_route, debug_insertion=debug_assignment):
                from second_level import calculate_z2_score
                cost = calculate_z2_score(test_route)
                if cost < best_cost:
                    best_cost = cost
                    best_route = test_route
        
        if best_route:
            current_route = best_route
        else:
            if debug_assignment:
                print(f"    DEBUG L1: Failed to insert pickup {pickup.id}")
            return None
    
    # Phase 2: Insert all deliveries after pickups (respecting individual precedence)
    for delivery in all_deliveries:
        best_cost = float('inf')
        best_route = None
        
        # Find corresponding pickup position for precedence
        pickup_pos = None
        for i, task in enumerate(current_route.tasks):
            if (hasattr(task, 'order_id') and hasattr(delivery, 'order_id') and 
                task.order_id == delivery.order_id and task.is_pickup()):
                pickup_pos = i
                break
        
        # Insert delivery after its pickup
        start_pos = (pickup_pos + 1) if pickup_pos is not None else len(all_pickups)
        
        for pos in range(start_pos, len(current_route.tasks) + 1):
            test_route = current_route.copy()
            test_route.insert_task_without_reordering(pos, delivery)
            
            # Use relaxed feasibility check during initialization to allow more exploration
            from second_level import is_feasible_for_insertion
            if is_feasible_for_insertion(test_route, debug_insertion=debug_assignment):
                from second_level import calculate_z2_score
                cost = calculate_z2_score(test_route)
                if cost < best_cost:
                    best_cost = cost
                    best_route = test_route
        
        if best_route:
            current_route = best_route
        else:
            if debug_assignment:
                print(f"    DEBUG L1: Failed to insert delivery {delivery.id}")
            return None
    
    # Add depot return task at the end if we have any actual cargo tasks
    if all_pickups or all_deliveries:
        # Get depot information from vehicle (same as start)
        depot_location_id = getattr(route.vehicle, 'depot_id', 'main_depot')
        depot_lat = getattr(route.vehicle, 'depot_lat', 44.9009)  # Default Asti coordinates (Via del Lavoro 38)
        depot_lon = getattr(route.vehicle, 'depot_lon', 8.2057)
        
        # Create and add depot return task at the end
        depot_return_task = Task.create_depot_return_task(
            vehicle_id=route.vehicle.id,
            depot_location_id=depot_location_id,
            depot_lat=depot_lat,
            depot_lon=depot_lon
        )
        current_route.insert_task_without_reordering(len(current_route.tasks), depot_return_task)
    
    if debug_assignment:
        print(f"    DEBUG L1: Successfully built clustered route with {len(current_route.tasks)} tasks")
    
    return current_route


def regret_k_initializer(orders: List['Order'], vehicles: List['Vehicle'], params: dict = None) -> 'Solution':
    """
    Regret-k Insertion Heuristic for Advanced Order Assignment.
    
    Prioritizes orders that have the fewest good placement options (high "regret").
    This prevents the solver from using up the best slots on easy-to-place orders.
    
    Args:
        orders: List of orders to be assigned
        vehicles: List of available vehicles
        params: Algorithm parameters (k value for regret calculation)
        
    Returns:
        Solution with orders assigned using regret-k strategy
    """
    if params is None:
        params = {}
    
    k = params.get('regret_k_value', 3)  # Default k=3 for regret calculation
    debug_regret = params.get('debug_regret', False)
    
    print(f"🧠 Starting regret-{k} initializer with {len(orders)} orders and {len(vehicles)} vehicles")
    
    solution = Solution()
    unassigned_orders = orders.copy()
    
    # Initialize empty routes for all vehicles
    for vehicle in vehicles:
        solution.add_route(vehicle.id, Route(vehicle=vehicle))
    
    # Main regret-k loop
    iteration = 0
    while unassigned_orders:
        iteration += 1
        if debug_regret and iteration % 10 == 1:
            print(f"  Regret iteration {iteration}: {len(unassigned_orders)} orders remaining")
        
        best_regret = -1
        best_order = None
        best_insertion = None
        order_costs = {}  # Use order.id as key instead of order object
        
        # Step 1: Calculate insertion costs for all unassigned orders
        for order in unassigned_orders:
            insertion_costs = []
            
            # Try inserting this order into every vehicle's route
            for vehicle in vehicles:
                current_route = solution.routes[vehicle.id]
                
                # Get all possible insertion positions for this order
                order_tasks = order.get_all_tasks()
                if not order_tasks:
                    continue
                
                # Find best insertion cost for this vehicle
                best_vehicle_cost = float('inf')
                best_vehicle_insertion = None
                
                # Try all possible insertion patterns (pickup positions + delivery positions)
                for pickup_task in order.get_pickups():
                    for delivery_task in order.get_deliveries():
                        # Try different insertion positions
                        for pickup_pos in range(len(current_route.tasks) + 1):
                            for delivery_pos in range(pickup_pos + 1, len(current_route.tasks) + 2):
                                # Create test route
                                test_route = current_route.copy()
                                
                                # Insert pickup first, then delivery
                                test_route.insert_task_without_reordering(pickup_pos, pickup_task)
                                test_route.insert_task_without_reordering(delivery_pos, delivery_task)
                                
                                # Check feasibility with lenient constraints during initialization
                                try:
                                    from second_level import is_feasible_for_insertion, calculate_z2_score
                                    if is_feasible_for_insertion(test_route, debug_insertion=False):
                                        cost = calculate_z2_score(test_route)
                                        if cost < best_vehicle_cost:
                                            best_vehicle_cost = cost
                                            best_vehicle_insertion = {
                                                'vehicle': vehicle,
                                                'route': test_route,
                                                'cost': cost,
                                                'pickup_pos': pickup_pos,
                                                'delivery_pos': delivery_pos,
                                                'order': order  # Store order reference
                                            }
                                except Exception as e:
                                    if debug_regret:
                                        print(f"    Warning: Error evaluating insertion for {order.id}: {e}")
                                    continue
                
                # Add this vehicle's best cost to the list
                if best_vehicle_cost < float('inf'):
                    insertion_costs.append(best_vehicle_cost)
                    if best_vehicle_insertion:
                        if order.id not in order_costs:
                            order_costs[order.id] = []
                        order_costs[order.id].append(best_vehicle_insertion)
            
            # Step 2: Calculate regret value for this order
            if len(insertion_costs) >= k:
                insertion_costs.sort()
                cost_1 = insertion_costs[0]  # Best insertion cost
                cost_k = insertion_costs[k-1]  # k-th best insertion cost
                regret = cost_k - cost_1
                
                if debug_regret:
                    print(f"    Order {order.id}: best_cost={cost_1:.2f}, {k}th_cost={cost_k:.2f}, regret={regret:.2f}")
                
                # Select order with highest regret
                if regret > best_regret:
                    best_regret = regret
                    best_order = order
                    # Find the best insertion among all vehicles for this order
                    if order.id in order_costs:
                        best_insertion = min(order_costs[order.id], key=lambda x: x['cost'])
        
        # Step 3: Insert the order with highest regret in its best position
        if best_order and best_insertion:
            vehicle_id = best_insertion['vehicle'].id
            solution.routes[vehicle_id] = best_insertion['route']
            unassigned_orders.remove(best_order)
            
            if debug_regret:
                print(f"  ✅ Inserted order {best_order.id} into vehicle {vehicle_id} (regret: {best_regret:.2f}, cost: {best_insertion['cost']:.2f})")
        else:
            # No feasible insertion found - try force assignment or break
            if unassigned_orders:
                problematic_order = unassigned_orders[0]
                print(f"⚠️  No feasible insertion found for order {problematic_order.id}, skipping...")
                unassigned_orders.remove(problematic_order)
                # Add to unassigned list in solution
                if not hasattr(solution, 'unassigned_orders'):
                    solution.unassigned_orders = set()
                solution.unassigned_orders.add(problematic_order.id)
    
    # Add depot tasks to all routes with actual orders
    for vehicle_id, route in solution.routes.items():
        if route.tasks and not any('depot' in str(task.id).lower() for task in route.tasks):
            _add_depot_tasks_to_route(route)
    
    print(f"✅ Regret-{k} initialization completed: {len(orders) - len(unassigned_orders)}/{len(orders)} orders assigned")
    return solution


def _add_depot_tasks_to_route(route: 'Route'):
    """Helper function to add depot start and return tasks to a route."""
    try:
        from epdt_data_structures import Task
        
        vehicle = route.vehicle
       
        depot_location_id = getattr(vehicle, 'depot_id', 'main_depot')
        depot_lat = getattr(vehicle, 'depot_lat', 44.9009)
        depot_lon = getattr(vehicle, 'depot_lon', 8.2057)
        
        # Add depot start at beginning
        depot_start = Task.create_depot_start_task(
            vehicle_id=vehicle.id,
            depot_location_id=depot_location_id,
            depot_lat=depot_lat,
            depot_lon=depot_lon
        )
        route.insert_task_without_reordering(0, depot_start)
        
        # Add depot return at end
        depot_return = Task.create_depot_return_task(
            vehicle_id=vehicle.id,
            depot_location_id=depot_location_id,
            depot_lat=depot_lat,
            depot_lon=depot_lon
        )
        route.insert_task_without_reordering(len(route.tasks), depot_return)
        
    except Exception as e:
        print(f"Warning: Could not add depot tasks to route: {e}")