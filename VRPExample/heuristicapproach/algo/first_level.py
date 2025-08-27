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
    from .epdt_data_structures import Route, Order, Vehicle, Solution, Task, TaskType
else:
    try:
        from .epdt_data_structures import Route, Order, Vehicle, Solution, Task, TaskType
    except ImportError:
        from epdt_data_structures import Route, Order, Vehicle, Solution, Task, TaskType

try:
    from second_level import l2_heuristic
except ImportError:
    try:
        from .second_level import l2_heuristic
    except ImportError:
        print("Warning: Could not import l2_heuristic")
        l2_heuristic = None

# Removed problematic circular imports - these modules will be imported locally where needed

def _create_base_route(vehicle: 'Vehicle') -> 'Route':
    """
    Creates a new, structurally valid route for a vehicle, pre-populated
    with DEPOT_START and DEPOT_RETURN tasks.
    
    FIXED: No longer uses hardcoded depot coordinates - gets them from vehicle object
    as recommended by technical review.
    
    This ensures all routes are valid from inception and prevents the need
    to add depot tasks later in the process, which was causing issues with
    neighborhood operators and route exploration.
    """
    route = Route(vehicle=vehicle)
    
    # FIXED: Get depot information from vehicle object instead of hardcoding
    # Fallback to Asti coordinates for backward compatibility
    depot_location_id = getattr(vehicle, 'depot_id', "DEPOT-ASTI")
    depot_lat = getattr(vehicle, 'depot_lat', 44.9009)  # Default Asti coordinates
    depot_lon = getattr(vehicle, 'depot_lon', 8.2057)

    start_task = Task(
        id=f"depot_start_order_{vehicle.id}",
        location_id=depot_location_id,
        task_type=TaskType.DEPOT_START,
        order_id=f"depot_start_order_{vehicle.id}",
        lat=depot_lat,
        lon=depot_lon,
        service_time=0.0,
        demand=0.0,
        volume=0.0
    )
    route.tasks.append(start_task)

    return_task = Task(
        id=f"depot_return_order_{vehicle.id}",
        location_id=depot_location_id,
        task_type=TaskType.DEPOT_RETURN,
        order_id=f"depot_return_order_{vehicle.id}",
        lat=depot_lat,
        lon=depot_lon,
        service_time=0.0,
        demand=0.0,
        volume=0.0
    )
    route.tasks.append(return_task)
    return route

def _add_depot_tasks_to_route(route: 'Route'):
    """
    Ensures a route has depot start and return tasks, adding them if missing.
    This is a defensive function to fix routes that were created incorrectly.
    
    FIXED: No longer uses hardcoded depot coordinates.
    """
    if not hasattr(route, 'tasks') or not route.tasks:
        # Don't add depot tasks to a completely empty route.
        # It will be filtered out later anyway.
        return

    # Defensive check: if route already has them, do nothing
    if route.tasks[0].is_depot_start() and route.tasks[-1].is_depot_return():
        return

    try:
        from .epdt_data_structures import Task, TaskType
    except ImportError:
        from epdt_data_structures import Task, TaskType

    # FIXED: Get depot information from vehicle object instead of hardcoding
    depot_location_id = getattr(route.vehicle, 'depot_id', "DEPOT-ASTI")
    depot_lat = getattr(route.vehicle, 'depot_lat', 44.9009)
    depot_lon = getattr(route.vehicle, 'depot_lon', 8.2057)

    # Check for and add start task
    if not route.tasks[0].is_depot_start():
        start_task = Task(
            id=f"depot_start_order_{route.vehicle.id}",
            location_id=depot_location_id,
            task_type=TaskType.DEPOT_START,
            order_id=f"depot_start_order_{route.vehicle.id}",
            lat=depot_lat,
            lon=depot_lon,
            service_time=0.0,
            demand=0.0,
            volume=0.0
        )
        route.tasks.insert(0, start_task)

    # Check for and add return task
    if not route.tasks[-1].is_depot_return():
        return_task = Task(
            id=f"depot_return_order_{route.vehicle.id}",
            location_id=depot_location_id,
            task_type=TaskType.DEPOT_RETURN,
            order_id=f"depot_return_order_{route.vehicle.id}",
            lat=depot_lat,
            lon=depot_lon,
            service_time=0.0,
            demand=0.0,
            volume=0.0
        )
        route.tasks.append(return_task)


def create_lightweight_solution_copy(solution, vehicles_to_copy: Optional[set] = None):
    """
    Create a lightweight copy of a solution, copying only specified vehicles or all vehicles.
    Much faster than deepcopy for large solutions.
    
    Args:
        solution: The solution to copy
        vehicles_to_copy: Set of vehicle IDs to copy. If None, copy all vehicles.
    
    Returns:
        New solution with copied routes
    """
    # Import here to avoid circular imports
    import sys
    import os
    try:
        from epdt_data_structures import Solution, Route
    except ImportError:
        # Fallback to current module path
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from epdt_data_structures import Solution, Route
    
    # Create new solution with basic attributes
    new_solution = Solution()
    new_solution.unassigned_orders = set(solution.unassigned_orders)  # Shallow copy of set
    new_solution.routes = {}
    
    # Copy only specified routes or all routes
    target_vehicles = vehicles_to_copy if vehicles_to_copy is not None else solution.routes.keys()
    
    for vehicle_id in target_vehicles:
        if vehicle_id in solution.routes:
            route = solution.routes[vehicle_id]
            new_route = Route(route.vehicle)
            
            # Copy task references (tasks themselves are immutable, so no need to copy)
            new_route.tasks = list(route.tasks)  # Shallow copy of task list
            
            # Copy other route attributes if they exist
            if hasattr(route, 'driver'):
                new_route.driver = route.driver
            if hasattr(route, 'total_cost'):
                new_route.total_cost = route.total_cost
            if hasattr(route, 'current_load'):
                new_route.current_load = route.current_load
                
            new_solution.routes[vehicle_id] = new_route
    
    # Copy routes not in target set without modification
    for vehicle_id, route in solution.routes.items():
        if vehicle_id not in target_vehicles:
            new_solution.routes[vehicle_id] = route  # Reference copy for unchanged routes
    
    return new_solution


def create_single_route_copy(solution, vehicle_id: str):
    """
    Create a solution copy with only one vehicle's route copied.
    All other routes are referenced, not copied.
    
    Args:
        solution: Original solution
        vehicle_id: ID of vehicle whose route to copy
        
    Returns:
        New solution with one copied route
    """
    return create_lightweight_solution_copy(solution, {vehicle_id})


def create_two_route_copy(solution, vehicle_id1: str, vehicle_id2: str):
    """
    Create a solution copy with only two vehicles' routes copied.
    All other routes are referenced, not copied.
    
    Args:
        solution: Original solution
        vehicle_id1: ID of first vehicle whose route to copy
        vehicle_id2: ID of second vehicle whose route to copy
        
    Returns:
        New solution with two copied routes
    """
    return create_lightweight_solution_copy(solution, {vehicle_id1, vehicle_id2})


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
        print("Using Regret-k initialization strategy")
        initial_solution = regret_k_initializer(orders, vehicles, params)
    else:
        print("Build  Using cluster-aware initialization strategy") 
        initial_solution = cluster_aware_initializer(orders, vehicles, params)
    
    # Check for destroy and repair if enabled and there are unassigned orders
    if params.get('enable_destroy_and_repair', False):
        unassigned_count = len(getattr(initial_solution, 'unassigned_orders', set()))
        if unassigned_count > 0:
            print(f"Enhanced Applying destroy and repair for {unassigned_count} unassigned orders")
            try:
                from destroy_and_repair import destroy_and_repair_large_orders
                initial_solution = destroy_and_repair_large_orders(initial_solution, orders, vehicles, params)
            except ImportError:
                print("Warning:  Warning: destroy_and_repair module not available")
    
    # NEW: HYBRID CLUSTER-CONSOLIDATION - Post-initialization route consolidation
    if params.get('enable_post_init_consolidation', False):
        print("\nBuild  HYBRID PHASE: POST-INITIALIZATION ROUTE CONSOLIDATION")
        print("Build  ================================================")
        
        target_idle_vehicles = params.get('target_idle_vehicles', 15)  
        vehicle_penalty = params.get('vehicle_penalty_per_vehicle', 0.0)
        
        if vehicle_penalty > 1000:  # Only consolidate if vehicle penalty is significant
            print(f"Build  Consolidating routes to create {target_idle_vehicles} idle vehicles...")
            initial_solution = consolidate_routes_for_idle_vehicles(initial_solution, vehicles, target_idle_vehicles, params)
        else:
            print(f"Build  Vehicle penalty ({vehicle_penalty}) too low for consolidation - skipping")
    
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
    print(f"RECALCULATING Starting L1 main optimization loop with M1={params['M1']}, M2={params['M2']}")
    while non_improving_iters < params['M1'] and total_iters < params['M2']:
        total_iters += 1
        improvement_found = False
        best_neighbors_pool = []
        
        # Add verbose logging (Step 3 from guide)
        verbose = params.get('verbose_logging', False)
        if verbose:
            print(f"\n--- L1 Heuristic: Iteration {total_iters} ---")
            print(f"  - Center Solution Score (Z1): {calculate_z1_score(center_solution, params, orders):.2f}")
            print(f"  - Non-improving iterations: {non_improving_iters}/{params['M1']}")
        
        
        if total_iters % 10 == 1:  # Print every 10 iterations
            print(f"RECALCULATING L1 Iteration {total_iters}: non_improving={non_improving_iters}, score={calculate_z1_score(center_solution, params, orders):.2f}")
        
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
            print(f"Running Convergence reached: No improvement for {time_since_improvement:.1f}s")
            break
            
        # 2. Quality threshold: Stop if we have good assignment rate and reasonable runtime
        if elapsed_time > 10 and len(l1_heuristic._score_history) >= 5:
            recent_scores = l1_heuristic._score_history[-5:]
            score_variance = max(recent_scores) - min(recent_scores)
            if score_variance < 1.0:  # Very stable scores
                print(f"OK: Solution stabilized: variance={score_variance:.2f}, time={elapsed_time:.1f}s")
                break
        
        # 3. Maximum time limit (60s as safety net, not hard production requirement)
        if elapsed_time > 60:
            print(f"Maximum time limit reached: {elapsed_time:.1f} seconds")
            break
            
        # Safety check to prevent infinite loops
        if total_iters > 20:  # Further reduced from 100 for production target
            print(f"Warning:  L1 reached 20 iterations, stopping for production performance")
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
            print(f"Debug Starting VND with {len(neighborhoods)} neighborhoods")
        
        # Track total neighbors evaluated across all neighborhoods in this VND iteration
        total_neighbors_this_vnd = 0
        max_neighbors_per_iteration = params.get('max_neighbors_per_iteration', 20)
        
        for neighborhood_idx, neighborhood_func in enumerate(neighborhoods):
            if verbose:
                print(f"\n  -> Exploring Neighborhood: {neighborhood_func.__name__}")
            if total_iters % 50 == 1:
                print(f"Debug Exploring neighborhood {neighborhood_idx+1}/{len(neighborhoods)}: {neighborhood_func.__name__}")
            
            
            # Check if we've already evaluated too many neighbors in this VND iteration
            if total_neighbors_this_vnd >= max_neighbors_per_iteration:
                if total_iters % 50 == 1:
                    print(f"Warning:  VND iteration reached {max_neighbors_per_iteration} neighbors limit, skipping remaining neighborhoods")
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
                        print(f"Warning:  Neighborhood {neighborhood_func.__name__} evaluated {max_neighbors}+ neighbors, breaking for sub-30s target")
                    break
                
                # Also check global VND iteration limit
                if total_neighbors_this_vnd >= max_neighbors_per_iteration:
                    if total_iters % 50 == 1:
                        print(f"Warning:  VND iteration reached {max_neighbors_per_iteration} total neighbors, breaking")
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
                            print(f"Warning:  First improvement found, stopping neighborhood search early")
                        break
                        
                elif not is_tabu:
                    # Add to pool for diversification if not tabu but not improving
                    best_neighbors_pool.append((neighbor, neighbor_score))
            
            # After iterating through a neighborhood's neighbors
            if verbose and best_neighbor_in_N:
                print(f"    - Best neighbor in this neighborhood has score: {best_neighbor_score:.2f}")
            
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
                
                # When an improving move is found and made
                if verbose:
                    print(f"  => Improvement found! New center score: {calculate_z1_score(center_solution, params, orders):.2f}. Restarting VND.")
                
                break # Go back to the first neighborhood (VND restart)
        
        if total_iters % 50 == 1:
            print(f"Debug VND completed. improvement_found={improvement_found}, neighbors_pool_size={len(best_neighbors_pool)}")
        
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
                
                # In the diversification step
                if verbose:
                    print(f"  => No improvement. Making diversification move. New center score: {calculate_z1_score(center_solution, params, orders):.2f}")
                
                # Allow flexible task ordering during search
                # for vehicle_id, route in center_solution.routes.items():
                #     route.ensure_pickup_first_ordering()
                
                # Add move attributes to tabu list (from previous center to new center)
                move_attrs = get_move_attributes(previous_center, center_solution)
                tabu_list.append(move_attrs)

    print(f"Complete L1 optimization completed after {total_iters} iterations")
    print(f"Complete Final score: {calculate_z1_score(best_solution, params, orders):.2f}")
    
    # Final enforcement of pickup-first ordering only if explicitly requested
    # Allow flexible task ordering by default for better efficiency
    enforce_final_ordering = params.get('enforce_pickup_first_ordering', False) if params else False
    if enforce_final_ordering:
        for vehicle_id, route in best_solution.routes.items():
            route.ensure_pickup_first_ordering()
    
    # FINAL DEPOT TASK FINALIZATION: Ensure all routes have proper depot start/end tasks
    print("Build  Performing final depot task finalization on all routes...")
    depot_tasks_added = 0
    for vehicle_id, route in best_solution.routes.items():
        if hasattr(route, 'tasks') and route.tasks:  # Only process routes with tasks
            # Check if route needs depot tasks
            needs_depot = False
            if not route.tasks[0].is_depot_start():
                needs_depot = True
            elif not route.tasks[-1].is_depot_return():
                needs_depot = True
            
            if needs_depot:
                _add_depot_tasks_to_route(route)
                depot_tasks_added += 1
    
    if depot_tasks_added > 0:
        print(f"Build  Added depot tasks to {depot_tasks_added} routes")
    else:
        print(f"OK: All routes already have proper depot structure")
    
    # FINAL VALIDATION: Strictly enforce HoS compliance on all routes
    print("Debug Performing final HoS validation on all routes...")
    final_solution = _validate_and_filter_solution(best_solution)
    
    return final_solution


def _validate_and_filter_solution(solution: 'Solution') -> 'Solution':
    """
    FIXED: Perform final strict validation on all routes as per technical review.
    
    The technical review recommendation: any route that does not pass strict feasibility 
    should be dismantled and its orders marked as unassigned. No moderate violations 
    are allowed in the final solution.
    
    This ensures that the final solution only contains routes that are 100% feasible 
    according to all hard constraints.
    """
    try:
        from second_level import is_feasible
    except ImportError:
        try:
            from .second_level import is_feasible
        except ImportError:
            print("Warning: Could not import is_feasible function for final validation")
            return solution
    
    validated_routes = {}
    removed_routes_count = 0
    
    print(f"Validating {len(solution.routes)} routes with STRICT feasibility checks...")
    
    for vehicle_id, route in solution.routes.items():
        if hasattr(route, 'tasks') and route.tasks:  # Only validate routes with tasks
            # Check if route has only depot tasks (should be filtered out)
            if len(route.tasks) <= 2:
                # Route has only depot tasks (or is truly empty), filter it out
                removed_routes_count += 1
                print(f"Empty route for vehicle {getattr(route.vehicle, 'id', 'unknown')} removed (only depot tasks)")
                continue
            
            # Route has customer tasks, validate with STRICT feasibility check
            # FIXED: No soft violations allowed in final solution
            feasible_result = is_feasible(route, debug_feasibility=False, return_reason=True, allow_soft_violations=False)
            
            if isinstance(feasible_result, tuple):
                feasible, reason = feasible_result
            else:
                feasible = feasible_result
                reason = "Unknown validation failure"
                
            if feasible:
                validated_routes[vehicle_id] = route
            else:
                # FIXED: Any infeasible route is rejected (no exceptions for moderate violations)
                removed_routes_count += 1
                print(f"Warning:  Route for vehicle {getattr(route.vehicle, 'id', 'unknown')} removed due to constraint violation: {reason}")
                
                # Add orders from removed route back to unassigned
                for task in route.tasks:
                    if hasattr(task, 'order_id') and task.order_id and 'depot' not in str(task.order_id).lower():
                        if not hasattr(solution, 'unassigned_orders'):
                            solution.unassigned_orders = set()
                        solution.unassigned_orders.add(task.order_id)
        else:
            # Route has no tasks at all, filter it out
            removed_routes_count += 1
            print(f"Completely empty route for vehicle {getattr(route.vehicle, 'id', 'unknown')} removed")

    # Update solution with validated routes
    solution.routes = validated_routes
    
    if removed_routes_count > 0:
        print(f"STRICT Final validation complete: {removed_routes_count} routes removed due to constraint violations")
    else:
        print("OK: STRICT Final validation complete: All routes are 100% feasible")
    
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
        # Handle None values for unassigned orders
        from_vehicle = from_vehicle if from_vehicle is not None else 'unassigned'
        to_vehicle = to_vehicle if to_vehicle is not None else 'unassigned'
        return ('relocation', order_id, from_vehicle, to_vehicle)
    elif len(changed_orders) == 2:
        # Two orders swap (2S)
        order1_id, from_vehicle1, to_vehicle1 = changed_orders[0]
        order2_id, from_vehicle2, to_vehicle2 = changed_orders[1]
        # Handle None values for unassigned orders
        from_vehicle1 = from_vehicle1 if from_vehicle1 is not None else 'unassigned'
        from_vehicle2 = from_vehicle2 if from_vehicle2 is not None else 'unassigned'
        # Normalize the order for consistent tabu representation
        if order1_id < order2_id:
            return ('swap', order1_id, from_vehicle1, order2_id, from_vehicle2)
        else:
            return ('swap', order2_id, from_vehicle2, order1_id, from_vehicle1)
    elif len(changed_orders) > 2:
        # Multiple order relocation (mR) or complex move
        order_ids = [order_id for order_id, _, _ in changed_orders]
        source_vehicles = [from_vehicle for _, from_vehicle, _ in changed_orders if from_vehicle is not None]
        # Handle case where source_vehicles might be empty due to None filtering
        if not source_vehicles:
            source_vehicles = ['unassigned']
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
    print(f"Build  Starting best insertion initializer with {len(orders)} orders and {len(vehicles)} vehicles")
    
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
        initial_route = _create_base_route(vehicle)
        
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
                    # FIXED: Get depot coordinates from vehicle instead of hardcoding
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
                    current_route = _create_base_route(vehicle)
                
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
            
            # Special debugging for Order 7
            if str(best_order.id) == "7":
                print(f"    *** ORDER 7 SUCCESSFULLY ASSIGNED TO {best_vehicle_idx} IN INITIAL PHASE ***")
                print(f"    Score improvement: {best_score_improvement}")
        else:
            # No feasible insertion found - add to unassigned orders
            if hasattr(solution, 'unassigned_orders'):
                solution.unassigned_orders.add(unassigned_orders[0].id)
            else:
                solution.unassigned_orders = {unassigned_orders[0].id}
            
            # Special debugging for Order 7
            order_id = unassigned_orders[0].id
            if str(order_id) == "7":
                print(f"    *** ORDER 7 FAILED TO ASSIGN IN INITIAL PHASE ***")
                print(f"    Reason: No feasible vehicle found among candidates")
                print(f"    Total vehicles checked: {len(vehicles)}")
                # Check if FF235DM and XA819VA were in the vehicle list
                ff235dm_present = any(v.id == 'FF235DM' for v in vehicles)
                xa819va_present = any(v.id == 'XA819VA' for v in vehicles)
                print(f"    FF235DM in candidate list: {ff235dm_present}")
                print(f"    XA819VA in candidate list: {xa819va_present}")
                
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
        initial_route = _create_base_route(vehicle)
        
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
                current_route = _create_base_route(current_vehicle)
            
            # Try to insert order using L2 heuristic with enhanced debugging
            print(f"    🔍 Attempting to assign Order {order.id} to Vehicle {current_vehicle.id}")
            print(f"       Order: {order.get_total_demand():.0f}kg, {order.get_total_volume():.1f}m³")
            print(f"       Vehicle: {getattr(current_vehicle, 'weight_capacity', 'unknown')}kg capacity")
            
            new_route = l2_heuristic(current_route, order, debug_assignment=True, enhanced_diagnostics=True)
            
            if new_route is not None:  # Feasible insertion
                solution.add_route(current_vehicle_id, new_route)
                unassigned_orders.remove(order)
                print(f"    ✅ Successfully assigned Order {order.id} to Vehicle {current_vehicle.id}")
                orders_assigned_to_class += 1
            else:
                print(f"    ❌ Failed to assign Order {order.id} to Vehicle {current_vehicle.id}")
                print(f"       Trying next vehicle...")
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
    Uses unified hard constraint checker for consistency.
    
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
    
    # Use unified hard constraint checker from second_level module
    from algo.second_level import check_hard_constraints
    is_valid, reason = check_hard_constraints(temp_route, debug=False)
    if not is_valid:
        return False
    
    # Check if the new task order is valid (additional soft constraint)
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
    
    # Strategy 2: Try assigning to idle or empty vehicles
    used_vehicle_ids = set(solution.routes.keys())
    idle_vehicles = []
    for v in vehicles:
        route = solution.routes.get(v.id)
        if not route or len(route.tasks) <= 2:
            idle_vehicles.append(v)
    
    print(f"Debug Found {len(idle_vehicles)} idle vehicles for unassigned order insertion")
    
    for unassigned_order in unassigned_orders:
        for idle_vehicle in idle_vehicles:
            # Create a new empty route for the idle vehicle
            from epdt_data_structures import Route
            empty_route = _create_base_route(idle_vehicle)
            
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
                
                # Log only debug info since this is just a neighbor exploration, not final assignment
                # print(f"OK: Successfully assigned unassigned order {unassigned_order.id} to idle vehicle {idle_vehicle.id}")
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
                
                # Create a lightweight copy with only affected vehicles
                new_solution = create_two_route_copy(solution, from_vehicle_id, to_vehicle_id)
                
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
        
        # Strategy 2: Try relocating orders to idle or empty vehicles  
        used_vehicle_ids = set(solution.routes.keys())
        idle_vehicles = []
        for v in vehicles:
            route = solution.routes.get(v.id)
            if not route or len(route.tasks) <= 2:
                idle_vehicles.append(v)
        
        if idle_vehicles:
            print(f"Debug Found {len(idle_vehicles)} idle vehicles for order relocation")
            
            # For each order in the current route
            for order_id, tasks in orders_in_route.items():
                for idle_vehicle in idle_vehicles:
                    # Create a lightweight copy with only affected vehicles
                    new_solution = create_two_route_copy(solution, from_vehicle_id, idle_vehicle.id)
                    
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
                    empty_route = _create_base_route(idle_vehicle)
                    
                    # Get the order object and use L2 heuristic to insert
                    order = next((o for o in orders if o.id == order_id), None)
                    if order:
                        optimized_route = l2_heuristic(empty_route, order)
                        if optimized_route and optimized_route.tasks:
                            new_solution.routes[idle_vehicle.id] = optimized_route
                            # Removed misleading "Successfully relocated" message during neighborhood exploration
                            # This function explores potential moves, not final assignments
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
                    # Create a lightweight copy with only affected vehicles
                    new_solution = create_two_route_copy(solution, route1_vehicle_id, route2_vehicle_id)
                    
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
                    
                    # Create a lightweight copy with only affected vehicles
                    new_solution = create_two_route_copy(solution, from_vehicle_id, to_vehicle_id)
                    
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
            # Create lightweight copy with only affected vehicle
            new_solution = create_single_route_copy(solution, vehicle_id)
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
    
    FLEXIBLE PRECEDENCE LOGIC:
    - For simple orders (1P-1D): pickup must come before delivery
    - For multiple simple orders: allows flexible interleaving (p1-p2-d2-d1)
    - For complex orders: conservative approach (all P before all D)
    
    Args:
        route: Route to validate
        
    Returns:
        True if route order is valid, False otherwise
    """
    if not route or not route.tasks:
        return True
    
    # Group tasks by order
    orders_tasks = {}
    
    for pos, task in enumerate(route.tasks):
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
        
        orders_tasks[order_id].append({'task_type': task_type, 'position': pos, 'task': task})
    
    # Check precedence constraints for each order
    for order_id, tasks in orders_tasks.items():
        pickups = [t for t in tasks if t['task_type'] == 'pickup']
        deliveries = [t for t in tasks if t['task_type'] == 'delivery']
        
        # Skip orders without both pickup and delivery
        if not pickups or not deliveries:
            continue
        
        # SIMPLE ORDERS: 1 pickup + 1 delivery - just check precedence
        if len(pickups) == 1 and len(deliveries) == 1:
            if pickups[0]['position'] >= deliveries[0]['position']:
                return False
        
        # COMPLEX ORDERS: Multiple pickups/deliveries - use conservative approach
        else:
            # For now, require all pickups before all deliveries for complex orders
            # This can be refined later with proper task pairing logic
            max_pickup_pos = max(p['position'] for p in pickups)
            min_delivery_pos = min(d['position'] for d in deliveries)
            
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
    Enhanced initializer that creates efficient pickup->pickup->delivery->delivery patterns.
    
    This initializer addresses the bouncing problem by:
    1. Grouping multiple orders per vehicle based on capacity
    2. Building complete routes with efficient task clustering  
    3. Creating pickup->pickup->delivery->delivery patterns instead of pickup->delivery->pickup->delivery
    """
    print(f"Build  Starting cluster-aware initializer with {len(orders)} orders and {len(vehicles)} vehicles")
    
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
            current_pallets = sum(order.get_total_pallets() for order in vehicle_assignments[vehicle.id])
            
            # Select orders that fit within vehicle capacity (including pallets)
            compatible_orders = []
            for order in unassigned_orders:
                weight_ok = current_weight + order.get_total_demand() <= vehicle.weight_capacity
                volume_ok = current_volume + order.get_total_volume() <= vehicle.volume_capacity
                pallets_ok = (vehicle.pallet_capacity is None or 
                            current_pallets + order.get_total_pallets() <= vehicle.pallet_capacity)
                
                if weight_ok and volume_ok and pallets_ok:
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
                          f"(total load: {current_weight + best_order.get_total_demand():.1f}kg, "
                          f"{current_volume + best_order.get_total_volume():.1f}m³, "
                          f"{current_pallets + best_order.get_total_pallets()} pallets)")
        
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
        current_route = _create_base_route(vehicle)
        
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
    
    This creates depot_start -> pickup->pickup->delivery->delivery -> depot_return patterns.
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
    
    # Phase 1: Insert all pickups in a cluster
    for pickup in all_pickups:
        best_cost = float('inf')
        best_route = None
        
        # Try inserting at each position using relaxed feasibility check
        # Skip position 0 (before depot start) and last position (after depot return)
        depot_start_positions = 1 if current_route.tasks and current_route.tasks[0].is_depot_start() else 0
        depot_return_offset = 1 if current_route.tasks and current_route.tasks[-1].is_depot_return() else 0
        for pos in range(depot_start_positions, len(current_route.tasks) - depot_return_offset + 1):
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
        
        # Respect depot boundaries for delivery insertion
        depot_return_offset = 1 if current_route.tasks and current_route.tasks[-1].is_depot_return() else 0
        max_pos = len(current_route.tasks) - depot_return_offset
        
        for pos in range(start_pos, max_pos + 1):
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
    
    print(f"Starting regret-{k} initializer with {len(orders)} orders and {len(vehicles)} vehicles")
    
    solution = Solution()
    unassigned_orders = orders.copy()
    
    # Initialize empty routes for all vehicles
    for vehicle in vehicles:
        solution.add_route(vehicle.id, _create_base_route(vehicle))
    
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
        
        # Step 1: Calculate insertion costs for all unassigned orders (optimized with L2 heuristic)
        for order in unassigned_orders:
            insertion_costs = []
            
            # Try inserting this order into every vehicle's route using L2 heuristic
            for vehicle in vehicles:
                current_route = solution.routes[vehicle.id]
                
                # Get all possible insertion positions for this order
                order_tasks = order.get_all_tasks()
                if not order_tasks:
                    continue
                
                # Use L2 heuristic to find optimal insertion for this vehicle
                try:
                    from second_level import l2_heuristic, calculate_z2_score
                    optimized_route = l2_heuristic(current_route, order, debug_assignment=False)
                    
                    if optimized_route:
                        cost = calculate_z2_score(optimized_route)
                        insertion_costs.append(cost)
                        
                        # Store the best insertion for this vehicle
                        if order.id not in order_costs:
                            order_costs[order.id] = []
                        order_costs[order.id].append({
                            'vehicle': vehicle,
                            'route': optimized_route,
                            'cost': cost,
                            'order': order
                        })
                        
                except Exception as e:
                    if debug_regret:
                        print(f"    Warning: Error evaluating insertion for {order.id}: {e}")
                    continue
            
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
                print(f"  OK: Inserted order {best_order.id} into vehicle {vehicle_id} (regret: {best_regret:.2f}, cost: {best_insertion['cost']:.2f})")
        else:
            # No feasible insertion found in this iteration - order will be retried or marked for alternative handling
            if unassigned_orders:
                problematic_order = unassigned_orders[0]
                print(f"Warning:  REGRET-{k} SKIP: Order {problematic_order.id} cannot be assigned in current iteration (will retry with alternative methods)")
                
                # Get actual order cargo details using proper methods
                total_weight = problematic_order.get_total_demand()
                total_volume = problematic_order.get_total_volume()
                total_pallets = sum(task.pallets for task in problematic_order.get_all_tasks())
                
                print(f"   Order details: Weight={total_weight:.1f}kg, Volume={total_volume:.2f}m³, Pallets={total_pallets}")
                
                # Extract time window information from actual tasks
                pickup_window = "N/A"
                delivery_window = "N/A"
                
                # Get time windows from pickup tasks
                if problematic_order.pickup_tasks:
                    pickup_task = problematic_order.pickup_tasks[0]
                    if pickup_task.earliest_time is not None or pickup_task.latest_time is not None:
                        earliest = f"{pickup_task.earliest_time:.0f}" if pickup_task.earliest_time is not None else "∞"
                        latest = f"{pickup_task.latest_time:.0f}" if pickup_task.latest_time is not None else "∞"
                        pickup_window = f"[{earliest}-{latest}]min"
                
                # Get time windows from delivery tasks
                if problematic_order.delivery_tasks:
                    delivery_task = problematic_order.delivery_tasks[0]
                    if delivery_task.earliest_time is not None or delivery_task.latest_time is not None:
                        earliest = f"{delivery_task.earliest_time:.0f}" if delivery_task.earliest_time is not None else "∞"
                        latest = f"{delivery_task.latest_time:.0f}" if delivery_task.latest_time is not None else "∞"
                        delivery_window = f"[{earliest}-{latest}]min"
                
                print(f"   Time constraints: Pickup {pickup_window}, Delivery {delivery_window}")
                print(f"   Priority: {problematic_order.priority}, Mandatory: {problematic_order.is_mandatory}")
                
                # Detailed diagnostic for each vehicle
                print(f"   DIAGNOSTIC BREAKDOWN FOR ALL {len(vehicles)} VEHICLES:")
                failed_reasons = {'capacity_weight': 0, 'capacity_volume': 0, 'time_window': 0, 'l2_heuristic': 0, 'other': 0}
                
                vehicles_shown = 0
                max_vehicles_to_show = 5  # Show detailed info for first 5 vehicles, count all failures
                
                for vehicle in vehicles:
                    current_route = solution.routes[vehicle.id]
                    failure_reasons = []
                    
                    # Calculate current weight load from route tasks
                    current_weight = 0
                    current_volume = 0
                    if hasattr(current_route, 'tasks') and current_route.tasks:
                        for task in current_route.tasks:
                            if hasattr(task, 'demand'):
                                current_weight += task.demand
                            if hasattr(task, 'volume'):
                                current_volume += task.volume
                    
                    available_weight = vehicle.weight_capacity - current_weight
                    available_volume = vehicle.volume_capacity - current_volume
                    order_weight = problematic_order.get_total_demand()
                    order_volume = problematic_order.get_total_volume()
                    
                    if order_weight > available_weight:
                        failure_reasons.append(f"Weight exceed: need {order_weight:.1f}kg, available {available_weight:.1f}kg")
                        failed_reasons['capacity_weight'] += 1
                    
                    if order_volume > available_volume:
                        failure_reasons.append(f"Volume exceed: need {order_volume:.2f}m³, available {available_volume:.2f}m³")
                        failed_reasons['capacity_volume'] += 1
                        failed_reasons['capacity_volume'] += 1
                    
                    # Check time window compatibility
                    order_earliest_day = getattr(problematic_order, 'pickup_day', 1)
                    order_latest_day = getattr(problematic_order, 'delivery_day', 1)
                    if order_earliest_day > 4 or order_latest_day > 4:  # Beyond 4-day planning horizon
                        failure_reasons.append(f"Time window: Days {order_earliest_day}-{order_latest_day} beyond planning horizon")
                        failed_reasons['time_window'] += 1
                    
                    # Try L2 heuristic and capture specific failure
                    l2_failed = False
                    try:
                        from second_level import l2_heuristic
                        order_tasks = problematic_order.get_all_tasks() if hasattr(problematic_order, 'get_all_tasks') else []
                        if order_tasks:
                            # Use enhanced diagnostics for the first few vehicles to show detailed L2 analysis
                            use_enhanced_diagnostics = (vehicles_shown < 2)  # Show detailed L2 analysis for first 2 vehicles
                            optimized_route = l2_heuristic(current_route, problematic_order, debug_assignment=False, enhanced_diagnostics=use_enhanced_diagnostics)
                            if not optimized_route:
                                failure_reasons.append("L2 heuristic rejection (infeasible route)")
                                failed_reasons['l2_heuristic'] += 1
                                l2_failed = True
                    except Exception as e:
                        failure_reasons.append(f"L2 error: {str(e)[:50]}")
                        failed_reasons['other'] += 1
                        l2_failed = True
                    
                    # If no specific failures detected but still can't assign, mark as other
                    if not failure_reasons and not l2_failed:
                        failure_reasons.append("Unknown constraint violation")
                        failed_reasons['other'] += 1
                    
                    # Show detailed breakdown for first few vehicles only
                    if vehicles_shown < max_vehicles_to_show and failure_reasons:
                        vehicle_type = getattr(vehicle, 'vehicle_type', 'unknown')
                        capacity_info = f"{getattr(vehicle, 'weight_capacity', 'N/A')}kg/{getattr(vehicle, 'volume_capacity', 'N/A')}m³"
                        print(f"     Vehicle {vehicle.id} ({vehicle_type}, {capacity_info}): {'; '.join(failure_reasons)}")
                        vehicles_shown += 1
                    
                # Show summary for all vehicles
                total_failures = sum(failed_reasons.values())
                print(f"   FAILURE SUMMARY (across {len(vehicles)} vehicles): Weight capacity: {failed_reasons['capacity_weight']}, Volume capacity: {failed_reasons['capacity_volume']}, Time windows: {failed_reasons['time_window']}, L2 heuristic: {failed_reasons['l2_heuristic']}, Other: {failed_reasons['other']}")
                
                if vehicles_shown < len(vehicles):
                    print(f"   (Detailed breakdown shown for first {vehicles_shown} vehicles only)")
                
                # ADVANCED ORDER HANDLING: Try advanced strategies before giving up
                print(f"\nATTEMPTING ADVANCED ASSIGNMENT for order {problematic_order.id}...")
                was_force_assigned = handle_unassigned_order(problematic_order, vehicles, solution)
                
                if was_force_assigned:
                    print(f"ADVANCED SUCCESS: Order {problematic_order.id} assigned using advanced strategies!")
                    # Remove from unassigned list and continue to next iteration
                    unassigned_orders.remove(problematic_order)
                else:
                    print(f"ADVANCED FAILURE: Order {problematic_order.id} could not be assigned even with advanced strategies.")
                    # Standard failure handling: add to unassigned list
                    unassigned_orders.remove(problematic_order)
                    if not hasattr(solution, 'unassigned_orders'):
                        solution.unassigned_orders = set()
                    solution.unassigned_orders.add(problematic_order.id)
    
    # Add depot tasks to all routes with actual orders
    for vehicle_id, route in solution.routes.items():
        if route.tasks and not any('depot' in str(task.id).lower() for task in route.tasks):
            _add_depot_tasks_to_route(route)
    
    assigned_count = len(orders) - len(unassigned_orders)
    if len(unassigned_orders) > 0:
        print(f"OK: Regret-{k} initialization completed: {assigned_count}/{len(orders)} orders assigned ({len(unassigned_orders)} remain for alternative assignment methods)")
    else:
        print(f"OK: Regret-{k} initialization completed: {assigned_count}/{len(orders)} orders assigned - PERFECT INITIALIZATION!")
    return solution


def _add_depot_tasks_to_route(route: 'Route'):
    """
    Helper function to add depot start and return tasks to a route.
    
    This function implements defensive programming by:
    1. Checking if depot tasks already exist before adding
    2. Handling edge cases (empty routes, None tasks)
    3. Ensuring proper error handling
    """
    try:
        from epdt_data_structures import Task
        
        # Defensive check: ensure route has a tasks list
        if not hasattr(route, 'tasks') or route.tasks is None:
            route.tasks = []
        
        vehicle = route.vehicle
        if not vehicle:
            print(f"Warning: Cannot add depot tasks to route without vehicle")
            return
            
        depot_location_id = getattr(vehicle, 'depot_id', 'main_depot')
        # FIXED: Get depot coordinates from vehicle instead of hardcoding
        depot_lat = getattr(vehicle, 'depot_lat', 44.9009)
        depot_lon = getattr(vehicle, 'depot_lon', 8.2057)
        
        # Defensive check: only add depot start if not already present
        needs_depot_start = True
        if route.tasks:
            first_task = route.tasks[0]
            if hasattr(first_task, 'is_depot_start') and first_task.is_depot_start():
                needs_depot_start = False
        
        if needs_depot_start:
            # Add depot start at beginning
            depot_start = Task.create_depot_start_task(
                vehicle_id=vehicle.id,
                depot_location_id=depot_location_id,
                depot_lat=depot_lat,
                depot_lon=depot_lon
            )
            route.insert_task_without_reordering(0, depot_start)
        
        # Defensive check: only add depot return if not already present
        needs_depot_return = True
        if route.tasks:
            last_task = route.tasks[-1]
            if hasattr(last_task, 'is_depot_return') and last_task.is_depot_return():
                needs_depot_return = False
        
        if needs_depot_return:
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


# ===== ADVANCED ORDER SPLITTING AND SEQUENCING FUNCTIONS =====

def handle_unassigned_order(order, vehicles, solution):
    """
    Orchestrates a series of advanced strategies to assign a difficult order.
    
    Args:
        order: The problematic order that failed standard insertion
        vehicles: List of available vehicles
        solution: Current solution to modify
        
    Returns:
        bool: True if order was successfully assigned, False otherwise
    """
    print(f"Executing advanced assignment for order {order.id}...")

    # Strategy 1: Try different single-vehicle sequencing
    if try_single_vehicle_strategies(order, vehicles, solution):
        print(f"SUCCESS: Assigned order {order.id} using a single-vehicle strategy.")
        return True

    # Strategy 2: Try splitting the order across multiple vehicles
    print(f"  Single-vehicle strategies failed. Attempting multi-vehicle splitting...")
    if try_multi_vehicle_splitting(order, vehicles, solution):
        print(f"SUCCESS: Assigned order {order.id} by splitting across multiple vehicles.")
        return True

    # Strategy 3: Try vehicle reallocation to free up large capacity vehicles
    print(f"  Multi-vehicle splitting failed. Attempting vehicle reallocation strategy...")
    if try_vehicle_reallocation(order, vehicles, solution):
        print(f"SUCCESS: Assigned order {order.id} using vehicle reallocation.")
        return True
    
    # Strategy 4: EMERGENCY MODE - Force assignment with minimal constraints
    print(f"  All strategies failed. Entering EMERGENCY MODE for order {order.id}...")
    if emergency_force_assignment(order, vehicles, solution):
        print(f"SUCCESS: Assigned order {order.id} using EMERGENCY MODE.")
        return True
    
    # Strategy 5: FINAL RESORT - Split into individual tasks
    print(f"  Emergency mode failed. FINAL RESORT: splitting order {order.id} into individual tasks...")
    if emergency_individual_task_splitting(order, vehicles, solution):
        print(f"SUCCESS: Assigned order {order.id} using individual task splitting.")
        return True

    print(f"FAILURE: All advanced strategies failed for order {order.id}.")
    return False


def try_single_vehicle_strategies(order, vehicles, solution):
    """
    Tries to assign all tasks of an order to a single vehicle using different sequencing.
    
    Args:
        order: Order to assign
        vehicles: List of available vehicles
        solution: Current solution to modify
        
    Returns:
        bool: True if successful assignment found, False otherwise
    """
    from second_level import l2_heuristic, calculate_z2_score
    
    print(f"  Trying single-vehicle strategies for order {order.id}...")
    
    # Iterate through all available vehicles
    for vehicle in vehicles:
        # Get the current route for this vehicle
        base_route = solution.routes[vehicle.id]
        
        print(f"    Testing vehicle {vehicle.id} (capacity: {vehicle.weight_capacity}kg, {vehicle.volume_capacity}m³)")
        
        # Strategy 1.1: Try clustered sequencing (P-P-D-D) PREFERRED
        print(f"      Trying clustered sequencing (P-P-D-D)...")
        clustered_route = l2_heuristic(base_route, order, sequencing_strategy='clustered')
        if clustered_route and clustered_route.is_feasible():
            solution.routes[vehicle.id] = clustered_route
            print(f"      SUCCESS: Clustered sequencing worked for vehicle {vehicle.id}")
            return True

        # Strategy 1.2: Try interleaved sequencing (P-D-P-D) FALLBACK ONLY
        print(f"      Clustered failed, trying interleaved fallback (P-D-P-D)...")
        interleaved_route = l2_heuristic(base_route, order, sequencing_strategy='interleaved')
        if interleaved_route and interleaved_route.is_feasible():
            solution.routes[vehicle.id] = interleaved_route
            print(f"      FALLBACK: Interleaved sequencing worked for vehicle {vehicle.id}")
            return True
            
        print(f"      FAILED: Neither sequencing strategy worked for vehicle {vehicle.id}")
    
    print(f"  FAILED: No single-vehicle strategy worked for order {order.id}")
    return False


def _create_base_route(vehicle):
    """
    Helper function to create a base route for a vehicle.
    Uses the same pattern as the existing code.
    
    Args:
        vehicle: Vehicle to create route for
        
    Returns:
        Route: Base route with depot tasks
    """
    try:
        from epdt_data_structures import Route
        route = Route(vehicle)
        # Add depot tasks if needed
        _add_depot_tasks_to_route(route)
        return route
    except Exception as e:
        print(f"Warning: Could not create base route for vehicle {vehicle.id}: {e}")
        # Fallback: create minimal route
        from epdt_data_structures import Route
        return Route(vehicle)


def try_multi_vehicle_splitting(order, vehicles, solution):
    """
    Tries to assign an order by splitting its tasks across multiple vehicles.
    This follows the advanced splitting instructions specifications.
    
    Args:
        order: Order to split across multiple vehicles
        vehicles: Available vehicles
        solution: Current solution to modify
        
    Returns:
        bool: True if successful assignment found, False otherwise
    """
    from itertools import combinations
    from second_level import l2_heuristic, calculate_z2_score
    
    print(f"  Attempting multi-vehicle task-based splitting for order {order.id}...")
    
    tasks = order.get_all_tasks()
    num_tasks = len(tasks)

    if num_tasks < 2:
        print(f"    Order has only {num_tasks} tasks - cannot split")
        return False
        
    print(f"    Order has {num_tasks} tasks to split across vehicles")

    # Iterate from a 50/50 split down to a 1 vs. N-1 split
    for i in range(num_tasks // 2, 0, -1):
        print(f"    Trying splits with {i} vs {num_tasks-i} tasks...")
        
        # Generate all combinations of `i` tasks for the first group
        combination_count = 0
        for task_group_1_indices in combinations(range(num_tasks), i):
            combination_count += 1
            if combination_count > 100:  # INCREASED from 50 - try more combinations for 100% assignment
                print(f"    Stopping after 100 combinations for performance")
                break
                
            task_group_1 = [tasks[j] for j in task_group_1_indices]
            task_group_2 = [tasks[j] for j in range(num_tasks) if j not in task_group_1_indices]

            # Create temporary orders for each task group
            order_split_1 = create_temp_order_from_tasks(task_group_1, f"{order.id}_split1", order)
            order_split_2 = create_temp_order_from_tasks(task_group_2, f"{order.id}_split2", order)
            
            if not order_split_1 or not order_split_2:
                continue

            # Find the best assignment for this specific split
            best_assignment = find_best_vehicle_pair_for_split(
                order_split_1, order_split_2, vehicles, solution
            )

            if best_assignment:
                # If a valid assignment is found, apply it to the main solution
                v1_id, route1, v2_id, route2 = best_assignment
                solution.routes[v1_id] = route1
                solution.routes[v2_id] = route2
                print(f"    SUCCESS: Split order into {v1_id} and {v2_id}")
                return True
                
    print(f"    FAILED: No feasible task-based split found")
    return False


def find_best_vehicle_pair_for_split(order1, order2, vehicles, solution):
    """
    Finds the best pair of vehicles to serve a two-way order split.
    
    Args:
        order1: First split order
        order2: Second split order
        vehicles: Available vehicles
        solution: Current solution
        
    Returns:
        Tuple of (v1_id, route1, v2_id, route2) or None if no valid assignment
    """
    from itertools import combinations
    from second_level import l2_heuristic, calculate_z2_score
    
    best_cost = float('inf')
    best_assignment = None

    # Iterate through all unique pairs of vehicles
    pair_count = 0
    for v1, v2 in combinations(vehicles, 2):
        pair_count += 1
        if pair_count > 50:  # INCREASED from 20 - try more vehicle pairs for 100% assignment
            break
            
        # Get base routes for vehicles (current routes or empty routes)
        base_route1 = solution.routes.get(v1.id, _create_base_route(v1))
        base_route2 = solution.routes.get(v2.id, _create_base_route(v2))
        
        # Try assigning order1 to v1 and order2 to v2
        route1 = l2_heuristic(base_route1, order1)
        if route1:
            route2 = l2_heuristic(base_route2, order2)
            if route2:
                cost = calculate_z2_score(route1) + calculate_z2_score(route2)
                if cost < best_cost:
                    best_cost = cost
                    best_assignment = (v1.id, route1, v2.id, route2)

        # Try assigning order1 to v2 and order2 to v1
        route1_rev = l2_heuristic(base_route2, order1)
        if route1_rev:
            route2_rev = l2_heuristic(base_route1, order2)
            if route2_rev:
                cost = calculate_z2_score(route1_rev) + calculate_z2_score(route2_rev)
                if cost < best_cost:
                    best_cost = cost
                    best_assignment = (v2.id, route1_rev, v1.id, route2_rev)

    return best_assignment


def consolidate_routes_for_idle_vehicles(solution, vehicles, target_idle_vehicles, params):
    """
    HYBRID CLUSTER-CONSOLIDATION: Post-initialization route consolidation.
    
    Merges routes to create idle vehicles while maintaining geographical efficiency.
    This gives us the speed of cluster-aware initialization with the utilization benefits of regret-k.
    
    Args:
        solution: Initial solution from cluster-aware initialization
        target_idle_vehicles: Number of idle vehicles to create
        params: Algorithm parameters
        
    Returns:
        Solution with consolidated routes and more idle vehicles
    """
    if not solution or not solution.routes:
        return solution
        
    print(f"Build    Starting consolidation: {len(solution.routes)} routes")
    
    # Get all routes with tasks (exclude depot-only routes)
    active_routes = []
    idle_vehicles = []
    
    for vehicle_id, route in solution.routes.items():
        non_depot_tasks = [task for task in route.tasks if not (task.is_depot_start() or task.is_depot_return())]
        if len(non_depot_tasks) > 0:
            total_weight = sum(abs(task.demand) for task in non_depot_tasks)
            total_volume = sum(abs(task.volume) for task in non_depot_tasks)  
            total_pallets = sum(abs(task.pallets) for task in non_depot_tasks)
            active_routes.append((vehicle_id, route, total_weight, total_volume, total_pallets, len(non_depot_tasks)))
        else:
            idle_vehicles.append(vehicle_id)
    
    print(f"Build    Current state: {len(active_routes)} active routes, {len(idle_vehicles)} idle vehicles")
    
    # If we already have enough idle vehicles, return
    current_idle = len(idle_vehicles) + (len(solution.routes.keys()) - len(active_routes))
    if current_idle >= target_idle_vehicles:
        print(f"Build    Already have {current_idle} idle vehicles (target: {target_idle_vehicles}) - no consolidation needed")
        return solution
    
    # Sort routes by load (lightest first) - these are best candidates for merging
    active_routes.sort(key=lambda x: x[5])  # Sort by number of tasks
    
    vehicles_to_free = target_idle_vehicles - current_idle
    consolidation_distance_penalty = params.get('consolidation_distance_penalty', 2.0)
    
    print(f"Build    Need to free {vehicles_to_free} vehicles through consolidation")
    
    consolidated_count = 0
    i = 0
    
    # Try to consolidate lightest routes into heavier ones
    while i < len(active_routes) and consolidated_count < vehicles_to_free:
        source_vehicle_id, source_route, s_weight, s_volume, s_pallets, s_tasks = active_routes[i]
        
        # Find a target route that can absorb this route
        best_target_idx = None
        
        for j in range(i + 1, len(active_routes)):
            target_vehicle_id, target_route, t_weight, t_volume, t_pallets, t_tasks = active_routes[j]
            
            # Get target vehicle capacity
            target_vehicle = next((v for v in vehicles if v.id == target_vehicle_id), None)
            if not target_vehicle:
                continue
                
            # Check if target can handle the combined load (with strict pallet constraints)
            combined_weight = s_weight + t_weight  
            combined_volume = s_volume + t_volume
            combined_pallets = s_pallets + t_pallets
            
            weight_ok = combined_weight <= target_vehicle.weight_capacity * 1.5  # Allow some overload
            volume_ok = combined_volume <= target_vehicle.volume_capacity * 1.5  # Allow some overload  
            pallets_ok = (target_vehicle.pallet_capacity is None or 
                         combined_pallets <= target_vehicle.pallet_capacity)  # STRICT pallet limit
            
            if weight_ok and volume_ok and pallets_ok:
                best_target_idx = j
                break
        
        if best_target_idx is not None:
            # Consolidate source route into target route
            target_vehicle_id, target_route, _, _, _, _ = active_routes[best_target_idx]
            
            # Move all non-depot tasks from source to target
            source_tasks = [task for task in source_route.tasks if not (task.is_depot_start() or task.is_depot_return())]
            
            for task in source_tasks:
                # Insert before the depot return task
                insert_pos = len(target_route.tasks) - 1  # Before depot return
                target_route.tasks.insert(insert_pos, task)
            
            # Clear the source route (keep only depot tasks)  
            depot_tasks = [task for task in source_route.tasks if (task.is_depot_start() or task.is_depot_return())]
            source_route.tasks = depot_tasks
            
            print(f"Build    Consolidated route {source_vehicle_id} ({s_tasks} tasks) into {target_vehicle_id}")
            consolidated_count += 1
            
            # Remove source from active list
            active_routes.pop(i)
            
            # Continue without incrementing i (since we removed an element)
        else:
            i += 1
    
    # Update the solution
    new_idle_count = current_idle + consolidated_count
    print(f"Build    Consolidation completed: {consolidated_count} routes consolidated, {new_idle_count} idle vehicles available")
    
    return solution


def emergency_individual_task_splitting(order, vehicles, solution):
    """
    Final resort for 100% assignment - split order into individual tasks and assign separately.
    This creates micro-orders for each pickup-delivery pair.
    
    Args:
        order: Order to split into individual tasks
        vehicles: Available vehicles  
        solution: Current solution to modify
        
    Returns:
        bool: True if at least one task was assigned, False otherwise
    """
    from second_level import l2_heuristic
    
    print(f"  FINAL RESORT: Individual task splitting for order {order.id}...")
    
    # Create individual orders for each pickup-delivery pair
    individual_success_count = 0
    
    # Process pickup tasks
    for i, pickup_task in enumerate(order.pickup_tasks):
        # Find corresponding delivery
        corresponding_delivery = None
        for delivery_task in order.delivery_tasks:
            if (hasattr(pickup_task, 'order_id') and hasattr(delivery_task, 'order_id') and
                pickup_task.order_id == delivery_task.order_id):
                corresponding_delivery = delivery_task
                break
        
        # Create micro-order
        micro_order_id = f"{order.id}_individual_{i+1}"
        try:
            from epdt_data_structures import Order
            micro_order = Order(
                id=micro_order_id,
                pickup_tasks=[pickup_task] if pickup_task else [],
                delivery_tasks=[corresponding_delivery] if corresponding_delivery else [],
                priority=order.priority,
                is_urgent=order.is_urgent,
                is_mandatory=order.is_mandatory
            )
            
            print(f"    Created micro-order {micro_order_id}")
            
            # Try to assign this micro-order using emergency force assignment
            if emergency_force_assignment(micro_order, vehicles, solution):
                individual_success_count += 1
                print(f"    SUCCESS: Assigned micro-order {micro_order_id}")
            else:
                print(f"    FAILED: Could not assign micro-order {micro_order_id}")
                
        except Exception as e:
            print(f"    ERROR creating micro-order: {e}")
            continue
    
    if individual_success_count > 0:
        print(f"  INDIVIDUAL SPLITTING SUCCESS: {individual_success_count}/{len(order.pickup_tasks)} tasks assigned")
        return True
    else:
        print(f"  INDIVIDUAL SPLITTING FAILED: No tasks could be assigned")
        return False


def emergency_force_assignment(order, vehicles, solution):
    """
    Emergency mode for 100% assignment - uses minimal constraints.
    This is the last resort method that will assign orders with almost no restrictions.
    
    Args:
        order: Order to assign  
        vehicles: Available vehicles
        solution: Current solution to modify
        
    Returns:
        bool: True if successful assignment found, False otherwise
    """
    from second_level import l2_heuristic
    
    print(f"  EMERGENCY MODE for order {order.id}...")
    
    # Sort vehicles by available capacity (largest first) and prioritize truly idle vehicles
    available_vehicles = []
    for vehicle in vehicles:
        current_route = solution.routes.get(vehicle.id)
        if current_route:
            # Calculate remaining capacity - prioritize vehicles with more remaining capacity
            current_weight = sum(task.demand for task in current_route.tasks if hasattr(task, 'demand'))
            remaining_weight = vehicle.weight_capacity - current_weight
            
            # Check if vehicle is essentially idle (only depot tasks)
            non_depot_tasks = [task for task in current_route.tasks if not hasattr(task, 'task_type') or task.task_type not in ['DEPOT_START', 'DEPOT_RETURN']]
            is_idle = len(non_depot_tasks) == 0
            
            # Prioritize idle vehicles by giving them artificially high remaining capacity
            priority_weight = remaining_weight + (10000 if is_idle else 0)
            available_vehicles.append((vehicle, remaining_weight, is_idle, priority_weight))
        else:
            # Vehicle has no route - completely idle, highest priority
            available_vehicles.append((vehicle, vehicle.weight_capacity, True, vehicle.weight_capacity + 20000))
    
    # Sort by priority (idle vehicles first, then by remaining capacity)
    available_vehicles.sort(key=lambda x: x[3], reverse=True)
    
    print(f"    Emergency vehicle prioritization:")
    for i, (vehicle, remaining, is_idle, priority) in enumerate(available_vehicles[:5]):
        status = "IDLE" if is_idle else f"{remaining:.0f}kg remaining"
        print(f"      {i+1}. {vehicle.id}: {status}")
    
    # Try each vehicle with MINIMAL constraints, prioritizing idle vehicles
    for vehicle, remaining_capacity, is_idle, priority_weight in available_vehicles[:15]:  # Try top 15 vehicles
        base_route = solution.routes.get(vehicle.id, _create_base_route(vehicle))
        
        print(f"    Emergency attempt: vehicle {vehicle.id} ({'IDLE' if is_idle else f'{remaining_capacity:.0f}kg remaining'})")
        
        # Try with MAXIMUM constraint relaxation
        # This bypasses most feasibility checks
        try:
            # Create a modified route by direct task insertion
            emergency_route = base_route.copy()
            
            # Insert all order tasks at the end of route (before depot return)
            insert_position = len(emergency_route.tasks) - 1  # Before depot return
            
            # Add all pickup tasks first
            for pickup_task in order.pickup_tasks:
                emergency_route.insert_task_without_reordering(insert_position, pickup_task)
                insert_position += 1
            
            # Add all delivery tasks after pickups
            for delivery_task in order.delivery_tasks:
                emergency_route.insert_task_without_reordering(insert_position, delivery_task)
                insert_position += 1
            
            # Basic feasibility check - only reject if truly impossible
            order_weight = order.get_total_demand()
            
            # Be more lenient for idle vehicles
            weight_tolerance = 2.0 if is_idle else 1.0  # Idle vehicles can handle more overload
            if remaining_capacity > -vehicle.weight_capacity * weight_tolerance:  
                solution.routes[vehicle.id] = emergency_route
                print(f"    EMERGENCY SUCCESS: Assigned to {vehicle.id} ({'idle vehicle' if is_idle else 'active vehicle'})")
                return True
            else:
                print(f"    Emergency rejected: would exceed {int(weight_tolerance*100)}% capacity limit")
                
        except Exception as e:
            print(f"    Emergency assignment failed for {vehicle.id}: {e}")
            continue
    
    print(f"  EMERGENCY MODE FAILED: Could not assign order {order.id}")
    return False


def create_temp_order_from_tasks(tasks, new_order_id, original_order):
    """
    Helper to create a temporary Order object from a list of tasks.
    This is needed to pass to the l2_heuristic.
    
    Args:
        tasks: List of tasks for the new order
        new_order_id: ID for the temporary order
        original_order: Original order for copying attributes
        
    Returns:
        Order object or None if creation fails
    """
    if not tasks:
        return None
        
    try:
        # Import here to avoid circular imports
        from epdt_data_structures import Order
        
        # Separate pickups and deliveries
        pickup_tasks = [t for t in tasks if t.is_pickup()]
        delivery_tasks = [t for t in tasks if t.is_delivery()]
        
        # Create new order with split tasks
        temp_order = Order(
            id=new_order_id,
            pickup_tasks=pickup_tasks,
            delivery_tasks=delivery_tasks,
            priority=original_order.priority,
            is_urgent=original_order.is_urgent,
            is_mandatory=original_order.is_mandatory
        )
        
        return temp_order
        
    except Exception as e:
        print(f"      ERROR creating temp order: {e}")
        return None
        print(f"    Could not identify pickup-delivery pairs")
        return False
        
    print(f"    Identified {len(pickup_delivery_pairs)} pickup-delivery pairs")
    
    # Step 2: Try different ways to split pairs across vehicles
    # Start with the most balanced split and work toward more uneven splits
    
    num_pairs = len(pickup_delivery_pairs)
    
    # PRIORITY ENHANCEMENT: Sort pairs by weight/volume to prioritize larger pairs
    pickup_delivery_pairs_prioritized = sorted(pickup_delivery_pairs, 
                                               key=lambda pair: sum(task.demand for task in [pair[0]] if task.is_pickup()), 
                                               reverse=True)  # Largest pairs first
    
    print(f"    Prioritized pairs by weight (largest first):")
    for i, (pickup, delivery) in enumerate(pickup_delivery_pairs_prioritized):
        pair_weight = pickup.demand if pickup.is_pickup() else 0
        print(f"      Pair {i+1}: {pair_weight:.1f}kg from {getattr(pickup, 'location_id', 'unknown')} to {getattr(delivery, 'location_id', 'unknown')}")
    
    # Use prioritized pairs for splitting
    pickup_delivery_pairs = pickup_delivery_pairs_prioritized
    
    # Try splitting into different numbers of groups (2 to min(num_pairs, num_vehicles))
    max_groups = min(num_pairs, len(vehicles), 4)  # Limit to 4 vehicles max for performance
    
    for num_groups in range(2, max_groups + 1):
        print(f"    Trying to split {num_pairs} pairs into {num_groups} groups...")
        
        # Generate all ways to partition pairs into num_groups
        pair_partitions = generate_pair_partitions(pickup_delivery_pairs, num_groups)
        
        # Try assigning each partition to vehicles
        for partition in pair_partitions:
            if try_assign_pair_partition(partition, vehicles, solution, order.id):
                print(f"    SUCCESS: Split order into {len(partition)} vehicle assignments")
                return True
    
    print(f"    FAILED: No feasible pair-based split found")
    return False


def identify_pickup_delivery_pairs(pickups, deliveries):
    """
    Identify which pickups match with which deliveries to form logical pairs.
    
    This handles different pairing strategies:
    1. Same location pairs (pickup and delivery at same location)
    2. Route-based pairs (pickup at A, delivery at B)
    3. Task sequence pairs (based on task numbering or ordering)
    
    Returns:
        List of (pickup, delivery) tuples representing pairs
    """
    pairs = []
    
    # Strategy 1: Try pairing by location proximity
    # This works for round-trip scenarios (same pickup/delivery locations)
    used_delivery_ids = set()  # Use task IDs instead of Task objects
    
    for pickup in pickups:
        best_delivery = None
        best_distance = float('inf')
        
        for delivery in deliveries:
            if delivery.id in used_delivery_ids:  # Use task ID for comparison
                continue
                
            # Calculate distance between pickup and delivery
            if hasattr(pickup, 'lat') and hasattr(delivery, 'lat'):
                distance = calculate_distance(pickup.lat, pickup.lon, delivery.lat, delivery.lon)
                if distance < best_distance:
                    best_distance = distance
                    best_delivery = delivery
        
        if best_delivery and best_distance < 50:  # 50km threshold for pairing
            pairs.append((pickup, best_delivery))
            used_delivery_ids.add(best_delivery.id)  # Use task ID
            print(f"      Paired pickup {getattr(pickup, 'id', 'unknown')} with delivery {getattr(best_delivery, 'id', 'unknown')} (distance: {best_distance:.1f}km)")
    
    # Strategy 2: If location pairing didn't work well, try sequential pairing
    if len(pairs) < min(len(pickups), len(deliveries)):
        print(f"      Location pairing only found {len(pairs)} pairs, trying sequential pairing...")
        pairs.clear()
        used_delivery_ids.clear()
        
        # Simple sequential pairing: pair pickups[i] with deliveries[i]
        for i in range(min(len(pickups), len(deliveries))):
            pairs.append((pickups[i], deliveries[i]))
            print(f"      Sequentially paired pickup {i} with delivery {i}")
    
    return pairs


def calculate_distance(lat1, lon1, lat2, lon2):
    """Simple Haversine distance calculation"""
    import math
    R = 6371  # Earth radius in km
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat/2) * math.sin(dlat/2) + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon/2) * math.sin(dlon/2))
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance


def generate_pair_partitions(pairs, num_groups):
    """
    Generate different ways to partition pairs into groups.
    
    For efficiency, we limit to reasonable partitioning strategies:
    1. Balanced partitioning (distribute pairs evenly)
    2. Sequential partitioning (group consecutive pairs)
    
    Args:
        pairs: List of (pickup, delivery) pairs
        num_groups: Number of groups to create
        
    Returns:
        List of partitions, where each partition is a list of groups
    """
    if num_groups >= len(pairs):
        # One pair per group
        return [[[pair] for pair in pairs]]
    
    partitions = []
    
    # Strategy 1: Balanced distribution
    # Distribute pairs as evenly as possible across groups
    pairs_per_group = len(pairs) // num_groups
    remainder = len(pairs) % num_groups
    
    balanced_partition = []
    start_idx = 0
    
    for group_idx in range(num_groups):
        group_size = pairs_per_group + (1 if group_idx < remainder else 0)
        group_pairs = pairs[start_idx:start_idx + group_size]
        if group_pairs:  # Only add non-empty groups
            balanced_partition.append(group_pairs)
        start_idx += group_size
    
    if balanced_partition:
        partitions.append(balanced_partition)
        print(f"      Generated balanced partition: {[len(group) for group in balanced_partition]} pairs per group")
    
    return partitions


def try_assign_pair_partition(partition, vehicles, solution, order_id):
    """
    Try to assign each group of pairs in the partition to different vehicles.
    
    Args:
        partition: List of groups, where each group contains pairs
        vehicles: Available vehicles  
        solution: Current solution to modify
        order_id: Original order ID for tracking
        
    Returns:
        bool: True if successful assignment found
    """
    from second_level import l2_heuristic
    
    # Find the best vehicle assignment for each group
    group_assignments = []
    
    for group_idx, pair_group in enumerate(partition):
        print(f"      Finding vehicle for group {group_idx+1} with {len(pair_group)} pairs...")
        
        # Create temporary order from this group of pairs
        temp_order = create_temp_order_from_pair_group(pair_group, f"{order_id}_group{group_idx+1}")
        
        if not temp_order:
            print(f"        Failed to create temporary order for group {group_idx+1}")
            return False
        
        # Calculate requirements for this group using correct Task attributes
        group_weight = sum(task.demand for task in temp_order.pickup_tasks)  # demand = weight in kg
        group_volume = sum(task.volume for task in temp_order.pickup_tasks)  # volume = volume in m³
        group_pallets = sum(task.pallets for task in temp_order.pickup_tasks)  # pallets = number of pallets
        print(f"        Group {group_idx+1} requirements: {group_weight:.1f}kg, {group_volume:.2f}m³, {group_pallets} pallets")
        
        # DEBUG: Add detailed info about the tasks in this group
        print(f"          DEBUG: Group {group_idx+1} contains {len(temp_order.pickup_tasks)} pickups and {len(temp_order.delivery_tasks)} deliveries")
        for i, pickup in enumerate(temp_order.pickup_tasks):
            print(f"            DEBUG: Pickup {i+1}: {pickup.demand:.1f}kg, {pickup.volume:.2f}m³, {pickup.pallets}p at {getattr(pickup, 'location_id', 'unknown')}")
        for i, delivery in enumerate(temp_order.delivery_tasks):
            print(f"            DEBUG: Delivery {i+1}: {delivery.demand:.1f}kg, {delivery.volume:.2f}m³, {delivery.pallets}p at {getattr(delivery, 'location_id', 'unknown')}")
            
        # DEBUG: Check if any tasks have time window constraints
        has_time_constraints = any(hasattr(task, 'time_window') and task.time_window for task in temp_order.pickup_tasks + temp_order.delivery_tasks)
        if has_time_constraints:
            print(f"          DEBUG: Group {group_idx+1} has time window constraints")
            for task in temp_order.pickup_tasks + temp_order.delivery_tasks:
                if hasattr(task, 'time_window') and task.time_window:
                    print(f"            DEBUG: Task {task.id} window: {task.time_window}")
        else:
            print(f"          DEBUG: Group {group_idx+1} has no time window constraints")
        
        # Find best vehicle for this group
        best_vehicle = None
        best_route = None
        best_cost = float('inf')
        tested_vehicles = 0
        
        for vehicle in vehicles:
            # Skip vehicles already assigned in this partition
            if any(assignment[0].id == vehicle.id for assignment in group_assignments):
                continue
                
            base_route = solution.routes[vehicle.id]
            tested_vehicles += 1
            
            # Quick capacity check before L2 heuristic
            if (group_weight > vehicle.weight_capacity or 
                group_volume > vehicle.volume_capacity or
                group_pallets > vehicle.pallet_capacity):
                print(f"        Vehicle {vehicle.id}: CAPACITY EXCEEDED - need {group_weight:.1f}kg/{group_volume:.2f}m³/{group_pallets}p, available {vehicle.weight_capacity:.1f}kg/{vehicle.volume_capacity:.2f}m³/{vehicle.pallet_capacity}p")
                continue
            
            # Try assigning this group to this vehicle with enhanced soft time window strategies
            print(f"          DEBUG: Testing group {group_idx+1} on vehicle {vehicle.id} with soft time window strategies...")
            
            best_strategy_route = None
            best_strategy_cost = float('inf')
            
            # STRATEGY 1: Interleaved sequencing (P-D-P-D) - better for tight time windows
            test_route_interleaved = l2_heuristic(base_route, temp_order, sequencing_strategy='interleaved', enhanced_diagnostics=True)
            
            # Check if this route can be made feasible by allowing time window violations
            if test_route_interleaved:
                route_acceptable = False
                if test_route_interleaved.is_feasible():
                    route_acceptable = True
                else:
                    # Check if it's only time window violations that make it infeasible
                    try:
                        from second_level import is_feasible
                        feasibility_result = is_feasible(test_route_interleaved, debug_feasibility=True, return_reason=True, allow_time_window_violations=True)
                        if feasibility_result[0]:  # Feasible if we ignore time window violations
                            print(f"            DEBUG: Strategy 1 route is feasible except for time windows - ACCEPTING with penalty")
                            test_route_interleaved._force_feasible_for_time_violations = True
                            route_acceptable = True
                    except Exception as e:
                        print(f"            DEBUG: Could not check soft feasibility for strategy 1: {e}")
                
                if route_acceptable:
                    from second_level import calculate_z2_score
                    cost_interleaved = calculate_z2_score(test_route_interleaved)
                    
                    # Apply priority boost for large orders (lower cost = higher priority)
                    if group_weight > 20000:  # Large orders get priority boost
                        cost_interleaved *= 0.8  # 20% cost reduction for large orders
                        print(f"            DEBUG: Applied large order priority boost (20% cost reduction)")
                    
                    if cost_interleaved < best_strategy_cost:
                        best_strategy_cost = cost_interleaved
                        best_strategy_route = test_route_interleaved
                        print(f"            DEBUG: Strategy 1 (interleaved P-D-P-D) succeeded with cost: {cost_interleaved:.2f}")
            
            # STRATEGY 2: Clustered sequencing (P-P-D-D) if strategy 1 failed
            if not best_strategy_route:
                test_route_clustered = l2_heuristic(base_route, temp_order, sequencing_strategy='clustered', enhanced_diagnostics=True)
                
                if test_route_clustered:
                    route_acceptable = False
                    if test_route_clustered.is_feasible():
                        route_acceptable = True
                    else:
                        # Check if it's only time window violations that make it infeasible
                        try:
                            from second_level import is_feasible
                            feasibility_result = is_feasible(test_route_clustered, debug_feasibility=True, return_reason=True, allow_time_window_violations=True)
                            if feasibility_result[0]:  # Feasible if we ignore time window violations
                                print(f"            DEBUG: Strategy 2 route is feasible except for time windows - ACCEPTING with penalty")
                                test_route_clustered._force_feasible_for_time_violations = True
                                route_acceptable = True
                        except Exception as e:
                            print(f"            DEBUG: Could not check soft feasibility for strategy 2: {e}")
                    
                    if route_acceptable:
                        from second_level import calculate_z2_score
                        cost_clustered = calculate_z2_score(test_route_clustered)
                        
                        # Apply priority boost for large orders
                        if group_weight > 20000:
                            cost_clustered *= 0.8
                            print(f"            DEBUG: Applied large order priority boost (20% cost reduction)")
                        
                        if cost_clustered < best_strategy_cost:
                            best_strategy_cost = cost_clustered
                            best_strategy_route = test_route_clustered
                            print(f"            DEBUG: Strategy 2 (clustered P-P-D-D) succeeded with cost: {cost_clustered:.2f}")
            
            # STRATEGY 3: Standard fallback if both sequencing strategies failed
            if not best_strategy_route:
                test_route = l2_heuristic(base_route, temp_order, enhanced_diagnostics=True)
                
                if test_route:
                    route_acceptable = False
                    if test_route.is_feasible():
                        route_acceptable = True
                    else:
                        # Check if it's only time window violations that make it infeasible
                        try:
                            from second_level import is_feasible
                            feasibility_result = is_feasible(test_route, debug_feasibility=True, return_reason=True, allow_time_window_violations=True)
                            if feasibility_result[0]:  # Feasible if we ignore time window violations
                                print(f"            DEBUG: Strategy 3 route is feasible except for time windows - ACCEPTING with penalty")
                                test_route._force_feasible_for_time_violations = True
                                route_acceptable = True
                        except Exception as e:
                            print(f"            DEBUG: Could not check soft feasibility for strategy 3: {e}")
                    
                    if route_acceptable:
                        from second_level import calculate_z2_score
                        cost_standard = calculate_z2_score(test_route)
                        
                        # Apply priority boost for large orders
                        if group_weight > 20000:
                            cost_standard *= 0.8
                            print(f"            DEBUG: Applied large order priority boost (20% cost reduction)")
                        
                        if cost_standard < best_strategy_cost:
                            best_strategy_cost = cost_standard
                            best_strategy_route = test_route
                            print(f"            DEBUG: Strategy 3 (standard) succeeded with cost: {cost_standard:.2f}")
            
            # Evaluate the best strategy result
            final_acceptable = best_strategy_route and (best_strategy_route.is_feasible() or getattr(best_strategy_route, '_force_feasible_for_time_violations', False))
            
            if final_acceptable:
                if best_strategy_cost < best_cost:
                    best_cost = best_strategy_cost
                    best_vehicle = vehicle
                    best_route = best_strategy_route
                    print(f"        Vehicle {vehicle.id}: FEASIBLE assignment found (cost: {best_strategy_cost:.2f}) with soft time windows")
            else:
                print(f"        Vehicle {vehicle.id}: All L2 sequencing strategies rejected or have non-time-window violations")
                
                # DEBUG: Add detailed failure analysis
                if test_route:
                    print(f"          DEBUG: L2 returned route but is_feasible() = False")
                    try:
                        from second_level import is_feasible
                        feasibility_result = is_feasible(test_route, debug_feasibility=True, return_reason=True)
                        print(f"          DEBUG: Feasibility failure: {feasibility_result}")
                    except Exception as e:
                        print(f"          DEBUG: Could not get feasibility details: {e}")
                else:
                    print(f"          DEBUG: L2 heuristic returned None (complete rejection)")
        
        print(f"        Tested {tested_vehicles} vehicles for group {group_idx+1}")
        
        if best_vehicle and best_route:
            group_assignments.append((best_vehicle, best_route))
            print(f"        Group {group_idx+1} assigned to vehicle {best_vehicle.id} (cost: {best_cost:.2f})")
        else:
            print(f"        Failed to assign group {group_idx+1} to any vehicle")
            
            # NEW: Try vehicle reallocation for this specific pair group
            print(f"        Attempting vehicle reallocation for individual pair group {group_idx+1}...")
            if try_vehicle_reallocation_for_pair(temp_order, vehicles, solution):
                print(f"        SUCCESS: Pair group {group_idx+1} assigned using vehicle reallocation!")
                # The reallocation function will have modified the solution directly
                # Add a placeholder assignment to continue the process
                group_assignments.append(("reallocation_success", None))
            else:
                print(f"        FAILED: Vehicle reallocation also failed for pair group {group_idx+1}")
                return False
    
    # If we successfully assigned all groups, apply the assignments
    if len(group_assignments) == len(partition):
        for vehicle, route in group_assignments:
            # Handle reallocation success cases (where assignments are already applied)
            if vehicle != "reallocation_success":
                solution.routes[vehicle.id] = route
        
        print(f"      Successfully applied all {len(group_assignments)} group assignments")
        return True
    
    return False


def create_temp_order_from_pair_group(pair_group, new_order_id):
    """
    Create a temporary order from a group of pickup-delivery pairs.
    
    Args:
        pair_group: List of (pickup, delivery) tuples
        new_order_id: ID for the temporary order
        
    Returns:
        Order: Temporary order containing all pairs, or None if creation fails
    """
    try:
        from epdt_data_structures import Order
        
        if not pair_group:
            return None
        
        # Create new order
        temp_order = Order(id=new_order_id)
        
        # Add all pickups and deliveries from the pairs
        for pickup, delivery in pair_group:
            temp_order.pickup_tasks.append(pickup)
            temp_order.delivery_tasks.append(delivery)
        
        return temp_order
        
    except Exception as e:
        print(f"    Warning: Could not create temporary order from pair group: {e}")
        return None


def try_vehicle_reallocation(order, vehicles, solution):
    """
    Vehicle reallocation strategy: Free up large capacity vehicles by moving their current
    orders to smaller vehicles, then try to assign the large problematic order to the freed vehicle.
    
    This strategy targets scenarios where large vehicles are assigned to small orders while
    large orders cannot be assigned due to lack of large vehicle capacity.
    
    Args:
        order: Large order that failed assignment
        vehicles: Available vehicles
        solution: Current solution to modify
        
    Returns:
        bool: True if successful reallocation and assignment found, False otherwise
    """
    from second_level import l2_heuristic
    
    print(f"  Attempting vehicle reallocation strategy for order {order.id}...")
    
    # Step 1: Calculate the requirements of the problematic order
    order_weight = sum(task.demand for task in order.pickup_tasks)
    order_volume = sum(task.volume for task in order.pickup_tasks)
    order_pallets = sum(task.pallets for task in order.pickup_tasks)
    
    print(f"    Order requirements: {order_weight:.1f}kg, {order_volume:.2f}m³, {order_pallets} pallets")
    
    # Step 2: Identify large capacity vehicles that could handle this order
    large_vehicles = []
    for vehicle in vehicles:
        if (vehicle.weight_capacity >= order_weight and 
            vehicle.volume_capacity >= order_volume and
            vehicle.pallet_capacity >= order_pallets):
            large_vehicles.append(vehicle)
    
    if not large_vehicles:
        print(f"    No vehicles found with sufficient capacity for order {order.id}")
        return False
    
    print(f"    Found {len(large_vehicles)} vehicles with sufficient capacity")
    
    # Step 3: For each large vehicle, check if its current order can be moved elsewhere
    for large_vehicle in large_vehicles:
        print(f"    Analyzing vehicle {large_vehicle.id} (capacity: {large_vehicle.weight_capacity}kg, {large_vehicle.volume_capacity:.1f}m³, {large_vehicle.pallet_capacity}p)")
        
        current_route = solution.routes[large_vehicle.id]
        
        # Skip if vehicle has no customer tasks
        customer_tasks = [task for task in current_route.tasks if not (task.is_depot_start() or task.is_depot_return())]
        if not customer_tasks:
            print(f"      Vehicle {large_vehicle.id} is empty, attempting direct assignment...")
            # Try direct assignment to empty vehicle
            base_route = solution.routes[large_vehicle.id]
            test_route = l2_heuristic(base_route, order)
            if test_route and test_route.is_feasible():
                solution.routes[large_vehicle.id] = test_route
                print(f"      SUCCESS: Direct assignment to empty vehicle {large_vehicle.id}")
                return True
            continue
        
        # Step 4: Identify the current order on this vehicle
        current_order_tasks = customer_tasks
        if not current_order_tasks:
            continue
        
        # Calculate current order requirements
        current_weight = sum(task.demand for task in current_order_tasks if task.is_pickup())
        current_volume = sum(task.volume for task in current_order_tasks if task.is_pickup())
        current_pallets = sum(task.pallets for task in current_order_tasks if task.is_pickup())
        
        print(f"      Current assignment: {current_weight:.1f}kg, {current_volume:.2f}m³, {current_pallets} pallets")
        
        # Step 5: Try to move current order to a smaller vehicle
        candidate_vehicles = []
        for candidate in vehicles:
            if (candidate.id != large_vehicle.id and 
                candidate.weight_capacity >= current_weight and
                candidate.volume_capacity >= current_volume and
                candidate.pallet_capacity >= current_pallets):
                candidate_vehicles.append(candidate)
        
        if not candidate_vehicles:
            print(f"      No suitable alternative vehicles found for current order")
            continue
        
        print(f"      Found {len(candidate_vehicles)} candidate vehicles for reallocation")
        
        # Step 6: Try reallocation
        successful_reallocation = False
        
        for candidate_vehicle in candidate_vehicles:
            # Check if candidate vehicle is empty or has space
            candidate_route = solution.routes[candidate_vehicle.id]
            candidate_tasks = [task for task in candidate_route.tasks if not (task.is_depot_start() or task.is_depot_return())]
            
            if candidate_tasks:
                print(f"        Candidate vehicle {candidate_vehicle.id} is already occupied, skipping...")
                continue
            
            # Try to assign current order to candidate vehicle
            # First, create a temporary order from current tasks
            temp_order = create_temp_order_from_tasks(current_order_tasks, f"temp_order_{large_vehicle.id}")
            if not temp_order:
                continue
            
            test_route = l2_heuristic(candidate_route, temp_order)
            if test_route and test_route.is_feasible():
                print(f"        Current order can be moved to vehicle {candidate_vehicle.id}")
                
                # Step 7: Now try to assign the large problematic order to the freed large vehicle
                # Create empty route for the large vehicle
                empty_route = _create_base_route(large_vehicle)
                large_order_route = l2_heuristic(empty_route, order)
                
                if large_order_route and large_order_route.is_feasible():
                    # Success! Apply both assignments
                    solution.routes[candidate_vehicle.id] = test_route
                    solution.routes[large_vehicle.id] = large_order_route
                    
                    print(f"      SUCCESS: Moved current order to vehicle {candidate_vehicle.id} and assigned large order to vehicle {large_vehicle.id}")
                    successful_reallocation = True
                    break
                else:
                    print(f"        Large order still cannot be assigned to freed vehicle {large_vehicle.id}")
        
        if successful_reallocation:
            return True
    
    print(f"    FAILED: Vehicle reallocation strategy could not assign order {order.id}")
    return False


def try_vehicle_reallocation_for_pair(pair_order, vehicles, solution):
    """
    Vehicle reallocation strategy specifically for individual pickup-delivery pairs.
    
    This is optimized for smaller orders (individual pairs) that should fit in large
    vehicles if we can free them up from smaller orders.
    
    Args:
        pair_order: Individual pickup-delivery pair order
        vehicles: Available vehicles
        solution: Current solution to modify
        
    Returns:
        bool: True if successful reallocation and assignment found
    """
    from second_level import l2_heuristic
    
    print(f"    Attempting pair-level vehicle reallocation for order {pair_order.id}...")
    
    # Calculate requirements for this pair
    pair_weight = sum(task.demand for task in pair_order.pickup_tasks)
    pair_volume = sum(task.volume for task in pair_order.pickup_tasks)
    pair_pallets = sum(task.pallets for task in pair_order.pickup_tasks)
    
    print(f"      Pair requirements: {pair_weight:.1f}kg, {pair_volume:.2f}m³, {pair_pallets} pallets")
    
    # Find large vehicles that could handle this pair
    suitable_vehicles = []
    for vehicle in vehicles:
        if (vehicle.weight_capacity >= pair_weight and 
            vehicle.volume_capacity >= pair_volume and
            vehicle.pallet_capacity >= pair_pallets):
            suitable_vehicles.append(vehicle)
    
    if not suitable_vehicles:
        print(f"      No vehicles found with capacity for pair {pair_order.id}")
        return False
    
    print(f"      Found {len(suitable_vehicles)} vehicles with sufficient capacity")
    
    # Sort suitable vehicles by capacity (largest first) to prioritize freeing up the biggest vehicles
    suitable_vehicles.sort(key=lambda v: v.weight_capacity, reverse=True)
    
    # Try to free up suitable vehicles by moving their current assignments
    for target_vehicle in suitable_vehicles:
        print(f"      Analyzing large vehicle {target_vehicle.id} (capacity: {target_vehicle.weight_capacity}kg)")
        
        current_route = solution.routes[target_vehicle.id]
        customer_tasks = [task for task in current_route.tasks if not (task.is_depot_start() or task.is_depot_return())]
        
        if not customer_tasks:
            print(f"        Vehicle {target_vehicle.id} is empty, attempting direct assignment...")
            
            print(f"          DEBUG: Calling L2 heuristic for pair {pair_order.id}")
            print(f"          DEBUG: Vehicle capacity: {target_vehicle.weight_capacity}kg, {target_vehicle.volume_capacity:.2f}m³, {target_vehicle.pallet_capacity}p")
            print(f"          DEBUG: Pair requirements: {pair_weight:.1f}kg, {pair_volume:.2f}m³, {pair_pallets}p")
            print(f"          DEBUG: Base route has {len(current_route.tasks)} tasks: {[task.task_type for task in current_route.tasks]}")
            
            # STRATEGY 1: Try with interleaved sequencing (P-D-P-D pattern) + allow time window violations
            print(f"          DEBUG: Strategy 1 - Trying L2 with interleaved sequencing + soft time windows...")
            test_route = l2_heuristic(current_route, pair_order, debug_assignment=True, sequencing_strategy='interleaved', enhanced_diagnostics=True)
            
            # If route is created but fails feasibility due to time windows, check if it's just time window violations
            if test_route and not test_route.is_feasible():
                print(f"          DEBUG: L2 route created but not feasible - checking if it's only time window violations...")
                try:
                    from second_level import is_feasible
                    feasibility_result = is_feasible(test_route, debug_feasibility=True, return_reason=True, allow_time_window_violations=True)
                    if feasibility_result[0]:  # Feasible if we ignore time window violations
                        print(f"          DEBUG: Route is feasible except for time windows - ACCEPTING with penalty")
                        # Override feasibility check - we'll accept this route and let penalties handle it
                        test_route._force_feasible_for_time_violations = True
                except Exception as e:
                    print(f"          DEBUG: Could not check soft feasibility: {e}")
            
            if not (test_route and (test_route.is_feasible() or getattr(test_route, '_force_feasible_for_time_violations', False))):
                # STRATEGY 2: Try with clustered sequencing (P-P-D-D pattern) + allow time window violations
                print(f"          DEBUG: Strategy 2 - Trying L2 with clustered sequencing + soft time windows...")
                test_route = l2_heuristic(current_route, pair_order, debug_assignment=True, sequencing_strategy='clustered', enhanced_diagnostics=True)
                
                # If route is created but fails feasibility due to time windows, check if it's just time window violations
                if test_route and not test_route.is_feasible():
                    print(f"          DEBUG: L2 route created but not feasible - checking if it's only time window violations...")
                    try:
                        from second_level import is_feasible
                        feasibility_result = is_feasible(test_route, debug_feasibility=True, return_reason=True, allow_time_window_violations=True)
                        if feasibility_result[0]:  # Feasible if we ignore time window violations
                            print(f"          DEBUG: Route is feasible except for time windows - ACCEPTING with penalty")
                            # Override feasibility check - we'll accept this route and let penalties handle it
                            test_route._force_feasible_for_time_violations = True
                    except Exception as e:
                        print(f"          DEBUG: Could not check soft feasibility: {e}")
            
            if not (test_route and (test_route.is_feasible() or getattr(test_route, '_force_feasible_for_time_violations', False))):
                # STRATEGY 3: Try standard L2 heuristic + allow time window violations
                print(f"          DEBUG: Strategy 3 - Trying standard L2 heuristic + soft time windows...")
                test_route = l2_heuristic(current_route, pair_order, debug_assignment=True, enhanced_diagnostics=True)
                
                # If route is created but fails feasibility due to time windows, check if it's just time window violations
                if test_route and not test_route.is_feasible():
                    print(f"          DEBUG: L2 route created but not feasible - checking if it's only time window violations...")
                    try:
                        from second_level import is_feasible
                        feasibility_result = is_feasible(test_route, debug_feasibility=True, return_reason=True, allow_time_window_violations=True)
                        if feasibility_result[0]:  # Feasible if we ignore time window violations
                            print(f"          DEBUG: Route is feasible except for time windows - ACCEPTING with penalty")
                            # Override feasibility check - we'll accept this route and let penalties handle it
                            test_route._force_feasible_for_time_violations = True
                    except Exception as e:
                        print(f"          DEBUG: Could not check soft feasibility: {e}")
            
            # Final check: Accept route if feasible or if it has only time window violations
            final_route_acceptable = test_route and (test_route.is_feasible() or getattr(test_route, '_force_feasible_for_time_violations', False))
            
            if final_route_acceptable:
                solution.routes[target_vehicle.id] = test_route
                print(f"        SUCCESS: Direct assignment to empty vehicle {target_vehicle.id} (with potential time window penalties)")
                return True
            else:
                print(f"          DEBUG: All strategies failed - route cannot be created or has non-time-window constraint violations")
            continue
        
        # Calculate current assignment requirements
        current_weight = sum(task.demand for task in customer_tasks if task.is_pickup())
        current_volume = sum(task.volume for task in customer_tasks if task.is_pickup())
        current_pallets = sum(task.pallets for task in customer_tasks if task.is_pickup())
        
        print(f"        Current assignment: {current_weight:.1f}kg, {current_volume:.2f}m³, {current_pallets}p")
        
        # Find smaller vehicles that could take the current assignment
        alternative_vehicles = []
        for alt_vehicle in vehicles:
            if (alt_vehicle.id != target_vehicle.id and
                alt_vehicle.weight_capacity >= current_weight and
                alt_vehicle.volume_capacity >= current_volume and
                alt_vehicle.pallet_capacity >= current_pallets):
                
                # Check if alternative vehicle is empty
                alt_route = solution.routes[alt_vehicle.id]
                alt_customer_tasks = [task for task in alt_route.tasks if not (task.is_depot_start() or task.is_depot_return())]
                if not alt_customer_tasks:  # Only use empty vehicles
                    alternative_vehicles.append(alt_vehicle)
        
        if not alternative_vehicles:
            print(f"        No empty alternative vehicles found for current assignment")
            continue
        
        print(f"        Found {len(alternative_vehicles)} empty alternative vehicles")
        
        # Try the reallocation
        for alt_vehicle in alternative_vehicles:
            print(f"          Testing reallocation to vehicle {alt_vehicle.id} (capacity: {alt_vehicle.weight_capacity}kg)")
            
            # Create temporary order from current tasks
            temp_order = create_temp_order_from_tasks(customer_tasks, f"temp_{target_vehicle.id}_to_{alt_vehicle.id}")
            if not temp_order:
                continue
            
            # Test if current assignment can move to alternative vehicle
            alt_route = solution.routes[alt_vehicle.id]
            test_alt_route = l2_heuristic(alt_route, temp_order)
            
            if test_alt_route and test_alt_route.is_feasible():
                # Test if pair can be assigned to the freed target vehicle
                empty_target_route = _create_base_route(target_vehicle)
                test_pair_route = l2_heuristic(empty_target_route, pair_order)
                
                if test_pair_route and test_pair_route.is_feasible():
                    # Success! Apply both assignments
                    solution.routes[alt_vehicle.id] = test_alt_route
                    solution.routes[target_vehicle.id] = test_pair_route
                    
                    print(f"          SUCCESS: Moved assignment to {alt_vehicle.id}, assigned pair to {target_vehicle.id}")
                    return True
                else:
                    print(f"          Pair still cannot be assigned to freed vehicle {target_vehicle.id}")
                    
                    # DEBUG: Add detailed info about why pair assignment failed
                    if test_pair_route:
                        print(f"            DEBUG: L2 returned route but is_feasible() = False")
                        try:
                            from second_level import is_feasible
                            feasibility_result = is_feasible(test_pair_route, debug_feasibility=True, return_reason=True)
                            print(f"            DEBUG: Feasibility failure reason: {feasibility_result}")
                        except Exception as e:
                            print(f"            DEBUG: Could not get feasibility details: {e}")
                    else:
                        print(f"            DEBUG: L2 heuristic returned None for freed vehicle")
            else:
                print(f"          Current assignment cannot be moved to alternative vehicle {alt_vehicle.id}")
    
    print(f"      FAILED: Pair-level vehicle reallocation could not assign {pair_order.id}")
    return False


def create_temp_order_from_tasks(tasks, new_order_id):
    """
    Helper to create a temporary Order object from a list of tasks.
    
    Args:
        tasks: List of tasks to include in the order
        new_order_id: ID for the new temporary order
        
    Returns:
        Order: Temporary order containing the tasks, or None if creation fails
    """
    try:
        from epdt_data_structures import Order
        
        if not tasks:
            return None
        
        # Create new order with ID as first positional argument (dataclass format)
        temp_order = Order(id=new_order_id)
        
        # Copy task properties
        for task in tasks:
            # Create new task with same properties
            if task.is_pickup():
                temp_order.pickup_tasks.append(task)
            else:
                temp_order.delivery_tasks.append(task)
        
        return temp_order
        
    except Exception as e:
        print(f"    Warning: Could not create temporary order from tasks: {e}")
        return None