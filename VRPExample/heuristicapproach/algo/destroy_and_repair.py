
from typing import List, Optional, Iterator, Callable, TYPE_CHECKING
import copy

if TYPE_CHECKING:
    from route import Route
    from order import Order
    from solution import Solution
    from vehicle import Vehicle

from first_level import l1_heuristic, _is_valid_route_order, _get_orders_from_route, _merge_routes

def destroy_and_repair(solution: 'Solution', params: dict = None) -> 'Solution':
    """
    Implement the Destroy and Repair phase (Section 5.9).
    
    This meta-heuristic:
    1. Checks for infeasible routes in the solution
    2. Destroys problematic parts by moving orders to unassigned list
    3. Repairs the solution by re-running L1 heuristic
    
    Args:
        solution: Current solution to apply destroy and repair to
        params: Algorithm parameters
        
    Returns:
        Repaired solution with better feasibility
    """
    if not solution or not solution.routes:
        return solution
    
    if params is None:
        params = {}
    
    # Step 1: Check for infeasibilities
    infeasible_routes = []
    problematic_orders = set()
    
    for route_idx, route in enumerate(solution.routes):
        if route and route.tasks:
            if not _is_route_feasible(route):
                infeasible_routes.append(route_idx)
                # Identify problematic orders in this route
                problematic_orders.update(_identify_problematic_orders(route))
    
    if not problematic_orders:
        return solution  # No repair needed
    
    # Step 2: Destroy - Remove problematic orders
    destroyed_solution = copy.deepcopy(solution)
    unassigned_orders = []
    
    for route_idx in infeasible_routes:
        route = destroyed_solution.routes[route_idx]
        orders_to_remove = _get_orders_from_route(route)
        
        # Remove problematic orders from the route
        for order_id in problematic_orders:
            if order_id in orders_to_remove:
                # Get the order object
                order = next((o for o in solution.orders if o.id == order_id), None)
                if order:
                    unassigned_orders.append(order)
                    
                # Remove all tasks for this order from the route
                for task in orders_to_remove[order_id]:
                    route.remove_task(task)
    
    # Step 3: Repair - Re-insert unassigned orders using L1 heuristic
    if unassigned_orders:
        # Create a list of vehicles for repair
        vehicles = [route.vehicle for route in destroyed_solution.routes if route and route.vehicle]
        
        # Configure repair parameters
        repair_params = params.copy()
        repair_params.update({
            'M1': params.get('repair_M1', 100),  # Fewer iterations for repair
            'M2': params.get('repair_M2', 500),
            'tabu_tenure': params.get('repair_tabu_tenure', 5),  # Shorter tabu tenure
            'enable_advanced_neighborhoods': False  # Disable for faster repair
        })
        
        # Use L1 heuristic to repair the solution
        repaired_solution = l1_heuristic(unassigned_orders, vehicles, repair_params)
        
        # Merge the repaired assignments back into the destroyed solution
        repaired_solution = _merge_solutions(destroyed_solution, repaired_solution)
        
        return repaired_solution
    
    return destroyed_solution

def _is_route_feasible(route: 'Route') -> bool:
    """
    Check if a route is feasible considering all constraints.
    
    Args:
        route: Route to check for feasibility
        
    Returns:
        True if route is feasible, False otherwise
    """
    if not route or not route.tasks:
        return True
    
    # Check pickup/delivery order constraints
    if not _is_valid_route_order(route):
        return False
    
    # Check capacity constraints
    if not _check_capacity_constraints(route):
        return False
    
    # Check time window constraints
    if not _check_time_window_constraints(route):
        return False
    
    # Check vehicle-specific constraints
    if not _check_vehicle_constraints(route):
        return False
    
    return True

def _identify_problematic_orders(route: 'Route') -> set:
    """
    Identify orders that cause infeasibility in a route.
    
    Args:
        route: Route to analyze for problematic orders
        
    Returns:
        Set of order IDs that contribute to infeasibility
    """
    problematic_orders = set()
    
    if not route or not route.tasks:
        return problematic_orders
    
    # Check each order's contribution to constraint violations
    orders_in_route = _get_orders_from_route(route)
    
    for order_id, tasks in orders_in_route.items():
        # Create a temporary route without this order
        temp_route = copy.deepcopy(route)
        for task in tasks:
            temp_route.remove_task(task)
        
        # If removing this order makes the route more feasible, it's problematic
        if _count_constraint_violations(temp_route) < _count_constraint_violations(route):
            problematic_orders.add(order_id)
    
    return problematic_orders

def _count_constraint_violations(route: 'Route') -> int:
    """
    Count the number of constraint violations in a route.
    
    Args:
        route: Route to count violations for
        
    Returns:
        Number of constraint violations
    """
    violations = 0
    
    if not route or not route.tasks:
        return violations
    
    # Count pickup/delivery order violations
    if not _is_valid_route_order(route):
        violations += len(route.tasks)  # Severe penalty for order violations
    
    # Count capacity violations
    if not _check_capacity_constraints(route):
        violations += 10  # Penalty for capacity violations
    
    # Count time window violations
    violations += _count_time_window_violations(route)
    
    # Count vehicle constraint violations
    if not _check_vehicle_constraints(route):
        violations += 5  # Penalty for vehicle violations
    
    return violations

def _check_capacity_constraints(route: 'Route') -> bool:
    """Check if route respects vehicle capacity constraints."""
    if not route or not route.vehicle:
        return True
    
    # Simplified capacity check - in practice would track load throughout route
    total_demand = 0
    for task in route.tasks:
        demand = getattr(task, 'demand', 0)
        if getattr(task, 'task_type', None) == 'pickup':
            total_demand += demand
        elif getattr(task, 'task_type', None) == 'delivery':
            total_demand -= demand
    
    vehicle_capacity = getattr(route.vehicle, 'capacity', float('inf'))
    return total_demand <= vehicle_capacity

def _check_time_window_constraints(route: 'Route') -> bool:
    """Check if route respects time window constraints."""
    if not route or not route.tasks:
        return True
    
    current_time = getattr(route.vehicle, 'start_time', 0) if route.vehicle else 0
    
    for task in route.tasks:
        # Add travel time to this task (simplified)
        travel_time = getattr(task, 'travel_time', 10)  # Default travel time
        current_time += travel_time
        
        # Check time window
        time_window = getattr(task, 'time_window', None)
        if time_window:
            earliest = getattr(time_window, 'start', 0)
            latest = getattr(time_window, 'end', float('inf'))
            
            if current_time < earliest:
                current_time = earliest  # Wait until time window opens
            elif current_time > latest:
                return False  # Violation
        
        # Add service time
        service_time = getattr(task, 'service_time', 5)  # Default service time
        current_time += service_time
    
    return True

def _count_time_window_violations(route: 'Route') -> int:
    """Count time window violations in a route."""
    violations = 0
    
    if not route or not route.tasks:
        return violations
    
    current_time = getattr(route.vehicle, 'start_time', 0) if route.vehicle else 0
    
    for task in route.tasks:
        travel_time = getattr(task, 'travel_time', 10)
        current_time += travel_time
        
        time_window = getattr(task, 'time_window', None)
        if time_window:
            latest = getattr(time_window, 'end', float('inf'))
            if current_time > latest:
                violations += 1
        
        service_time = getattr(task, 'service_time', 5)
        current_time += service_time
    
    return violations

def _check_vehicle_constraints(route: 'Route') -> bool:
    """Check vehicle-specific constraints."""
    if not route or not route.vehicle:
        return True
    
    # Check if vehicle can handle all task types in the route
    for task in route.tasks:
        task_type = getattr(task, 'task_type', None)
        vehicle_capabilities = getattr(route.vehicle, 'capabilities', [])
        
        if task_type and vehicle_capabilities:
            if task_type not in vehicle_capabilities:
                return False
    
    return True

def _merge_solutions(base_solution: 'Solution', repair_solution: 'Solution') -> 'Solution':
    """
    Merge a repair solution back into the base solution.
    
    Args:
        base_solution: Base solution with destroyed parts
        repair_solution: Solution from repair process
        
    Returns:
        Merged solution combining both
    """
    merged_solution = copy.deepcopy(base_solution)
    
    # Merge routes from repair solution
    if repair_solution and repair_solution.routes:
        for route_idx, repair_route in enumerate(repair_solution.routes):
            if repair_route and repair_route.tasks:
                if route_idx < len(merged_solution.routes):
                    # Merge with existing route
                    existing_route = merged_solution.routes[route_idx]
                    merged_route = _merge_routes(existing_route, repair_route)
                    merged_solution.routes[route_idx] = merged_route
                else:
                    # Add new route
                    merged_solution.routes.append(repair_route)
    
    return merged_solution
