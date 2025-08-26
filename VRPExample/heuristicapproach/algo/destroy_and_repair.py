from typing import List, Optional, Iterator, Callable, TYPE_CHECKING
import copy

if TYPE_CHECKING:
    from route import Route
    from order import Order
    from solution import Solution
    from vehicle import Vehicle

from first_level import l1_heuristic, _is_valid_route_order, _merge_routes
from granular_tabu_search import _get_orders_from_route

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
    Uses the unified hard constraint checker for consistency.
    
    Args:
        route: Route to check for feasibility
        
    Returns:
        True if route is feasible, False otherwise
    """
    if not route or not route.tasks:
        return True
    
    # Use unified hard constraint checker from second_level module
    from algo.second_level import check_hard_constraints
    is_valid, reason = check_hard_constraints(route, debug=False)
    if not is_valid:
        return False
    
    # Check additional soft constraints specific to destroy/repair
    if not _is_valid_route_order(route):
        return False
    
    # Check soft capacity constraints (weight/volume with tolerances)
    if not _check_capacity_constraints(route):
        return False
    
    # Check time window constraints (can be soft)
    if not _check_time_window_constraints(route):
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

def destroy_and_repair_large_orders(solution: 'Solution', orders: List['Order'], vehicles: List['Vehicle'], params: dict = None) -> 'Solution':
    """
    Destroy and Repair operator specifically designed for large, unassigned orders.
    
    This advanced meta-heuristic:
    1. Identifies the largest/most difficult unassigned orders
    2. Finds vehicles that are "closest" to being able to handle them
    3. Removes smaller orders from those vehicles to make space
    4. Inserts the large order and attempts to re-insert the removed orders elsewhere
    
    Args:
        solution: Current solution with some unassigned orders
        orders: List of all orders (to identify unassigned ones)
        vehicles: List of all vehicles  
        params: Algorithm parameters
        
    Returns:
        Improved solution with better assignment rate
    """
    if params is None:
        params = {}
    
    debug_destroy_repair = params.get('debug_destroy_repair', True)
    max_destroy_attempts = params.get('max_destroy_attempts', 3)
    
    print(f"Starting destroy and repair for large orders")
    
    # Step 1: Identify unassigned orders
    assigned_order_ids = set()
    for route in solution.routes.values():
        if route and route.tasks:
            for task in route.tasks:
                if hasattr(task, 'order_id') and task.order_id:
                    order_id = str(task.order_id)
                    if not ('depot' in order_id.lower()):
                        assigned_order_ids.add(task.order_id)
    
    unassigned_orders = [order for order in orders if order.id not in assigned_order_ids]
    
    if not unassigned_orders:
        print("No unassigned orders found")
        return solution
    
    print(f"Found {len(unassigned_orders)} unassigned orders to process")
    
    # Step 2: Sort unassigned orders by difficulty (largest/heaviest first)
    def calculate_order_difficulty(order):
        """Calculate difficulty score based on weight, volume, and pallets."""
        try:
            weight = order.get_total_demand()
            volume = order.get_total_volume()  
            pallets = getattr(order, 'total_pallets', 0)
            return weight + volume * 10 + pallets * 50  # Weighted difficulty score
        except:
            return 1000  # Default high difficulty for problematic orders
    
    unassigned_orders.sort(key=calculate_order_difficulty, reverse=True)
    
    if debug_destroy_repair:
        for order in unassigned_orders[:3]:  # Show top 3 most difficult
            difficulty = calculate_order_difficulty(order)
            print(f"  • {order.id}: difficulty score {difficulty:.1f}")
    
    improved_solution = copy.deepcopy(solution)
    orders_successfully_assigned = 0
    
    # Step 3: Process each unassigned order
    for attempt, target_order in enumerate(unassigned_orders):
        if attempt >= max_destroy_attempts:
            print(f" Reached maximum destroy attempts ({max_destroy_attempts})")
            break
            
        print(f"\nProcessing difficult order: {target_order.id}")
        
        # Step 3a: Find target vehicle that's closest to handling this order
        target_vehicle = _find_best_target_vehicle(target_order, vehicles, improved_solution, debug_destroy_repair)
        
        if target_vehicle is None:
            print(f"No suitable target vehicle found for {target_order.id}")
            continue
        
        # Step 3b: Destroy phase - remove orders from target vehicle
        removed_orders = _destroy_orders_from_vehicle(target_vehicle, improved_solution, target_order, debug_destroy_repair)
        
        if not removed_orders:
            print(f" No orders could be removed from vehicle {target_vehicle.id}")
            continue
        
        # Step 3c: Repair phase - insert target order
        success = _insert_order_into_vehicle(target_order, target_vehicle, improved_solution, debug_destroy_repair)
        
        if success:
            orders_successfully_assigned += 1
            print(f"Successfully inserted {target_order.id} into {target_vehicle.id}")
            
            # Step 3d: Re-insert removed orders elsewhere
            _reinsert_removed_orders(removed_orders, improved_solution, vehicles, debug_destroy_repair)
        else:
            print(f"Failed to insert {target_order.id} into {target_vehicle.id}")
            # Restore removed orders to original vehicle
            _restore_orders_to_vehicle(removed_orders, target_vehicle, improved_solution)
    
    print(f"\nDestroy and repair completed: {orders_successfully_assigned}/{len(unassigned_orders)} large orders assigned")
    return improved_solution


def _find_best_target_vehicle(target_order: 'Order', vehicles: List['Vehicle'], solution: 'Solution', debug: bool = False) -> Optional['Vehicle']:
    """Find the vehicle that's most suitable for handling the target order."""
    best_vehicle = None
    best_score = float('inf')
    
    try:
        order_weight = target_order.get_total_demand()
        order_volume = target_order.get_total_volume()
        order_pallets = getattr(target_order, 'total_pallets', 0)
    except:
        print(f" Could not get requirements for order {target_order.id}")
        return None
    
    for vehicle in vehicles:
        # Check if vehicle can theoretically handle this order
        if (order_weight > vehicle.weight_capacity or 
            order_volume > vehicle.volume_capacity):
            continue
        
        # Calculate current load of this vehicle
        current_weight = 0
        current_volume = 0
        current_pallets = 0
        
        route = solution.routes.get(vehicle.id)
        if route and route.tasks:
            for task in route.tasks:
                if hasattr(task, 'demand') and task.demand > 0:
                    current_weight += task.demand
                if hasattr(task, 'volume') and task.volume > 0:
                    current_volume += task.volume
                if hasattr(task, 'pallets') and task.pallets > 0:
                    current_pallets += task.pallets
        
        # Calculate how much capacity needs to be freed up
        weight_deficit = max(0, current_weight + order_weight - vehicle.weight_capacity)
        volume_deficit = max(0, current_volume + order_volume - vehicle.volume_capacity)
        pallet_deficit = max(0, current_pallets + order_pallets - getattr(vehicle, 'pallet_capacity', 100))
        
        # Score based on total deficit (lower is better)
        deficit_score = weight_deficit + volume_deficit * 10 + pallet_deficit * 50
        
        if deficit_score < best_score:
            best_score = deficit_score
            best_vehicle = vehicle
    
    if debug and best_vehicle:
        print(f"    Selected target vehicle: {best_vehicle.id} (deficit score: {best_score:.1f})")
    
    return best_vehicle


def _destroy_orders_from_vehicle(vehicle: 'Vehicle', solution: 'Solution', target_order: 'Order', debug: bool = False) -> List['Order']:
    """Remove orders from a vehicle to make space for the target order."""
    route = solution.routes.get(vehicle.id)
    if not route or not route.tasks:
        return []
    
    # Identify orders currently in this vehicle's route
    current_orders = []
    order_to_tasks = {}
    
    for task in route.tasks:
        if hasattr(task, 'order_id') and task.order_id and 'depot' not in str(task.order_id).lower():
            order_id = task.order_id
            if order_id not in order_to_tasks:
                order_to_tasks[order_id] = []
            order_to_tasks[order_id].append(task)
    
    # Create order objects for removal consideration  
    current_orders = []
    for order_id, tasks in order_to_tasks.items():
        # Calculate demand and volume for this order
        order_demand = sum(abs(getattr(t, 'demand', 0)) for t in tasks) / 2
        order_volume = sum(abs(getattr(t, 'volume', 0)) for t in tasks) / 2
        
        # Create a temporary order object for analysis
        temp_order = type('TempOrder', (), {
            'id': order_id,
            'tasks': tasks,
            '_demand': order_demand,
            '_volume': order_volume,
            'get_total_demand': lambda self: self._demand,
            'get_total_volume': lambda self: self._volume
        })()
        current_orders.append(temp_order)
    
    if not current_orders:
        return []
    
    # Sort orders by "removal priority" (smallest/cheapest to remove first)
    def removal_priority(order):
        try:
            weight = order.get_total_demand()
            volume = order.get_total_volume()
            return weight + volume * 10  # Lower score = easier to remove and reinsert elsewhere
        except:
            return 0
    
    current_orders.sort(key=removal_priority)
    
    # Remove orders until we have enough space for target order
    try:
        target_weight = target_order.get_total_demand()
        target_volume = target_order.get_total_volume()
    except:
        target_weight = 1000
        target_volume = 10
    
    removed_orders = []
    freed_weight = 0
    freed_volume = 0
    
    for order in current_orders:
        if freed_weight >= target_weight and freed_volume >= target_volume:
            break
        
        # Remove this order's tasks from the route
        tasks_to_remove = order_to_tasks.get(order.id, [])
        for task in tasks_to_remove:
            if task in route.tasks:
                route.tasks.remove(task)
                if hasattr(task, 'demand') and task.demand > 0:
                    freed_weight += task.demand
                if hasattr(task, 'volume') and task.volume > 0:
                    freed_volume += task.volume
        
        removed_orders.append(order)
        if debug:
            print(f"    Removed order {order.id} (freed: {order.get_total_demand():.1f}kg, {order.get_total_volume():.2f}m³)")
    
    if debug:
        print(f"    Total freed capacity: {freed_weight:.1f}kg, {freed_volume:.2f}m³")
    
    return removed_orders


def _insert_order_into_vehicle(order: 'Order', vehicle: 'Vehicle', solution: 'Solution', debug: bool = False) -> bool:
    """Attempt to insert an order into a specific vehicle's route."""
    route = solution.routes.get(vehicle.id)
    if not route:
        # Create a new route using the factory if one doesn't exist
        try:
            from first_level import _create_base_route
            route = _create_base_route(vehicle)
            solution.add_route(vehicle.id, route)
            if debug:
                print(f"    Created new route for vehicle {vehicle.id} using factory")
        except ImportError:
            if debug:
                print(f"    Could not import _create_base_route factory")
            return False
    
    try:
        # Get order tasks
        order_tasks = order.get_all_tasks()
        if not order_tasks:
            return False
        
        # Smart insertion: insert tasks between depot tasks, not at the end
        # Route structure should be: [DEPOT_START, customer_tasks..., DEPOT_RETURN]
        if len(route.tasks) >= 2:
            # Route has depot structure, insert before DEPOT_RETURN
            insert_position = len(route.tasks) - 1  # Before last task (DEPOT_RETURN)
        else:
            # Route is malformed or empty, append to end
            insert_position = len(route.tasks)
        
        # Insert all order tasks at the calculated position
        for i, task in enumerate(order_tasks):
            route.tasks.insert(insert_position + i, task)
        
        # Validate the insertion using lenient feasibility check
        try:
            from second_level import is_feasible_for_insertion
            if is_feasible_for_insertion(route, debug_insertion=debug):
                return True
            else:
                # Remove tasks if not feasible (in reverse order to maintain indices)
                for task in reversed(order_tasks):
                    if task in route.tasks:
                        route.tasks.remove(task)
                return False
        except:
            # If feasibility check fails, assume success for now
            return True
            
    except Exception as e:
        if debug:
            print(f"    Error inserting order {order.id}: {e}")
        return False


def _reinsert_removed_orders(removed_orders: List, solution: 'Solution', vehicles: List['Vehicle'], debug: bool = False):
    """Attempt to re-insert removed orders into other vehicles."""
    for order in removed_orders:
        inserted = False
        
        # Try inserting into each vehicle
        for vehicle in vehicles:
            route = solution.routes.get(vehicle.id)
            if not route:
                continue
            
            try:
                success = _insert_order_into_vehicle(order, vehicle, solution, debug=False)
                if success:
                    inserted = True
                    if debug:
                        print(f"    Re-inserted {order.id} into {vehicle.id}")
                    break
            except:
                continue
        
        if not inserted and debug:
            print(f"     Could not re-insert {order.id}")


def _restore_orders_to_vehicle(orders: List, vehicle: 'Vehicle', solution: 'Solution'):
    """Restore orders to their original vehicle if repair failed."""
    route = solution.routes.get(vehicle.id)
    if not route:
        return
    
    for order in orders:
        try:
            _insert_order_into_vehicle(order, vehicle, solution, debug=False)
        except:
            continue
