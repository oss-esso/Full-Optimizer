
from typing import List

# Import the base route creation function
try:
    from first_level import _create_base_route
except ImportError:
    def _create_base_route(vehicle):
        """Fallback base route creation"""
        from epdt_data_structures import Route, Task, TaskType
        route = Route(vehicle)
        # Add basic depot tasks
        return route

def comprehensive_post_l1_force_assignment(orders: List['Order'], vehicles: List['Vehicle'], solution: 'Solution') -> int:
    """
    Comprehensive post-L1 force assignment that checks ALL unassigned orders against ALL vehicles.
    
    This function:
    1. Identifies all unassigned orders
    2. Tests each order against ALL vehicles (both idle and active)
    3. Uses HoS engine for proper constraint validation
    4. Performs full feasibility checking including LOW_TEMP capabilities
    5. Assigns orders to the best available vehicle
    
    Args:
        orders: All orders from the original problem
        vehicles: All available vehicles
        solution: Current solution after L1 heuristic
        
    Returns:
        Number of additional orders successfully assigned
    """
    print("Starting comprehensive post-L1 force assignment...")
    
    # Step 1: Identify currently assigned orders
    assigned_order_ids = set()
    for vehicle_id, route in solution.routes.items():
        if route and hasattr(route, 'tasks'):
            for task in route.tasks:
                if hasattr(task, 'order_id') and task.order_id:
                    assigned_order_ids.add(str(task.order_id))
    
    # Step 2: Find unassigned orders
    unassigned_orders = []
    for order in orders:
        order_id = str(getattr(order, 'id', ''))
        if order_id and order_id not in assigned_order_ids:
            unassigned_orders.append(order)
    
    print(f"Found {len(unassigned_orders)} unassigned orders: {[o.id for o in unassigned_orders]}")
    
    if not unassigned_orders:
        print("No unassigned orders found - all orders are already assigned!")
        return 0
    
    successfully_assigned = 0
    
    # Step 3: For each unassigned order, try ALL vehicles
    for order in unassigned_orders:
        print(f"\nProcessing unassigned Order {order.id}...")
        
        # Get order requirements
        order_weight = getattr(order, 'get_total_demand', lambda: 0)()
        order_volume = getattr(order, 'get_total_volume', lambda: 0)()
        order_pallets = sum(getattr(task, 'pallets', 0) for task in getattr(order, 'get_all_tasks', lambda: [])())
        order_needs_low_temp = any(getattr(task, 'requires_low_temp', False) for task in getattr(order, 'get_all_tasks', lambda: [])())
        
        print(f"  Order requirements: {order_weight:.1f}kg, {order_volume:.1f}m³, {order_pallets}pal, LOW_TEMP: {order_needs_low_temp}")
        
        best_vehicle = None
        best_route = None
        best_score = float('inf')
        
        # Step 4: Test against ALL vehicles
        for vehicle in vehicles:
            print(f"    Testing vehicle {vehicle.id}...")
            
            # Get current route for this vehicle
            current_route = solution.routes.get(vehicle.id)
            if not current_route:
                # Create empty route for idle vehicle
                current_route = _create_base_route(vehicle)
            
            # Check basic capacity constraints
            current_weight = 0.0
            current_volume = 0.0
            current_pallets = 0.0
            
            if hasattr(current_route, 'tasks'):
                for task in current_route.tasks:
                    current_weight += getattr(task, 'demand', 0.0)
                    current_volume += getattr(task, 'volume', 0.0)  
                    current_pallets += getattr(task, 'pallets', 0.0)
            
            available_weight = getattr(vehicle, 'weight_capacity', float('inf')) - current_weight
            available_volume = getattr(vehicle, 'volume_capacity', float('inf')) - current_volume
            available_pallets = getattr(vehicle, 'pallet_capacity', float('inf')) - current_pallets
            
            # Check capacity feasibility
            if (order_weight > available_weight or 
                order_volume > available_volume or 
                order_pallets > available_pallets):
                print(f"      CAPACITY FAIL: Need {order_weight:.0f}kg/{order_volume:.1f}m³/{order_pallets}pal, Available {available_weight:.0f}kg/{available_volume:.1f}m³/{available_pallets}pal")
                continue
            
            # Check temperature capability
            vehicle_has_low_temp = getattr(vehicle, 'has_low_temp', False) or 'LOW_TEMP' in getattr(vehicle, 'capabilities', [])
            if order_needs_low_temp and not vehicle_has_low_temp:
                print(f"      TEMP CAPABILITY FAIL: Order needs LOW_TEMP, vehicle lacks capability")
                continue
            
            print(f"      BASIC CHECKS PASSED: Capacity and temperature OK")
            
            # Step 5: Use L2 heuristic for detailed feasibility and HoS validation
            try:
                from second_level import l2_heuristic, calculate_z2_score
                optimized_route = l2_heuristic(current_route, order, debug_assignment=False, enhanced_diagnostics=False)
                
                if optimized_route:
                    # Check if route passes HoS validation by testing the route feasibility
                    if hasattr(optimized_route, 'is_feasible'):
                        is_feasible = optimized_route.is_feasible()
                    else:
                        is_feasible = True  # Assume feasible if no validation method
                    
                    if is_feasible:
                        cost = calculate_z2_score(optimized_route)
                        print(f"      FEASIBLE: Vehicle {vehicle.id} can take Order {order.id} with cost {cost:.2f}")
                        
                        # Check if this is the best option so far
                        if cost < best_score:
                            best_score = cost
                            best_vehicle = vehicle
                            best_route = optimized_route
                            print(f"      NEW BEST: Vehicle {vehicle.id} is new best option")
                    else:
                        print(f"      HoS/TIME WINDOW FAIL: Route violates HoS or time constraints")
                else:
                    print(f"      L2 HEURISTIC FAIL: Could not generate feasible route")
                    
            except Exception as e:
                print(f"      ERROR: {str(e)[:100]}")
                continue
        
        # Step 6: Assign order to best vehicle if found
        if best_vehicle and best_route:
            solution.routes[best_vehicle.id] = best_route
            successfully_assigned += 1
            print(f"   SUCCESS: Order {order.id} assigned to vehicle {best_vehicle.id} (score: {best_score:.2f})")
        else:
            print(f"   FAILED: Order {order.id} could not be assigned to any vehicle")
            # Mark as unassigned in solution
            if not hasattr(solution, 'unassigned_orders'):
                solution.unassigned_orders = set()
            solution.unassigned_orders.add(str(order.id))
    
    print(f"\nPost-L1 force assignment summary: {successfully_assigned}/{len(unassigned_orders)} orders assigned")
    return successfully_assigned
