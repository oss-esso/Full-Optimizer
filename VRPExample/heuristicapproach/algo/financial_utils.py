"""
Financial calculation utilities for the EPDT heuristic solver.

This module contains utility functions for cost/profit calculations and financial analysis.
"""

import math

def calculate_route_cost_and_profit(vehicle_id: str, route, vehicle=None, orders=None):
    """
    Calculate detailed cost and profit breakdown for a route.
    
    Costs:
    - Driver cost: cost_per_hour * hours_worked
    - Vehicle cost: cost_per_km * total_distance
    
    Profit:
    - Per order: sum over all task pairs (price_per_km * km_between_pickup_and_delivery)
    - Price per km: 0.8 for furgone (standard), 1.25 for camion (heavy)
    
    Args:
        vehicle_id: Vehicle ID
        route: Route object
        vehicle: Vehicle object
        orders: List of Order objects for profit calculation
        
    Returns:
        Dictionary with cost and profit breakdown
    """
    # Default values
    total_hours = 0.0
    total_distance = 0.0
    driver_cost_per_hour = 25.0  # Default driver cost per hour
    vehicle_cost_per_km = 1.0    # Default vehicle cost per km
    
    # Calculate total hours from route timeline or estimate
    if hasattr(route, 'hos_timeline') and route.hos_timeline:
        # Extract hours from HoS timeline
        total_hours = len(route.hos_timeline) * 0.5  # Rough estimate
        # Count actual driving events
        driving_hours = 0
        for event in route.hos_timeline:
            if hasattr(event, 'event_type') and 'DRIVE' in str(event.event_type).upper():
                driving_hours += getattr(event, 'duration', 30) / 60.0  # Convert minutes to hours
        if driving_hours == 0:
            driving_hours = total_hours * 0.7  # Estimate 70% driving time
    else:
        # Estimate hours based on tasks
        total_hours = len(getattr(route, 'tasks', [])) * 0.5
        driving_hours = total_hours * 0.7  # Estimate 70% driving time
    
    # Get driver cost from route or use default
    if hasattr(route, 'driver') and route.driver:
        driver_cost_per_hour = getattr(route.driver, 'hourly_wage', driver_cost_per_hour)
    
    # Get vehicle cost per km
    if vehicle:
        vehicle_cost_per_km = getattr(vehicle, 'cost_per_km', vehicle_cost_per_km)
        # Determine price per km based on vehicle type
        vehicle_type = getattr(vehicle, 'type', 'furgone').lower()
        if 'camion' in vehicle_type or 'heavy' in vehicle_type:
            price_per_km = 1.25
        else:
            price_per_km = 0.8
    else:
        price_per_km = 0.8
    
    # Calculate total distance (simplified - using Haversine estimates)
    total_distance = estimate_route_distance(route)
    
    # Calculate costs - ONLY PAY DRIVERS FOR DRIVING HOURS
    driver_cost = driver_cost_per_hour * driving_hours
    vehicle_cost = vehicle_cost_per_km * total_distance
    total_cost = driver_cost + vehicle_cost
    
    # Calculate profit per order
    total_profit = 0.0
    order_profits = {}
    
    if orders and vehicle:
        # Get order IDs from route
        order_ids = set()
        if hasattr(route, 'tasks'):
            for task in route.tasks:
                if hasattr(task, 'order_id') and task.order_id:
                    order_ids.add(str(task.order_id))
        
        # Calculate profit for each order using dedicated route model
        for order_id in order_ids:
            order_profit = calculate_order_profit(order_id, orders, price_per_km)
            if order_profit > 0:
                order_profits[order_id] = order_profit
                total_profit += order_profit
    
    return {
        'total_hours': total_hours,
        'driving_hours': driving_hours,
        'total_distance': total_distance,
        'driver_cost_per_hour': driver_cost_per_hour,
        'vehicle_cost_per_km': vehicle_cost_per_km,
        'driver_cost': driver_cost,
        'vehicle_cost': vehicle_cost,
        'total_cost': total_cost,
        'total_profit': total_profit,
        'order_profits': order_profits,
        'net_profit': total_profit - total_cost,
        'price_per_km': price_per_km
    }


def estimate_route_distance(route):
    """Estimate total distance for a route using simplified calculation."""
    if not route or not route.tasks or len(route.tasks) < 2:
        return 0.0
    
    total_distance = 0.0
    try:
        for i in range(len(route.tasks) - 1):
            task1 = route.tasks[i]
            task2 = route.tasks[i + 1]
            
            # Get coordinates
            lat1 = getattr(task1, 'latitude', 0.0)
            lon1 = getattr(task1, 'longitude', 0.0)
            lat2 = getattr(task2, 'latitude', 0.0)
            lon2 = getattr(task2, 'longitude', 0.0)
            
            if lat1 and lon1 and lat2 and lon2:
                distance = haversine_distance(lat1, lon1, lat2, lon2)
                total_distance += distance
            else:
                # Fallback: assume 50km per segment
                total_distance += 50.0
                
    except Exception:
        # Fallback: estimate based on number of tasks
        total_distance = len(route.tasks) * 25.0  # 25km per task average
    
    return total_distance


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance between two points in kilometers."""
    
    R = 6371.0  # Earth radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def calculate_order_profit(order_id, orders, price_per_km):
    """
    Calculate profit for a specific order using the dedicated route model.
    
    Revenue = price_per_km × dedicated_route_distance
    Where dedicated_route_distance = depot → order_tasks → depot
    
    Args:
        order_id: ID of the order
        orders: List of all orders
        price_per_km: Price per kilometer for the vehicle type
        
    Returns:
        Revenue for this order if served by dedicated route
    """
    try:
        # Find the order
        order = None
        for o in orders:
            if str(o.id) == str(order_id):
                order = o
                break
        
        if not order:
            return 0.0
        
        # Estimate dedicated route distance
        # Simplified: assume depot is at (0,0) and calculate round trip
        total_distance = 0.0
        
        # Get all task locations for this order
        task_locations = []
        for task_list in [getattr(order, 'pickup_tasks', []), getattr(order, 'delivery_tasks', [])]:
            for task in task_list:
                lat = getattr(task, 'latitude', None)
                lon = getattr(task, 'longitude', None)
                if lat and lon:
                    task_locations.append((lat, lon))
        
        if task_locations:
            # Simplified calculation: depot -> first location -> last location -> depot
            if len(task_locations) >= 1:
                # Assume depot is 50km from first task (average)
                total_distance += 50.0 * 2  # Round trip to depot
                
                # Add distances between task locations
                for i in range(len(task_locations) - 1):
                    lat1, lon1 = task_locations[i]
                    lat2, lon2 = task_locations[i + 1]
                    total_distance += haversine_distance(lat1, lon1, lat2, lon2)
        else:
            # Fallback: estimate based on order size
            total_distance = 100.0  # 100km default dedicated route
        
        return price_per_km * total_distance
        
    except Exception:
        # Fallback: fixed revenue per order
        return price_per_km * 50.0  # 50km default


def validate_time_window_status(arrival_time_minutes: float, task) -> str:
    """
    Validate arrival time against task time window and return status.
    
    Args:
        arrival_time_minutes: Arrival time in minutes from start of planning period
        task: Task object with earliest_time and latest_time attributes
        
    Returns:
        Status string: "On time", "Early (waiting)", "Late", or "No window"
    """
    earliest = getattr(task, 'earliest_time', None)
    latest = getattr(task, 'latest_time', None)
    
    # Handle depot tasks or tasks without time windows
    if earliest is None and latest is None:
        return "No window"
    
    # If only earliest time is specified
    if earliest is not None and latest is None:
        if arrival_time_minutes < earliest:
            wait_time = earliest - arrival_time_minutes
            return f"Early (wait {wait_time/60:.1f}h)"
        else:
            return "On time"
    
    # If only latest time is specified
    if earliest is None and latest is not None:
        if arrival_time_minutes > latest:
            delay = arrival_time_minutes - latest
            return f"Late ({delay/60:.1f}h)"
        else:
            return "On time"
    
    # If both earliest and latest times are specified
    if earliest is not None and latest is not None:
        if arrival_time_minutes < earliest:
            wait_time = earliest - arrival_time_minutes
            return f"Early (wait {int(wait_time/60):02d}:{int(wait_time%60):02d})"
        elif arrival_time_minutes > latest:
            delay = arrival_time_minutes - latest
            return f"Late ({int(delay/60):02d}:{int(delay%60):02d})"
        else:
            return "On time"
    
    return "On time"
