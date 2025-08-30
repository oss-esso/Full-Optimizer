"""
Route calculation utility functions for the EPDT heuristic solver.

This module contains utility functions for route calculations, travel time estimation,
and route utilization analysis.
"""

# Global counter for tracking route calculations
route_calculation_count = 0

def calculate_travel_time_with_counter(prev_task, curr_task, vehicle):
    """
    Wrapper for calculate_travel_time_between_tasks that increments the global counter.
    This helps track how many route calculations are made.
    
    Args:
        prev_task: Previous task in route
        curr_task: Current task in route  
        vehicle: object
        
    Returns:
        Travel time in minutes
    """
    global route_calculation_count
    route_calculation_count += 1
    
    try:
        from route_provider import calculate_travel_time_between_tasks
        return calculate_travel_time_between_tasks(prev_task, curr_task, vehicle)
    except ImportError:
        try:
            from algo.route_provider import calculate_travel_time_between_tasks
            return calculate_travel_time_between_tasks(prev_task, curr_task, vehicle)
        except ImportError:
            # Fallback: simple time estimation
            return 60  # 1 hour default

def reset_route_calculation_counter():
    """Reset the global route calculation counter."""
    global route_calculation_count
    route_calculation_count = 0

def get_route_calculation_count():
    """Get the current route calculation count."""
    global route_calculation_count
    return route_calculation_count


def calculate_peak_route_utilization(route):
    """Calculate the peak utilization for a route (highest point during route execution)."""
    if not route or not route.tasks:
        return {
            'peak_weight': 0,
            'peak_volume': 0,
            'peak_pallets': 0,
            'utilization_weight': 0.0,
            'utilization_volume': 0.0,
            'utilization_pallets': 0.0
        }
    
    current_weight = 0
    current_volume = 0
    current_pallets = 0
    
    peak_weight = 0
    peak_volume = 0
    peak_pallets = 0
    
    # Debug: Count tasks
    pickup_count = 0
    delivery_count = 0
    depot_count = 0
    
    # Simulate route execution to find peak loads
    for task in route.tasks:
        if task.is_depot_start() or task.is_depot_return():
            depot_count += 1
            continue
        
        # Debug: Check task types and values
        task_weight = abs(getattr(task, 'demand', 0))
        task_volume = abs(getattr(task, 'volume', 0))
        task_pallets = abs(getattr(task, 'pallets', 0))
            
        # Update current loads (pickups add, deliveries subtract)
        if task.is_pickup():
            pickup_count += 1
            current_weight += task_weight
            current_volume += task_volume
            current_pallets += task_pallets
        elif task.is_delivery():
            delivery_count += 1
            current_weight -= task_weight
            current_volume -= task_volume
            current_pallets -= task_pallets
        
        # Track peaks
        peak_weight = max(peak_weight, current_weight)
        peak_volume = max(peak_volume, current_volume)
        peak_pallets = max(peak_pallets, current_pallets)
    
    # Debug: Log if we have tasks but no peak load
    if len(route.tasks) > 2 and peak_weight == 0 and peak_volume == 0 and peak_pallets == 0:
        print(f"DEBUG: Vehicle {route.vehicle.id} has {len(route.tasks)} tasks ({pickup_count}P, {delivery_count}D, {depot_count} depot) but zero peak load")
    
    # Calculate utilization percentages based on vehicle capacity
    vehicle = route.vehicle
    utilization_weight = 0.0
    utilization_volume = 0.0
    utilization_pallets = 0.0
    
    if hasattr(vehicle, 'weight_capacity') and vehicle.weight_capacity and vehicle.weight_capacity > 0:
        utilization_weight = (peak_weight / vehicle.weight_capacity) * 100
    
    if hasattr(vehicle, 'volume_capacity') and vehicle.volume_capacity and vehicle.volume_capacity > 0:
        utilization_volume = (peak_volume / vehicle.volume_capacity) * 100
    
    if hasattr(vehicle, 'pallet_capacity') and vehicle.pallet_capacity and vehicle.pallet_capacity > 0:
        utilization_pallets = (peak_pallets / vehicle.pallet_capacity) * 100
    
    return {
        'peak_weight': peak_weight,
        'peak_volume': peak_volume,
        'peak_pallets': peak_pallets,
        'utilization_weight': utilization_weight,
        'utilization_volume': utilization_volume,
        'utilization_pallets': utilization_pallets
    }


def calculate_route_load(route):
    """Calculate total load changes for a route."""
    if not route or not route.tasks:
        return {
            'total_weight_change': 0,
            'total_volume_change': 0,
            'total_pallets_change': 0
        }
    
    total_weight_change = 0
    total_volume_change = 0
    total_pallets_change = 0
    
    for task in route.tasks:
        if task.is_depot_start() or task.is_depot_return():
            continue
            
        # Get absolute values of the changes
        weight_change = abs(getattr(task, 'demand', 0))
        volume_change = abs(getattr(task, 'volume', 0))
        pallets_change = abs(getattr(task, 'pallets', 0))
        
        total_weight_change += weight_change
        total_volume_change += volume_change
        total_pallets_change += pallets_change
    
    return {
        'total_weight_change': total_weight_change,
        'total_volume_change': total_volume_change,
        'total_pallets_change': total_pallets_change
    }


def print_simulated_hos_breakdown(travel_time_minutes: float):
    """
    Print a simulated HoS breakdown for a travel segment.
    This provides the visual breakdown the user wants when HoS timeline is not available.
    
    Args:
        travel_time_minutes: Total travel time in minutes
    """
    if travel_time_minutes <= 0:
        return
    
    # Import the time formatting function
    try:
        from algo.time_utils import format_time_hhmm
    except ImportError:
        def format_time_hhmm(minutes):
            if minutes < 0:
                return "00:00"
            hours = int(minutes // 60)
            mins = int(minutes % 60)
            return f"{hours:02d}:{mins:02d}"
    
    # Simulate realistic HoS breakdown based on EU regulations
    remaining_time = travel_time_minutes
    segment_num = 1
    
    while remaining_time > 0:
        if remaining_time <= 270:  # 4.5 hours or less - can drive continuously
            drive_time = remaining_time
            print(f"               {segment_num}: DRIVE - {format_time_hhmm(drive_time)}")
            break
        else:
            # Need break after 4.5 hours of driving
            drive_time = 270  # 4.5 hours
            print(f"               {segment_num}: DRIVE - {format_time_hhmm(drive_time)}")
            
            remaining_time -= drive_time
            
            if remaining_time > 0:
                print(f"               {segment_num+1}: BREAK - 00:45")  # 45 min break
                segment_num += 2
            else:
                break
