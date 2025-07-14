"""
Data Adapter for EPDT Algorithm

This module provides data transformation functionality to convert VRPInstance objects
(used by the test scenarios) into EPDT algorithm data structures.

The main function `convert_instance_to_epdt_input` bridges the gap between:
- VRPInstance format: Location, Vehicle, RideRequest
- EPDT format: Order, Vehicle, Task, Route, Solution

Transformation Logic:
1. VRPInstance.ride_requests → List[Order] (with pickup/delivery tasks)  
2. VRPInstance.vehicles → List[Vehicle] (with EPDT vehicle structure)
3. VRPInstance.locations → Task location data (coordinates, time windows)
"""

import sys
import os
from typing import List, Tuple, Dict, Set
from dataclasses import replace

# Add the necessary paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from VRPInstance.vrp_data_models import VRPInstance, Location, Vehicle as VRPVehicle, RideRequest
except ImportError:
    try:
        from vrp_data_models import VRPInstance, Location, Vehicle as VRPVehicle, RideRequest
    except ImportError:
        print("Warning: Could not import VRP data models. Ensure VRPInstance module is available.")
        # Define minimal stubs for testing
        class VRPInstance:
            def __init__(self):
                self.locations = {}
                self.vehicles = {}
                self.ride_requests = []
        
        class Location:
            def __init__(self):
                pass
        
        class VRPVehicle:
            def __init__(self):
                pass
        
        class RideRequest:
            def __init__(self):
                pass

from .epdt_data_structures import (
    Task, TaskType, Order, Vehicle, Route, Solution, EPDTParameters
)


def convert_instance_to_epdt_input(instance: VRPInstance) -> Tuple[List[Order], List[Vehicle]]:
    """
    Convert a VRPInstance to EPDT algorithm input format.
    
    Args:
        instance: VRPInstance object from test scenarios
        
    Returns:
        Tuple of (orders, vehicles) ready for EPDT algorithm
        
    Transformation Process:
    1. Convert each RideRequest to an Order with pickup and delivery tasks
    2. Convert each VRPInstance Vehicle to EPDT Vehicle format
    3. Populate task location data from VRPInstance.locations
    """
    print(f"🔄 Converting VRPInstance to EPDT format...")
    print(f"   📍 Locations: {len(instance.locations)}")
    print(f"   🚛 Vehicles: {len(instance.vehicles)}")  
    print(f"   📦 Ride Requests: {len(instance.ride_requests)}")
    
    # Convert vehicles first
    epdt_vehicles = _convert_vehicles(instance.vehicles, instance.locations)
    
    # Convert ride requests to orders
    epdt_orders = _convert_ride_requests_to_orders(instance.ride_requests, instance.locations)
    
    print(f"✅ Conversion complete:")
    print(f"   📦 Orders created: {len(epdt_orders)}")
    print(f"   🚛 Vehicles converted: {len(epdt_vehicles)}")
    
    # Validate conversion
    _validate_conversion(epdt_orders, epdt_vehicles, instance)
    
    return epdt_orders, epdt_vehicles


def _convert_vehicles(vrp_vehicles: Dict[str, VRPVehicle], locations: Dict[str, Location]) -> List[Vehicle]:
    """Convert VRPInstance vehicles to EPDT Vehicle objects."""
    epdt_vehicles = []
    
    for vehicle_id, vrp_vehicle in vrp_vehicles.items():
        # Extract vehicle attributes with defaults
        weight_capacity = getattr(vrp_vehicle, 'capacity', 3500)  # Default 3.5t
        volume_capacity = getattr(vrp_vehicle, 'volume_capacity', 20.0)  # Default 20m³
        depot_id = getattr(vrp_vehicle, 'depot_id', 'depot')
        max_time = getattr(vrp_vehicle, 'max_time', 480.0)  # Default 8 hours
        cost_per_km = getattr(vrp_vehicle, 'cost_per_km', 1.0)
        vehicle_type = getattr(vrp_vehicle, 'vehicle_type', 'standard')
        
        # Map vehicle regulatory constraints
        if vehicle_type == "heavy":
            max_driving_time = 540.0   # 9 hours for heavy vehicles  
            max_work_time = 900.0      # 15 hours max duty time
            break_frequency = 270.0    # 4.5 hours
        else:
            max_driving_time = 480.0   # 8 hours for standard vehicles
            max_work_time = 780.0      # 13 hours max duty time  
            break_frequency = 270.0    # 4.5 hours
        
        epdt_vehicle = Vehicle(
            id=vehicle_id,
            depot_id=depot_id,
            weight_capacity=weight_capacity,
            volume_capacity=volume_capacity,
            max_time=max_time,
            cost_per_km=cost_per_km,
            vehicle_type=vehicle_type,
            max_driving_time=max_driving_time,
            required_break_time=45.0,
            max_work_time=max_work_time,
            break_frequency=break_frequency
        )
        
        epdt_vehicles.append(epdt_vehicle)
    
    return epdt_vehicles


def _convert_ride_requests_to_orders(ride_requests: List[RideRequest], locations: Dict[str, Location]) -> List[Order]:
    """Convert VRPInstance ride requests to EPDT Order objects with Tasks."""
    epdt_orders = []
    
    for request in ride_requests:
        # Extract request attributes
        order_id = getattr(request, 'id', f"order_{len(epdt_orders)}")
        pickup_location_id = getattr(request, 'pickup_location', None)
        dropoff_location_id = getattr(request, 'dropoff_location', None)
        cargo_weight = getattr(request, 'passengers', 1.0)  # 'passengers' field represents weight
        cargo_volume = getattr(request, 'volume', cargo_weight / 200.0)  # Default density 200kg/m³
        
        if not pickup_location_id or not dropoff_location_id:
            print(f"⚠️  Skipping request {order_id}: missing pickup or dropoff location")
            continue
        
        # Get location details
        pickup_location = locations.get(pickup_location_id)
        dropoff_location = locations.get(dropoff_location_id)
        
        if not pickup_location or not dropoff_location:
            print(f"⚠️  Skipping request {order_id}: location not found")
            continue
        
        # Create pickup task
        pickup_task = _create_task_from_location(
            task_id=f"{order_id}_pickup",
            location_id=pickup_location_id,
            location=pickup_location,
            task_type=TaskType.PICKUP,
            order_id=order_id,
            demand=cargo_weight,
            volume=cargo_volume
        )
        
        # Create delivery task  
        delivery_task = _create_task_from_location(
            task_id=f"{order_id}_delivery",
            location_id=dropoff_location_id,
            location=dropoff_location,
            task_type=TaskType.DELIVERY,
            order_id=order_id,
            demand=-cargo_weight,  # Negative for delivery
            volume=-cargo_volume   # Negative for delivery
        )
        
        # Create order with tasks
        order = Order(
            id=order_id,
            pickup_tasks=[pickup_task],
            delivery_tasks=[delivery_task],
            priority=1,
            is_mandatory=True,
            earliest_pickup=getattr(request, 'earliest_pickup', None),
            latest_delivery=getattr(request, 'latest_dropoff', None)
        )
        
        epdt_orders.append(order)
    
    return epdt_orders


def _create_task_from_location(task_id: str, location_id: str, location: Location, 
                             task_type: TaskType, order_id: str, 
                             demand: float, volume: float) -> Task:
    """Create a Task object from a Location."""
    # Extract location attributes with defaults
    lat = getattr(location, 'lat', 0.0)
    lon = getattr(location, 'lon', 0.0)
    service_time = getattr(location, 'service_time', 15.0)  # Default 15 minutes
    
    # Time windows (convert from location time windows if available)
    time_window_start = getattr(location, 'time_window_start', None)
    time_window_end = getattr(location, 'time_window_end', None)
    
    # Convert hours to minutes if needed
    if time_window_start is not None and time_window_start > 0:
        earliest_time = time_window_start if time_window_start < 1440 else time_window_start / 60
    else:
        earliest_time = None
        
    if time_window_end is not None and time_window_end > 0:
        latest_time = time_window_end if time_window_end < 1440 else time_window_end / 60
    else:
        latest_time = None
    
    return Task(
        id=task_id,
        location_id=location_id,
        task_type=task_type,
        order_id=order_id,
        lat=lat,
        lon=lon,
        service_time=service_time,
        earliest_time=earliest_time,
        latest_time=latest_time,
        demand=demand,
        volume=volume,
        priority=1
    )


def _validate_conversion(orders: List[Order], vehicles: List[Vehicle], instance: VRPInstance) -> None:
    """Validate the conversion was successful."""
    print(f"\n🔍 Validation Results:")
    
    # Check order consistency
    total_pickup_weight = sum(order.get_total_demand() for order in orders)
    total_pickup_volume = sum(order.get_total_volume() for order in orders)
    
    print(f"   📦 Total cargo weight: {total_pickup_weight:,.1f} kg")
    print(f"   📦 Total cargo volume: {total_pickup_volume:,.1f} m³")
    
    # Check vehicle capacity
    total_weight_capacity = sum(vehicle.weight_capacity for vehicle in vehicles)
    total_volume_capacity = sum(vehicle.volume_capacity for vehicle in vehicles)
    
    print(f"   🚛 Total fleet weight capacity: {total_weight_capacity:,.1f} kg")
    print(f"   🚛 Total fleet volume capacity: {total_volume_capacity:,.1f} m³")
    
    # Capacity utilization
    weight_utilization = (total_pickup_weight / total_weight_capacity) * 100 if total_weight_capacity > 0 else 0
    volume_utilization = (total_pickup_volume / total_volume_capacity) * 100 if total_volume_capacity > 0 else 0
    
    print(f"   📊 Weight utilization: {weight_utilization:.1f}%")
    print(f"   📊 Volume utilization: {volume_utilization:.1f}%")
    
    # Check for issues
    issues = []
    if weight_utilization > 100:
        issues.append(f"Weight capacity exceeded by {weight_utilization - 100:.1f}%")
    if volume_utilization > 100:
        issues.append(f"Volume capacity exceeded by {volume_utilization - 100:.1f}%")
    
    if issues:
        print(f"   ⚠️  Issues found:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print(f"   ✅ No capacity issues detected")
    
    # Validate task consistency
    total_tasks = sum(len(order.get_all_tasks()) for order in orders)
    expected_tasks = len(instance.ride_requests) * 2  # Each request = pickup + delivery
    
    print(f"   🎯 Tasks created: {total_tasks} (expected: {expected_tasks})")
    
    if total_tasks != expected_tasks:
        print(f"   ⚠️  Task count mismatch - some requests may have been skipped")


def create_empty_solution(vehicles: List[Vehicle]) -> Solution:
    """Create an empty solution with routes for all vehicles."""
    solution = Solution()
    
    for vehicle in vehicles:
        empty_route = Route(vehicle=vehicle)
        solution.add_route(vehicle.id, empty_route)
    
    return solution


def get_default_parameters() -> EPDTParameters:
    """Get default algorithm parameters for testing."""
    return EPDTParameters(
        tabu_tenure=10,
        max_non_improving_iterations=50,
        max_total_iterations=500,
        exploration_strategy="vnd",
        enable_advanced_neighborhoods=True,
        enable_granular_search=False,  # Disable for initial testing
        enable_parallelization=False,  # Disable for initial testing
        local_search_strategy="first_improvement",
        initialization_method="best_insertion"
    )


# Utility functions for solution analysis
def print_conversion_summary(orders: List[Order], vehicles: List[Vehicle]) -> None:
    """Print a summary of the converted data."""
    print(f"\n📋 EPDT Conversion Summary")
    print(f"=" * 50)
    
    print(f"🚛 Fleet Overview:")
    vehicle_types = {}
    for vehicle in vehicles:
        vtype = vehicle.vehicle_type
        vehicle_types[vtype] = vehicle_types.get(vtype, 0) + 1
    
    for vtype, count in vehicle_types.items():
        print(f"   - {vtype.title()} vehicles: {count}")
    
    print(f"\n📦 Orders Overview:")
    print(f"   - Total orders: {len(orders)}")
    print(f"   - Total pickup tasks: {sum(len(order.pickup_tasks) for order in orders)}")
    print(f"   - Total delivery tasks: {sum(len(order.delivery_tasks) for order in orders)}")
    
    # Cargo analysis
    total_weight = sum(order.get_total_demand() for order in orders)
    total_volume = sum(order.get_total_volume() for order in orders)
    
    print(f"\n📊 Cargo Analysis:")
    print(f"   - Total weight: {total_weight:,.1f} kg")
    print(f"   - Total volume: {total_volume:,.1f} m³")
    print(f"   - Average order weight: {total_weight/len(orders):,.1f} kg")
    
    print(f"\n🎯 Data Ready for EPDT Algorithm")
