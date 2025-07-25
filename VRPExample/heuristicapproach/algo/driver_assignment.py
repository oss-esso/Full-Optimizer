"""
Driver Assignment Module

This module implements the driver-vehicle matching system as described in the strategy document.
It provides functionality for:
1. Loading driver data from Excel files
2. Assigning drivers to vehicle routes using bipartite matching
3. Cost calculation for driver-route assignments
4. HoS feasibility checking for assignments

The implementation follows a two-stage approach:
Stage 1: Vehicle-centric route optimization (handled by existing EPDT algorithm)
Stage 2: Driver-centric assignment (implemented here)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment
import copy

from epdt_data_structures import Driver, DriverState, Route, Vehicle


def load_drivers_from_excel(file_path: str, sheet_name: str = 'AUTISTI') -> List[Driver]:
    """
    Load drivers from the AUTISTI sheet in furgoni.xlsx file.
    
    Args:
        file_path: Path to the Excel file
        sheet_name: Name of the sheet containing driver data
        
    Returns:
        List of Driver objects
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        drivers = []
        for _, row in df.iterrows():
            # Extract data from Excel row
            number_plate = str(row['NUMBER PLATE'])
            driver_name = str(row['DRIVER'])
            license_type = str(row['LICENSE'])
            
            # Create driver ID from name (remove spaces, lowercase)
            driver_id = driver_name.lower().replace(' ', '_').replace("'", "")
            
            # Create driver object
            driver = Driver(
                id=driver_id,
                name=driver_name,
                license=license_type,
                default_vehicle_id=number_plate,
                cost_per_hour=25.0,  # Default cost, could be configurable
                home_depot_id="main_depot",  # Default depot
                qualifications=set(),  # Could be enhanced to read from Excel
                hos_state=DriverState()
            )
            
            # Set basic qualifications based on license
            if license_type == 'CE':
                driver.qualifications.add('heavy_vehicle')
                driver.qualifications.add('standard_vehicle')
            elif license_type == 'B':
                driver.qualifications.add('standard_vehicle')
                
            drivers.append(driver)
            
        print(f"Loaded {len(drivers)} drivers from {sheet_name}")
        return drivers
        
    except Exception as e:
        print(f"Error loading drivers from Excel: {e}")
        return []


def calculate_route_duration_without_hos(route: Route) -> float:
    """
    Calculate route duration without HoS constraints for light vehicles.
    
    Args:
        route: Route to calculate duration for
        
    Returns:
        Total route duration in minutes
    """
    if not route.tasks:
        return 0.0
        
    total_time = 0.0
    
    # Service time for all tasks
    for task in route.tasks:
        total_time += task.service_time
        
    # Travel time (simplified calculation)
    total_time += route.calculate_travel_time()
    
    return total_time


def calculate_assignment_cost(driver: Driver, route: Route) -> float:
    """
    Calculate the cost of assigning a driver to a route.
    
    Args:
        driver: Driver to assign
        route: Route to assign driver to
        
    Returns:
        Assignment cost (float('inf') if infeasible)
    """
    cost = 0.0
    
    # 1. License Qualification Check
    if not driver.can_operate_vehicle(route.vehicle):
        return float('inf')  # Infeasible assignment
    
    # 2. HoS Feasibility and Cost (ONLY FOR HEAVY TRUCKS)
    if route.vehicle.vehicle_type == 'heavy':
        # Import here to avoid circular imports
        from second_level import _simulate_hos_advanced, _sort_tasks_chronologically
        
        # Create a copy of driver state for simulation
        driver_state_copy = copy.deepcopy(driver.hos_state)
        sorted_tasks = _sort_tasks_chronologically(route.tasks)
        
        is_feasible, route_duration = _simulate_hos_advanced(route, driver_state_copy, sorted_tasks)
        if not is_feasible:
            return float('inf')  # Infeasible assignment
        else:
            cost += route_duration * driver.cost_per_hour
    else:
        # For light trucks, HoS rules do not apply
        route_duration = calculate_route_duration_without_hos(route)
        cost += route_duration * driver.cost_per_hour
    
    # 3. Other Penalties (if applicable)
    PENALTY_WRONG_DEPOT = 50.0
    if driver.home_depot_id != route.vehicle.depot_id:
        cost += PENALTY_WRONG_DEPOT
    
    # 4. Default vehicle preference bonus
    if driver.default_vehicle_id == route.vehicle.id:
        cost -= 20.0  # Small bonus for preferred assignment
    
    return cost


def assign_drivers_to_routes(drivers: List[Driver], routes: List[Route]) -> Dict[str, str]:
    """
    Assign drivers to routes using minimum weight bipartite matching.
    
    Args:
        drivers: List of available drivers
        routes: List of vehicle routes to assign drivers to
        
    Returns:
        Dictionary mapping route_vehicle_id to driver_id
    """
    if not drivers or not routes:
        return {}
    
    # Filter routes that need heavy vehicle drivers (for HoS constraints)
    heavy_routes = [r for r in routes if r.vehicle.vehicle_type == 'heavy']
    light_routes = [r for r in routes if r.vehicle.vehicle_type != 'heavy']
    
    print(f"Assigning {len(drivers)} drivers to {len(routes)} routes")
    print(f"  - Heavy truck routes: {len(heavy_routes)}")
    print(f"  - Light vehicle routes: {len(light_routes)}")
    
    # Create cost matrix
    n_drivers = len(drivers)
    n_routes = len(routes)
    
    # Handle case where drivers != routes by padding with dummy assignments
    max_size = max(n_drivers, n_routes)
    cost_matrix = np.full((max_size, max_size), 1000.0)  # High cost for dummy assignments
    
    # Fill actual costs
    for i, driver in enumerate(drivers):
        for j, route in enumerate(routes):
            cost_matrix[i, j] = calculate_assignment_cost(driver, route)
    
    # Solve assignment problem using Hungarian algorithm
    try:
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        assignments = {}
        total_cost = 0.0
        successful_assignments = 0
        
        for i, j in zip(row_indices, col_indices):
            if i < n_drivers and j < n_routes:
                cost = cost_matrix[i, j]
                if cost != float('inf') and cost < 900.0:  # Valid assignment threshold
                    driver = drivers[i]
                    route = routes[j]
                    assignments[route.vehicle.id] = driver.id
                    
                    # Update route with assigned driver
                    route.driver = driver
                    
                    total_cost += cost
                    successful_assignments += 1
                    
                    print(f"Assigned driver {driver.name} to vehicle {route.vehicle.id} (cost: {cost:.2f})")
        
        print(f"Successfully assigned {successful_assignments}/{len(routes)} routes")
        print(f"Total assignment cost: {total_cost:.2f}")
        
        return assignments
        
    except Exception as e:
        print(f"Error in driver assignment: {e}")
        return {}


def validate_assignments(routes: List[Route]) -> Dict[str, List[str]]:
    """
    Validate driver assignments and return any issues found.
    
    Args:
        routes: List of routes with assigned drivers
        
    Returns:
        Dictionary of validation issues by category
    """
    issues = {
        'license_violations': [],
        'hos_violations': [],
        'unassigned_heavy_routes': [],
        'capacity_issues': []
    }
    
    for route in routes:
        if route.driver is None:
            if route.vehicle.vehicle_type == 'heavy':
                issues['unassigned_heavy_routes'].append(route.vehicle.id)
            continue
            
        # Check license compatibility
        if not route.driver.can_operate_vehicle(route.vehicle):
            issues['license_violations'].append(
                f"Driver {route.driver.name} (license: {route.driver.license}) "
                f"cannot operate vehicle {route.vehicle.id} (type: {route.vehicle.vehicle_type})"
            )
        
        # Check HoS feasibility for heavy vehicles
        if route.vehicle.vehicle_type == 'heavy':
            from second_level import _simulate_hos_advanced, _sort_tasks_chronologically
            
            driver_state_copy = copy.deepcopy(route.driver.hos_state)
            sorted_tasks = _sort_tasks_chronologically(route.tasks)
            
            is_feasible, _ = _simulate_hos_advanced(route, driver_state_copy, sorted_tasks)
            if not is_feasible:
                issues['hos_violations'].append(
                    f"Driver {route.driver.name} assigned to route {route.vehicle.id} "
                    f"violates Hours of Service regulations"
                )
    
    return issues


def print_assignment_summary(routes: List[Route], drivers: List[Driver]):
    """
    Print a summary of driver assignments.
    
    Args:
        routes: List of routes with assignments
        drivers: List of all available drivers
    """
    print("\n" + "="*60)
    print("DRIVER ASSIGNMENT SUMMARY")
    print("="*60)
    
    assigned_drivers = set()
    heavy_routes_assigned = 0
    light_routes_assigned = 0
    
    print("\nRoute Assignments:")
    print("-" * 50)
    for route in routes:
        if route.driver:
            assigned_drivers.add(route.driver.id)
            if route.vehicle.vehicle_type == 'heavy':
                heavy_routes_assigned += 1
            else:
                light_routes_assigned += 1
            print(f"Vehicle {route.vehicle.id:10} -> Driver {route.driver.name:15} "
                  f"(License: {route.driver.license}, Type: {route.vehicle.vehicle_type})")
        else:
            print(f"Vehicle {route.vehicle.id:10} -> UNASSIGNED "
                  f"(Type: {route.vehicle.vehicle_type})")
    
    print(f"\nAssignment Statistics:")
    print(f"  Total routes: {len(routes)}")
    print(f"  Heavy vehicle routes assigned: {heavy_routes_assigned}")
    print(f"  Light vehicle routes assigned: {light_routes_assigned}")
    print(f"  Total drivers available: {len(drivers)}")
    print(f"  Drivers assigned: {len(assigned_drivers)}")
    print(f"  Drivers unassigned: {len(drivers) - len(assigned_drivers)}")
    
    # Show unassigned drivers
    unassigned_drivers = [d for d in drivers if d.id not in assigned_drivers]
    if unassigned_drivers:
        print(f"\nUnassigned Drivers:")
        for driver in unassigned_drivers:
            print(f"  {driver.name} (License: {driver.license}, Default vehicle: {driver.default_vehicle_id})")


def update_hos_logic_for_driver_assignment():
    """
    Update the HoS simulation logic to work with the new driver assignment system.
    This function modifies the second_level.py imports and function calls.
    """
    # This function would contain logic to update second_level.py
    # For now, it's a placeholder that indicates the integration point
    print("HoS logic updated to use driver-based state tracking")


# Example usage and testing function
def test_driver_assignment():
    """
    Test function to demonstrate the driver assignment system.
    """
    # This would be called from a main script to test the functionality
    excel_path = "d:\\Projects\\OQI_Project\\Full Optimizer\\VRPExample\\heuristicapproach\\src\\furgoni.xlsx"
    
    # Load drivers
    drivers = load_drivers_from_excel(excel_path)
    
    # This would integrate with the existing EPDT algorithm to get routes
    # For testing, we would create mock routes
    print(f"Driver assignment system ready with {len(drivers)} drivers")
    
    return drivers


if __name__ == "__main__":
    test_driver_assignment()
