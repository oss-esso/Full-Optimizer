"""
Main Driver-Vehicle Matching System Implementation

This script demonstrates the complete two-stage driver-vehicle matching system
as outlined in the strategy document:

Stage 1: Vehicle-Centric Route Optimization (using existing EPDT algorithm)
Stage 2: Driver-Centric Assignment (using bipartite matching)

Usage:
    python driver_vehicle_matching_demo.py

This script will:
1. Load drivers from the Excel file
2. Create sample routes (in a real system, these would come from Stage 1)
3. Assign drivers to routes using the matching algorithm
4. Validate assignments and show results
"""

import sys
import os

# Add the algo directory to the Python path
algo_path = os.path.join(os.path.dirname(__file__), '..', 'algo')
sys.path.insert(0, algo_path)

from driver_assignment import (
    load_drivers_from_excel,
    assign_drivers_to_routes,
    validate_assignments,
    print_assignment_summary
)
from epdt_data_structures import Vehicle, Route, Task, TaskType, Driver, DriverState
import pandas as pd


def create_sample_vehicles() -> list:
    """Create sample vehicles for demonstration."""
    vehicles = []
    
    # Load vehicle data from Excel to create realistic vehicles
    try:
        excel_path = "d:\\Projects\\OQI_Project\\Full Optimizer\\VRPExample\\heuristicapproach\\src\\furgoni.xlsx"
        df = pd.read_excel(excel_path, sheet_name='VEICOLI')
        
        for _, row in df.iterrows():
            number_plate = str(row['NUMBER PLATE'])
            vehicle_type_raw = str(row['TYPE OF VEHICLE'])
            max_load_kg = float(row['MAX LOAD KG'])
            max_volume = float(row['MAX LOAD VOLUME M^3'])
            
            # Map Italian vehicle types to our system
            if 'CAMION' in vehicle_type_raw.upper():
                vehicle_type = 'heavy'
            elif 'FURGONE' in vehicle_type_raw.upper():
                vehicle_type = 'standard'
            else:
                vehicle_type = 'standard'  # Default
            
            # Create capabilities set
            capabilities = set()
            if pd.notna(row.get('LOADER')) and str(row['LOADER']).upper() == 'YES':
                capabilities.add('loader')
            if pd.notna(row.get('LOW TEMP')) and str(row['LOW TEMP']).upper() == 'YES':
                capabilities.add('low_temp')
            if pd.notna(row.get('HANGERS')) and str(row['HANGERS']).upper() == 'YES':
                capabilities.add('hangers')
            
            vehicle = Vehicle(
                id=number_plate,
                depot_id="main_depot",
                weight_capacity=max_load_kg,
                volume_capacity=max_volume,
                pallet_capacity=int(row.get('PALLET', 10)) if pd.notna(row.get('PALLET')) else 10,
                vehicle_type=vehicle_type,
                capabilities=capabilities,
                cost_per_km=1.2,
                cost_per_hour=50.0
            )
            
            vehicles.append(vehicle)
            
        print(f"Created {len(vehicles)} vehicles from Excel data")
        return vehicles
        
    except Exception as e:
        print(f"Error loading vehicles from Excel: {e}")
        # Fallback to hardcoded vehicles
        return create_fallback_vehicles()


def create_fallback_vehicles() -> list:
    """Create fallback vehicles if Excel loading fails."""
    vehicles = [
        Vehicle(
            id="XA359KW",
            depot_id="main_depot",
            weight_capacity=3500.0,
            volume_capacity=20.0,
            vehicle_type="heavy",
            capabilities={"loader"},
            cost_per_km=1.5,
            cost_per_hour=60.0
        ),
        Vehicle(
            id="FX192HX",
            depot_id="main_depot", 
            weight_capacity=1500.0,
            volume_capacity=10.0,
            vehicle_type="standard",
            capabilities=set(),
            cost_per_km=1.0,
            cost_per_hour=40.0
        ),
        Vehicle(
            id="FX194HX",
            depot_id="main_depot",
            weight_capacity=3500.0,
            volume_capacity=18.0,
            vehicle_type="heavy",
            capabilities={"low_temp"},
            cost_per_km=1.4,
            cost_per_hour=55.0
        )
    ]
    
    print(f"Created {len(vehicles)} fallback vehicles")
    return vehicles


def create_sample_tasks() -> list:
    """Create sample tasks for demonstration."""
    tasks = [
        Task(
            id="pickup_1",
            location_id="loc_1",
            task_type=TaskType.PICKUP,
            order_id="order_1",
            lat=45.4642,
            lon=9.1900,
            service_time=15.0,
            demand=500.0,
            volume=2.5,
            earliest_time=480.0,  # 8:00 AM
            latest_time=600.0     # 10:00 AM
        ),
        Task(
            id="delivery_1",
            location_id="loc_2",
            task_type=TaskType.DELIVERY,
            order_id="order_1",
            lat=45.4784,
            lon=9.2010,
            service_time=10.0,
            demand=-500.0,
            volume=-2.5,
            earliest_time=720.0,  # 12:00 PM
            latest_time=900.0     # 3:00 PM
        ),
        Task(
            id="pickup_2",
            location_id="loc_3",
            task_type=TaskType.PICKUP,
            order_id="order_2",
            lat=45.4500,
            lon=9.1800,
            service_time=20.0,
            demand=800.0,
            volume=4.0,
            requires_low_temp=True,
            earliest_time=540.0,  # 9:00 AM
            latest_time=660.0     # 11:00 AM
        )
    ]
    
    return tasks


def create_sample_routes(vehicles: list, tasks: list) -> list:
    """Create sample routes for demonstration (normally from Stage 1)."""
    routes = []
    
    # Route 1: Heavy vehicle with demanding tasks
    if len(vehicles) > 0 and len(tasks) > 1:
        heavy_vehicles = [v for v in vehicles if v.vehicle_type == 'heavy']
        if heavy_vehicles:
            route1 = Route(
                vehicle=heavy_vehicles[0],
                tasks=[tasks[0], tasks[1]] if len(tasks) > 1 else [tasks[0]]
            )
            routes.append(route1)
    
    # Route 2: Standard vehicle with lighter tasks  
    light_vehicles = [v for v in vehicles if v.vehicle_type != 'heavy']
    if light_vehicles and len(tasks) > 2:
        route2 = Route(
            vehicle=light_vehicles[0],
            tasks=[tasks[2]] if len(tasks) > 2 else []
        )
        routes.append(route2)
    
    # Route 3: Another heavy vehicle if available
    heavy_vehicles = [v for v in vehicles if v.vehicle_type == 'heavy']
    if len(heavy_vehicles) > 1:
        route3 = Route(
            vehicle=heavy_vehicles[1],
            tasks=[]  # Empty route for demonstration
        )
        routes.append(route3)
    
    return routes


def demonstrate_driver_vehicle_matching():
    """Main demonstration function."""
    print("="*80)
    print("DRIVER-VEHICLE MATCHING SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Step 1: Load drivers from Excel
    print("\nStep 1: Loading Drivers from Excel")
    print("-" * 40)
    excel_path = "d:\\Projects\\OQI_Project\\Full Optimizer\\VRPExample\\heuristicapproach\\src\\furgoni.xlsx"
    drivers = load_drivers_from_excel(excel_path)
    
    if not drivers:
        print("ERROR: Could not load drivers from Excel file")
        return
    
    print(f"Loaded drivers:")
    for driver in drivers:
        print(f"  - {driver.name} (License: {driver.license}, Default Vehicle: {driver.default_vehicle_id})")
    
    # Step 2: Create sample vehicles and routes (Stage 1 simulation)
    print(f"\nStep 2: Creating Sample Routes (Stage 1 Simulation)")
    print("-" * 40)
    vehicles = create_sample_vehicles()
    tasks = create_sample_tasks()
    routes = create_sample_routes(vehicles, tasks)
    
    print(f"Created {len(routes)} routes:")
    for i, route in enumerate(routes):
        task_count = len(route.tasks)
        print(f"  Route {i+1}: Vehicle {route.vehicle.id} ({route.vehicle.vehicle_type}) with {task_count} tasks")
    
    # Step 3: Assign drivers to routes (Stage 2)
    print(f"\nStep 3: Assigning Drivers to Routes (Stage 2)")
    print("-" * 40)
    assignments = assign_drivers_to_routes(drivers, routes)
    
    # Step 4: Validate assignments
    print(f"\nStep 4: Validating Assignments")
    print("-" * 40)
    issues = validate_assignments(routes)
    
    if any(issues.values()):
        print("Validation Issues Found:")
        for category, issue_list in issues.items():
            if issue_list:
                print(f"  {category.replace('_', ' ').title()}:")
                for issue in issue_list:
                    print(f"    - {issue}")
    else:
        print("✓ All assignments are valid!")
    
    # Step 5: Print summary
    print_assignment_summary(routes, drivers)
    
    # Step 6: Show integration points
    print(f"\nStep 6: Integration with Existing EPDT System")
    print("-" * 50)
    print("Integration Points:")
    print("1. Route data structures now include driver assignments")
    print("2. HoS logic updated to use driver-specific state")
    print("3. Cost calculations use driver hourly rates")
    print("4. Feasibility checks include driver license validation")
    print("5. Assignment algorithm can be called after Stage 1 optimization")
    
    print(f"\nDemonstration completed successfully!")
    print("="*80)


if __name__ == "__main__":
    demonstrate_driver_vehicle_matching()
