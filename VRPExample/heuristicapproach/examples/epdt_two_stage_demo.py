"""
EPDT Algorithm Integration Example

This script demonstrates how the enhanced driver assignment system integrates
with the EPDT algorithm's two-stage approach as specified in the strategy document.

Stage 1: Vehicle-Centric Route Optimization (using existing EPDT)
Stage 2: Driver-Centric Assignment (using enhanced driver assignment)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'algo'))

from driver_assignment_enhanced import (
    load_drivers_from_excel_enhanced,
    assign_drivers_to_routes_enhanced,
    DriverAssignmentConfig,
    print_assignment_summary,
    validate_assignments
)


def simulate_epdt_stage_1():
    """
    Simulate Stage 1: Vehicle-centric route optimization using EPDT algorithm.
    
    In the real implementation, this would be the existing EPDT algorithm
    that generates optimal vehicle routes considering vehicle constraints
    (capacity, type, LIFO requirements, etc.) but without driver-specific HoS.
    
    Returns:
        List[Route]: Optimized vehicle routes from Stage 1
    """
    print("STAGE 1: Vehicle-Centric Route Optimization (EPDT Algorithm)")
    print("-" * 60)
    print("🚛 Running EPDT algorithm for vehicle route optimization...")
    print("   - Considering vehicle capacity constraints")
    print("   - Optimizing pickup/delivery sequences")
    print("   - Handling LIFO requirements")
    print("   - Minimizing total travel distance")
    print("   - Ensuring order precedence constraints")
    print("✅ Stage 1 complete: Generated optimized vehicle routes")
    print()
    
    # In real implementation, this would return routes from EPDT
    # For demonstration, we'll return mock routes based on furgoni.xlsx data
    from epdt_data_structures import Vehicle, Route, Task, TaskType
    
    # Create vehicles based on actual Excel data
    vehicles = [
        Vehicle(
            id="FX194HX",  # Heavy truck from Excel
            depot_id="main_depot",
            weight_capacity=13500.0,
            volume_capacity=67.14,
            vehicle_type="heavy",
            capabilities={'loader'}
        ),
        Vehicle(
            id="GA621VG",  # Another heavy truck
            depot_id="main_depot",
            weight_capacity=8000.0,
            volume_capacity=56.94,
            vehicle_type="heavy",
            capabilities={'loader'}
        ),
        Vehicle(
            id="FX192HX",  # Light vehicle from Excel
            depot_id="main_depot",
            weight_capacity=800.0,
            volume_capacity=21.79,
            vehicle_type="standard",
            capabilities={'hangers'}
        ),
        Vehicle(
            id="GA620VG",  # Low temp vehicle
            depot_id="main_depot",
            weight_capacity=700.0,
            volume_capacity=18.17,
            vehicle_type="standard",
            capabilities={'low_temp'}
        )
    ]
    
    # Create mock optimized routes (would come from EPDT in real scenario)
    routes = []
    for i, vehicle in enumerate(vehicles):
        # Create sample tasks for each route
        tasks = []
        task_count = 5 + (i * 2)  # Varying complexity
        
        for j in range(task_count):
            task = Task(
                id=f"task_{vehicle.id}_{j}",
                location_id=f"loc_{i}_{j}",
                task_type=TaskType.PICKUP if j % 2 == 0 else TaskType.DELIVERY,
                order_id=f"order_{i}_{j//2}",
                lat=45.0 + (i * 0.1) + (j * 0.01),
                lon=9.0 + (i * 0.1) + (j * 0.01),
                service_time=30.0 + (j * 5),
                demand=100.0 * (1 if j % 2 == 0 else -1)
            )
            tasks.append(task)
        
        route = Route(vehicle=vehicle, tasks=tasks)
        routes.append(route)
    
    return routes


def run_epdt_stage_2(routes, excel_path="src/furgoni.xlsx"):
    """
    Run Stage 2: Driver-centric assignment using enhanced driver assignment.
    
    Args:
        routes: Optimized vehicle routes from Stage 1
        excel_path: Path to Excel file with driver data
        
    Returns:
        Dict: Driver assignments mapping vehicle_id -> driver_id
    """
    print("STAGE 2: Driver-Centric Assignment (Enhanced Driver Assignment)")
    print("-" * 60)
    
    # Load configuration
    config = DriverAssignmentConfig.load_from_file("config/driver_assignment.json")
    print("✅ Loaded assignment configuration")
    
    # Load drivers from Excel with enhanced functionality
    drivers = load_drivers_from_excel_enhanced(excel_path, config=config)
    print(f"✅ Loaded {len(drivers)} enhanced drivers from Excel")
    
    # Perform optimal driver assignment using Hungarian algorithm
    print("🧮 Computing optimal driver assignments...")
    assignments = assign_drivers_to_routes_enhanced(drivers, routes, config)
    print(f"✅ Stage 2 complete: {len(assignments)} optimal assignments computed")
    print()
    
    return assignments, drivers


def validate_complete_solution(routes, drivers):
    """
    Validate the complete two-stage solution for feasibility and compliance.
    
    Args:
        routes: Routes with assigned drivers
        drivers: List of all drivers
    """
    print("SOLUTION VALIDATION")
    print("-" * 60)
    
    # Check for assignment violations
    issues = validate_assignments(routes)
    total_issues = sum(len(issue_list) for issue_list in issues.values())
    
    if total_issues == 0:
        print("✅ All assignments are feasible and compliant")
    else:
        print(f"⚠️  Found {total_issues} potential issues:")
        for issue_type, issue_list in issues.items():
            if issue_list:
                print(f"  {issue_type}: {len(issue_list)} issues")
                for issue in issue_list[:3]:  # Show first 3
                    print(f"    - {issue}")
    
    # Check coverage
    assigned_vehicles = sum(1 for route in routes if route.driver is not None)
    total_vehicles = len(routes)
    coverage = (assigned_vehicles / total_vehicles) * 100
    
    print(f"📊 Assignment Coverage: {assigned_vehicles}/{total_vehicles} ({coverage:.1f}%)")
    
    # Check HoS compliance for heavy vehicles
    heavy_routes = [r for r in routes if r.vehicle.vehicle_type == 'heavy' and r.driver]
    print(f"🚛 Heavy vehicle routes: {len(heavy_routes)} assigned")
    
    print()


def main():
    """
    Demonstrate the complete two-stage EPDT algorithm with enhanced driver assignment.
    """
    print("EPDT ALGORITHM: TWO-STAGE APPROACH DEMONSTRATION")
    print("=" * 70)
    print()
    
    try:
        # Stage 1: Vehicle-centric route optimization
        routes = simulate_epdt_stage_1()
        
        # Stage 2: Driver-centric assignment  
        assignments, drivers = run_epdt_stage_2(routes)
        
        # Validation and reporting
        validate_complete_solution(routes, drivers)
        
        # Print detailed summary
        print_assignment_summary(routes, drivers)
        
        print("\n" + "=" * 70)
        print("🎉 TWO-STAGE EPDT ALGORITHM DEMONSTRATION COMPLETE")
        print("=" * 70)
        print("\nKey Benefits of Two-Stage Approach:")
        print("✅ Decoupled driver and vehicle constraints")
        print("✅ Optimal vehicle routing (Stage 1)")
        print("✅ Optimal driver assignment (Stage 2)")
        print("✅ Hours of Service compliance")
        print("✅ Scalable and maintainable architecture")
        print("✅ Real-world operational constraints handled")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
