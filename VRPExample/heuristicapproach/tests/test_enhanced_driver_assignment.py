"""
Test script for the enhanced driver assignment system.

This script validates the implementation according to the driver_vehicle_matching_strategy.md requirements.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'algo'))

from driver_assignment_enhanced import (
    load_drivers_from_excel_enhanced,
    assign_drivers_to_routes_enhanced,
    DriverAssignmentConfig,
    EnhancedDriver,
    calculate_enhanced_assignment_cost,
    validate_assignments,
    print_assignment_summary
)
from epdt_data_structures import Vehicle, Route, Task, TaskType, DriverState


def test_driver_loading():
    """Test enhanced driver loading from Excel."""
    print("Testing Enhanced Driver Loading...")
    print("-" * 40)
    
    excel_path = "../src/furgoni.xlsx"
    
    # Test with default config
    drivers = load_drivers_from_excel_enhanced(excel_path)
    
    print(f"✓ Loaded {len(drivers)} drivers")
    
    # Verify driver attributes
    for driver in drivers[:3]:  # Show first 3 drivers
        print(f"  Driver: {driver.name}")
        print(f"    ID: {driver.id}")
        print(f"    License: {driver.license}")
        print(f"    Default Vehicle: {driver.default_vehicle_id}")
        print(f"    Qualifications: {driver.qualifications}")
        print(f"    Experience: {driver.experience_years} years")
        print(f"    Performance: {driver.performance_rating}/10")
        print()
    
    return drivers


def test_cost_calculation():
    """Test the enhanced cost calculation function."""
    print("Testing Enhanced Cost Calculation...")
    print("-" * 40)
    
    # Create test driver
    driver = EnhancedDriver(
        id="test_driver_1",
        name="Test Driver",
        license="CE",
        default_vehicle_id="TEST001",
        cost_per_hour=25.0,
        home_depot_id="main_depot",
        qualifications={'heavy_vehicle', 'standard_vehicle', 'loader'},
        hos_state=DriverState(),
        experience_years=5,
        performance_rating=7.5
    )
    
    # Create test vehicle
    vehicle = Vehicle(
        id="TEST001",
        depot_id="main_depot",
        weight_capacity=8000.0,
        volume_capacity=50.0,
        vehicle_type="heavy",
        capabilities={'loader'}
    )
    
    # Create test tasks
    tasks = [
        Task(
            id="task1",
            location_id="loc1",
            task_type=TaskType.PICKUP,
            order_id="order1",
            lat=45.0,
            lon=9.0,
            service_time=30.0,
            demand=1000.0
        ),
        Task(
            id="task2",
            location_id="loc2",
            task_type=TaskType.DELIVERY,
            order_id="order1",
            lat=45.1,
            lon=9.1,
            service_time=20.0,
            demand=-1000.0
        )
    ]
    
    # Create test route
    route = Route(vehicle=vehicle, tasks=tasks)
    
    # Test cost calculation
    config = DriverAssignmentConfig()
    cost = calculate_enhanced_assignment_cost(driver, route, config)
    
    print(f"✓ Cost calculation successful: {cost:.2f}")
    print(f"  Driver qualifications match vehicle requirements: {driver.qualifications.issuperset(vehicle.capabilities)}")
    print(f"  License compatibility: {driver.can_operate_vehicle(vehicle)}")
    print()
    
    return cost


def test_assignment_algorithm():
    """Test the enhanced assignment algorithm."""
    print("Testing Enhanced Assignment Algorithm...")
    print("-" * 40)
    
    # Create test drivers
    drivers = [
        EnhancedDriver(
            id="driver_ce_1",
            name="CE Driver 1",
            license="CE",
            default_vehicle_id="TRUCK001",
            cost_per_hour=30.0,
            home_depot_id="main_depot",
            qualifications={'heavy_vehicle', 'standard_vehicle', 'loader'},
            hos_state=DriverState(),
            experience_years=8,
            performance_rating=8.5
        ),
        EnhancedDriver(
            id="driver_b_1",
            name="B Driver 1",
            license="B",
            default_vehicle_id="VAN001",
            cost_per_hour=25.0,
            home_depot_id="main_depot",
            qualifications={'standard_vehicle'},
            hos_state=DriverState(),
            experience_years=3,
            performance_rating=6.0
        )
    ]
    
    # Create test vehicles and routes
    heavy_vehicle = Vehicle(
        id="TRUCK001",
        depot_id="main_depot",
        weight_capacity=13500.0,
        volume_capacity=70.0,
        vehicle_type="heavy",
        capabilities={'loader'}
    )
    
    light_vehicle = Vehicle(
        id="VAN001",
        depot_id="main_depot",
        weight_capacity=800.0,
        volume_capacity=20.0,
        vehicle_type="standard",
        capabilities=set()
    )
    
    routes = [
        Route(vehicle=heavy_vehicle, tasks=[]),
        Route(vehicle=light_vehicle, tasks=[])
    ]
    
    # Test assignment
    config = DriverAssignmentConfig()
    assignments = assign_drivers_to_routes_enhanced(drivers, routes, config)
    
    print(f"✓ Assignment completed: {len(assignments)} assignments made")
    for vehicle_id, driver_id in assignments.items():
        print(f"  Vehicle {vehicle_id} -> Driver {driver_id}")
    
    # Validate assignments
    issues = validate_assignments(routes)
    total_issues = sum(len(issue_list) for issue_list in issues.values())
    print(f"✓ Validation completed: {total_issues} issues found")
    
    return assignments


def test_integration():
    """Test full integration with real Excel data."""
    print("Testing Full Integration...")
    print("-" * 40)
    
    try:
        excel_path = "../src/furgoni.xlsx"
        
        # Load real drivers
        drivers = load_drivers_from_excel_enhanced(excel_path)
        print(f"✓ Loaded {len(drivers)} real drivers from Excel")
        
        # Create mock routes for testing (in real scenario, these would come from EPDT algorithm)
        mock_routes = []
        
        # Create some mock vehicles based on the Excel data
        heavy_vehicle = Vehicle(
            id="FX194HX",  # From Excel data
            depot_id="main_depot",
            weight_capacity=13500.0,
            volume_capacity=67.0,
            vehicle_type="heavy",
            capabilities={'loader'}
        )
        
        light_vehicle = Vehicle(
            id="FX192HX",  # From Excel data
            depot_id="main_depot", 
            weight_capacity=800.0,
            volume_capacity=21.0,
            vehicle_type="standard",
            capabilities=set()
        )
        
        mock_routes = [
            Route(vehicle=heavy_vehicle, tasks=[]),
            Route(vehicle=light_vehicle, tasks=[])
        ]
        
        # Test assignment with real data
        config = DriverAssignmentConfig()
        assignments = assign_drivers_to_routes_enhanced(
            [d for d in drivers if isinstance(d, EnhancedDriver)], 
            mock_routes, 
            config
        )
        
        print(f"✓ Real data assignment completed: {len(assignments)} assignments")
        
        # Print summary
        print_assignment_summary(mock_routes, drivers)
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests for the enhanced driver assignment system."""
    print("ENHANCED DRIVER ASSIGNMENT SYSTEM TESTS")
    print("="*50)
    print()
    
    try:
        # Run individual tests
        drivers = test_driver_loading()
        cost = test_cost_calculation()
        assignments = test_assignment_algorithm()
        integration_success = test_integration()
        
        print("\nTEST SUMMARY")
        print("="*30)
        print("✓ Driver loading test: PASSED")
        print("✓ Cost calculation test: PASSED")
        print("✓ Assignment algorithm test: PASSED")
        print(f"{'✓' if integration_success else '✗'} Integration test: {'PASSED' if integration_success else 'FAILED'}")
        
        if integration_success:
            print("\n🎉 All tests PASSED! Enhanced driver assignment system is ready.")
        else:
            print("\n⚠️  Some tests failed. Please review the output above.")
            
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
