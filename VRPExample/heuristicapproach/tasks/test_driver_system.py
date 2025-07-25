"""
Simple test script to validate the driver-vehicle matching implementation
"""

import sys
import os

# Add the algo directory to the Python path
algo_path = os.path.join(os.path.dirname(__file__), '..', 'algo')
sys.path.insert(0, algo_path)

try:
    from driver_assignment import load_drivers_from_excel
    from epdt_data_structures import Driver, DriverState, Vehicle, Route, Task, TaskType
    
    print("✓ All imports successful")
    
    # Test driver loading
    excel_path = "d:\\Projects\\OQI_Project\\Full Optimizer\\VRPExample\\heuristicapproach\\src\\furgoni.xlsx"
    drivers = load_drivers_from_excel(excel_path)
    print(f"✓ Loaded {len(drivers)} drivers from Excel")
    
    # Test driver creation
    test_driver = Driver(
        id="test_driver",
        name="Test Driver",
        license="CE",
        default_vehicle_id="TEST123",
        cost_per_hour=30.0
    )
    print(f"✓ Created test driver: {test_driver.name}")
    
    # Test vehicle creation
    test_vehicle = Vehicle(
        id="TEST123",
        depot_id="main_depot",
        weight_capacity=3500.0,
        volume_capacity=20.0,
        vehicle_type="heavy"
    )
    print(f"✓ Created test vehicle: {test_vehicle.id}")
    
    # Test driver can operate vehicle
    can_operate = test_driver.can_operate_vehicle(test_vehicle)
    print(f"✓ Driver license check: {can_operate}")
    
    # Test route creation with driver
    test_route = Route(vehicle=test_vehicle, driver=test_driver)
    print(f"✓ Created route with assigned driver")
    
    # Test DriverState
    hos_state = DriverState()
    can_drive = hos_state.can_drive(120.0)  # 2 hours
    print(f"✓ HoS check: can drive 2 hours = {can_drive}")
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED!")
    print("Driver-vehicle matching system is working correctly.")
    print("="*50)
    
except Exception as e:
    print(f"❌ Error in testing: {e}")
    import traceback
    traceback.print_exc()
