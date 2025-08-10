"""
Basic test for HoS simulation to verify the implementation works.
"""
import sys
import os

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
algo_dir = os.path.join(heuristic_root, 'algo')
sys.path.insert(0, heuristic_root)
sys.path.insert(0, algo_dir)

try:
    from epdt_data_structures import DriverState, Driver, Route, Vehicle, Task, TaskType
    from second_level import _check_hos_multiday
    print("✅ Successfully imported required modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_driver_state_copy():
    """Test that DriverState has copy method."""
    print("\n🔹 Testing DriverState copy method...")
    
    driver_state = DriverState()
    driver_state.drive_today = 100.0
    driver_state.work_today = 150.0
    
    try:
        copied_state = driver_state.copy()
        print(f"   Original drive_today: {driver_state.drive_today}")
        print(f"   Copied drive_today: {copied_state.drive_today}")
        
        # Modify original
        driver_state.drive_today = 200.0
        print(f"   After modifying original: {driver_state.drive_today}")
        print(f"   Copy should be unchanged: {copied_state.drive_today}")
        
        if copied_state.drive_today == 100.0:
            print("   ✅ Copy method works correctly")
            return True
        else:
            print("   ❌ Copy method failed - copy was modified")
            return False
    except Exception as e:
        print(f"   ❌ Copy method failed: {e}")
        return False

def test_driver_state_reset_daily():
    """Test that DriverState has reset_daily method."""
    print("\n🔹 Testing DriverState reset_daily method...")
    
    driver_state = DriverState()
    driver_state.drive_today = 100.0
    driver_state.work_today = 150.0
    driver_state.drive_since_break = 50.0
    driver_state.work_since_break = 75.0
    
    try:
        driver_state.reset_daily()
        if (driver_state.drive_today == 0.0 and 
            driver_state.work_today == 0.0 and 
            driver_state.drive_since_break == 0.0 and 
            driver_state.work_since_break == 0.0):
            print("   ✅ reset_daily method works correctly")
            return True
        else:
            print("   ❌ reset_daily method failed - counters not reset")
            return False
    except Exception as e:
        print(f"   ❌ reset_daily method failed: {e}")
        return False

def test_basic_hos_check():
    """Test basic HoS check function."""
    print("\n🔹 Testing basic HoS check...")
    
    try:
        # Create minimal objects for testing
        vehicle = Vehicle(
            id="TEST_VAN",
            depot_id="main_depot",
            weight_capacity=1000.0,
            volume_capacity=10.0,
            cost_per_hour=25.0
        )
        
        driver = Driver(
            id="test_driver",
            name="Test Driver",
            license="CE",
            default_vehicle_id="TEST_VAN",
            cost_per_hour=25.0
        )
        
        route = Route(vehicle=vehicle, driver=driver)
        
        # Create simple tasks
        task1 = Task(
            id="DEPOT_START",
            location_id="depot_start",
            task_type=TaskType.DEPOT_START,
            order_id="depot_order",
            lat=44.9,
            lon=8.2,
            service_time=0,
            demand=0,
            volume=0
        )
        
        task2 = Task(
            id="DEPOT_RETURN",
            location_id="depot_return", 
            task_type=TaskType.DEPOT_RETURN,
            order_id="depot_order",
            lat=44.9,
            lon=8.2,
            service_time=0,
            demand=0,
            volume=0
        )
        
        # Add tasks to route
        route.tasks = [task1, task2]
        
        # Test with minimal task list
        driver_state = DriverState()
        tasks = [task1, task2]
        
        feasible, cost = _check_hos_multiday(route, driver_state, tasks)
        
        print(f"   Route feasible: {feasible}")
        print(f"   HoS cost: {cost}")
        print("   ✅ Basic HoS check completed successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Basic HoS check failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

def main():
    """Run all basic tests."""
    print("🧪 Testing Basic HoS Implementation")
    print("=" * 50)
    
    all_passed = True
    
    # Test 1: DriverState copy method
    if not test_driver_state_copy():
        all_passed = False
    
    # Test 2: DriverState reset_daily method
    if not test_driver_state_reset_daily():
        all_passed = False
    
    # Test 3: Basic HoS check
    if not test_basic_hos_check():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All basic tests PASSED!")
        return 0
    else:
        print("❌ Some tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
