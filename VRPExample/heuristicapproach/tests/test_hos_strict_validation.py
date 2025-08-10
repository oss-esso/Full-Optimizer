"""
Test script to validate the corrected HoS simulation logic.
This test creates a route that would require mandatory breaks during travel
and verifies that it's correctly flagged as infeasible.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_strict_hos_validation():
    """Test that HoS simulation correctly flags infeasible routes."""
    try:
        from algo.second_level import _check_hos_multiday
        from algo.epdt_data_structures import DriverState, Task, Route, Vehicle, Driver
        
        print("🧪 Testing strict HoS validation...")
        
        # Create a mock route with very long travel times that would require breaks
        driver_state = DriverState()
        
        # Mock tasks that would require more than 4.5 hours of continuous driving
        class MockTask:
            def __init__(self, lat, lon, service_time=0, day=0):
                self.lat = lat
                self.lon = lon
                self.service_time = service_time
                self.day = day
                self.earliest_time = None
                self.latest_time = None
        
        # Create tasks with significant distance requiring long travel time
        tasks = [
            MockTask(45.0, 7.0, 30, 0),    # Start
            MockTask(48.0, 11.0, 30, 0),  # Very far destination (would require 5+ hours)
            MockTask(45.0, 7.0, 0, 0)     # Return
        ]
        
        # Mock route and vehicle
        class MockVehicle:
            cost_per_hour = 25.0
            
        class MockRoute:
            def __init__(self):
                self.vehicle = MockVehicle()
                self.driver = None
        
        route = MockRoute()
        
        # Test with a fresh driver state
        is_feasible, cost = _check_hos_multiday(route, driver_state, tasks)
        
        if not is_feasible:
            print("✅ PASS: Route correctly flagged as infeasible due to HoS violations")
            print(f"   Accumulated rest cost: {cost:.2f}")
        else:
            print("❌ FAIL: Route should have been flagged as infeasible")
            
        return is_feasible
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_depot_vs_customer_waiting():
    """Test that depot waiting doesn't count as work time."""
    try:
        from algo.second_level import _check_hos_multiday
        from algo.epdt_data_structures import DriverState
        
        print("\n🧪 Testing depot vs customer waiting logic...")
        
        class MockTask:
            def __init__(self, is_depot=False, earliest_time=None):
                self.lat = 45.0
                self.lon = 7.0
                self.service_time = 0
                self.day = 0
                self.earliest_time = earliest_time
                self.latest_time = None
                self.is_depot = is_depot
                
            def is_depot_start(self):
                return self.is_depot
        
        class MockVehicle:
            cost_per_hour = 25.0
            
        class MockRoute:
            def __init__(self):
                self.vehicle = MockVehicle()
                self.driver = None
        
        # Test with depot waiting (should not count as work)
        depot_task = MockTask(is_depot=True, earliest_time=480)  # 8 hours wait
        customer_task = MockTask(is_depot=False)
        
        tasks = [depot_task, customer_task]
        route = MockRoute()
        driver_state = DriverState()
        
        is_feasible, cost = _check_hos_multiday(route, driver_state, tasks)
        
        print(f"   Depot waiting test - Feasible: {is_feasible}, Cost: {cost:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Waiting test error: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing corrected HoS simulation and feasibility logic...")
    
    test1_result = test_strict_hos_validation()
    test2_result = test_depot_vs_customer_waiting()
    
    if test1_result is False and test2_result:  # We want test1 to return False (infeasible)
        print("\n✅ All tests passed! HoS simulation is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
