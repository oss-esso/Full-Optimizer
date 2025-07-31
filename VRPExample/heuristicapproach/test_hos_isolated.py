#!/usr/bin/env python3

"""
Isolated test to verify the HoS (_check_hos_multiday) function behavior.
"""

import sys
import os

# Add the current directory to the path to import from the algo package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'algo'))

from algo.second_level import _check_hos_multiday
from algo.epdt_data_structures import DriverState

# Mock task class for testing
class MockTask:
    def __init__(self, lat, lon, service_time=30, day=0):
        self.lat = lat
        self.lon = lon
        self.service_time = service_time  # minutes
        self.day = day
        self.earliest_time = None
        self.latest_time = None

# Mock route class
class MockRoute:
    def __init__(self, tasks):
        self.tasks = tasks

def test_hos_function_signatures():
    """Test which HoS function is actually being called."""
    print("=== Testing HoS Function Signatures ===")
    
    # Create a simple route with tasks that should violate HoS
    tasks = [
        MockTask(44.9, 8.2, 30, 0),  # Start at depot
        MockTask(45.0, 8.5, 600, 0),  # 10 hours of service (should exceed work limits)
        MockTask(44.9, 8.2, 30, 0),  # Return to depot
    ]
    
    route = MockRoute(tasks)
    driver_state = DriverState()
    
    print(f"Task 1: lat={tasks[0].lat}, lon={tasks[0].lon}, service_time={tasks[0].service_time}")
    print(f"Task 2: lat={tasks[1].lat}, lon={tasks[1].lon}, service_time={tasks[1].service_time}")
    print(f"Task 3: lat={tasks[2].lat}, lon={tasks[2].lon}, service_time={tasks[2].service_time}")
    
    # Test the 2-parameter version (should be the stub)
    try:
        result_2_param = _check_hos_multiday(route, debug_feasibility=True)
        print(f"✅ 2-parameter version called successfully: {result_2_param}")
    except Exception as e:
        print(f"❌ 2-parameter version failed: {e}")
    
    # Test the 3-parameter version (should be the real implementation)
    try:
        result_3_param = _check_hos_multiday(route, driver_state, tasks)
        print(f"✅ 3-parameter version called successfully: {result_3_param}")
        print(f"    Driver state after: drive_today={driver_state.drive_today}, work_today={driver_state.work_today}")
    except Exception as e:
        print(f"❌ 3-parameter version failed: {e}")
        import traceback
        traceback.print_exc()

def test_extreme_hos_violation():
    """Test with an extreme HoS violation that should definitely fail."""
    print("\n=== Testing Extreme HoS Violation ===")
    
    # Create a route with extreme violations (over 14 hours work)
    tasks = [
        MockTask(44.9, 8.2, 30, 0),     # Start at depot
        MockTask(45.0, 8.5, 900, 0),   # 15 hours of service (should exceed work limits)
        MockTask(44.9, 8.2, 30, 0),    # Return to depot
    ]
    
    route = MockRoute(tasks)
    driver_state = DriverState()
    
    print(f"Testing route with 15-hour service task (should violate work limits)")
    result = _check_hos_multiday(route, driver_state, tasks)
    print(f"Result: {result} (should be False)")
    print(f"Driver state: drive_today={driver_state.drive_today}, work_today={driver_state.work_today}")

def test_realistic_violation():
    """Test with violation similar to what we see in the comprehensive test."""
    print("\n=== Testing Realistic Violation (18+ hours work) ===")
    
    # Create tasks that simulate long driving + service times
    tasks = [
        MockTask(44.9, 8.2, 30, 0),      # Start at depot
        MockTask(42.0, 12.5, 300, 0),    # Rome (5 hours service + ~6 hours driving)
        MockTask(45.7, 9.7, 300, 0),     # Milan (5 hours service + ~6 hours driving)
        MockTask(44.9, 8.2, 30, 0),      # Return to depot (~2 hours driving)
    ]
    
    route = MockRoute(tasks)
    driver_state = DriverState()
    
    print(f"Testing route with multiple distant locations (should accumulate >14 hours)")
    result = _check_hos_multiday(route, driver_state, tasks)
    print(f"Result: {result}")
    print(f"Driver state: drive_today={driver_state.drive_today:.1f}min, work_today={driver_state.work_today:.1f}min")
    print(f"Total work hours: {driver_state.work_today/60:.1f}h (limit: 14h)")

def test_work_limit_boundary():
    """Test exactly at the work limit boundary."""
    print("\n=== Testing Work Limit Boundary ===")
    
    # Test exactly 14 hours (840 minutes) - should pass
    tasks = [
        MockTask(44.9, 8.2, 30, 0),      # Start at depot
        MockTask(45.0, 8.5, 810, 0),     # 13.5 hours service (plus travel should hit ~14h total)
        MockTask(44.9, 8.2, 30, 0),      # Return to depot
    ]
    
    route = MockRoute(tasks)
    driver_state = DriverState()
    
    print(f"Testing route with ~13.5h service + travel (should hit ~14h limit)")
    result = _check_hos_multiday(route, driver_state, tasks)
    print(f"Result: {result}")
    print(f"Driver state: drive_today={driver_state.drive_today:.1f}min, work_today={driver_state.work_today:.1f}min")
    print(f"Total work hours: {driver_state.work_today/60:.1f}h (limit: 14h)")
    
    # Test slightly over 14 hours - should fail
    tasks_over = [
        MockTask(44.9, 8.2, 30, 0),      # Start at depot
        MockTask(45.0, 8.5, 850, 0),     # 14.2 hours service (plus travel should exceed 14h)
        MockTask(44.9, 8.2, 30, 0),      # Return to depot
    ]
    
    route_over = MockRoute(tasks_over)
    driver_state_over = DriverState()
    
    print(f"\nTesting route with 14.2h service + travel (should exceed 14h limit)")
    result_over = _check_hos_multiday(route_over, driver_state_over, tasks_over)
    print(f"Result: {result_over} (should be False)")
    print(f"Driver state: drive_today={driver_state_over.drive_today:.1f}min, work_today={driver_state_over.work_today:.1f}min")
    print(f"Total work hours: {driver_state_over.work_today/60:.1f}h (limit: 14h)")

if __name__ == "__main__":
    test_hos_function_signatures()
    test_extreme_hos_violation()
    test_realistic_violation()
    test_work_limit_boundary()
