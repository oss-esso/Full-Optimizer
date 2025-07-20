#!/usr/bin/env python3
"""
Quick test to verify the advanced HoS simulation is working correctly.

This test creates a simple scenario and verifies that the advanced HoS 
simulation (_simulate_hos_advanced) is properly integrated and functioning.
"""

import sys
import os

# Add path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
algo_dir = os.path.join(heuristic_root, 'algo')
sys.path.insert(0, heuristic_root)
sys.path.insert(0, algo_dir)

def test_advanced_hos_import():
    """Test that advanced HoS functions can be imported."""
    try:
        # Try direct import first
        import second_level
        _simulate_hos_advanced = second_level._simulate_hos_advanced
        DriverState = second_level.DriverState
        print("✅ Successfully imported _simulate_hos_advanced and DriverState")
        return True, _simulate_hos_advanced, DriverState
    except ImportError:
        try:
            # Try from algo directory
            from algo import second_level
            _simulate_hos_advanced = second_level._simulate_hos_advanced
            DriverState = second_level.DriverState
            print("✅ Successfully imported _simulate_hos_advanced and DriverState from algo")
            return True, _simulate_hos_advanced, DriverState
        except ImportError as e:
            print(f"❌ Failed to import advanced HoS functions: {e}")
            return False, None, None

def test_driver_state_enhanced_fields():
    """Test that DriverState has all the enhanced fields."""
    success, _, DriverState = test_advanced_hos_import()
    if not success:
        return False
    
    # Create a test driver state
    driver = DriverState()
    
    # Check for enhanced fields
    enhanced_fields = [
        'time_in_daily_period',
        'work_this_week', 
        'time_since_weekly_rest',
        'daily_rest_reductions_used',
        'is_weekly_rest_reduction_taken'
    ]
    
    missing_fields = []
    for field in enhanced_fields:
        if not hasattr(driver, field):
            missing_fields.append(field)
    
    if missing_fields:
        print(f"❌ DriverState missing enhanced fields: {missing_fields}")
        return False
    
    print("✅ DriverState has all enhanced fields")
    return True

def test_simulate_hos_advanced_basic():
    """Test basic functionality of _simulate_hos_advanced."""
    success, simulate_func, DriverState = test_advanced_hos_import()
    if not success:
        return False
    
    try:
        # Create minimal test data
        driver_state = DriverState()
        sorted_tasks = []  # Empty task list
        
        # This should work with empty task list
        result = simulate_func(None, driver_state, sorted_tasks) 
        
        if isinstance(result, tuple) and len(result) == 2:
            feasible, time_elapsed = result
            if feasible and time_elapsed >= 0:
                print("✅ _simulate_hos_advanced basic test passed")
                return True
        
        print("❌ _simulate_hos_advanced returned unexpected result")
        return False
        
    except Exception as e:
        print(f"❌ _simulate_hos_advanced basic test failed: {e}")
        return False

def main():
    """Run all HoS tests."""
    print("🧪 Testing Advanced HoS Simulation Integration")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_advanced_hos_import),
        ("Enhanced Fields Test", test_driver_state_enhanced_fields),
        ("Basic Function Test", test_simulate_hos_advanced_basic),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name}...")
        if test_func()[0] if test_name == "Import Test" else test_func():
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Advanced HoS simulation is ready for --scenario furgoni")
        return True
    else:
        print("⚠️  Some tests failed. Check integration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
