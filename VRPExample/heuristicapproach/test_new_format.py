#!/usr/bin/env python3
"""
Quick test script to verify the new driver assignment Excel format works correctly.
"""

import sys
import os
from pathlib import Path

# Add paths for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "utils"))
sys.path.insert(0, str(current_dir / "algo"))

def test_import():
    """Test that imports work correctly."""
    try:
        from scenario_creator import create_scenario_from_excel
        from epdt_data_structures import Driver, Vehicle, Order
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_driver_creation():
    """Test driver creation with mock data."""
    try:
        import pandas as pd
        from scenario_creator import create_driver_from_row
        
        # Create mock driver data
        driver_data = {
            'DRIVER_ID': 'DRV-001',
            'DRIVER_NAME': 'Test Driver',
            'COST_PER_HOUR': 28.50,
            'MAX_SHIFT_HOURS': 13.0,
            'MAX_DRIVING_HOURS': 9.0,
            'CAPABILITIES': 'ADR_CERTIFIED, FORKLIFT_LICENSE'
        }
        
        row = pd.Series(driver_data)
        driver = create_driver_from_row(row)
        
        if driver:
            print(f"✅ Driver created: {driver.id} - {driver.name}")
            print(f"   Cost: €{driver.cost_per_hour}/hour")
            print(f"   Max shift: {driver.max_shift_hours}h, Max driving: {driver.max_driving_hours}h")
            print(f"   Capabilities: {driver.qualifications}")
            return True
        else:
            print("❌ Failed to create driver")
            return False
    except Exception as e:
        print(f"❌ Driver creation error: {e}")
        return False

def test_enhanced_vehicle():
    """Test vehicle creation with new cost fields."""
    try:
        import pandas as pd
        from scenario_creator import create_vehicle_from_row
        
        # Create mock vehicle data with new format
        vehicle_data = {
            'NUMBER PLATE': 'AB123CD',
            'TYPE OF VEHICLE': 'Van',
            'MAX LOAD KG': 3500,
            'PALLET': 8,
            'MAX LOAD VOLUME M^3': 15,
            'COST_PER_KM': 0.55,
            'FIXED_COST': 50,
            'CAPABILITIES': 'LOW_TEMP, LOADER',
            'REGULATIONS': 'YES'
        }
        
        row = pd.Series(vehicle_data)
        vehicle = create_vehicle_from_row(row)
        
        if vehicle:
            print(f"✅ Vehicle created: {vehicle.id}")
            print(f"   Cost: €{vehicle.cost_per_km}/km + €{vehicle.fixed_cost} fixed")
            print(f"   Capacity: {vehicle.weight_capacity}kg, {vehicle.volume_capacity}m³")
            print(f"   Capabilities: {vehicle.capabilities}")
            return True
        else:
            print("❌ Failed to create vehicle")
            return False
    except Exception as e:
        print(f"❌ Vehicle creation error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing new driver assignment Excel format implementation")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_import),
        ("Driver Creation Test", test_driver_creation), 
        ("Enhanced Vehicle Test", test_enhanced_vehicle)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The implementation is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
