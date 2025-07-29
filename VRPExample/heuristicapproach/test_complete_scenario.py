#!/usr/bin/env python3
"""
Test the complete scenario creation with the new Excel format.
"""

import sys
import os
from pathlib import Path

# Add paths for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "utils"))
sys.path.insert(0, str(current_dir / "algo"))

def test_complete_scenario():
    """Test complete scenario creation with the new format."""
    try:
        from scenario_creator import create_scenario_from_excel
        
        excel_file = "sample_new_format.xlsx"
        if not Path(excel_file).exists():
            print(f"❌ Excel file not found: {excel_file}")
            return False
        
        print(f"📊 Testing scenario creation with {excel_file}")
        
        # Create scenario (this will use mock coordinates since we don't have real geocoding)
        orders, vehicles, drivers = create_scenario_from_excel(excel_file)
        
        print(f"✅ Scenario created successfully!")
        print(f"   📦 Orders: {len(orders)}")
        print(f"   🚚 Vehicles: {len(vehicles)}")  
        print(f"   👨‍💼 Drivers: {len(drivers)}")
        
        # Test order details
        if orders:
            order = orders[0]
            print(f"\n📋 Sample Order: {order.id}")
            print(f"   Tasks: {len(order.get_all_tasks())}")
            for task in order.get_all_tasks():
                print(f"   - {task.id}: {task.task_type.value} at ({task.lat:.4f}, {task.lon:.4f})")
        
        # Test vehicle details
        if vehicles:
            vehicle = vehicles[0]
            print(f"\n🚚 Sample Vehicle: {vehicle.id}")
            print(f"   Type: {vehicle.vehicle_type}")
            print(f"   Capacity: {vehicle.weight_capacity}kg, {vehicle.volume_capacity}m³")
            print(f"   Costs: €{vehicle.cost_per_km}/km + €{vehicle.fixed_cost} fixed")
            print(f"   Capabilities: {vehicle.capabilities}")
        
        # Test driver details
        if drivers:
            driver = drivers[0]
            print(f"\n👨‍💼 Sample Driver: {driver.id}")
            print(f"   Name: {driver.name}")
            print(f"   License: {driver.license}")
            print(f"   Cost: €{driver.cost_per_hour}/hour")
            print(f"   Limits: {driver.max_shift_hours}h shift, {driver.max_driving_hours}h driving")
            print(f"   Capabilities: {driver.qualifications}")
        
        return True
        
    except Exception as e:
        print(f"❌ Scenario creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the complete test."""
    print("🧪 Testing Complete Scenario Creation with New Format")
    print("=" * 60)
    
    if test_complete_scenario():
        print("\n🎉 Complete test passed! The new driver assignment format is working correctly.")
        return 0
    else:
        print("\n❌ Test failed. Check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
