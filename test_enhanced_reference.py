#!/usr/bin/env python3
"""
Test script for enhanced reference file handling
Tests the updated scenario_creator with furgoni2.xlsx reference file
"""

import sys
import os

# Add the correct path to the VRPExample module
sys.path.append(os.path.join(os.path.dirname(__file__), 'VRPExample'))

from heuristicapproach.utils.scenario_creator import create_scenario_from_excel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_enhanced_reference_file():
    """Test the enhanced implementation with the reference file"""
    excel_file = 'VRPExample/heuristicapproach/src/furgoni2.xlsx'
    
    if not os.path.exists(excel_file):
        print(f"ERROR: Reference file not found: {excel_file}")
        return False
    
    print(f"Testing enhanced implementation with: {excel_file}")
    print("=" * 50)
    
    try:
        # Load the scenario
        orders, vehicles, drivers = create_scenario_from_excel(excel_file)
        
        print(f"\nSUCCESS: Loaded scenario from {excel_file}")
        print(f"- Orders: {len(orders)}")
        print(f"- Vehicles: {len(vehicles)}")
        print(f"- Drivers: {len(drivers)}")
        
        # Analyze driver-vehicle linkage
        print("\n--- Driver-Vehicle Analysis ---")
        linked_drivers = [d for d in drivers if d.default_vehicle_id]
        unlinked_drivers = [d for d in drivers if not d.default_vehicle_id]
        
        print(f"Linked drivers: {len(linked_drivers)}")
        print(f"Unlinked drivers: {len(unlinked_drivers)}")
        
        # Show first few driver examples
        print("\n--- Sample Drivers ---")
        for i, driver in enumerate(drivers[:5]):
            print(f"Driver {i+1}: {driver.id}")
            print(f"  Name: {driver.name}")
            print(f"  Default Vehicle: {driver.default_vehicle_id}")
            print(f"  License: {driver.license}")
            print(f"  Cost/Hour: {driver.cost_per_hour}")
            print()
        
        # Show first few vehicle examples
        print("--- Sample Vehicles ---")
        for i, vehicle in enumerate(vehicles[:5]):
            print(f"Vehicle {i+1}: {vehicle.id}")
            print(f"  Weight Capacity: {vehicle.weight_capacity}")
            print(f"  Volume Capacity: {vehicle.volume_capacity}")
            print(f"  Cost/km: {vehicle.cost_per_km}")
            print(f"  Fixed Cost: {vehicle.fixed_cost}")
            print(f"  Vehicle Type: {vehicle.vehicle_type}")
            print()
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to load scenario: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_enhanced_reference_file()
