#!/usr/bin/env python3
"""
Test the implementation with the reference Excel file furgoni2.xlsx.
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add paths for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "utils"))
sys.path.insert(0, str(current_dir / "algo"))

def inspect_excel_structure(excel_path):
    """Inspect the structure of the Excel file to see what sheets and columns it has."""
    print(f"🔍 Inspecting Excel file: {excel_path}")
    
    try:
        # Get all sheet names
        excel_file = pd.ExcelFile(excel_path)
        sheets = excel_file.sheet_names
        print(f"📊 Found {len(sheets)} sheets: {sheets}")
        
        # Inspect each sheet
        for sheet_name in sheets:
            print(f"\n📋 Sheet: {sheet_name}")
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {list(df.columns)}")
            
            # Show first few rows
            if len(df) > 0:
                print(f"   Sample data:")
                for col in df.columns[:5]:  # Show first 5 columns
                    if not df[col].empty:
                        sample_val = df[col].iloc[0] if pd.notna(df[col].iloc[0]) else "N/A"
                        print(f"     {col}: {sample_val}")
                
    except Exception as e:
        print(f"❌ Error inspecting Excel file: {e}")
        return False
        
    return True

def test_with_reference_file():
    """Test scenario creation with the reference Excel file."""
    excel_path = "src/furgoni2.xlsx"
    
    if not Path(excel_path).exists():
        print(f"❌ Reference file not found: {excel_path}")
        return False
    
    print(f"🧪 Testing with reference file: {excel_path}")
    
    # First inspect the structure
    if not inspect_excel_structure(excel_path):
        return False
    
    print(f"\n🔄 Attempting scenario creation...")
    
    try:
        from scenario_creator import create_scenario_from_excel
        
        # Try to create scenario
        result = create_scenario_from_excel(excel_path)
        
        if len(result) == 3:
            orders, vehicles, drivers = result
            print(f"✅ Scenario created successfully with new format!")
            print(f"   📦 Orders: {len(orders)}")
            print(f"   🚚 Vehicles: {len(vehicles)}")
            print(f"   👨‍💼 Drivers: {len(drivers)}")
            
            # Show sample data
            if orders:
                order = orders[0]
                print(f"\n📋 Sample Order: {order.id}")
                all_tasks = order.get_all_tasks()
                print(f"   Tasks: {len(all_tasks)}")
                if all_tasks:
                    task = all_tasks[0]
                    print(f"   Sample Task: {task.id} ({task.task_type.value})")
                    
            if vehicles:
                vehicle = vehicles[0]
                print(f"\n🚚 Sample Vehicle: {vehicle.id}")
                print(f"   Type: {vehicle.vehicle_type}")
                print(f"   Costs: €{vehicle.cost_per_km}/km + €{vehicle.fixed_cost} fixed")
                
            if drivers:
                driver = drivers[0]
                print(f"\n👨‍💼 Sample Driver: {driver.id}")
                print(f"   Name: {driver.name}")
                print(f"   Cost: €{driver.cost_per_hour}/hour")
            
            return True
        else:
            print(f"❌ Unexpected return format. Got {len(result)} items instead of 3")
            return False
            
    except Exception as e:
        print(f"❌ Scenario creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_format_compliance():
    """Check if the reference file follows the proposed format."""
    excel_path = "src/furgoni2.xlsx"
    
    print(f"\n🔍 Checking format compliance for: {excel_path}")
    
    try:
        excel_file = pd.ExcelFile(excel_path)
        sheets = excel_file.sheet_names
        
        # Check for required sheets
        required_sheets = ['CONSEGNE', 'VEICOLI', 'DRIVERS']
        missing_sheets = []
        
        for required_sheet in required_sheets:
            if required_sheet not in sheets:
                missing_sheets.append(required_sheet)
        
        if missing_sheets:
            print(f"⚠️  Missing sheets for new format: {missing_sheets}")
            print(f"   Available sheets: {sheets}")
            print(f"   This appears to be a legacy format file")
            return False
        else:
            print(f"✅ All required sheets found: {required_sheets}")
            
            # Check column compliance for each sheet
            for sheet_name in required_sheets:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                print(f"\n📋 {sheet_name} sheet columns: {list(df.columns)}")
                
                if sheet_name == 'CONSEGNE':
                    required_cols = ['ORDER_ID', 'COMPANY_NAME', 'DELIVERY_OR_PICKUP']
                    for col in required_cols:
                        if col in df.columns:
                            print(f"   ✅ {col}")
                        else:
                            print(f"   ❌ Missing: {col}")
                            
                elif sheet_name == 'VEICOLI':
                    required_cols = ['NUMBER PLATE', 'COST_PER_KM', 'FIXED_COST']
                    for col in required_cols:
                        if col in df.columns:
                            print(f"   ✅ {col}")
                        else:
                            print(f"   ❌ Missing: {col}")
                            
                elif sheet_name == 'DRIVERS':
                    required_cols = ['DRIVER_ID', 'COST_PER_HOUR']
                    for col in required_cols:
                        if col in df.columns:
                            print(f"   ✅ {col}")
                        else:
                            print(f"   ❌ Missing: {col}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error checking format compliance: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Testing Implementation with Reference File")
    print("=" * 60)
    
    # Test 1: Inspect structure
    print("\n1️⃣ Inspecting file structure...")
    
    # Test 2: Check format compliance
    print("\n2️⃣ Checking format compliance...")
    is_new_format = check_format_compliance()
    
    # Test 3: Test scenario creation
    print("\n3️⃣ Testing scenario creation...")
    success = test_with_reference_file()
    
    if success:
        if is_new_format:
            print("\n🎉 SUCCESS: Reference file follows new format and works perfectly!")
        else:
            print("\n✅ SUCCESS: Implementation works with legacy format (backward compatibility)")
        return 0
    else:
        print("\n❌ FAILED: Issues found with reference file processing")
        return 1

if __name__ == "__main__":
    sys.exit(main())
