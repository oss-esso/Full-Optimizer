"""
Quick validation script to verify the fixes
"""
import sys
import os

# Test 1: Verify the is_depot() method fix
print("Test 1: Checking is_depot() method fix...")
try:
    # Read the first_level.py file to check if the fix is present
    first_level_path = "../algo/first_level.py"
    if os.path.exists(first_level_path):
        with open(first_level_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if "task.is_depot()" in content:
            print("❌ FAILED: Old is_depot() calls still present")
        elif "task.is_depot_start() or task.is_depot_return()" in content:
            print("✅ PASSED: is_depot() calls properly replaced")
        else:
            print("? UNKNOWN: Could not verify fix")
    else:
        print("❌ FAILED: first_level.py not found")
        
except Exception as e:
    print(f"❌ ERROR: {e}")

# Test 2: Verify the route filtering fix
print("\nTest 2: Checking route filtering fix...")
try:
    # Read the comprehensive_integration_test.py file to check if the fix is present
    test_file_path = "comprehensive_integration_test.py"
    if os.path.exists(test_file_path):
        with open(test_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if "Skipping infeasible route" in content:
            print("❌ FAILED: Old filtering logic still present")
        elif "keeping route with warning" in content:
            print("✅ PASSED: Route filtering logic updated")
        else:
            print("? UNKNOWN: Could not verify fix")
    else:
        print("❌ FAILED: comprehensive_integration_test.py not found")
        
except Exception as e:
    print(f"❌ ERROR: {e}")

# Test 3: Check Unicode character fixes
print("\nTest 3: Checking Unicode character fixes...")
try:
    driver_file_path = "../algo/driver_assignment_enhanced.py"
    if os.path.exists(driver_file_path):
        with open(driver_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        unicode_chars = ['❌', '€', '✓', '•']
        found_unicode = []
        for char in unicode_chars:
            if char in content:
                found_unicode.append(char)
                
        if found_unicode:
            print(f"❌ FAILED: Unicode characters still present: {found_unicode}")
        else:
            print("✅ PASSED: Problematic Unicode characters removed")
    else:
        print("❌ FAILED: driver_assignment_enhanced.py not found")
        
except Exception as e:
    print(f"❌ ERROR: {e}")

print("\n" + "="*50)
print("Validation completed!")
print("Note: Run the full test to verify complete functionality.")
