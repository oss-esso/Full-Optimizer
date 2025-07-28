#!/usr/bin/env python3
"""
Simple test to verify the Test Summary functionality works correctly.
"""
import os
import sys

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
sys.path.insert(0, heuristic_root)

# Global counter for tracking Haversine distance calculations
haversine_call_count = 0

def reset_haversine_counter():
    """Reset the global Haversine call counter."""
    global haversine_call_count
    haversine_call_count = 0

def get_haversine_call_count():
    """Get the current Haversine call count."""
    global haversine_call_count
    return haversine_call_count

def test_summary_display():
    """Test the Test Summary display functionality."""
    print("Testing Test Summary display...")
    
    # Mock data
    total_orders = 42
    total_vehicles = 55
    runtime_seconds = 0.11
    
    # Set a test counter value
    global haversine_call_count
    haversine_call_count = 150
    
    # Test Summary with Haversine call count for OSRM estimation - ALWAYS DISPLAY
    print("DEBUG: About to display Test Summary...")
    try:
        print(f"\n📊 Test Summary:")
        print(f"   • Scenario source: furgoni.xlsx")
        print(f"   • Orders processed: {total_orders}")
        print(f"   • Vehicles available: {total_vehicles}")
        if runtime_seconds is not None:
            print(f"   • Total runtime: {runtime_seconds:.2f} seconds")
        print(f"   • Haversine distance calls: {get_haversine_call_count()}")
        print(f"   • Estimated OSRM API calls: ~{get_haversine_call_count()} (when switching to production)")
        print("DEBUG: Test Summary displayed successfully!")
    except Exception as e:
        print(f"\n📊 Test Summary: Error displaying summary: {e}")
        # Fallback summary even if there are errors
        print(f"   • Basic info: {total_orders} orders, {total_vehicles} vehicles")
        if runtime_seconds is not None:
            print(f"   • Runtime: {runtime_seconds:.2f} seconds")
    
    print("\nTest Summary display test completed!")

if __name__ == "__main__":
    test_summary_display()
