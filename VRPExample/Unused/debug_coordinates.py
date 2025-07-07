#!/usr/bin/env python3
"""
Quick test to examine location coordinates and understand the distance calculation issue.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from VRPExample.vrp_scenarios import get_all_scenarios

def examine_locations():
    """Examine location coordinates to understand distance calculations."""
    print("🔍 EXAMINING LOCATION COORDINATES")
    print("=" * 50)
    
    # Get the scenario
    scenarios = get_all_scenarios()
    scenario = scenarios['MODA_small']
    
    print(f"Scenario: {scenario.name}")
    print(f"Total locations: {len(scenario.locations)}")
    print()
    
    # Print first few locations to understand coordinates
    print("📍 LOCATION COORDINATES:")
    for i, (loc_id, location) in enumerate(scenario.locations.items()):
        if i < 10:  # Just first 10 locations
            print(f"   {loc_id}: ({location.x:.2f}, {location.y:.2f})")
        elif i == 10:
            print("   ... (more locations)")
            break
    
    print()
    
    # Calculate some sample distances
    locations = list(scenario.locations.values())
    print("📏 SAMPLE DISTANCE CALCULATIONS:")
    
    depot_1 = locations[0]  # Should be depot_1
    pickup_1 = None
    dropoff_1 = None
    
    for loc in locations:
        if loc.id == 'pickup_1':
            pickup_1 = loc
        elif loc.id == 'dropoff_1':
            dropoff_1 = loc
    
    if depot_1 and pickup_1:
        dx = abs(depot_1.x - pickup_1.x)
        dy = abs(depot_1.y - pickup_1.y)
        manhattan_dist = dx + dy
        print(f"   {depot_1.id} to {pickup_1.id}: dx={dx:.2f}, dy={dy:.2f}, Manhattan={manhattan_dist:.2f}")
        print(f"   OR-Tools scaled distance: {int(manhattan_dist * 100)}")
        print(f"   OR-Tools time (distance/100): {int(manhattan_dist * 100) // 100}")
    
    if pickup_1 and dropoff_1:
        dx = abs(pickup_1.x - dropoff_1.x)
        dy = abs(pickup_1.y - dropoff_1.y)
        manhattan_dist = dx + dy
        print(f"   {pickup_1.id} to {dropoff_1.id}: dx={dx:.2f}, dy={dy:.2f}, Manhattan={manhattan_dist:.2f}")
        print(f"   OR-Tools scaled distance: {int(manhattan_dist * 100)}")
        print(f"   OR-Tools time (distance/100): {int(manhattan_dist * 100) // 100}")
    
    print()
    print("🔍 ANALYSIS:")
    print("   If Manhattan distances are < 0.01, then:")
    print("   - Scaled distance (x100) < 1")
    print("   - Time calculation (distance/100) = 0")
    print("   - This explains why final_time = 0!")
    
    print()
    print("💡 SOLUTION:")
    print("   - Increase scaling factor in distance calculation")
    print("   - Or ensure minimum travel time between locations")
    print("   - Or add more realistic coordinate spacing")

if __name__ == "__main__":
    examine_locations()
