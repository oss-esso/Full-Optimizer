#!/usr/bin/env python3
"""
Test and fix the return-to-depot logic in the sequential VRP optimizer.

This test focuses specifically on the issue where vehicles stay at their overnight 
positions instead of returning to the depot on the final day.
"""

import os
import sys
import time
import json
import importlib.util
from datetime import datetime

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def test_return_to_depot_fix():
    """Test the return-to-depot logic fix."""
    print("🧪 Testing Return-to-Depot Logic Fix")
    print("=" * 60)
    
    try:
        # Import the scenario and VRP solver
        from vrp_scenarios import create_furgoni_scenario
        
        # Import vrp_multiday_sequential
        spec = importlib.util.spec_from_file_location("vrp_multiday", 
                                                    os.path.join(parent_dir, 
                                                                "vrp_multiday_sequential.py"))
        vrp_multiday = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vrp_multiday)
        
        print("📦 Creating test scenario with existing database...")
        scenario = create_furgoni_scenario()
        
        # Convert to format needed for Sequential Multi-Day VRP
        locations = []
        for loc_id, loc in scenario.locations.items():
            x = getattr(loc, 'x', None) or getattr(loc, 'lon', None) or 0
            y = getattr(loc, 'y', None) or getattr(loc, 'lat', None) or 0
            lat = getattr(loc, 'lat', None) or getattr(loc, 'latitude', None) or y
            lon = getattr(loc, 'lon', None) or getattr(loc, 'longitude', None) or x
            
            locations.append({
                'id': str(loc_id),
                'x': x, 'y': y,
                'lat': lat, 'lon': lon,
                'address': getattr(loc, 'address', f'Location {loc_id}'),
                'service_time': getattr(loc, 'service_time', 15)
            })
        
        # Convert vehicles
        from vrp_scenarios import DEFAULT_TRUCK_SPEED_RATIOS
        vehicles = []
        for vehicle_id, vehicle in scenario.vehicles.items():
            vehicle_type = getattr(vehicle, 'type', 'furgone').lower()
            if 'camion' in vehicle_type or 'truck' in vehicle_type or vehicle.capacity > 5000:
                truck_ratios = DEFAULT_TRUCK_SPEED_RATIOS['camion']
            else:
                truck_ratios = DEFAULT_TRUCK_SPEED_RATIOS['furgone']

            vehicles.append({
                'id': str(vehicle_id),
                'capacity': getattr(vehicle, 'capacity', 3000),
                'truck_speed_ratios': truck_ratios,
                'cost_per_km': getattr(vehicle, 'cost_per_km', 1.2)
            })
        
        print(f"📊 Test scenario prepared:")
        print(f"  - Locations: {len(locations)}")
        print(f"  - Vehicles: {len(vehicles)}")
        
        # Initialize the VRP solver with existing database to save time
        print(f"🚛 Initializing Sequential Multi-Day VRP...")
        start_time = time.time()
        
        # Use existing database if it exists
        db_path = "moda_routes_fixed.db"
        if os.path.exists(db_path):
            print(f"  📂 Using existing OSRM database: {db_path}")
        else:
            print(f"  📂 Will create new database: {db_path}")
        
        # Initialize with fewer vehicles to speed up testing
        test_vehicles = vehicles[:6]  # Use only 6 vehicles for faster testing
        
        vrp_solver = vrp_multiday.SequentialMultiDayVRP(
            vehicles=test_vehicles,
            locations=locations,
            use_truck_speeds=True,
            db_path=db_path
        )
        
        print(f"  ⏱️ VRP solver initialized in {time.time() - start_time:.2f}s")
        
        # Test the issue: Run with max_days=10 to see if vehicles return
        print(f"\n🔧 Testing with max_days=10 to check return-to-depot logic...")
        start_time = time.time()
        
        solution = vrp_solver.solve_sequential_multiday(max_days=10)
        
        solve_time = time.time() - start_time
        print(f"  ⏱️ Solution found in {solve_time:.1f}s")
        
        if not solution:
            print("❌ No solution found!")
            return False
        
        # Analyze the final day vehicle status
        print(f"\n📊 Analyzing final day vehicle status...")
        
        final_day = max(vrp_solver.daily_solutions.keys())
        final_solution = vrp_solver.daily_solutions[final_day]
        
        print(f"📅 Final day: {final_day}")
        print(f"📍 Vehicle locations on final day:")
        
        vehicles_at_depot = 0
        vehicles_still_out = 0
        
        for vehicle_id, vehicle_state in vrp_solver.vehicle_states.items():
            is_active = vehicle_state.get('is_active', False)
            current_location = vehicle_state.get('current_location_idx', 0)
            overnight_position = vehicle_state.get('overnight_position', None)
            
            if current_location == 0 and not is_active:
                print(f"  ✅ {vehicle_id}: At depot (inactive)")
                vehicles_at_depot += 1
            elif is_active:
                print(f"  ⚠️ {vehicle_id}: Still active at location {current_location}")
                vehicles_still_out += 1
            else:
                print(f"  ❓ {vehicle_id}: Inactive at location {current_location}")
                if current_location != 0:
                    vehicles_still_out += 1
                else:
                    vehicles_at_depot += 1
        
        print(f"\n📊 Summary:")
        print(f"  - Vehicles at depot: {vehicles_at_depot}")
        print(f"  - Vehicles still out: {vehicles_still_out}")
        print(f"  - Total days used: {final_day}")
        
        # Check if the fix is needed
        if vehicles_still_out > 0:
            print(f"\n🔧 ISSUE DETECTED: {vehicles_still_out} vehicles are still out!")
            print(f"   This confirms the return-to-depot logic needs fixing.")
            
            # Analyze what went wrong
            print(f"\n🔍 Analyzing the issue...")
            
            # Check if the final day had customers or was return-only
            final_visited = final_solution.get('visited_customers', set())
            if not final_visited:
                print(f"   - Final day had no customer visits (return-only day)")
                print(f"   - Issue: Vehicles set to inactive but didn't actually travel to depot")
            else:
                print(f"   - Final day visited {len(final_visited)} customers")
                print(f"   - Issue: Some vehicles couldn't make it back to depot in time")
            
            # Show specific vehicle problems
            print(f"\n🚛 Vehicles still out:")
            for vehicle_id, vehicle_state in vrp_solver.vehicle_states.items():
                current_location = vehicle_state.get('current_location_idx', 0)
                is_active = vehicle_state.get('is_active', False)
                
                if current_location != 0 or is_active:
                    if current_location < len(locations):
                        location_info = locations[current_location]
                        print(f"  ⚠️ {vehicle_id}: At {location_info['id']} ({location_info['x']:.2f}, {location_info['y']:.2f})")
                    else:
                        print(f"  ⚠️ {vehicle_id}: At unknown location index {current_location}")
            
            return False
        else:
            print(f"\n✅ SUCCESS: All vehicles returned to depot!")
            print(f"   The return-to-depot logic is working correctly.")
            return True
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def propose_fix():
    """Propose the fix for the return-to-depot logic."""
    print(f"\n🔧 PROPOSED FIX:")
    print(f"=" * 60)
    print(f"The issue is in the solve_single_day method when force_return_to_depot=True")
    print(f"and there are no customers left to visit.")
    print(f"")
    print(f"Current behavior:")
    print(f"  1. Vehicles at overnight positions have 'no work to do'")
    print(f"  2. They stay at their current position instead of traveling to depot")
    print(f"  3. The sequential solver marks them as 'returned to depot' incorrectly")
    print(f"")
    print(f"Required fix:")
    print(f"  1. When force_return_to_depot=True and no customers remain")
    print(f"  2. Vehicles at overnight positions should travel back to depot")
    print(f"  3. Create return routes even if 'no deliveries' to make")
    print(f"  4. Handle time limits appropriately for final day")
    print(f"")
    print(f"Fix location: vrp_multiday_sequential.py, solve_single_day method")
    print(f"Around line 275-295 where it checks 'has_deliveries'")

if __name__ == "__main__":
    print("🧪 Return-to-Depot Logic Test")
    print("=" * 60)
    
    success = test_return_to_depot_fix()
    
    if not success:
        propose_fix()
        print(f"\n💡 Next step: Apply the fix to vrp_multiday_sequential.py")
    else:
        print(f"\n🎉 No fix needed - logic is already working!")
