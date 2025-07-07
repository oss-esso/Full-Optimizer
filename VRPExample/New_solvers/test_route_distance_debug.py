#!/usr/bin/env python3
"""
Debug test to check exact route distance calculation in sequential VRP solution.
"""

import os
import sys
import importlib.util
import json

def debug_route_distances():
    """Debug exactly how route distances are being calculated and stored."""
    print("🔍 Debugging Route Distance Calculation")
    print("=" * 60)
    
    # Import necessary modules
    try:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)
        from vrp_scenarios import create_furgoni_scenario, DEFAULT_TRUCK_SPEED_RATIOS
        print("✅ Successfully imported VRP scenarios")
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        return
    
    # Create test scenario
    scenario = create_furgoni_scenario()
    
    # Take just 4 locations for simplicity
    test_locations = []
    location_items = list(scenario.locations.items())[:4]  
    
    for i, (loc_id, loc) in enumerate(location_items):
        x = getattr(loc, 'x', None) or getattr(loc, 'lon', None) or 0
        y = getattr(loc, 'y', None) or getattr(loc, 'lat', None) or 0
        lat = getattr(loc, 'lat', None) or y
        lon = getattr(loc, 'lon', None) or x
        
        test_locations.append({
            'id': str(loc_id),
            'x': x, 'y': y,
            'lat': lat, 'lon': lon,
            'address': getattr(loc, 'address', f'Location {loc_id}'),
            'service_time': 15
        })
    
    print(f"\n📊 Test locations ({len(test_locations)}):")
    for i, loc in enumerate(test_locations):
        print(f"  {i}: {loc['id']} at ({loc['x']:.4f}, {loc['y']:.4f})")
    
    # Create test vehicle
    test_vehicles = [{
        'id': 'debug_vehicle',
        'capacity': 1000,
        'depot_id': test_locations[0]['id'],
        'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['standard'],
        'max_time': 8 * 60
    }]
    
    # Import and run sequential VRP
    try:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location("vrp_multiday", 
                                                    os.path.join(parent_dir, 
                                                                "vrp_multiday_sequential.py"))
        vrp_multiday = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vrp_multiday)
        
        print("\n🚀 Running Sequential Multi-Day VRP...")
        
        # Create sequential VRP solver
        db_path = "debug_routes.db"
        sequential_vrp = vrp_multiday.SequentialMultiDayVRP(test_vehicles, test_locations, use_truck_speeds=True, db_path=db_path)
        
        # Solve and get detailed solution
        solution = sequential_vrp.solve_sequential_multiday(max_days=2)
        
        if solution:
            print(f"\n📊 DETAILED SOLUTION ANALYSIS:")
            print(f"Solution keys: {list(solution.keys())}")
            
            # Print full solution structure for debugging
            print(f"\n🔍 Raw solution structure:")
            print(json.dumps(solution, indent=2, default=str))
            
            # Analyze daily solutions
            print(f"\n📅 Daily Solutions Analysis:")
            for day_num, day_solution in solution.get('daily_solutions', {}).items():
                print(f"\nDay {day_num}:")
                print(f"  Routes: {list(day_solution.get('routes', {}).keys())}")
                
                for vehicle_id, route_data in day_solution.get('routes', {}).items():
                    print(f"\n  🚛 Vehicle {vehicle_id}:")
                    print(f"    Distance: {route_data.get('distance', 'N/A')} km")
                    print(f"    Total demand: {route_data.get('total_demand', 'N/A')}")
                    print(f"    Ends with overnight: {route_data.get('ends_with_overnight', 'N/A')}")
                    print(f"    Stops: {len(route_data.get('stops', []))}")
                    
                    for i, stop in enumerate(route_data.get('stops', [])):
                        print(f"      Stop {i}: {stop.get('location_id', 'N/A')} at {stop.get('coordinates', 'N/A')}")
                        print(f"               Overnight: {stop.get('is_overnight', False)}")
            
            # Analyze vehicle routes in final solution
            print(f"\n🚛 Vehicle Routes Analysis:")
            for vehicle_id, vehicle_data in solution.get('vehicle_routes', {}).items():
                print(f"\nVehicle {vehicle_id}:")
                print(f"  Total distance: {vehicle_data.get('total_distance', 'N/A')} km")
                print(f"  Total overnight stays: {vehicle_data.get('total_overnight_stays', 'N/A')}")
                print(f"  Days active: {vehicle_data.get('days_active', 'N/A')}")
                
                full_route = vehicle_data.get('full_route', [])
                print(f"  Full route stops: {len(full_route)}")
                
                for i, stop in enumerate(full_route):
                    print(f"    Stop {i}: {stop.get('location_id', 'N/A')} at {stop.get('coordinates', 'N/A')}")
                    print(f"             Day marker: {stop.get('is_day_marker', False)}")
                    print(f"             Overnight: {stop.get('is_overnight', False)}")
            
            # Now manually verify distance calculation using OSRM matrix
            print(f"\n🧮 Manual Distance Verification:")
            distance_matrix = sequential_vrp.distance_calculator.distance_matrix
            
            print(f"Distance matrix size: {len(distance_matrix)}x{len(distance_matrix[0]) if distance_matrix else 0}")
            
            # Show OSRM distance matrix
            print(f"\n📊 OSRM Distance Matrix:")
            for i in range(len(test_locations)):
                for j in range(len(test_locations)):
                    if i != j:
                        dist = distance_matrix[i][j] if i < len(distance_matrix) and j < len(distance_matrix[i]) else 0
                        print(f"  {test_locations[i]['id']} → {test_locations[j]['id']}: {dist:.2f} km")
            
            # Trace through each daily route and calculate distance manually
            manual_total = 0
            for day_num, day_solution in solution.get('daily_solutions', {}).items():
                print(f"\n🔍 Manual calculation for Day {day_num}:")
                
                for vehicle_id, route_data in day_solution.get('routes', {}).items():
                    day_manual_distance = 0
                    stops = route_data.get('stops', [])
                    
                    print(f"  Vehicle {vehicle_id} stops:")
                    for i, stop in enumerate(stops):
                        print(f"    {i}: {stop.get('location_id', 'N/A')} at {stop.get('coordinates', 'N/A')}")
                    
                    # Calculate distances between consecutive stops
                    for i in range(len(stops) - 1):
                        stop1 = stops[i]
                        stop2 = stops[i + 1]
                        
                        # Find the location indices
                        loc1_idx = None
                        loc2_idx = None
                        
                        for idx, loc in enumerate(test_locations):
                            if str(loc['id']) == str(stop1.get('location_id', '')):
                                loc1_idx = idx
                            if str(loc['id']) == str(stop2.get('location_id', '')):
                                loc2_idx = idx
                        
                        if loc1_idx is not None and loc2_idx is not None:
                            segment_distance = distance_matrix[loc1_idx][loc2_idx]
                            day_manual_distance += segment_distance
                            print(f"    {test_locations[loc1_idx]['id']} → {test_locations[loc2_idx]['id']}: {segment_distance:.2f} km")
                        else:
                            print(f"    ❌ Could not find indices for {stop1.get('location_id', 'N/A')} → {stop2.get('location_id', 'N/A')}")
                    
                    reported_distance = route_data.get('distance', 0)
                    print(f"  Day {day_num} {vehicle_id}: Manual={day_manual_distance:.2f} km, Reported={reported_distance:.2f} km")
                    
                    manual_total += day_manual_distance
            
            reported_total = sum(route.get('total_distance', 0) for route in solution.get('vehicle_routes', {}).values())
            
            print(f"\n🎯 FINAL VERIFICATION:")
            print(f"  Manual total: {manual_total:.2f} km")
            print(f"  Reported total: {reported_total:.2f} km")
            print(f"  Difference: {abs(manual_total - reported_total):.2f} km")
            
            if abs(manual_total - reported_total) < 0.1:
                print(f"  ✅ DISTANCES MATCH - The optimizer IS using OSRM distances correctly!")
            else:
                print(f"  ❌ DISTANCES DON'T MATCH - There may be an issue with distance calculation")
                
        else:
            print("❌ No solution found")
            
    except Exception as e:
        print(f"❌ Error in debug test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_route_distances()
