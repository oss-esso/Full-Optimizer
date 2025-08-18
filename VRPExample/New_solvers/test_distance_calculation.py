#!/usr/bin/env python3
"""
Test to verify if the sequential VRP optimizer uses OSRM routes in objective calculation
or falls back to straight-line distances.

This test will:
1. Create a simple VRP problem with 3-4 locations
2. Run the sequential VRP optimizer 
3. Extract the actual route distances used in optimization
4. Compare with straight-line distances to see which one matches the 11,000km issue
"""

import os
import sys
import importlib.util
import math
import time

def calculate_euclidean_distance(loc1, loc2):
    """Calculate straight-line distance between two locations."""
    x1, y1 = loc1['x'], loc1['y']
    x2, y2 = loc2['x'], loc2['y']
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def test_distance_calculation_methods():
    """Test to see which distance calculation method is being used in the objective."""
    print("🧪 Testing Distance Calculation Methods in Sequential VRP")
    print("=" * 70)
    
    # Import necessary modules
    try:
        # Import from parent directory (VRPExample)
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)
        from vrp_scenarios import create_furgoni_scenario, DEFAULT_TRUCK_SPEED_RATIOS
        
        print("✅ Successfully imported VRP scenarios")
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        return
    
    # Create a small test scenario
    print("\n📦 Creating MODA furgoni scenario for distance analysis...")
    scenario = create_furgoni_scenario()
    
    # Get a small subset of locations for testing (first 5 locations)
    test_locations = []
    location_items = list(scenario.locations.items())[:6]  # Take first 6 locations
    
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
    
    print(f"📊 Test locations ({len(test_locations)}):")
    for i, loc in enumerate(test_locations):
        print(f"  {i}: {loc['id']} at ({loc['x']:.4f}, {loc['y']:.4f})")
    
    # Create a small vehicle fleet (just 1 vehicle for simplicity)
    test_vehicles = [{
        'id': 'test_vehicle',
        'capacity': 1000,
        'depot_id': test_locations[0]['id'],  # First location as depot
        'truck_speed_ratios': DEFAULT_TRUCK_SPEED_RATIOS['standard'],
        'max_time': 8 * 60  # 8 hours
    }]
    
    # Calculate straight-line distances for comparison
    print(f"\n📏 Straight-line distances between all location pairs:")
    euclidean_distances = {}
    total_euclidean = 0
    
    for i in range(len(test_locations)):
        for j in range(len(test_locations)):
            if i != j:
                loc1, loc2 = test_locations[i], test_locations[j]
                distance = calculate_euclidean_distance(loc1, loc2)
                euclidean_distances[(i, j)] = distance
                total_euclidean += distance
                print(f"  {loc1['id']} -> {loc2['id']}: {distance:.2f} km")
    
    print(f"📊 Total sum of all Euclidean distances: {total_euclidean:.2f} km")
    
    # Import and run the sequential multi-day VRP solver
    try:
        # Import vrp_multiday_sequential from parent VRPExample directory  
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location("vrp_multiday", 
                                                    os.path.join(parent_dir, 
                                                                "vrp_multiday_sequential.py"))
        vrp_multiday = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vrp_multiday)
        
        print("\n🚀 Testing Sequential Multi-Day VRP distance calculations...")
        
        # Create sequential VRP solver
        db_path = "test_routes.db"
        sequential_vrp = vrp_multiday.SequentialMultiDayVRP(test_vehicles, test_locations, use_truck_speeds=True, db_path=db_path)
        
        # Get OSRM distance matrix for comparison
        print(f"\n🗺️ OSRM distances from distance calculator:")
        osrm_distances = {}
        total_osrm = 0
        
        if hasattr(sequential_vrp, 'distance_calculator') and hasattr(sequential_vrp.distance_calculator, 'distance_matrix'):
            distance_matrix = sequential_vrp.distance_calculator.distance_matrix
            
            for i in range(len(test_locations)):
                for j in range(len(test_locations)):
                    if i != j and i < len(distance_matrix) and j < len(distance_matrix[i]):
                        distance = distance_matrix[i][j]
                        osrm_distances[(i, j)] = distance
                        total_osrm += distance
                        
                        euclidean_dist = euclidean_distances.get((i, j), 0)
                        ratio = distance / euclidean_dist if euclidean_dist > 0 else 1
                        
                        print(f"  {test_locations[i]['id']} -> {test_locations[j]['id']}:")
                        print(f"    OSRM: {distance:.2f} km, Euclidean: {euclidean_dist:.2f} km (ratio: {ratio:.2f})")
            
            print(f"📊 Total sum of all OSRM distances: {total_osrm:.2f} km")
            print(f"📊 OSRM vs Euclidean ratio: {total_osrm/total_euclidean:.2f}")
        else:
            print("❌ Could not access OSRM distance matrix")
        
        # Now solve a simple VRP problem and check which distances are used in the solution
        print(f"\n🔍 Solving VRP and analyzing solution distances...")
        
        # Solve the problem
        solution = sequential_vrp.solve_sequential_multiday(max_days=2)
        
        if solution:
            print(f"✅ Solution found!")
            
            # Extract solution routes and calculate distances
            solution_distance = 0
            euclidean_solution_distance = 0
            
            print(f"\n📊 Analyzing solution routes:")
            
            for vehicle_id, route_data in solution.get('vehicle_routes', {}).items():
                print(f"\n🚛 Vehicle {vehicle_id}:")
                vehicle_distance = route_data.get('total_distance', 0)
                print(f"  Reported total distance: {vehicle_distance:.2f} km")
                
                # Check if we can find the actual route sequence
                daily_routes = route_data.get('daily_routes', {})
                if daily_routes:
                    for day, day_route in daily_routes.items():
                        stops = day_route.get('stops', [])
                        print(f"  Day {day} stops: {len(stops)}")
                        
                        # Calculate actual distances for this route
                        day_distance_osrm = 0
                        day_distance_euclidean = 0
                        
                        for i in range(len(stops) - 1):
                            stop1 = stops[i]
                            stop2 = stops[i + 1]
                            
                            # Find location indices
                            loc1_idx = None
                            loc2_idx = None
                            
                            for idx, loc in enumerate(test_locations):
                                if str(loc['id']) == str(stop1.get('location_id', stop1.get('id', ''))):
                                    loc1_idx = idx
                                if str(loc['id']) == str(stop2.get('location_id', stop2.get('id', ''))):
                                    loc2_idx = idx
                            
                            if loc1_idx is not None and loc2_idx is not None:
                                # Get OSRM distance
                                osrm_dist = osrm_distances.get((loc1_idx, loc2_idx), 0)
                                euclidean_dist = euclidean_distances.get((loc1_idx, loc2_idx), 0)
                                
                                day_distance_osrm += osrm_dist
                                day_distance_euclidean += euclidean_dist
                                
                                print(f"    {test_locations[loc1_idx]['id']} -> {test_locations[loc2_idx]['id']}: "
                                      f"OSRM {osrm_dist:.2f}km, Euclidean {euclidean_dist:.2f}km")
                        
                        print(f"  Day {day} total - OSRM: {day_distance_osrm:.2f}km, Euclidean: {day_distance_euclidean:.2f}km")
                        solution_distance += day_distance_osrm
                        euclidean_solution_distance += day_distance_euclidean
            
            print(f"\n🎯 FINAL COMPARISON:")
            print(f"  Solution reported distance: {sum(route.get('total_distance', 0) for route in solution.get('vehicle_routes', {}).values()):.2f} km")
            print(f"  Manual OSRM calculation: {solution_distance:.2f} km")
            print(f"  Manual Euclidean calculation: {euclidean_solution_distance:.2f} km")
            
            # Determine which method matches the reported solution
            reported_total = sum(route.get('total_distance', 0) for route in solution.get('vehicle_routes', {}).values())
            
            osrm_diff = abs(reported_total - solution_distance)
            euclidean_diff = abs(reported_total - euclidean_solution_distance)
            
            print(f"\n📊 ANALYSIS:")
            print(f"  Difference from OSRM: {osrm_diff:.2f} km")
            print(f"  Difference from Euclidean: {euclidean_diff:.2f} km")
            
            if osrm_diff < euclidean_diff:
                print(f"  ✅ CONCLUSION: Solution is using OSRM distances")
            else:
                print(f"  ❌ CONCLUSION: Solution is using Euclidean distances (explains 11,000km issue)")
                
        else:
            print("❌ No solution found")
            
    except Exception as e:
        print(f"❌ Error in distance calculation test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_distance_calculation_methods()
