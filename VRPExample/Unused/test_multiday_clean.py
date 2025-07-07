from VRPExample.vrp_optimizer_multiday import CleanVRPOptimizer


def test_moda_small_scenario():
    """Test the clean optimizer with MODA_small VRPPD scenario, twice with different vehicle orders."""
    print("🧪 Testing Clean VRP Optimizer with MODA_small scenario (order sensitivity test)")
    print("=" * 60)
    try:
        from vrp_scenarios import create_furgoni_scenario
    except ImportError:
        print("❌ Could not import vrp_scenarios. Make sure the file is available.")
        return
    scenario = create_furgoni_scenario()
    print(f"📊 Scenario details:")
    print(f"  - Locations: {len(scenario.locations)}")
    print(f"  - Vehicles: {len(scenario.vehicles)}")
    print(f"  - Ride requests: {len(scenario.ride_requests)}")

    # --- Get vehicle list from scenario ---
    vehicle_ids = list(scenario.vehicles.keys())
    vehicles_from_scenario = [scenario.vehicles[vid] for vid in vehicle_ids]
    # Convert to dicts for CleanVRPOptimizer
    vehicles_dicts = [{
        'id': v.id,
        'capacity': v.capacity,
        'start_location': v.depot_id,
        'end_location': v.depot_id,
        'max_time': getattr(v, 'max_time', 24 * 60)
    } for v in vehicles_from_scenario]

    # --- First run: original order ---
    print("\n================= RUN 1: Original vehicle order ================")
    print("RUN 1 vehicle order:", [v['id'] for v in vehicles_dicts])
    print("RUN 1 vehicle capacities:", [v['capacity'] for v in vehicles_dicts])
    optimizer1 = CleanVRPOptimizer(vehicles=vehicles_dicts, locations=None, vrp_instance=scenario)
    optimizer1.ride_requests = scenario.ride_requests
    result1, status1, applied_constraints1 = optimizer1.solve(constraint_level="full", verbose=False)
    print(f"\n=== RUN 1 RESULT ===")
    if result1:
        print(f"✅ SUCCESS - Status: {status1}")
        print(f"   Constraints applied: {applied_constraints1}")
        print(f"   Objective value: {result1['objective_value']}")
        print(f"   Total distance: {result1['total_distance']:.1f} km")
        for vehicle_id, route_data in result1['routes'].items():
            route = route_data['route']
            print(f"   Vehicle {vehicle_id}: {len(route)} stops, distance: {route_data['distance']:.1f} km")
        # Plot the solution
        optimizer1.plot_solution(result1, title="Furgoni VRP Solution - Single Day")
    else:
        print(f"❌ FAILED - Status: {status1}")
        print(f"   Constraints applied: {applied_constraints1}")

    # --- Test different max_days values on the MODA scenario ---
    max_days_to_test = [1, 2, 3]
    results = {}
    
    for max_days in max_days_to_test:
        print(f"\n================= TESTING max_days={max_days} ================")
        print(f"Testing MODA furgoni scenario with max_days={max_days}")
        print("Vehicle order:", [v['id'] for v in vehicles_dicts])
        
        optimizer = CleanVRPOptimizer(vehicles=vehicles_dicts, locations=None, vrp_instance=scenario)
        optimizer.ride_requests = scenario.ride_requests
        result, status, applied_constraints = optimizer.solve(
            constraint_level="full", 
            verbose=False, 
            max_days=max_days
        )
        
        results[max_days] = {
            'result': result,
            'status': status,
            'constraints': applied_constraints
        }
        
        print(f"\n=== max_days={max_days} RESULT ===")
        if result:
            print(f"✅ SUCCESS - Status: {status}")
            print(f"   Constraints applied: {applied_constraints}")
            print(f"   Objective value: {result['objective_value']:,}")
            print(f"   Total distance: {result['total_distance']:.1f} km")
            
            # Count day transitions
            day_transitions = 0
            active_vehicles = 0
            for vehicle_id, route_data in result['routes'].items():
                route = route_data['route']
                if len(route) > 1:  # Only count vehicles with actual routes
                    active_vehicles += 1
                    for stop in route:
                        if 'night' in stop['location_id'] or 'morning' in stop['location_id']:
                            day_transitions += 1
                    print(f"   Vehicle {vehicle_id}: {len(route)} stops, distance: {route_data['distance']:.1f} km")
            
            print(f"   📊 Active vehicles: {active_vehicles}")
            if day_transitions > 0:
                print(f"   🌙 Day transitions: {day_transitions} (multi-day scheduling used)")
            else:
                print(f"   📅 No day transitions (single day was sufficient)")
                
        else:
            print(f"❌ FAILED - Status: {status}")
            print(f"   Constraints applied: {applied_constraints}")
    
    # --- Compare all results ---
    print(f"\n📊 COMPARISON: Different max_days values")
    print("="*60)
    successful_results = [(days, data) for days, data in results.items() if data['result'] is not None]
    
    if successful_results:
        print("Successful solutions:")
        best_objective = float('inf')
        best_days = None
        
        for max_days, data in successful_results:
            result = data['result']
            objective = result['objective_value']
            distance = result['total_distance']
            
            # Count day transitions
            day_transitions = 0
            for vehicle_id, route_data in result['routes'].items():
                for stop in route_data['route']:
                    if 'night' in stop['location_id'] or 'morning' in stop['location_id']:
                        day_transitions += 1
            
            print(f"   max_days={max_days}: Objective={objective:,}, Distance={distance:.1f}km, Transitions={day_transitions}")
            
            if objective < best_objective:
                best_objective = objective
                best_days = max_days
        
        if best_days:
            print(f"\n🏆 Best solution: max_days={best_days} with objective {best_objective:,}")
            
            # Plot the best solution
            best_result = results[best_days]['result']
            optimizer_best = CleanVRPOptimizer(vehicles=vehicles_dicts, locations=None, vrp_instance=scenario)
            optimizer_best.ride_requests = scenario.ride_requests
            optimizer_best.plot_solution(best_result, title=f"Furgoni VRP Solution - Best (max_days={best_days})")
    else:
        print("❌ No successful solutions found for any max_days value")



def test_moda_inverted_scenario():
    """Test the clean optimizer with MODA_small VRPPD scenario, twice with different vehicle orders."""
    print("🧪 Testing Clean VRP Optimizer with MODA_small scenario (order sensitivity test)")
    print("=" * 60)
    try:
        from vrp_scenarios import create_moda_small_scenario
    except ImportError:
        print("❌ Could not import vrp_scenarios. Make sure the file is available.")
        return
    scenario = create_moda_small_scenario()
    print(f"📊 Scenario details:")
    print(f"  - Locations: {len(scenario.locations)}")
    print(f"  - Vehicles: {len(scenario.vehicles)}")
    print(f"  - Ride requests: {len(scenario.ride_requests)}")

    # --- Get vehicle list from scenario ---
    vehicle_ids = list(scenario.vehicles.keys())
    vehicles_from_scenario = [scenario.vehicles[vid] for vid in vehicle_ids]
    # Convert to dicts for CleanVRPOptimizer
    vehicles_dicts = [{
        'id': v.id,
        'capacity': v.capacity,
        'start_location': v.depot_id,
        'end_location': v.depot_id,
        'max_time': getattr(v, 'max_time', 24 * 60)
    } for v in vehicles_from_scenario]

    # --- First run: original order ---
    print("\n================= RUN 1: Original vehicle order ================")
    print("RUN 1 vehicle order:", [v['id'] for v in vehicles_dicts])
    print("RUN 1 vehicle capacities:", [v['capacity'] for v in vehicles_dicts])
    optimizer1 = CleanVRPOptimizer(vehicles=vehicles_dicts, locations=None, vrp_instance=scenario)
    optimizer1.ride_requests = scenario.ride_requests
    result1, status1, applied_constraints1 = optimizer1.solve(constraint_level="full", verbose=False)
    print(f"\n=== RUN 1 RESULT ===")
    if result1:
        print(f"✅ SUCCESS - Status: {status1}")
        print(f"   Constraints applied: {applied_constraints1}")
        print(f"   Objective value: {result1['objective_value']}")
        print(f"   Total distance: {result1['total_distance']:.1f} km")
        for vehicle_id, route_data in result1['routes'].items():
            route = route_data['route']
            print(f"   Vehicle {vehicle_id}: {len(route)} stops, distance: {route_data['distance']:.1f} km")
    else:
        print(f"❌ FAILED - Status: {status1}")
        print(f"   Constraints applied: {applied_constraints1}")

    # --- Second run: reversed order ---
    vehicles_dicts_reversed = list(reversed(vehicles_dicts))
    print("\n================= RUN 2: Reversed vehicle order ================")
    print("RUN 2 vehicle order:", [v['id'] for v in vehicles_dicts_reversed])
    print("RUN 2 vehicle capacities:", [v['capacity'] for v in vehicles_dicts_reversed])
    optimizer2 = CleanVRPOptimizer(vehicles=vehicles_dicts_reversed, locations=None, vrp_instance=scenario)
    optimizer2.ride_requests = scenario.ride_requests
    result2, status2, applied_constraints2 = optimizer2.solve(constraint_level="full", verbose=False)
    print(f"\n=== RUN 2 RESULT ===")
    if result2:
        print(f"✅ SUCCESS - Status: {status2}")
        print(f"   Constraints applied: {applied_constraints2}")
        print(f"   Objective value: {result2['objective_value']}")
        print(f"   Total distance: {result2['total_distance']:.1f} km")
        for vehicle_id, route_data in result2['routes'].items():
            route = route_data['route']
            print(f"   Vehicle {vehicle_id}: {len(route)} stops, distance: {route_data['distance']:.1f} km")
    else:
        print(f"❌ FAILED - Status: {status2}")
        print(f"   Constraints applied: {applied_constraints2}")



def test_constraint_levels():
    """Test the clean optimizer with MODA_small VRPPD scenario, twice with different vehicle orders."""
    print("🧪 Testing Clean VRP Optimizer with MODA_small scenario (order sensitivity test)")
    print("=" * 60)
    try:
        from vrp_scenarios import create_furgoni_scenario
    except ImportError:
        print("❌ Could not import vrp_scenarios. Make sure the file is available.")
        return
    scenario = create_furgoni_scenario()
    print(f"📊 Scenario details:")
    print(f"  - Locations: {len(scenario.locations)}")
    print(f"  - Vehicles: {len(scenario.vehicles)}")
    print(f"  - Ride requests: {len(scenario.ride_requests)}")

    # --- Get vehicle list from scenario ---
    vehicle_ids = list(scenario.vehicles.keys())
    vehicles_from_scenario = [scenario.vehicles[vid] for vid in vehicle_ids]
    # Convert to dicts for CleanVRPOptimizer
    vehicles_dicts = [{
        'id': v.id,
        'capacity': v.capacity,
        'start_location': v.depot_id,
        'end_location': v.depot_id,
        'max_time': getattr(v, 'max_time', 24 * 60)
    } for v in vehicles_from_scenario]

    levels = ["none",  "pickup_delivery", "time_windows", "capacity", "full"]
    for i, level in enumerate(levels):

        # --- First run: original order ---
        print(f"\n================= RUN {i+1}: Level: {level} ================")
        print("RUN 1 vehicle order:", [v['id'] for v in vehicles_dicts])
        print("RUN 1 vehicle capacities:", [v['capacity'] for v in vehicles_dicts])
        optimizer1 = CleanVRPOptimizer(vehicles=vehicles_dicts, locations=None, vrp_instance=scenario)
        optimizer1.ride_requests = scenario.ride_requests
        result1, status1, applied_constraints1 = optimizer1.solve(constraint_level=level, verbose=False)
        print(f"\n=== RUN {i+1} RESULT ===")
        if result1:
            print(f"✅ SUCCESS - Status: {status1}")
            print(f"   Constraints applied: {applied_constraints1}")
            print(f"   Objective value: {result1['objective_value']}")
            print(f"   Total distance: {result1['total_distance']:.1f} km")
            for vehicle_id, route_data in result1['routes'].items():
                route = route_data['route']
                print(f"   Vehicle {vehicle_id}: {len(route)} stops, distance: {route_data['distance']:.1f} km")
        else:
            print(f"❌ FAILED - Status: {status1}")
            print(f"   Constraints applied: {applied_constraints1}")

def test_moda_first_scenario():
    """Test the clean optimizer with MODA_first VRPPD scenario."""
    print("🧪 Testing Clean VRP Optimizer with MODA_first scenario")
    print("=" * 60)
    try:
        from vrp_scenarios import create_moda_first_scenario
    except ImportError:
        print("❌ Could not import vrp_scenarios. Make sure the file is available.")
        return
    scenario = create_moda_first_scenario()
    print(f"📊 Scenario details:")
    print(f"  - Locations: {len(scenario.locations)}")
    print(f"  - Vehicles: {len(scenario.vehicles)}")
    print(f"  - Ride requests: {len(scenario.ride_requests)}")

    # --- Get vehicle list from scenario ---
    vehicle_ids = list(scenario.vehicles.keys())
    vehicles_from_scenario = [scenario.vehicles[vid] for vid in vehicle_ids]
    # Convert to dicts for CleanVRPOptimizer
    vehicles_dicts = [{
        'id': v.id,
        'capacity': v.capacity,
        'start_location': v.depot_id,
        'end_location': v.depot_id,
        'max_time': getattr(v, 'max_time', 24 * 60)
    } for v in vehicles_from_scenario]

    print("\n================= RUN: MODA_first scenario ================")
    print("Vehicle order:", [v['id'] for v in vehicles_dicts])
    print("Vehicle capacities:", [v['capacity'] for v in vehicles_dicts])
    optimizer = CleanVRPOptimizer(vehicles=vehicles_dicts, locations=None, vrp_instance=scenario)
    optimizer.ride_requests = scenario.ride_requests
    result, status, applied_constraints = optimizer.solve(constraint_level="full", verbose=False)
    print(f"\n=== MODA_first RESULT ===")
    if result:
        print(f"✅ SUCCESS - Status: {status}")
        print(f"   Constraints applied: {applied_constraints}")
        print(f"   Objective value: {result['objective_value']}")
        print(f"   Total distance: {result['total_distance']:.1f} km")
        for vehicle_id, route_data in result['routes'].items():
            route = route_data['route']
            print(f"   Vehicle {vehicle_id}: {route_data}")
    else:
        print(f"❌ FAILED - Status: {status}")
        print(f"   Constraints applied: {applied_constraints}")
    """Test the clean optimizer with different constraint levels."""
    import sys
    import os
    
    print("🧪 Testing Clean VRP Optimizer with different constraint levels")
    print("=" * 60)
    
    # Create a simple test scenario
    # 1. Define the Vehicle List
    vehicle_list = [
        {"id": 0, "capacity": 15, "start_location": "A", "end_location": "A"},
        {"id": 1, "capacity": 15, "start_location": "A", "end_location": "A"},
    ]

    # 2. Define the Location List
    location_list = [
        {"id": "A", "demand": 0, "time_window": (0, 0)},  # Depot
        {"id": "B", "demand": -1, "time_window": (7, 12), "pickup": "C"},
        {"id": "C", "demand": 1, "time_window": (7, 12), "delivery": "B"},
        {"id": "D", "demand": 2, "time_window": (8, 15)},
        {"id": "E", "demand": 1, "time_window": (9, 14)},
    ]
    
    optimizer = CleanVRPOptimizer(
        vehicles=vehicle_list,
        locations=location_list,
        distance_matrix_provider="google"
    )
    
    # Test each constraint level
    levels = ["none", "capacity", "pickup_delivery", "time_windows", "full"]
    
    for level in levels:
        print(f"\n{'='*20} TESTING LEVEL: {level.upper()} {'='*20}")
        
        try:
            solution = optimizer.solve(constraint_level=level)
            
            if solution:
                print(f"✅ SUCCESS - {level} constraints work!")
                print(f"   Objective value: {solution['objective_value']}")
                print(f"   Total distance: {solution['total_distance']}")
                
                # Show first few routes
                for vehicle_id, route_data in list(solution['routes'].items())[:2]:
                    route = route_data['route']
                    print(f"   {vehicle_id}: {len(route)} stops")
                    for stop in route[:3]:
                        print(f"     - {stop['location_id']} (load: {stop['load']})")
                    if len(route) > 3:
                        print(f"     - ... and {len(route)-3} more stops")
            else:
                print(f"❌ FAILED - {level} constraints cause infeasibility!")
                print("   🛑 STOPPING HERE to debug")
                break
                
        except Exception as e:
            print(f"💥 ERROR at {level} level: {str(e)}")
            print("   🛑 STOPPING HERE to debug")
            break


def test_multi_day_scenario():
    """Test multi-day scheduling capability."""
    print("\n" + "="*80)
    print("🧪 TESTING MULTI-DAY VRP SCHEDULING")
    print("="*80)
    
    # Create a challenging test scenario that REQUIRES multi-day scheduling
    locations = [
        {'id': 'depot', 'x': 0, 'y': 0, 'demand': 0, 'service_time': 0, 'time_window': (0, 1440), 'address': 'Main Depot'},
        {'id': 'location_1', 'x': 0.5, 'y': 0.5, 'demand': 10, 'service_time': 60, 'time_window': (60, 1440), 'address': 'Location 1'},
        {'id': 'location_2', 'x': 1.0, 'y': 1.0, 'demand': 15, 'service_time': 60, 'time_window': (120, 1440), 'address': 'Location 2'},
        {'id': 'location_3', 'x': 1.5, 'y': 1.5, 'demand': 20, 'service_time': 60, 'time_window': (180, 1440), 'address': 'Location 3'},
        {'id': 'location_4', 'x': 2.0, 'y': 2.0, 'demand': 25, 'service_time': 60, 'time_window': (240, 1440), 'address': 'Location 4'},
        {'id': 'location_5', 'x': 2.5, 'y': 2.5, 'demand': 30, 'service_time': 60, 'time_window': (300, 1440), 'address': 'Location 5'},
        {'id': 'location_6', 'x': 3.0, 'y': 3.0, 'demand': 35, 'service_time': 60, 'time_window': (360, 1440), 'address': 'Location 6'},
    ]
    
    vehicles = [
        {'id': 'vehicle_1', 'capacity': 300, 'start_location': 'depot', 'end_location': 'depot', 'max_time': 180}  # 3 hours max - very restrictive!
    ]
    
    print(f"📍 Test scenario:")
    print(f"  - {len(locations)} locations (including depot)")
    print(f"  - {len(vehicles)} vehicle with 3-hour daily limit (very restrictive!)")
    print(f"  - Locations spread out with 1-hour service times")
    print(f"  - Single day should struggle, multi-day should visit more locations")
    
    # Test single day (should struggle with far locations)
    print(f"\n--- Single Day Test ---")
    optimizer_1day = CleanVRPOptimizer(vehicles=vehicles, locations=locations)
    result_1day, status_1day, constraints_1day = optimizer_1day.solve(
        constraint_level="time_windows", 
        verbose=False, 
        max_days=1
    )
    
    if result_1day:
        print(f"✅ Single day solution found")
        print(f"   Objective: {result_1day['objective_value']}")
        for vehicle_id, route_data in result_1day['routes'].items():
            print(f"   {vehicle_id}: {len(route_data['route'])} stops")
    else:
        print(f"❌ Single day solution failed: {status_1day}")
    
    # Test multi-day (should handle far locations better)
    print(f"\n--- Multi-Day Test (3 days) ---")
    optimizer_3day = CleanVRPOptimizer(vehicles=vehicles, locations=locations)
    result_3day, status_3day, constraints_3day = optimizer_3day.solve(
        constraint_level="time_windows", 
        verbose=False, 
        max_days=3
    )
    
    if result_3day:
        print(f"✅ Multi-day solution found")
        print(f"   Objective: {result_3day['objective_value']}")
        print(f"   Total distance: {result_3day['total_distance']:.1f} km")
        
        day_transitions = 0
        for vehicle_id, route_data in result_3day['routes'].items():
            route = route_data['route']
            print(f"   {vehicle_id}: {len(route)} stops, distance: {route_data['distance']:.1f} km")
            # Show day transitions
            for i, stop in enumerate(route):
                location_id = stop['location_id']
                if 'night' in location_id or 'morning' in location_id:
                    day_transitions += 1
                    arrival_time = stop.get('arrival_time', 0)
                    print(f"     Stop {i+1}: {location_id} (day transition at {arrival_time}min)")
        
        print(f"   🌙 Total day transitions: {day_transitions}")
        
        # Plot the multi-day solution
        print(f"\n📊 Plotting multi-day solution...")
        optimizer_3day.plot_solution(result_3day, title="Multi-Day VRP Solution (3 days max)")
        
    else:
        print(f"❌ Multi-day solution failed: {status_3day}")
    
    # Compare solutions if both exist
    if result_1day and result_3day:
        print(f"\n📊 COMPARISON: Single Day vs Multi-Day")
        print(f"   Single Day  - Objective: {result_1day['objective_value']:,}, Distance: {result_1day['total_distance']:.1f} km")
        print(f"   Multi-Day   - Objective: {result_3day['objective_value']:,}, Distance: {result_3day['total_distance']:.1f} km")
        
        if result_3day['objective_value'] < result_1day['objective_value']:
            improvement = ((result_1day['objective_value'] - result_3day['objective_value']) / result_1day['objective_value']) * 100
            print(f"   🎉 Multi-day improved objective by {improvement:.1f}%")
        elif result_3day['objective_value'] > result_1day['objective_value']:
            degradation = ((result_3day['objective_value'] - result_1day['objective_value']) / result_1day['objective_value']) * 100
            print(f"   📈 Multi-day objective is {degradation:.1f}% higher (trade-off for flexibility)")
        else:
            print(f"   ⚖️ Both solutions have same objective value")
    
    print(f"\n🏁 Multi-day test completed!")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test multi-day functionality with simple scenario first
    #test_multi_day_scenario()
    
    # Run the main test with multi-day capability
    #test_moda_inverted_scenario()
    #test_moda_small_scenario()
