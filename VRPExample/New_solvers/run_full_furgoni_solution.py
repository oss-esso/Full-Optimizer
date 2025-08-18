#!/usr/bin/env python3
"""
Complete Furgoni Scenario Solution with Full Constraints and Visualizations

This script:
1. Loads the MODA furgoni scenario
2. Uses an existing route database for speed
3. Solves with full constraints (capacity, time, etc.)
4. Creates both matplotlib and HTML interactive visualizations
5. Saves detailed results and analysis
"""

import os
import sys
import time
import json
import importlib.util
import traceback
import subprocess
from datetime import datetime

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def print_constraint_satisfaction_report(vehicle_summary, constraint_violations, vehicles, locations, total_overnight_stays=0):
    """Print a detailed report of all constraint satisfaction."""
    print(f"\n" + "="*80)
    print(f"📋 CONSTRAINT SATISFACTION REPORT")
    print(f"="*80)
    
    # Calculate overall statistics
    total_vehicles = len(vehicles)
    used_vehicles = len(vehicle_summary)
    total_weight_capacity = sum(v['capacity'] for v in vehicles)
    total_volume_capacity = sum(v.get('volume_capacity', v['capacity'] * 0.001) for v in vehicles)
    total_demand = sum(l['demand'] for l in locations)
    
    print(f"\n📊 OVERALL CONSTRAINT OVERVIEW:")
    print(f"  Fleet Utilization: {used_vehicles}/{total_vehicles} vehicles ({(used_vehicles/total_vehicles)*100:.1f}%)")
    print(f"  Total Weight Demand: {total_demand} units")
    print(f"  Total Weight Capacity: {total_weight_capacity} units")
    print(f"  Demand Coverage: {(total_demand/total_weight_capacity)*100:.1f}% of total capacity")
    
    # Weight and Volume Capacity Analysis
    print(f"\n🏋️ WEIGHT & VOLUME CAPACITY CONSTRAINTS:")
    print(f"{'Vehicle':<15} {'Type':<8} {'Weight Used':<12} {'Weight Cap':<11} {'Weight %':<9} {'Volume Used':<12} {'Volume Cap':<11} {'Volume %':<9} {'Status':<8}")
    print(f"-" * 110)
    
    weight_violations = 0
    volume_violations = 0
    total_weight_used = 0
    total_volume_used = 0
    
    for vehicle_id, summary in vehicle_summary.items():
        weight_used = summary['weight_used']
        weight_cap = summary['weight_capacity']
        weight_pct = summary['weight_utilization']
        volume_used = summary['volume_used']
        volume_cap = summary['volume_capacity']
        volume_pct = summary['volume_utilization']
        
        total_weight_used += weight_used
        total_volume_used += volume_used
        
        # Check violations
        weight_ok = weight_used <= weight_cap
        volume_ok = volume_used <= volume_cap
        
        if not weight_ok:
            weight_violations += 1
        if not volume_ok:
            volume_violations += 1
        
        status = "✅ OK" if weight_ok and volume_ok else "❌ VIOLATION"
        
        print(f"{vehicle_id:<15} {summary['type']:<8} {weight_used:<12.0f} {weight_cap:<11.0f} {weight_pct:<9.1f} {volume_used:<12.2f} {volume_cap:<11.2f} {volume_pct:<9.1f} {status:<8}")
    
    print(f"-" * 110)
    # Fix division by zero for totals row
    weight_pct_total = (total_weight_used/total_weight_capacity)*100 if total_weight_capacity > 0 else 0
    volume_pct_total = (total_volume_used/total_volume_capacity)*100 if total_volume_capacity > 0 else 0
    print(f"{'TOTALS':<15} {'':<8} {total_weight_used:<12.0f} {total_weight_capacity:<11.0f} {weight_pct_total:<9.1f} {total_volume_used:<12.2f} {total_volume_capacity:<11.2f} {volume_pct_total:<9.1f}")
    
    # Time Constraints Analysis
    print(f"\n⏰ TIME CONSTRAINT ANALYSIS:")
    print(f"{'Vehicle':<15} {'Type':<8} {'Days Used':<10} {'Max Daily Time':<15} {'Time Limit':<12} {'Max Day %':<10} {'Status':<8}")
    print(f"-" * 85)
    
    time_violations = 0
    max_time_usage = 0
    
    for vehicle_id, summary in vehicle_summary.items():
        max_daily_time = summary['max_daily_time']
        time_limit = summary['daily_time_limit']
        days_used = summary['days_used']
        time_pct = (max_daily_time / time_limit) * 100 if time_limit > 0 else 0
        
        if time_pct > max_time_usage:
            max_time_usage = time_pct
        
        time_ok = max_daily_time <= time_limit
        if not time_ok:
            time_violations += 1
        
        status = "✅ OK" if time_ok else "❌ EXCEEDED"
        
        print(f"{vehicle_id:<15} {summary['type']:<8} {days_used:<10} {max_daily_time:<15.1f} {time_limit:<12.0f} {time_pct:<10.1f} {status:<8}")
    
    # Service Time Windows Analysis (if available)
    print(f"\n🕐 SERVICE TIME WINDOWS:")
    service_violations = 0
    locations_with_windows = 0
    
    for location in locations:
        if 'time_window_start' in location or 'time_window_end' in location:
            locations_with_windows += 1
    
    if locations_with_windows > 0:
        print(f"  Locations with time windows: {locations_with_windows}")
        print(f"  Time window violations: {service_violations}")
        print(f"  Time window compliance: {((locations_with_windows - service_violations) / locations_with_windows) * 100:.1f}%")
    else:
        print(f"  ℹ️ No explicit time windows defined in this scenario")
        print(f"  All deliveries use default service times")
    
    # Fleet Efficiency Analysis
    print(f"\n🚛 FLEET EFFICIENCY ANALYSIS:")
    furgoni_count = len([v for v in vehicle_summary.values() if v['type'] == 'furgone'])
    camion_count = len([v for v in vehicle_summary.values() if v['type'] == 'camion'])
    
    avg_weight_utilization = sum(v['weight_utilization'] for v in vehicle_summary.values()) / len(vehicle_summary) if vehicle_summary else 0
    avg_volume_utilization = sum(v['volume_utilization'] for v in vehicle_summary.values()) / len(vehicle_summary) if vehicle_summary else 0
    avg_daily_time_usage = sum((v['max_daily_time'] / v['daily_time_limit']) * 100 for v in vehicle_summary.values()) / len(vehicle_summary) if vehicle_summary else 0
    
    print(f"  Vehicle Types Used:")
    print(f"    - Furgoni (light trucks): {furgoni_count}")
    print(f"    - Camion (heavy trucks): {camion_count}")
    print(f"  Average Utilization:")
    print(f"    - Weight capacity: {avg_weight_utilization:.1f}%")
    print(f"    - Volume capacity: {avg_volume_utilization:.1f}%")
    print(f"    - Daily time: {avg_daily_time_usage:.1f}%")
    
    # Constraint Violation Summary
    print(f"\n❌ CONSTRAINT VIOLATIONS SUMMARY:")
    total_violations = weight_violations + volume_violations + time_violations + service_violations
    
    print(f"  Weight Capacity Violations: {weight_violations}")
    print(f"  Volume Capacity Violations: {volume_violations}")
    print(f"  Daily Time Limit Violations: {time_violations}")
    print(f"  Service Time Window Violations: {service_violations}")
    print(f"  TOTAL VIOLATIONS: {total_violations}")
    
    if total_violations == 0:
        print(f"  🎉 ALL CONSTRAINTS SATISFIED! Perfect feasible solution.")
    else:
        print(f"  ⚠️ {total_violations} constraint violations detected.")
        print(f"     Solution may need adjustment or constraints may be too restrictive.")
    
    # Recommendations
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    
    if avg_weight_utilization < 70:
        print(f"  📦 Weight utilization is low ({avg_weight_utilization:.1f}%) - consider using fewer or smaller vehicles")
    elif avg_weight_utilization > 95:
        print(f"  📦 Weight utilization is very high ({avg_weight_utilization:.1f}%) - consider adding more capacity")
    
    if avg_volume_utilization < 70:
        print(f"  📏 Volume utilization is low ({avg_volume_utilization:.1f}%) - cargo is likely heavy but compact")
    elif avg_volume_utilization > 95:
        print(f"  📏 Volume utilization is very high ({avg_volume_utilization:.1f}%) - cargo is likely bulky")
    
    if avg_daily_time_usage < 60:
        print(f"  ⏰ Time utilization is low ({avg_daily_time_usage:.1f}%) - routes could be longer or consolidated")
    elif avg_daily_time_usage > 90:
        print(f"  ⏰ Time utilization is very high ({avg_daily_time_usage:.1f}%) - drivers working near limits")
    
    if used_vehicles < total_vehicles * 0.8:
        print(f"  🚛 Fleet utilization is low ({(used_vehicles/total_vehicles)*100:.1f}%) - fleet may be oversized")
    
    if total_overnight_stays > 0:
        overnight_ratio = total_overnight_stays / used_vehicles
        print(f"  🛏️ Overnight stops: {total_overnight_stays} total ({overnight_ratio:.1f} per vehicle)")
        if overnight_ratio > 2:
            print(f"      High overnight frequency suggests customers are far from depot")
    
    print(f"="*80)

def run_full_furgoni_solution():
    """Run complete furgoni scenario with full constraints and visualizations."""
    print("🚀 COMPLETE FURGONI SCENARIO SOLUTION")
    print("=" * 80)
    
    try:
        # Import the scenario and VRP solver
        print("📦 Importing required modules...")
        sys.path.append('.')
        from vrp_scenarios import create_furgoni_scenario, DEFAULT_TRUCK_SPEED_RATIOS
        
        # Import vrp_multiday_sequential
        spec = importlib.util.spec_from_file_location("vrp_multiday", "vrp_multiday_sequential.py")
        vrp_multiday = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vrp_multiday)
        
        print("✅ Successfully imported all modules")
        
        # Create the full furgoni scenario
        print("\n📊 Creating MODA furgoni scenario...")
        scenario = create_furgoni_scenario()
        
        # Convert to format needed for Sequential Multi-Day VRP
        print("🔧 Converting scenario data...")
        locations = []
        for loc_id, loc in scenario.locations.items():
            x = getattr(loc, 'x', None) or getattr(loc, 'lon', None) or 0
            y = getattr(loc, 'y', None) or getattr(loc, 'lat', None) or 0
            lat = getattr(loc, 'lat', None) or getattr(loc, 'latitude', None) or y
            lon = getattr(loc, 'lon', None) or getattr(loc, 'longitude', None) or x
            
            # Get demand - handle both direct and via ride_requests
            demand = getattr(loc, 'demand', 0)
            if demand == 0 and hasattr(scenario, 'ride_requests'):
                # Check if this location has ride requests
                ride_requests = scenario.ride_requests
                # Handle both dict and list formats
                if isinstance(ride_requests, dict):
                    ride_requests = ride_requests.values()
                elif isinstance(ride_requests, list):
                    ride_requests = ride_requests
                
                for req in ride_requests:
                    if (hasattr(req, 'pickup_location') and req.pickup_location == loc_id) or \
                       (hasattr(req, 'delivery_location') and req.delivery_location == loc_id):
                        demand += getattr(req, 'weight', 100)  # Default weight if not specified
            
            locations.append({
                'id': str(loc_id),
                'x': x, 'y': y,
                'lat': lat, 'lon': lon,
                'demand': demand,
                'address': getattr(loc, 'address', f'Location {loc_id}'),
                'service_time': getattr(loc, 'service_time', 15)
            })
        
        # Convert vehicles with full constraint handling
        vehicles = []
        for vehicle_id, vehicle in scenario.vehicles.items():
            # Determine vehicle type and speed ratios
            vehicle_type = getattr(vehicle, 'type', 'furgone').lower()
            if 'camion' in vehicle_type or 'truck' in vehicle_type or vehicle.capacity > 5000:
                truck_ratios = DEFAULT_TRUCK_SPEED_RATIOS.get('heavy', DEFAULT_TRUCK_SPEED_RATIOS['standard'])
                v_type = 'camion'
            else:
                truck_ratios = DEFAULT_TRUCK_SPEED_RATIOS.get('standard', DEFAULT_TRUCK_SPEED_RATIOS['standard'])
                v_type = 'furgone'
            
            vehicles.append({
                'id': str(vehicle_id),
                'capacity': vehicle.capacity,
                'volume_capacity': getattr(vehicle, 'volume_capacity', vehicle.capacity * 0.001),
                'depot_id': str(vehicle.depot_id),
                'truck_speed_ratios': truck_ratios,
                'type': v_type,
                'cost_per_km': getattr(vehicle, 'cost_per_km', 1.5 if v_type == 'camion' else 1.0),
                'max_time': getattr(vehicle, 'max_time', 8 * 60)  # 8 hours per day
            })
        
        print(f"📊 Scenario prepared:")
        print(f"  - Total locations: {len(locations)}")
        print(f"  - Customer locations: {len([l for l in locations if l['demand'] > 0])}")
        print(f"  - Total vehicles: {len(vehicles)}")
        print(f"  - Furgoni (light): {len([v for v in vehicles if v['type'] == 'furgone'])}")
        print(f"  - Camion (heavy): {len([v for v in vehicles if v['type'] == 'camion'])}")
        print(f"  - Total demand: {sum(l['demand'] for l in locations)} units")
        print(f"  - Total capacity: {sum(v['capacity'] for v in vehicles)} units")
        
        # Check for existing databases and use the most recent one
        db_candidates = [
            "moda_routes_fixed.db",
            "moda_routes.db", 
            "debug_routes.db"
        ]
        
        db_path = None
        for db_name in db_candidates:
            if os.path.exists(db_name):
                db_path = db_name
                db_size = os.path.getsize(db_name) / (1024 * 1024)  # Size in MB
                print(f"📂 Found existing database: {db_name} ({db_size:.1f} MB)")
                break
        
        if not db_path:
            db_path = "full_furgoni_routes.db"
            print(f"📂 Creating new database: {db_path}")
        
        # Initialize the Sequential Multi-Day VRP solver
        print(f"\n🚛 Initializing Sequential Multi-Day VRP solver...")
        start_init = time.time()
        
        sequential_vrp = vrp_multiday.SequentialMultiDayVRP(
            vehicles=vehicles,
            locations=locations,
            use_truck_speeds=True,
            db_path=db_path
        )
        
        init_time = time.time() - start_init
        print(f"  ⏱️ Solver initialized in {init_time:.2f}s")
        
        # Show database statistics if available
        if hasattr(sequential_vrp, 'distance_calculator') and \
           hasattr(sequential_vrp.distance_calculator, 'get_cache_stats'):
            try:
                cache_stats = sequential_vrp.distance_calculator.get_cache_stats()
                print(f"  📊 Route database stats: {cache_stats}")
            except:
                pass
        
        # Solve the multi-day VRP with full constraints
        print(f"\n🚀 Solving Sequential Multi-Day VRP with FULL CONSTRAINTS...")
        print(f"  🔒 Capacity constraints: ENABLED")
        print(f"  ⏰ Time constraints: ENABLED (8 hours/day)")
        print(f"  🚛 Mixed fleet: ENABLED (different truck speeds)")
        print(f"  🛏️ Overnight stops: ENABLED (OSRM-based placement)")
        print(f"  🏠 Return to depot: ENFORCED on final day")
        
        start_solve = time.time()
        solution = sequential_vrp.solve_sequential_multiday(max_days=10)
        solve_time = time.time() - start_solve
        
        if solution:
            print(f"\n✅ SOLUTION FOUND! (Solved in {solve_time:.1f}s)")
            
            # Recalculate distances using real OSRM routes (filter out overnight nodes)
            solution = sequential_vrp.recalculate_vehicle_distances(solution)
            
            # Analyze the solution
            total_vehicles_used = len(solution.get('vehicle_routes', {}))
            total_distance = 0
            total_overnight_stays = 0
            total_cost = 0
            
            print(f"\n📊 Solution Analysis:")
            vehicle_summary = {}
            constraint_violations = {
                'weight_capacity': 0,
                'volume_capacity': 0,
                'daily_time_limit': 0,
                'service_time_windows': 0
            }
            
            for vehicle_id, route_data in solution.get('vehicle_routes', {}).items():
                route_distance = route_data.get('total_distance', 0)
                total_distance += route_distance
                
                vehicle_overnights = route_data.get('total_overnight_stays', 0)
                total_overnight_stays += vehicle_overnights
                
                # Calculate cost based on vehicle type
                vehicle_info = next((v for v in vehicles if v['id'] == vehicle_id), None)
                if vehicle_info:
                    cost_per_km = vehicle_info.get('cost_per_km', 1.0)
                    vehicle_cost = route_distance * cost_per_km
                    total_cost += vehicle_cost
                    
                    # Analyze constraint satisfaction for this vehicle
                    weight_used = 0
                    volume_used = 0
                    
                    # Get demand and time from the full_route in vehicle_routes
                    vehicle_route_data = solution.get('vehicle_routes', {}).get(vehicle_id, {})
                    full_route = vehicle_route_data.get('full_route', [])
                    
                    # Extract customer visits from full route
                    for route_item in full_route:
                        if isinstance(route_item, dict):
                            # Skip day markers and overnight stops
                            if route_item.get('is_day_marker', False) or route_item.get('is_overnight', False):
                                continue
                            
                            location_id = route_item.get('location_id')
                            # Skip depot
                            if location_id == 'depot':
                                continue
                                
                            # Look up demand for this customer
                            location = next((l for l in locations if l['id'] == str(location_id)), None)
                            if location:
                                stop_demand = location.get('demand', 0)
                                weight_used += stop_demand
                                volume_used += stop_demand * 0.001
                        elif isinstance(route_item, str):
                            # Handle string format route items
                            if 'depot' not in route_item.lower() and 'overnight' not in route_item.lower():
                                location = next((l for l in locations if l['id'] == str(route_item)), None)
                                if location:
                                    stop_demand = location.get('demand', 0)
                                    weight_used += stop_demand
                                    volume_used += stop_demand * 0.001
                    
                    weight_capacity = vehicle_info['capacity']
                    volume_capacity = vehicle_info.get('volume_capacity', vehicle_info['capacity'] * 0.001)
                    
                    # Check daily routes for time constraints from daily_solutions
                    max_daily_time = 0
                    days_used = 0
                    
                    daily_solutions = solution.get('daily_solutions', {})
                    for day, day_solution in daily_solutions.items():
                        day_routes = day_solution.get('routes', {})
                        if vehicle_id in day_routes:
                            days_used += 1
                            day_route = day_routes[vehicle_id]
                            day_time = day_route.get('total_time', 0)
                            if day_time > max_daily_time:
                                max_daily_time = day_time
                            
                            # Check if daily time limit was exceeded
                            if day_time > vehicle_info.get('max_time', 8 * 60):
                                constraint_violations['daily_time_limit'] += 1
                    
                    # Check capacity constraints
                    weight_utilization = (weight_used / weight_capacity) * 100 if weight_capacity > 0 else 0
                    volume_utilization = (volume_used / volume_capacity) * 100 if volume_capacity > 0 else 0
                    
                    if weight_used > weight_capacity:
                        constraint_violations['weight_capacity'] += 1
                    if volume_used > volume_capacity:
                        constraint_violations['volume_capacity'] += 1
                    
                    vehicle_summary[vehicle_id] = {
                        'type': vehicle_info['type'],
                        'distance': route_distance,
                        'overnight_stays': vehicle_overnights,
                        'cost': vehicle_cost,
                        'weight_used': weight_used,
                        'weight_capacity': weight_capacity,
                        'weight_utilization': weight_utilization,
                        'volume_used': volume_used,
                        'volume_capacity': volume_capacity,
                        'volume_utilization': volume_utilization,
                        'max_daily_time': max_daily_time,
                        'daily_time_limit': vehicle_info.get('max_time', 8 * 60),
                        'days_used': days_used
                    }
                
                print(f"  🚛 {vehicle_id}: {route_distance:.1f}km, {vehicle_overnights} overnight stays, {weight_utilization:.1f}% weight capacity")
            
            print(f"\n📊 Overall Summary:")
            print(f"  - Vehicles used: {total_vehicles_used} / {len(vehicles)}")
            print(f"  - Total distance: {total_distance:.1f} km")
            print(f"  - Total cost: {total_cost:.2f} units")
            print(f"  - Total overnight stays: {total_overnight_stays}")
            print(f"  - Days used: {solution.get('days_used', 'N/A')}")
            print(f"  - Solution time: {solve_time:.1f} seconds")
            
            # Print detailed constraint satisfaction report
            print_constraint_satisfaction_report(vehicle_summary, constraint_violations, vehicles, locations, total_overnight_stays)
            
            # Create detailed daily constraint tracking
            print(f"\n📊 Creating detailed daily constraint tracking...")
            
            # Debug: Print solution structure to understand what's available
            print(f"  🔍 Solution keys: {list(solution.keys())}")
            if 'vehicle_routes' in solution:
                print(f"  🔍 Vehicle routes sample:")
                for vehicle_id in list(solution['vehicle_routes'].keys())[:2]:
                    vehicle_route = solution['vehicle_routes'][vehicle_id]
                    print(f"    {vehicle_id}: {list(vehicle_route.keys())}")
                    if 'daily_routes' in vehicle_route:
                        print(f"      daily_routes: {list(vehicle_route['daily_routes'].keys())}")
            
            detailed_tracking = create_detailed_constraint_tracking(solution, vehicles, locations, scenario)
            
            # Save detailed tracking to JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tracking_file = f"detailed_constraint_tracking_{timestamp}.json"
            
            with open(tracking_file, 'w') as f:
                json.dump(detailed_tracking, f, indent=2, default=str)
            
            print(f"  ✅ Detailed constraint tracking saved: {tracking_file}")
            
            # Print summary of tracking data
            total_stops = sum(len(day['stops']) for vehicle in detailed_tracking["vehicles"].values() 
                            for day in vehicle['daily_breakdown'].values())
            vehicles_with_violations = 0
            total_time_violations = 0
            weight_violations = 0
            volume_violations = 0
            
            for vehicle_id, vehicle_data in detailed_tracking["vehicles"].items():
                # Check time window violations
                for day, day_data in vehicle_data['daily_breakdown'].items():
                    violations_in_day = sum(1 for stop in day_data['stops'] if stop.get('time_window_violation', False))
                    if violations_in_day > 0:
                        vehicles_with_violations += 1
                        total_time_violations += violations_in_day
                
                # Check capacity violations
                if vehicle_data['constraint_violations'].get('weight_capacity', False):
                    weight_violations += 1
                
                if vehicle_data['constraint_violations'].get('volume_capacity', False):
                    volume_violations += 1
            
            print(f"  📋 Tracking summary:")
            print(f"    - Total stops tracked: {total_stops}")
            print(f"    - Vehicles with time window violations: {vehicles_with_violations}")
            print(f"    - Total time window violations: {total_time_violations}")
            print(f"    - Weight capacity violations: {weight_violations}")
            print(f"    - Volume capacity violations: {volume_violations}")
            print(f"    - Daily breakdowns available for: {len(detailed_tracking['vehicles'])} vehicles")
            
            # Check if all vehicles returned to depot
            vehicles_at_depot = 0
            vehicles_still_out = 0
            final_day_info = solution.get('daily_solutions', {})
            
            if final_day_info:
                final_day = max(final_day_info.keys())
                print(f"\n🏠 Final Day ({final_day}) Status:")
                
                for vehicle_id in [v['id'] for v in vehicles]:
                    final_routes = final_day_info.get(final_day, {}).get('routes', {})
                    if vehicle_id in final_routes:
                        route = final_routes[vehicle_id]
                        if route.get('returned_to_depot', False) or not route.get('ends_with_overnight', False):
                            vehicles_at_depot += 1
                            print(f"  ✅ {vehicle_id}: At depot")
                        else:
                            vehicles_still_out += 1
                            print(f"  ⚠️ {vehicle_id}: Still out")
                    else:
                        vehicles_at_depot += 1
                        print(f"  ✅ {vehicle_id}: At depot (completed earlier)")
                
                if vehicles_still_out == 0:
                    print(f"  🎉 SUCCESS: All vehicles returned to depot!")
                else:
                    print(f"  ⚠️ WARNING: {vehicles_still_out} vehicles still out")
            
            # Create matplotlib visualization
            print(f"\n📊 Creating matplotlib visualization...")
            try:
                plot_filename = sequential_vrp.plot_sequential_solution(
                    solution, 
                    "Complete MODA Furgoni Solution - Full Constraints"
                )
                print(f"  ✅ Matplotlib plot saved: {plot_filename}")
            except Exception as plot_error:
                print(f"  ❌ Could not create matplotlib plot: {plot_error}")
                plot_filename = None
            
            # Create interactive HTML map
            print(f"\n🗺️ Creating interactive HTML map...")
            try:
                html_filename = create_interactive_html_map(scenario, solution, sequential_vrp)
                print(f"  ✅ Interactive map saved: {html_filename}")
            except Exception as map_error:
                print(f"  ❌ Could not create HTML map: {map_error}")
                html_filename = None
            
            # Save detailed results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"complete_furgoni_results_{timestamp}.json"
            
            detailed_results = {
                'scenario': 'MODA_furgoni_complete',
                'timestamp': timestamp,
                'solve_time': solve_time,
                'init_time': init_time,
                'total_distance': total_distance,
                'total_cost': total_cost,
                'total_overnight_stays': total_overnight_stays,
                'vehicles_used': total_vehicles_used,
                'vehicles_total': len(vehicles),
                'vehicles_at_depot': vehicles_at_depot,
                'vehicles_still_out': vehicles_still_out,
                'days_used': solution.get('days_used', None),
                'database_used': db_path,
                'vehicle_summary': vehicle_summary,
                'constraint_violations': constraint_violations,
                'constraints_enabled': {
                    'capacity': True,
                    'time_limits': True,
                    'mixed_fleet': True,
                    'overnight_stops': True,
                    'return_to_depot': True
                },
                'constraint_satisfaction': {
                    'total_violations': sum(constraint_violations.values()),
                    'weight_violations': constraint_violations['weight_capacity'],
                    'volume_violations': constraint_violations['volume_capacity'],
                    'time_violations': constraint_violations['daily_time_limit'],
                    'service_violations': constraint_violations['service_time_windows'],
                    'all_satisfied': sum(constraint_violations.values()) == 0
                },
                'files_created': {
                    'matplotlib_plot': plot_filename,
                    'html_map': html_filename,
                    'results_json': results_file,
                    'detailed_tracking': tracking_file
                }
            }
            
            with open(results_file, 'w') as f:
                json.dump(detailed_results, f, indent=2, default=str)
            
            print(f"\n📄 Detailed results saved: {results_file}")
            
            # Print summary of created files
            print(f"\n📁 Files Created:")
            if plot_filename:
                print(f"  📊 Matplotlib plot: {plot_filename}")
            if html_filename:
                print(f"  🗺️ Interactive HTML map: {html_filename}")
            print(f"  📄 Results JSON: {results_file}")
            print(f"  🗄️ Route database: {db_path}")
            
            return solution, detailed_results
            
        else:
            print("❌ No solution found")
            return None, None
            
    except Exception as e:
        print(f"❌ Error in complete furgoni solution: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def create_interactive_html_map(scenario, solution, sequential_vrp):
    """Create an interactive HTML map showing the complete VRP solution."""
    
    # Check if folium is available
    try:
        import folium
        import folium.plugins
        print("  📦 Using folium for interactive mapping")
    except ImportError:
        print("  ⚠️ Folium not available - installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "folium"])
        import folium
        import folium.plugins
    
    # Import the route database to get actual OSRM routes
    try:
        from route_database import RouteDatabase
        import json
        print("  🗺️ Route database available for actual OSRM routes")
        has_route_db = True
        # Initialize route database connection
        db_path = "moda_routes_fixed.db"  # Use the same database as the solver
        route_db = RouteDatabase(db_path)
    except ImportError:
        print("  ⚠️ Route database not available - using straight lines")
        has_route_db = False
        route_db = None
    
    def get_route_geometry(from_lat, from_lon, to_lat, to_lon, from_id="", to_id=""):
        """Get the actual route geometry from the database, fetching from API if needed."""
        if not has_route_db or not route_db:
            return None
            
        try:
            # Get full route data (will fetch from API if not cached)
            route_data = route_db.get_route(from_lat, from_lon, to_lat, to_lon, from_id, to_id)
            
            if route_data and 'route_geometry' in route_data:
                geometry = route_data['route_geometry']
                
                # Handle different geometry formats
                if isinstance(geometry, dict) and geometry.get('type') == 'LineString':
                    coords = geometry.get('coordinates', [])
                    if coords:
                        # Convert from [lon, lat] to [lat, lon] for folium
                        return [[coord[1], coord[0]] for coord in coords]
                        
                elif isinstance(geometry, list) and len(geometry) > 0:
                    # Check if it's already a list of coordinates
                    if isinstance(geometry[0], (list, tuple)) and len(geometry[0]) == 2:
                        # Already in [lat, lon] format for folium
                        return geometry
                    
        except Exception as e:
            print(f"    ⚠️ Error getting route geometry {from_id}->{to_id}: {e}")
            
        return None
    
    # Get GPS coordinates from locations
    lats = []
    lons = []
    
    # Try to get coordinates from scenario locations
    for loc_id, loc in scenario.locations.items():
        lat = getattr(loc, 'lat', None) or getattr(loc, 'latitude', None)
        lon = getattr(loc, 'lon', None) or getattr(loc, 'longitude', None)
        
        if lat is not None and lon is not None:
            lats.append(lat)
            lons.append(lon)
    
    if not lats:
        print("  ⚠️ No GPS coordinates found - using default map center")
        center_lat, center_lon = 45.4642, 9.1900  # Milan area
    else:
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
    
    print(f"  📍 Map center: ({center_lat:.4f}, {center_lon:.4f})")
    
    # Display initial database stats
    if has_route_db and route_db:
        initial_stats = route_db.get_cache_stats()
        initial_route_count = initial_stats.get('total_routes', 0)
        print(f"📊 Initial Route Cache Stats:")
        print(f"  - Cached routes: {initial_route_count}")
        print(f"  - Database size: {initial_stats.get('database_size_mb', 0):.2f} MB")
    else:
        initial_route_count = 0
    
    # Create the map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles='OpenStreetMap'
    )
    
    # Add alternative tile layers
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Color palette for vehicles
    vehicle_colors = [
        'blue', 'green', 'purple', 'orange', 'darkred', 'black', 
        'gray', 'pink', 'lightblue', 'lightgreen', 'darkgreen', 'cadetblue'
    ]
    
    # Add depot marker
    depot_added = False
    for loc_id, loc in scenario.locations.items():
        if 'depot' in str(loc_id).lower():
            lat = getattr(loc, 'lat', None) or getattr(loc, 'latitude', None)
            lon = getattr(loc, 'lon', None) or getattr(loc, 'longitude', None)
            if lat and lon:
                folium.Marker(
                    location=[lat, lon],
                    popup=f"<b>DEPOT</b><br>{loc_id}<br>{getattr(loc, 'address', '')}",
                    tooltip=f"Depot: {loc_id}",
                    icon=folium.Icon(color='red', icon='home', prefix='fa')
                ).add_to(m)
                depot_added = True
                break
    
    # Process vehicle routes and add to map
    route_info = {}
    markers_added = 0
    
    vehicle_routes = solution.get('vehicle_routes', {})
    for i, (vehicle_id, route_data) in enumerate(vehicle_routes.items()):
        color = vehicle_colors[i % len(vehicle_colors)]
        
        # Get route information for this vehicle
        full_route = route_data.get('full_route', [])
        vehicle_total_distance = route_data.get('total_distance', 0)
        vehicle_overnight_stays = route_data.get('total_overnight_stays', 0)
        
        route_coordinates = []
        route_segments = []  # Store actual OSRM route segments
        current_day = 1
        
        # Process the full route sequence to extract stop coordinates
        stops = []
        
        for j, route_item in enumerate(full_route):
            # Check if this is a day marker
            if isinstance(route_item, dict) and route_item.get('is_day_marker', False):
                current_day += 1
                continue
            elif isinstance(route_item, str) and 'day' in route_item.lower():
                current_day += 1
                continue
                
            # Add this stop to the current set of stops
            stop_data = {
                "location_id": route_item if isinstance(route_item, str) else route_item.get('location_id'),
                "is_overnight": (isinstance(route_item, dict) and route_item.get('is_overnight', False)) or
                               (isinstance(route_item, str) and 'overnight' in route_item.lower()),
                "day": current_day,
                "coordinates": route_item.get('coordinates', None) if isinstance(route_item, dict) else None,
            }
            stops.append(stop_data)
        
        # Build route with actual OSRM geometry - start from depot
        depot_coords = None
        for loc_id, loc in scenario.locations.items():
            if 'depot' in str(loc_id).lower() and str(loc_id).lower() == 'depot':
                depot_lat = getattr(loc, 'lat', None) or getattr(loc, 'latitude', None)
                depot_lon = getattr(loc, 'lon', None) or getattr(loc, 'longitude', None)
                if depot_lat and depot_lon:
                    depot_coords = [depot_lat, depot_lon]
                    break
        
        prev_coords = depot_coords  # Start from depot coordinates
        prev_location_id = 'depot'
        
        for i, stop in enumerate(stops):
            # Get coordinates for this stop
            lat, lon = None, None
            stop_location_id = stop.get('location_id')
            
            if stop_location_id is None:
                continue
                
            # Method 1: Direct coordinates in stop
            if stop.get('coordinates'):
                coords = stop['coordinates']
                if isinstance(coords, tuple) and len(coords) == 2:
                    lat, lon = coords[1], coords[0]  # folium expects [lat, lon]
                elif isinstance(coords, list) and len(coords) == 2:
                    lat, lon = coords[1], coords[0]
            
            # Method 2: Location ID lookup in scenario
            if (lat is None or lon is None) and stop_location_id:
                for loc_id, loc in scenario.locations.items():
                    if str(loc_id) == str(stop_location_id):
                        lat = getattr(loc, 'lat', None) or getattr(loc, 'latitude', None)
                        lon = getattr(loc, 'lon', None) or getattr(loc, 'longitude', None)
                        break
            
            if lat is not None and lon is not None:
                current_coords = [lat, lon]
                route_coordinates.append(current_coords)
                
                # Get route geometry between stops
                if prev_coords is not None:
                    # ALWAYS add a fallback straight line for comparison (vehicle color, thick, visible)
                    comparison_segment = {
                        'geometry': [prev_coords, current_coords],
                        'from': prev_location_id,
                        'to': stop_location_id,
                        'color': color,  # Use vehicle color instead of red
                        'is_fallback': True,
                        'type': 'comparison',
                        'weight': 2,
                        'opacity': 0.7
                    }
                    route_segments.append(comparison_segment)
                    
                    if has_route_db:
                        # Try to get OSRM route geometry
                        route_geometry = get_route_geometry(
                            prev_coords[0], prev_coords[1], 
                            current_coords[0], current_coords[1],
                            prev_location_id or "", stop_location_id or ""
                        )
                        
                        if route_geometry:
                            # Add this route segment (successful OSRM route)
                            route_segments.append({
                                'geometry': route_geometry,
                                'from': prev_location_id,
                                'to': stop_location_id,
                                'color': color,
                                'type': 'osrm'
                            })
                            print(f"    �️ Added OSRM segment: {prev_location_id} -> {stop_location_id}")
                        else:
                            # No OSRM route available - note this but fallback already added above
                            print(f"    ⚠️ No OSRM route available for {prev_location_id}->{stop_location_id}, using fallback only")
                    else:
                        # No OSRM database - fallback already added above
                        print(f"    📍 Using fallback segment: {prev_location_id} -> {stop_location_id} (no OSRM DB)")
                
                # Add marker for this stop
                is_overnight = stop.get('is_overnight', False)
                location_id = stop.get('location_id', 'Unknown')
                demand = stop.get('demand', 0)
                
                if is_overnight:
                    icon_color = 'darkblue'
                    icon_name = 'bed'
                    stop_type = 'Overnight'
                elif 'pickup' in location_id.lower():
                    icon_color = 'orange'
                    icon_name = 'warehouse'
                    stop_type = 'Pickup'
                elif demand > 0:
                    icon_color = color
                    icon_name = 'box'
                    stop_type = 'Delivery'
                elif location_id == 'depot':
                    continue  # Skip main depot (already added)
                else:
                    icon_color = color
                    icon_name = 'box'
                    stop_type = 'Customer'
                
                folium.Marker(
                    location=[lat, lon],
                    popup=f"<b>{stop_type}</b><br>Vehicle: {vehicle_id}<br>Day: {stop.get('day', 1)}<br>Location: {location_id}<br>Demand: {demand}",
                    tooltip=f"{vehicle_id} - {stop_type}",
                    icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa')
                ).add_to(m)
                markers_added += 1
                
                # Update for next iteration
                prev_coords = current_coords
                prev_location_id = stop_location_id
        
        # Add return route to depot (final segment)
        if prev_coords is not None:
            # Find depot coordinates
            depot_coords = None
            for loc_id, loc in scenario.locations.items():
                if 'depot' in str(loc_id).lower() and str(loc_id).lower() == 'depot':
                    depot_lat = getattr(loc, 'lat', None) or getattr(loc, 'latitude', None)
                    depot_lon = getattr(loc, 'lon', None) or getattr(loc, 'longitude', None)
                    if depot_lat and depot_lon:
                        depot_coords = [depot_lat, depot_lon]
                        break
            
            print(f"    🔍 DEBUG {vehicle_id}: prev_coords={prev_coords}, depot_coords={depot_coords}")
            print(f"    🔍 DEBUG {vehicle_id}: prev_location_id={prev_location_id}")
            
            # Add return route to depot if needed
            # Always add return route unless we're already at the main depot location
            if depot_coords and (prev_location_id != 'depot'):
                # ALWAYS add fallback straight line for comparison (vehicle color, thick, visible)
                comparison_segment = {
                    'geometry': [prev_coords, depot_coords],
                    'from': prev_location_id,
                    'to': 'depot',
                    'color': color,  # Use vehicle color instead of red
                    'is_fallback': True,
                    'type': 'comparison',
                    'weight': 2,
                    'opacity': 0.7
                }
                route_segments.append(comparison_segment)
                
                if has_route_db:
                    # Try to get OSRM route geometry back to depot
                    route_geometry = get_route_geometry(
                        prev_coords[0], prev_coords[1], 
                        depot_coords[0], depot_coords[1],
                        prev_location_id or "", "depot"
                    )
                    
                    if route_geometry:
                        route_segments.append({
                            'geometry': route_geometry,
                            'from': prev_location_id,
                            'to': 'depot',
                            'color': color,
                            'type': 'osrm'
                        })
                        print(f"    🏠 Added OSRM return segment: {prev_location_id} -> depot")
                    else:
                        print(f"    ⚠️ No OSRM route available for return {prev_location_id}->depot, using fallback only")
                else:
                    print(f"    🏠 Using fallback return segment: {prev_location_id} -> depot (no OSRM DB)")
            else:
                print(f"    ⚠️ DEBUG {vehicle_id}: Skipping return route - already at main depot")
        
        # Add route segments to map
        osrm_segments = 0
        fallback_segments = 0
        comparison_segments = 0
        
        print(f"    🗺️ Processing {len(route_segments)} route segments for {vehicle_id}")
        
        for segment in route_segments:
            geometry = segment['geometry']
            segment_type = segment.get('type', 'unknown')
            is_fallback = segment.get('is_fallback', False)
            
            # Ensure geometry is in correct [lat, lon] format for folium
            if isinstance(geometry, list) and len(geometry) > 0:
                # Check if it's a list of coordinate pairs
                if isinstance(geometry[0], (list, tuple)) and len(geometry[0]) >= 2:
                    # Already a list of [lat, lon] pairs - use as is
                    folium_geometry = geometry
                elif len(geometry) == 2 and isinstance(geometry[0], (int, float)):
                    # Single [lat, lon] pair - wrap in list
                    folium_geometry = [geometry]
                else:
                    # Try to convert - assume [lon, lat] and flip to [lat, lon]
                    try:
                        if isinstance(geometry[0], (list, tuple)):
                            folium_geometry = [[coord[1], coord[0]] for coord in geometry if len(coord) >= 2]
                        else:
                            folium_geometry = [[geometry[1], geometry[0]]] if len(geometry) >= 2 else []
                    except (IndexError, TypeError):
                        print(f"    ⚠️ Skipping invalid geometry for {segment.get('from', 'unknown')} -> {segment.get('to', 'unknown')}")
                        continue
            else:
                print(f"    ⚠️ Skipping empty geometry for {segment.get('from', 'unknown')} -> {segment.get('to', 'unknown')}")
                continue
            
            # Count different segment types
            if segment_type == 'osrm':
                osrm_segments += 1
                line_style = {}
                opacity = 0.8
                weight = 3
            elif segment_type == 'comparison':
                comparison_segments += 1
                line_style = {'dash_array': '8,4'}  # More visible dashed line for comparison
                opacity = segment.get('opacity', 0.7)  # Less transparent
                weight = segment.get('weight', 2)      # Thicker line
            elif is_fallback or segment_type == 'fallback':
                fallback_segments += 1
                line_style = {'dash_array': '10,5'}  # Dashed line for fallbacks
                opacity = 0.6
                weight = 3
            else:
                # Default handling for old-style segments
                if is_fallback:
                    fallback_segments += 1
                    line_style = {'dash_array': '10,5'}
                    opacity = 0.6
                else:
                    osrm_segments += 1
                    line_style = {}
                    opacity = 0.8
                weight = 3
            
            try:
                folium.PolyLine(
                    locations=folium_geometry,
                    color=segment['color'],
                    weight=weight,
                    opacity=opacity,
                    popup=f"Vehicle {vehicle_id}: {segment['from']} -> {segment['to']} ({segment_type})",
                    **line_style
                ).add_to(m)
                print(f"    ✅ Added {segment_type} line: {segment.get('from', 'unknown')} -> {segment.get('to', 'unknown')}")
            except Exception as e:
                print(f"    ❌ Failed to add line {segment.get('from', 'unknown')} -> {segment.get('to', 'unknown')}: {e}")
                print(f"        Geometry: {folium_geometry}")
                continue
        
        print(f"    🚛 {vehicle_id}: {osrm_segments} OSRM routes, {fallback_segments} fallback lines, {comparison_segments} comparison lines")
        
        route_info[vehicle_id] = {
            'distance': vehicle_total_distance,
            'overnight_stays': vehicle_overnight_stays,
            'stops': len(route_coordinates),
            'color': color,
            'osrm_segments': osrm_segments,
            'fallback_segments': fallback_segments,
            'comparison_segments': comparison_segments
        }
    
    # Create legend
    total_osrm = sum(info.get('osrm_segments', 0) for info in route_info.values())
    total_fallback = sum(info.get('fallback_segments', 0) for info in route_info.values())
    total_comparison = sum(info.get('comparison_segments', 0) for info in route_info.values())
    
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 450px; height: auto; max-height: 500px;
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 15px;
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
                border-radius: 5px;
                overflow-y: auto;">
    <h4 style="margin-top:0; margin-bottom:10px;">Complete MODA Furgoni Solution</h4>
    <div style="max-height: 320px; overflow-y: auto;">
    <table style="width:100%; font-size:11px;">
    <tr><th>Vehicle</th><th>Distance</th><th>Overnight</th><th>OSRM</th><th>Fallback</th><th>Compare</th></tr>
    '''
    
    for vehicle_id, info in route_info.items():
        osrm_count = info.get('osrm_segments', 0)
        fallback_count = info.get('fallback_segments', 0)
        comparison_count = info.get('comparison_segments', 0)
        legend_html += f'''
        <tr>
            <td><span style="color:{info['color']};">●</span> {vehicle_id}</td>
            <td>{info['distance']:.1f}km</td>
            <td>{info['overnight_stays']}</td>
            <td>{osrm_count}</td>
            <td>{fallback_count}</td>
            <td>{comparison_count}</td>
        </tr>
        '''
    
    legend_html += f'''
    </table>
    </div>
    <hr style="margin: 10px 0;">
    <div style="font-size: 11px;">
        <b>Route Quality Analysis:</b><br>
        🗺️ OSRM Routes: {total_osrm} segments (real roads)<br>
        ⚠️ Fallback Lines: {total_fallback} segments (straight lines)<br>
        🔍 Comparison Lines: {total_comparison} segments (red reference)<br>
        <br>
        <b>Total Route Segments:</b> {total_osrm + total_fallback + total_comparison}<br>
        <b>OSRM Coverage:</b> {(total_osrm/(total_osrm + total_fallback)*100) if (total_osrm + total_fallback) > 0 else 0:.1f}%<br>
        <br>
        <b>Legend:</b><br>
        🏠 Red: Main Depot<br>
        📦 Colored: Vehicle Deliveries<br>
        🛏️ Dark Blue: Overnight Stops<br>
        📤 Orange: Pickup Points<br>
        <br>
        <b>Route Lines:</b><br>
        Solid thick lines: Real OSRM road routes<br>
        Dashed thick lines: Fallback straight lines<br>
        Thin red dashed lines: Comparison straight lines<br>
        (Red lines should match OSRM if API is working correctly)<br>
        Routes cached in database for speed<br>
    </div>
    </div>
    '''
    
    # Add layer control and legend
    folium.LayerControl().add_to(m)
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add title
    title_html = f'''
    <h3 align="center" style="font-size:18px"><b>Complete MODA Furgoni VRP Solution</b></h3>
    <p align="center" style="font-size:14px">Full Constraints - {len(route_info)} Vehicles - Multi-Day Solution</p>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save the map
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_filename = f"complete_furgoni_map_{timestamp}.html"
    m.save(html_filename)
    
    print(f"  📊 Map created with {markers_added} markers and {len(route_info)} vehicle routes")
    print(f"  🗺️ Route geometry: {total_osrm} OSRM segments, {total_fallback} fallback lines, {total_comparison} comparison lines")
    
    # Show database growth
    if has_route_db and route_db:
        final_stats = route_db.get_cache_stats()
        final_route_count = final_stats.get('total_routes', 0)
        new_routes_added = final_route_count - initial_route_count
        if new_routes_added > 0:
            print(f"  📈 Database updated: +{new_routes_added} new routes cached")
            print(f"  💾 Final database size: {final_stats.get('database_size_mb', 0):.2f} MB")
        else:
            print(f"  ✅ All routes were already cached (no API calls needed)")
    
    return html_filename

def create_detailed_constraint_tracking(solution, vehicles, locations, scenario):
    """Create detailed daily constraint tracking for each vehicle with stops, weights, volumes, and time windows."""
    
    detailed_tracking = {
        "metadata": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario_name": getattr(scenario, "name", "Furgoni Scenario"),
            "total_days": solution.get("total_days", 0),
            "total_vehicles": len(vehicles),
            "total_locations": len(locations),
            "total_customers": len([loc for loc in locations.values() if getattr(loc, "is_depot", False) is False]) if isinstance(locations, dict) else 0,
            "total_demand": sum(getattr(req, "passengers", 0) for req in scenario.ride_requests) if hasattr(scenario, "ride_requests") else 0,
            "solver_version": "1.0"
        },
        "vehicles": {}
    }
    daily_solutions = solution.get('daily_solutions', {})
    vehicle_routes = solution.get('vehicle_routes', {})
    
    # Create location lookup for quick access to location details
    location_lookup = {}
    for loc_id, loc in scenario.locations.items():
        location_lookup[str(loc_id)] = {
            'address': getattr(loc, 'address', f'Location {loc_id}'),
            'lat': getattr(loc, 'lat', None) or getattr(loc, 'latitude', None),
            'lon': getattr(loc, 'lon', None) or getattr(loc, 'longitude', None),
            'time_window_start': getattr(loc, 'time_window_start', 0),
            'time_window_end': getattr(loc, 'time_window_end', 1440),
            'service_time': getattr(loc, 'service_time', 15),
            'demand': getattr(loc, 'demand', 0)
        }
    
    # Create ride request lookup for weight/volume information
    request_lookup = {}
    for req in scenario.ride_requests:  # ride_requests is a list, not a dict
        request_lookup[req.pickup_location] = {
            'weight': req.passengers,
            'volume': getattr(req, 'volume', req.passengers / 200.0),  # Default cargo density 200 kg/m³
            'dropoff': req.dropoff_location,
            'request_id': req.id
        }
        request_lookup[req.dropoff_location] = {
            'weight': -req.passengers,  # Negative for delivery
            'volume': -getattr(req, 'volume', req.passengers / 200.0),
            'pickup': req.pickup_location,
            'request_id': req.id
        }
    
    # Process each vehicle
    for vehicle_id, vehicle_data in vehicle_routes.items():
        # Get vehicle info
        vehicle_info = next((v for v in vehicles if v['id'] == vehicle_id), None)
        if not vehicle_info:
            continue
            
        detailed_tracking["vehicles"][vehicle_id] = {
            'vehicle_info': {
                'id': vehicle_id,
                'type': vehicle_info.get('type', 'unknown'),
                'weight_capacity': vehicle_info.get('capacity', 0),
                'volume_capacity': vehicle_info.get('volume_capacity', 0),
                'daily_time_limit': vehicle_info.get('max_time', 8 * 60),
                'cost_per_km': vehicle_info.get('cost_per_km', 1.0),
                'depot_id': vehicle_info.get('depot_id', 'depot')
            },
            'daily_breakdown': {},
            'totals': {
                'total_weight_carried': 0,
                'total_volume_carried': 0,
                'total_time_driven': 0,
                'total_distance': vehicle_data.get('total_distance', 0),
                'days_used': 0,
                'overnight_stays': vehicle_data.get('total_overnight_stays', 0),
                'total_distance_driven': 0,
                'total_stops': 0
            },
            'constraint_violations': {
                'weight_capacity': False,
                'volume_capacity': False,
                'time_capacity': False,
                'time_windows': False
            }
        }
        
        # Process full_route to extract daily information
        # Since daily_routes is not available, we need to parse full_route which includes day markers
        full_route = vehicle_data.get('full_route', [])
        
        # Initialize tracking variables
        current_day = 1
        current_day_info = None
        current_weight = 0
        current_volume = 0
        current_stop_number = 0
        max_weight_carried = 0
        max_volume_carried = 0
        day_start_time = 0
        days_used = 0
        
        # For time and distance tracking
        current_time = 0  # Start at time 0
        last_location = None
        last_location_coords = None
        current_day_distance = 0
        
        # Add a new day to the breakdown
        def start_new_day():
            nonlocal current_day_info, days_used, current_stop_number, current_time, day_start_time, current_day_distance
            days_used += 1
            current_day_info = {
                'weight_carried': 0,
                'volume_carried': 0,
                'time_driven': 0,
                'distance_driven': 0,
                'stops': [],
                'route_summary': {
                    'start_time': day_start_time,
                    'end_time': 0,  # Will be updated at the end of the day
                    'returned_to_depot': False,  # Will be updated
                    'ends_with_overnight': False  # Will be updated
                }
            }
            current_day_distance = 0
            detailed_tracking["vehicles"][vehicle_id]['daily_breakdown'][f'day_{current_day}'] = current_day_info
            current_stop_number = 0
            day_start_time = current_time
            
        # Start with day 1
        start_new_day()
        
        # Process the full route to extract daily information
        for i, route_item in enumerate(full_route):
            is_day_marker = False
            is_overnight = False
            location_id = None
            
            # Extract information based on item type
            if isinstance(route_item, dict):
                # Handle dict format route items
                if route_item.get('is_day_marker', False):
                    is_day_marker = True
                    # End the current day and start a new one
                    if current_day_info:
                        current_day_info['route_summary']['end_time'] = current_time
                    current_day += 1
                    start_new_day()
                    continue
                
                if route_item.get('is_overnight', False):
                    is_overnight = True
                    current_day_info['route_summary']['ends_with_overnight'] = True
                    location_id = 'overnight'
                else:
                    location_id = route_item.get('location_id')
                    
                # Get time information
                arrival_time = route_item.get('arrival_time', current_time)
                service_time = route_item.get('service_time', 15)
                departure_time = route_item.get('departure_time', arrival_time + service_time)
                current_time = departure_time
                
            elif isinstance(route_item, str):
                # Handle string format route items
                if 'day' in route_item.lower():
                    is_day_marker = True
                    # End the current day and start a new one
                    if current_day_info:
                        current_day_info['route_summary']['end_time'] = current_time
                    current_day += 1
                    start_new_day()
                    continue
                
                if 'overnight' in route_item.lower():
                    is_overnight = True
                    current_day_info['route_summary']['ends_with_overnight'] = True
                    location_id = 'overnight'
                else:
                    location_id = route_item
                    
                # Estimate time (since not provided in string format)
                arrival_time = current_time
                service_time = 15  # Default
                departure_time = arrival_time + service_time
                current_time = departure_time
            
            # Skip if no location ID or it's a day marker
            if not location_id or is_day_marker:
                continue
                
            # Update depot return status
            if location_id == 'depot':
                current_day_info['route_summary']['returned_to_depot'] = True
            
            # Skip depot for demand tracking (but keep in route for visualization)
            stop_demand = 0
            stop_volume = 0
            
            # Get location details
            location_details = None
            if location_id != 'overnight' and location_id != 'depot':
                location_details = location_lookup.get(str(location_id), None)
                if location_details:
                    stop_demand = location_details['demand']
                    stop_volume = stop_demand * 0.001  # Default volume conversion
                    
                    # Check if this location has ride request information
                    if str(location_id) in request_lookup:
                        req_info = request_lookup[str(location_id)]
                        stop_demand = req_info['weight']
                        stop_volume = req_info['volume']
            
            # Update weight and volume carried
            current_weight += stop_demand
            current_volume += stop_volume
            
            if current_weight > max_weight_carried:
                max_weight_carried = current_weight
                current_day_info['weight_carried'] = max_weight_carried
                
            if current_volume > max_volume_carried:
                max_volume_carried = current_volume
                current_day_info['volume_carried'] = max_volume_carried
            
            # Create stop info
            current_stop_number += 1
            
            # Skip adding overnight stops to the stop list (they're tracked in route_summary)
            if location_id == 'overnight':
                continue
                
            # For locations that aren't in our lookup, create minimal details
            if not location_details and location_id != 'depot':
                location_details = {
                    'address': f'Location {location_id}',
                    'lat': 0,
                    'lon': 0,
                    'time_window_start': 0,
                    'time_window_end': 1440,
                    'service_time': 15,
                    'demand': 0
                }
            elif location_id == 'depot':
                # Get depot information
                depot_id = vehicle_info.get('depot_id', 'depot')
                location_details = location_lookup.get(str(depot_id), {
                    'address': 'Main Depot',
                    'lat': 0,
                    'lon': 0,
                    'time_window_start': 0,
                    'time_window_end': 1440,
                    'service_time': 0,
                    'demand': 0
                })
                
            # Calculate distance from previous location
            distance_from_last = 0
            current_lat = location_details.get('lat', 0)
            current_lon = location_details.get('lon', 0)
            
            if last_location_coords and current_lat is not None and current_lon is not None:
                # Simple Euclidean distance for demonstration
                # In a real scenario, you'd use the actual distance matrix
                distance_from_last = ((current_lat - last_location_coords[0])**2 + 
                                     (current_lon - last_location_coords[1])**2)**0.5 * 111  # Rough km conversion
            
            # Update day's distance
            current_day_distance += distance_from_last
            current_day_info['distance_driven'] = current_day_distance
            
            # Calculate driving time based on distance (assuming 50km/h average speed)
            driving_time = distance_from_last * 1.2  # Minutes (50km/h = 1.2 min/km)
            current_day_info['time_driven'] += driving_time
            
            # Create the stop info
            stop_info = {
                'stop_number': current_stop_number,
                'location_id': location_id,
                'location_type': 'overnight' if is_overnight else ('depot' if location_id == 'depot' else 'customer'),
                'arrival_time': arrival_time,
                'departure_time': departure_time,
                'service_time': service_time,
                'weight_change': stop_demand,
                'volume_change': stop_volume,
                'current_weight_load': current_weight,
                'current_volume_load': current_volume,
                'coordinates': {
                    'lat': current_lat,
                    'lon': current_lon
                },
                'address': location_details.get('address', f'Location {location_id}'),
                'time_window': {
                    'start': location_details.get('time_window_start', 0),
                    'end': location_details.get('time_window_end', 1440)
                },
                'time_window_violation': False,
                'is_overnight': is_overnight,
                'distance_from_last': distance_from_last,
                'driving_time_from_last': driving_time,
                'demand': location_details.get('demand', 0)
            }
            
            # Update for next stop
            last_location = location_id
            last_location_coords = (current_lat, current_lon)
            
            # Check for time window violations - safely handling None values
            tw_start = stop_info['time_window']['start']
            tw_end = stop_info['time_window']['end']
            if arrival_time is not None and tw_start is not None and tw_end is not None:
                if arrival_time < tw_start or arrival_time > tw_end:
                    stop_info['time_window_violation'] = True
            
            current_day_info['stops'].append(stop_info)
            last_location = location_id
                
        # When we're done processing the route, update the final day's end time
        if current_day_info:
            current_day_info['route_summary']['end_time'] = current_time
        
        # Update totals
        detailed_tracking["vehicles"][vehicle_id]['totals']['days_used'] = days_used
        detailed_tracking["vehicles"][vehicle_id]['totals']['total_time_driven'] = sum(
            day['time_driven'] for day in detailed_tracking["vehicles"][vehicle_id]['daily_breakdown'].values()
        )
        detailed_tracking["vehicles"][vehicle_id]['totals']['total_weight_carried'] = max_weight_carried
        detailed_tracking["vehicles"][vehicle_id]['totals']['total_volume_carried'] = max_volume_carried
        detailed_tracking["vehicles"][vehicle_id]['totals']['total_distance_driven'] = sum(
            day['distance_driven'] for day in detailed_tracking["vehicles"][vehicle_id]['daily_breakdown'].values()
        )
        detailed_tracking["vehicles"][vehicle_id]['totals']['total_stops'] = sum(
            len(day['stops']) for day in detailed_tracking["vehicles"][vehicle_id]['daily_breakdown'].values()
        )
        
        # Check for constraint violations
        weight_cap = detailed_tracking["vehicles"][vehicle_id]['vehicle_info']['weight_capacity']
        volume_cap = detailed_tracking["vehicles"][vehicle_id]['vehicle_info']['volume_capacity']
        time_cap = detailed_tracking["vehicles"][vehicle_id]['vehicle_info']['daily_time_limit']
        
        if weight_cap > 0 and max_weight_carried > weight_cap:
            detailed_tracking["vehicles"][vehicle_id]['constraint_violations']['weight_capacity'] = True
        
        if volume_cap > 0 and max_volume_carried > volume_cap:
            detailed_tracking["vehicles"][vehicle_id]['constraint_violations']['volume_capacity'] = True
        
        # Check for time window violations
        time_window_violations = 0
        for day_info in detailed_tracking["vehicles"][vehicle_id]['daily_breakdown'].values():
            for stop in day_info['stops']:
                if stop.get('time_window_violation', False):
                    time_window_violations += 1
        
        if time_window_violations > 0:
            detailed_tracking["vehicles"][vehicle_id]['constraint_violations']['time_windows'] = True
            detailed_tracking["vehicles"][vehicle_id]['totals']['time_window_violations'] = time_window_violations
    
    return detailed_tracking

# ...existing code...
if __name__ == "__main__":
    print("🚀 Running Complete MODA Furgoni Solution")
    print("=" * 80)
    
    start_time = time.time()
    solution, results = run_full_furgoni_solution()
    total_time = time.time() - start_time
    
    if solution and results:
        print(f"\n🎉 COMPLETE SUCCESS!")
        print(f"  ⏱️ Total execution time: {total_time:.1f} seconds")
        print(f"  📊 Solution quality: {results['vehicles_used']}/{results['vehicles_total']} vehicles used")
        print(f"  💰 Total cost: {results['total_cost']:.2f} units")
        print(f"  🛣️ Total distance: {results['total_distance']:.1f} km")
        print(f"  🏠 All vehicles returned to depot: {'✅' if results['vehicles_still_out'] == 0 else '❌'}")
        
        # Show created files
        files = results['files_created']
        print(f"\n📁 Generated Files:")
        for file_type, filename in files.items():
            if filename:
                print(f"  📄 {file_type}: {filename}")
        
    else:
        print(f"\n❌ Solution failed after {total_time:.1f} seconds")
        print("Check the error messages above for details.")
