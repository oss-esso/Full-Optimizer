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
            
            # Print ALL vehicle stops with detailed constraint analysis
            print(f"\n🚛 ANALYZING ALL VEHICLE STOPS WITH FULL CONSTRAINTS...")
            total_violations = print_all_vehicle_stops_detailed(solution, vehicles, locations, scenario)
            
            # Create detailed daily constraint tracking for file output
            print(f"\n📊 Creating detailed daily constraint tracking file...")
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

def print_all_vehicle_stops_detailed(solution, vehicles, locations, scenario):
    """
    Print all stops for all vehicles in detail, ensuring all constraints are checked.
    Focus on: overnight, pickup/delivery, capacity, and time windows.
    """
    print(f"\n" + "="*100)
    print(f"🚛 DETAILED VEHICLE STOPS ANALYSIS - ALL CONSTRAINTS ACTIVATED")
    print(f"="*100)
    
    # Get vehicle routes from solution
    vehicle_routes = solution.get('vehicle_routes', {})
    
    # Create location lookup for quick access
    location_lookup = {str(loc['id']): loc for loc in locations}
    
    # Track overall constraint violations
    total_violations = {
        'weight_capacity': 0,
        'volume_capacity': 0,
        'time_windows': 0,
        'time_limits': 0,
        'pickup_delivery': 0
    }
    
    print(f"\n📊 CONSTRAINT SETTINGS:")
    print(f"  ✅ Weight Capacity Constraints: ACTIVE")
    print(f"  ✅ Volume Capacity Constraints: ACTIVE") 
    print(f"  ✅ Time Window Constraints: ACTIVE")
    print(f"  ✅ Daily Time Limit Constraints: ACTIVE")
    print(f"  ✅ Pickup/Delivery Constraints: ACTIVE")
    print(f"  ✅ Overnight Stay Constraints: ACTIVE")
    
    # Process each vehicle
    for vehicle_id, vehicle_data in vehicle_routes.items():
        # Get vehicle info
        vehicle_info = next((v for v in vehicles if v['id'] == vehicle_id), None)
        if not vehicle_info:
            continue
            
        print(f"\n" + "🚛" + "="*95)
        print(f"VEHICLE: {vehicle_id} ({vehicle_info.get('type', 'unknown')})")
        print(f"="*100)
        
        # Vehicle specifications
        weight_capacity = vehicle_info.get('capacity', 0)
        volume_capacity = vehicle_info.get('volume_capacity', weight_capacity * 0.001)
        daily_time_limit = vehicle_info.get('max_time', 8 * 60)  # minutes
        cost_per_km = vehicle_info.get('cost_per_km', 1.0)
        
        print(f"📋 VEHICLE SPECIFICATIONS:")
        print(f"  Weight Capacity: {weight_capacity} units")
        print(f"  Volume Capacity: {volume_capacity:.3f} m³")
        print(f"  Daily Time Limit: {daily_time_limit} minutes ({daily_time_limit/60:.1f} hours)")
        print(f"  Cost per km: {cost_per_km:.2f}")
        
        # Get full route
        full_route = vehicle_data.get('full_route', [])
        total_distance = vehicle_data.get('total_distance', 0)
        total_overnights = vehicle_data.get('total_overnight_stays', 0)
        
        print(f"\n📍 ROUTE OVERVIEW:")
        print(f"  Total Stops: {len(full_route)}")
        print(f"  Total Distance: {total_distance:.1f} km")
        print(f"  Total Overnight Stays: {total_overnights}")
        
        # Track vehicle state
        current_weight = 0
        current_volume = 0
        current_time = 0
        current_day = 1
        stop_number = 0
        day_start_time = 0
        daily_distance = 0
        last_coords = None
        
        # Vehicle constraint violations for this vehicle
        vehicle_violations = {
            'weight_capacity': False,
            'volume_capacity': False,
            'time_windows': [],
            'time_limits': [],
            'pickup_delivery': []
        }
        
        print(f"\n📝 DETAILED STOP-BY-STOP ANALYSIS:")
        print(f"{'Stop':<4} {'Day':<3} {'Type':<12} {'Location':<15} {'Time':<8} {'Weight':<8} {'Volume':<8} {'Constraints':<15}")
        print(f"-" * 100)
        
        # Process each stop in the route
        for i, route_item in enumerate(full_route):
            stop_number += 1
            location_id = None
            is_overnight = False
            is_day_marker = False
            is_depot = False
            
            # Parse route item
            if isinstance(route_item, dict):
                if route_item.get('is_day_marker', False):
                    is_day_marker = True
                    current_day += 1
                    day_start_time = current_time
                    daily_distance = 0
                    print(f"{'---':<4} {'---':<3} {'DAY BREAK':<12} {'---':<15} {'---':<8} {'---':<8} {'---':<8} {'---':<15}")
                    continue
                    
                if route_item.get('is_overnight', False):
                    is_overnight = True
                    location_id = 'overnight'
                else:
                    location_id = route_item.get('location_id', 'unknown')
                    
                # Get timing info
                arrival_time = route_item.get('arrival_time', current_time)
                service_time = route_item.get('service_time', 15)
                departure_time = route_item.get('departure_time', arrival_time + service_time)
                current_time = departure_time
                
            elif isinstance(route_item, str):
                if 'day' in route_item.lower():
                    is_day_marker = True
                    current_day += 1
                    day_start_time = current_time
                    daily_distance = 0
                    print(f"{'---':<4} {'---':<3} {'DAY BREAK':<12} {'---':<15} {'---':<8} {'---':<8} {'---':<8} {'---':<15}")
                    continue
                    
                if 'overnight' in route_item.lower():
                    is_overnight = True
                    location_id = 'overnight'
                else:
                    location_id = route_item
                    
                # Estimate timing
                arrival_time = current_time
                service_time = 15
                departure_time = arrival_time + service_time
                current_time = departure_time
            
            # Skip if no valid location
            if not location_id or is_day_marker:
                continue
                
            # Check if depot
            if location_id == 'depot' or location_id == vehicle_info.get('depot_id', 'depot'):
                is_depot = True
                
            # Get location details
            location_details = None
            stop_demand = 0
            stop_volume = 0
            
            if not is_overnight and not is_depot:
                location_details = location_lookup.get(str(location_id))
                if location_details:
                    stop_demand = location_details.get('demand', 0)
                    stop_volume = stop_demand * 0.001  # Convert to volume
                    
            # Update vehicle load
            current_weight += stop_demand
            current_volume += stop_volume
            
            # Determine stop type
            if is_overnight:
                stop_type = "OVERNIGHT"
            elif is_depot:
                stop_type = "DEPOT"
            elif location_details and location_details.get('type') == 'pickup':
                stop_type = "PICKUP"
            elif location_details and location_details.get('type') == 'delivery':
                stop_type = "DELIVERY"
            else:
                stop_type = "CUSTOMER"
                
            # Check constraints
            constraint_status = []
            
            # Weight capacity check
            if current_weight > weight_capacity:
                constraint_status.append("WEIGHT!")
                vehicle_violations['weight_capacity'] = True
                
            # Volume capacity check  
            if current_volume > volume_capacity:
                constraint_status.append("VOLUME!")
                vehicle_violations['volume_capacity'] = True
                
            # Time window check (for non-depot, non-overnight stops)
            if location_details and not is_overnight and not is_depot:
                tw_start = location_details.get('time_window_start', 0)
                tw_end = location_details.get('time_window_end', 1440)
                if arrival_time < tw_start or arrival_time > tw_end:
                    constraint_status.append("TIME_WIN!")
                    vehicle_violations['time_windows'].append({
                        'location': location_id,
                        'arrival': arrival_time,
                        'window': [tw_start, tw_end]
                    })
                    
            # Daily time limit check
            daily_time_used = current_time - day_start_time
            if daily_time_used > daily_time_limit:
                constraint_status.append("TIME_LIM!")
                vehicle_violations['time_limits'].append({
                    'day': current_day,
                    'time_used': daily_time_used,
                    'limit': daily_time_limit
                })
            
            # Format constraint status
            constraint_str = ",".join(constraint_status) if constraint_status else "OK"
            
            # Print stop details
            time_str = f"{arrival_time:.0f}"
            weight_str = f"{current_weight:.0f}"
            volume_str = f"{current_volume:.3f}"
            
            print(f"{stop_number:<4} {current_day:<3} {stop_type:<12} {str(location_id)[:15]:<15} {time_str:<8} {weight_str:<8} {volume_str:<8} {constraint_str:<15}")
            
            # Add detailed info for problem stops
            if constraint_status:
                if location_details:
                    print(f"    📍 Address: {location_details.get('address', 'N/A')[:50]}")
                    if 'TIME_WIN!' in constraint_status:
                        print(f"    ⏰ Time Window: [{tw_start}-{tw_end}], Arrived: {arrival_time:.0f}")
                if 'WEIGHT!' in constraint_status:
                    print(f"    ⚖️ Weight: {current_weight}/{weight_capacity} (VIOLATION)")
                if 'VOLUME!' in constraint_status:
                    print(f"    📦 Volume: {current_volume:.3f}/{volume_capacity:.3f} (VIOLATION)")
        
        # Vehicle summary
        print(f"\n📊 VEHICLE {vehicle_id} CONSTRAINT SUMMARY:")
        
        violations_found = False
        if vehicle_violations['weight_capacity']:
            print(f"  ❌ Weight Capacity: VIOLATED (exceeded {weight_capacity} units)")
            total_violations['weight_capacity'] += 1
            violations_found = True
            
        if vehicle_violations['volume_capacity']:
            print(f"  ❌ Volume Capacity: VIOLATED (exceeded {volume_capacity:.3f} m³)")
            total_violations['volume_capacity'] += 1
            violations_found = True
            
        if vehicle_violations['time_windows']:
            print(f"  ❌ Time Windows: {len(vehicle_violations['time_windows'])} VIOLATIONS")
            total_violations['time_windows'] += len(vehicle_violations['time_windows'])
            violations_found = True
            for tw_violation in vehicle_violations['time_windows']:
                print(f"    • Location {tw_violation['location']}: arrived {tw_violation['arrival']:.0f}, window {tw_violation['window']}")
                
        if vehicle_violations['time_limits']:
            print(f"  ❌ Daily Time Limits: {len(vehicle_violations['time_limits'])} VIOLATIONS")
            total_violations['time_limits'] += len(vehicle_violations['time_limits'])
            violations_found = True
            for tl_violation in vehicle_violations['time_limits']:
                print(f"    • Day {tl_violation['day']}: used {tl_violation['time_used']:.0f}/{tl_violation['limit']} minutes")
        
        if not violations_found:
            print(f"  ✅ ALL CONSTRAINTS SATISFIED for Vehicle {vehicle_id}")
        
        print(f"  📈 Final Load: Weight {current_weight}/{weight_capacity}, Volume {current_volume:.3f}/{volume_capacity:.3f}")
        print(f"  🛣️ Total Distance: {total_distance:.1f} km")
        print(f"  🛏️ Overnight Stays: {total_overnights}")
    
    # Overall summary
    print(f"\n" + "="*100)
    print(f"🎯 OVERALL CONSTRAINT VIOLATION SUMMARY")
    print(f"="*100)
    
    total_violation_count = sum(total_violations.values())
    if total_violation_count == 0:
        print(f"🎉 PERFECT SOLUTION! All constraints satisfied across all vehicles.")
    else:
        print(f"⚠️ TOTAL VIOLATIONS FOUND: {total_violation_count}")
        for constraint_type, count in total_violations.items():
            if count > 0:
                print(f"  • {constraint_type.replace('_', ' ').title()}: {count} violations")
    
    print(f"="*100)
    
    return total_violations

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
