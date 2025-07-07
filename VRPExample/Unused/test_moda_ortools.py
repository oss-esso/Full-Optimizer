#!/usr/bin/env python3
"""
Quick test script to debug OR-Tools MODA scenarios
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from vrp_scenarios import get_all_scenarios
from vrp_optimizer_fixed import VRPQuantumOptimizer
from vrp_data_models import VRPObjective

def test_moda_scenarios():
    """Test MODA scenarios specifically with OR-Tools - comprehensive analysis"""
    print("🚀 TESTING MODA SCENARIOS WITH 10-HOUR TIME LIMITS")
    print("=" * 80)
    
    # Get scenarios
    scenarios = get_all_scenarios()
    moda_scenarios = {k: v for k, v in scenarios.items() if 'MODA' in k}
    
    for name, instance in moda_scenarios.items():
        print(f'\n🎯 TESTING {name.upper()}')
        print("-" * 60)
        
        # === SCENARIO OVERVIEW ===
        print(f'📊 Scenario Overview:')
        print(f'  Locations: {len(instance.locations)}')
        print(f'  Vehicles: {len(instance.vehicles)}')
        print(f'  Ride requests: {len(instance.ride_requests) if instance.ride_requests else 0}')
        
        # === VEHICLE TIME LIMITS ANALYSIS ===
        print(f'\n⏰ Vehicle Time Limits Analysis:')
        vehicle_times = [v.max_time for v in instance.vehicles.values()]
        unique_times = set(vehicle_times)
        print(f'  Time limits: {unique_times} minutes')
        
        # Convert to hours for clarity
        time_hours = {f"{t/60:.1f}h" for t in unique_times if t is not None}
        if time_hours:
            print(f'  Time limits: {time_hours}')
            
        # Check if using 10-hour limits as intended
        if 600 in unique_times:
            print(f'  ✅ Confirmed: Using 10-hour (600 min) time limits')
        elif 1080 in unique_times:
            print(f'  ⚠️  Still using 18-hour (1080 min) time limits - needs update')
        else:
            print(f'  ❓ Unexpected time limits: {unique_times}')
        
        # === CAPACITY ANALYSIS ===
        print(f'\n📦 Capacity Analysis:')
        if instance.ride_requests:
            total_demand = sum(req.passengers for req in instance.ride_requests)
            total_capacity = sum(vehicle.capacity for vehicle in instance.vehicles.values())
            capacity_ratio = total_demand / total_capacity if total_capacity > 0 else float('inf')
            
            print(f'  Total demand: {total_demand} passengers')
            print(f'  Total capacity: {total_capacity} seats')
            print(f'  Capacity utilization: {capacity_ratio:.1%}')
            
            if capacity_ratio > 1.0:
                print(f'  ⚠️  Over-capacity! May cause infeasibility')
            elif capacity_ratio > 0.9:
                print(f'  ⚠️  High utilization - may be challenging')
            else:
                print(f'  ✅ Reasonable capacity utilization')
        
        # === TIME WINDOW ANALYSIS ===
        print(f'\n⏰ Time Window Analysis:')
        locations_with_tw = sum(1 for loc in instance.locations.values() 
                               if hasattr(loc, 'time_window_start') and loc.time_window_start is not None)
        print(f'  Locations with time windows: {locations_with_tw}/{len(instance.locations)}')
        
        if locations_with_tw > 0:
            # Analyze time window span
            tw_starts = [loc.time_window_start for loc in instance.locations.values() 
                        if hasattr(loc, 'time_window_start') and loc.time_window_start is not None]
            tw_ends = [loc.time_window_end for loc in instance.locations.values() 
                      if hasattr(loc, 'time_window_end') and loc.time_window_end is not None]
            
            if tw_starts and tw_ends:
                earliest_start = min(tw_starts)
                latest_end = max(tw_ends)
                total_span = latest_end - earliest_start
                avg_window_size = sum(end - start for start, end in zip(tw_starts, tw_ends)) / len(tw_starts)
                
                print(f'  Time span: {earliest_start} to {latest_end} minutes ({total_span/60:.1f} hours)')
                print(f'  Average window size: {avg_window_size:.0f} minutes ({avg_window_size/60:.1f} hours)')
                
                # Check compatibility with vehicle time limits
                vehicle_time_limit = next(iter(unique_times)) if unique_times else None
                if vehicle_time_limit and total_span > vehicle_time_limit:
                    print(f'  ⚠️  Time span ({total_span:.0f}min) > vehicle limit ({vehicle_time_limit:.0f}min)')
                else:
                    print(f'  ✅ Time span compatible with vehicle limits')
        
        # === SAMPLE REQUESTS ===
        if instance.ride_requests:
            print(f'\n📋 Sample Ride Requests:')
            for i, req in enumerate(list(instance.ride_requests)[:3]):
                print(f'  Request {i+1}: {req.pickup_location} -> {req.dropoff_location} ({req.passengers} passengers)')
        
        # === OR-TOOLS OPTIMIZATION ===
        print(f'\n🔧 Running OR-Tools Optimization...')
        optimizer = VRPQuantumOptimizer(instance, VRPObjective.MINIMIZE_DISTANCE)
        
        try:
            import time
            start_time = time.time()
            result = optimizer.optimize_with_ortools()
            solve_time = time.time() - start_time
            
            # Check if solved
            solved = result.routes is not None and len(result.routes) > 0
            if solved:
                vehicles_used = len([r for r in result.routes.values() if len(r) > 2])
                solved = vehicles_used > 0
            else:
                vehicles_used = 0
            
            print(f'  Result: {"✅ SOLVED" if solved else "❌ NOT SOLVED"}')
            print(f'  Solve time: {solve_time:.3f}s')
            
            if solved:
                print(f'  Objective value: {result.objective_value:.2f}')
                print(f'  Distance: {result.metrics.get("total_distance", 0):.2f}')
                print(f'  Runtime: {result.runtime:.2f}ms')
                print(f'  Vehicles used: {vehicles_used}/{len(instance.vehicles)}')
                
                # === ROUTE ANALYSIS ===
                print(f'\n🛣️  Route Analysis:')
                total_stops = sum(len(route) for route in result.routes.values())
                print(f'  Total stops across all routes: {total_stops}')
                
                # Detailed route breakdown
                active_routes = [(vid, route) for vid, route in result.routes.items() if len(route) > 2]
                print(f'  Active routes: {len(active_routes)}')
                
                # Check for unreasonable routes
                long_routes = [(vid, len(route)) for vid, route in active_routes if len(route) > 20]
                if long_routes:
                    print(f'  ⚠️  Long routes detected:')
                    for vid, length in long_routes[:3]:  # Show first 3
                        print(f'    {vid}: {length} stops')
                        
                        # Show route preview for long routes
                        route = result.routes[vid]
                        pickups = [stop for stop in route if 'pickup' in stop]
                        dropoffs = [stop for stop in route if 'dropoff' in stop]
                        deliveries = [stop for stop in route if 'delivery' in stop]
                        
                        print(f'      Pickups: {len(pickups)}, Dropoffs: {len(dropoffs)}, Deliveries: {len(deliveries)}')
                        print(f'      Preview: {" -> ".join(route[:5])} ... {" -> ".join(route[-3:])}')
                    
                    if len(long_routes) > 3:
                        print(f'    ... and {len(long_routes)-3} more long routes')
                else:
                    print(f'  ✅ All route lengths appear reasonable')
                
                # === CONSTRAINT VALIDATION ===
                print(f'\n🔍 Constraint Validation:')
                
                # Pickup-delivery precedence constraints
                if instance.ride_requests:
                    violations = validate_pickup_delivery_constraints(result.routes, instance.ride_requests)
                    if violations > 0:
                        print(f'  ⚠️  Pickup-delivery violations: {violations}')
                    else:
                        print(f'  ✅ All pickup-delivery constraints satisfied')
                
                # Time window constraint validation (simplified check)
                time_violations = validate_time_constraints(result.routes, instance.locations, unique_times)
                if time_violations > 0:
                    print(f'  ⚠️  Potential time constraint violations: {time_violations}')
                else:
                    print(f'  ✅ Time constraints appear satisfied')
                
                # Plot the solution if it has GPS coordinates
                plot_solution(instance, result, name)
            else:
                print(f'  ❌ Optimization failed')
                if hasattr(result, 'status'):
                    print(f'  Status: {result.status}')
                    
        except Exception as e:
            print(f'  ❌ ERROR: {str(e)}')
            import traceback
            traceback.print_exc()
        
        print(f'\n' + "="*80)

def plot_solution(instance, result, scenario_name):
    """Plot the OR-Tools solution path"""
    # Check if locations have GPS coordinates
    has_gps = hasattr(instance.locations[list(instance.locations.keys())[0]], 'lat')
    
    if not has_gps:
        print(f'  No GPS coordinates available for {scenario_name}, skipping plot')
        return
    
    print(f'  Creating route visualization for {scenario_name}...')
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Color palette for different routes
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot all locations first
    depot_lats, depot_lons = [], []
    pickup_lats, pickup_lons = [], []
    dropoff_lats, dropoff_lons = [], []
    
    for loc_id, loc in instance.locations.items():
        if 'depot' in loc_id:
            depot_lats.append(loc.lat)
            depot_lons.append(loc.lon)
        elif 'pickup' in loc_id:
            pickup_lats.append(loc.lat)
            pickup_lons.append(loc.lon)
        elif 'dropoff' in loc_id:
            dropoff_lats.append(loc.lat)
            dropoff_lons.append(loc.lon)
    
    # Plot locations
    if depot_lats:
        ax.scatter(depot_lons, depot_lats, c='black', s=200, marker='s', label='Depots', zorder=5)
    if pickup_lats:
        ax.scatter(pickup_lons, pickup_lats, c='green', s=100, marker='^', label='Pickups', alpha=0.7, zorder=3)
    if dropoff_lats:
        ax.scatter(dropoff_lons, dropoff_lats, c='red', s=100, marker='v', label='Dropoffs', alpha=0.7, zorder=3)
    
    # Plot routes
    route_count = 0
    for vehicle_id, route in result.routes.items():
        if len(route) <= 2:  # Skip empty routes (just depot start/end)
            continue
            
        color = colors[route_count % len(colors)]
        route_lats, route_lons = [], []
        
        # Get coordinates for each location in the route
        for loc_id in route:
            if loc_id in instance.locations:
                loc = instance.locations[loc_id]
                route_lats.append(loc.lat)
                route_lons.append(loc.lon)
        
        if len(route_lats) > 1:
            # Plot route line
            ax.plot(route_lons, route_lats, color=color, linewidth=2, alpha=0.8, 
                   label=f'{vehicle_id} (stops: {len(route)-2})', zorder=2)
            
            # Add arrows to show direction
            for i in range(len(route_lons)-1):
                dx = route_lons[i+1] - route_lons[i]
                dy = route_lats[i+1] - route_lats[i]
                if abs(dx) > 0.001 or abs(dy) > 0.001:  # Only draw arrow if distance is significant
                    ax.annotate('', xy=(route_lons[i+1], route_lats[i+1]), 
                               xytext=(route_lons[i], route_lats[i]),
                               arrowprops=dict(arrowstyle='->', color=color, alpha=0.6, lw=1))
            
            route_count += 1
    
    # Customize plot
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'OR-Tools Solution - {scenario_name}\n'
                f'Distance: {result.metrics.get("total_distance", 0):.2f}, '
                f'Runtime: {result.runtime:.0f}ms, '
                f'Vehicles Used: {len([r for r in result.routes.values() if len(r) > 2])}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio for better visualization
    ax.set_aspect('equal', adjustable='box')
    
    # Tight layout
    plt.tight_layout()
    
    
    # Show plot
    plt.show()
    
    # Close figure to free memory
    plt.close()

def test_vehicle_capacities():
    """Test what vehicle capacities are set for MODA scenarios"""
    print("Testing vehicle capacities...")
    print("=" * 50)
    
    scenarios = get_all_scenarios()
    
    for name, instance in scenarios.items():
        if 'MODA' not in name:
            continue
            
        print(f'\n{name} vehicle capacities:')
        for vehicle_id, vehicle in list(instance.vehicles.items())[:5]:  # First 5 vehicles
            capacity = getattr(vehicle, 'capacity', 'NOT SET')
            print(f'  {vehicle_id}: {capacity}')
        
        print(f'  Total vehicles: {len(instance.vehicles)}')
        
        # Check ride request passenger counts
        if instance.ride_requests:
            passenger_counts = [request.passengers for request in list(instance.ride_requests)[:5]]
            print(f'  Sample passenger counts: {passenger_counts}')
            total_passengers = sum(request.passengers for request in instance.ride_requests)
            print(f'  Total passengers across all requests: {total_passengers}')

def validate_pickup_delivery_constraints(routes, ride_requests):
    """Validate that pickups come before dropoffs for each request."""
    violations = 0
    
    for vehicle_id, route in routes.items():
        if len(route) <= 2:
            continue
        
        # Check if pickups come before dropoffs for each request
        for request in ride_requests:
            pickup_pos = None
            dropoff_pos = None
            
            for i, stop in enumerate(route):
                if stop == request.pickup_location:
                    pickup_pos = i
                elif stop == request.dropoff_location:
                    dropoff_pos = i
            
            if pickup_pos is not None and dropoff_pos is not None:
                if pickup_pos >= dropoff_pos:
                    violations += 1
    
    return violations

def validate_time_constraints(routes, locations, vehicle_time_limits):
    """Basic validation of time constraints (simplified)."""
    violations = 0
    
    for vehicle_id, route in routes.items():
        if len(route) <= 2:
            continue
        
        # Estimate total route time (simplified - just count stops and distances)
        estimated_time = len(route) * 15  # Assume 15 minutes per stop including service time
        
        # Add travel time estimate (simplified)
        for i in range(len(route) - 1):
            # Simplified distance calculation - would need actual distance matrix
            estimated_time += 10  # Assume 10 minutes between stops
        
        # Check against vehicle time limit
        max_time = next(iter(vehicle_time_limits)) if vehicle_time_limits else 600
        if max_time and estimated_time > max_time:
            violations += 1
    
    return violations

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE MODA SCENARIO TESTING")
    print("=" * 80)
    print("Testing vehicle capacities and time constraints...")
    print()
    
    test_vehicle_capacities()
    test_moda_scenarios()
    
    print("\n💡 SUMMARY:")
    print("✅ Both MODA scenarios should now be using 10-hour (600 min) vehicle time limits")
    print("✅ Comprehensive analysis includes capacity, time windows, and constraint validation")
    print("✅ This addresses the requirement for 10 hours per vehicle in the large scenario")
    print("🔧 Any constraint violations are reported for further investigation")
    print("📊 Route analysis helps identify unreasonable solutions")
    print()
    print("🎯 Key metrics to watch:")
    print("  - Vehicle time limits: Should be 600 minutes (10 hours)")
    print("  - Capacity utilization: Should be < 100% for feasibility")
    print("  - Route lengths: Should be reasonable (< 50 stops per vehicle)")
    print("  - Constraint violations: Should be 0 for valid solutions")
