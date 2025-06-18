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
    """Test MODA scenarios specifically with OR-Tools"""
    print("Testing MODA scenarios with OR-Tools")
    print("=" * 50)
    
    # Get scenarios
    scenarios = get_all_scenarios()
    moda_scenarios = {k: v for k, v in scenarios.items() if 'MODA' in k}
    
    for name, instance in moda_scenarios.items():
        print(f'\nTesting {name}:')
        print(f'  Locations: {len(instance.locations)}')
        print(f'  Vehicles: {len(instance.vehicles)}')
        print(f'  Ride requests: {len(instance.ride_requests) if instance.ride_requests else 0}')
        
        # Print some ride requests to understand structure
        if instance.ride_requests:
            print(f'  Sample ride requests:')
            for i, req in enumerate(list(instance.ride_requests)[:3]):
                print(f'    Request {i+1}: {req.pickup_location} -> {req.dropoff_location}')
        
        optimizer = VRPQuantumOptimizer(instance, VRPObjective.MINIMIZE_DISTANCE)
        
        try:
            print('  Running OR-Tools...')
            result = optimizer.optimize_with_ortools()
            
            print(f'  OR-Tools result:')
            print(f'    Objective: {result.objective_value:.2f}')
            print(f'    Distance: {result.metrics.get("total_distance", 0):.2f}')
            print(f'    Runtime: {result.runtime:.2f}ms')
            
            # Print route details with analysis
            print(f'    Routes: {len(result.routes)} vehicles used')
            
            total_stops = 0
            for vehicle_id, route in result.routes.items():
                route_length = len(route)
                total_stops += route_length
                
                if len(route) > 2:  # More than just depot start/end
                    print(f'      {vehicle_id}: {route_length} stops')
                    
                    # Flag unreasonable routes
                    if route_length > 50:
                        print(f'        WARNING: {vehicle_id} has {route_length} stops - this is unreasonable!')
                        print(f'        Route preview: {" -> ".join(route[:10])} ... {" -> ".join(route[-5:])}')
                        
                        # Analyze pickup/delivery balance
                        pickups = [stop for stop in route if 'pickup' in stop]
                        dropoffs = [stop for stop in route if 'dropoff' in stop]
                        print(f'        Pickups: {len(pickups)}, Dropoffs: {len(dropoffs)}')
                    else:
                        if len(route) > 5:
                            route_str = " -> ".join(route[:5]) + f"... ({route_length} total)"
                        else:
                            route_str = " -> ".join(route)
                        print(f'        Route: {route_str}')
            
            print(f'    Total stops across all vehicles: {total_stops}')
            
            # Analyze pickup-delivery constraint violations
            if instance.ride_requests:
                print(f'    Analyzing pickup-delivery constraints...')
                violations = 0
                for vehicle_id, route in result.routes.items():
                    if len(route) <= 2:
                        continue
                    
                    # Check if pickups come before dropoffs for each request
                    for request in instance.ride_requests:
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
                                print(f'        VIOLATION: {request.pickup_location} at pos {pickup_pos} >= {request.dropoff_location} at pos {dropoff_pos}')
                
                print(f'    Pickup-delivery violations: {violations}')
            
            # Plot the solution if it has GPS coordinates
            plot_solution(instance, result, name)
                    
        except Exception as e:
            print(f'  ERROR: {str(e)}')
            import traceback
            traceback.print_exc()
          # Test both MODA scenarios
        if 'first' in name:
            continue  # Skip the large one for now to test the small one

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

if __name__ == "__main__":
    test_vehicle_capacities()
    test_moda_scenarios()
