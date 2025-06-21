#!/usr/bin/env python3
"""
MODA Scenario Visualization Tool
Creates side-by-side plots of MODA_first (large) and MODA_small scenarios
"""

import matplotlib.pyplot as plt
import numpy as np
from vrp_scenarios import create_moda_first_scenario, create_moda_small_scenario
from vrp_optimizer_fixed import VRPQuantumOptimizer
import matplotlib.patches as patches
from typing import Dict, List, Tuple
import os


def plot_scenario_overview(instance, ax, title):
    """Plot scenario overview without solving - just show locations and fleet"""
    
    # Check if locations have GPS coordinates
    has_gps = hasattr(instance.locations[list(instance.locations.keys())[0]], 'lat')
    
    if not has_gps:
        ax.text(0.5, 0.5, f'No GPS coordinates\navailable for {title}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        return
    
    # Separate locations by type
    depot_lats, depot_lons = [], []
    pickup_lats, pickup_lons = [], []
    dropoff_lats, dropoff_lons = [], []
    service_lats, service_lons = [], []
    
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
        elif any(word in loc_id for word in ['service', 'area', 'stop']):
            service_lats.append(loc.lat)
            service_lons.append(loc.lon)
    
    # Plot locations with different colors and shapes
    if depot_lats:
        ax.scatter(depot_lons, depot_lats, c='black', s=300, marker='s', 
                  label=f'Depots ({len(depot_lats)})', zorder=5, edgecolors='white', linewidth=2)
    if pickup_lats:
        ax.scatter(pickup_lons, pickup_lats, c='green', s=80, marker='^', 
                  label=f'Pickups ({len(pickup_lats)})', alpha=0.8, zorder=3, edgecolors='darkgreen')
    if dropoff_lats:
        ax.scatter(dropoff_lons, dropoff_lats, c='red', s=80, marker='v', 
                  label=f'Dropoffs ({len(dropoff_lats)})', alpha=0.8, zorder=3, edgecolors='darkred')
    if service_lats:
        ax.scatter(service_lons, service_lats, c='orange', s=60, marker='o', 
                  label=f'Service Areas ({len(service_lats)})', alpha=0.6, zorder=2)
      # Calculate scenario statistics
    total_demand = sum(req.passengers for req in instance.ride_requests)
    total_capacity = sum(vehicle.capacity for vehicle in instance.vehicles.values())
    heavy_trucks = len([v for v in instance.vehicles.values() if getattr(v, 'vehicle_type', '') == 'heavy'])
    standard_trucks = len([v for v in instance.vehicles.values() if getattr(v, 'vehicle_type', '') == 'standard'])
    
    # Customize plot
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'{title}\n'
                f'Fleet: {heavy_trucks}×24t + {standard_trucks}×4t trucks\n'
                f'Demand: {total_demand:,}kg | Capacity: {total_capacity:,}kg\n'
                f'Locations: {len(instance.locations)} | Requests: {len(instance.ride_requests)}')
    
    # Legend
    ax.legend(loc='upper left', bbox_to_anchor=(0, 0.95), fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')


def plot_solved_scenario(instance, result, ax, title):
    """Plot solved scenario with routes"""
    
    # Check if locations have GPS coordinates
    has_gps = hasattr(instance.locations[list(instance.locations.keys())[0]], 'lat')
    
    if not has_gps:
        ax.text(0.5, 0.5, f'No GPS coordinates\navailable for {title}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        return
    
    # Color palette for different routes
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
              'magenta', 'yellow', 'darkred', 'darkblue', 'darkgreen', 'darkorange', 'darkviolet']
    
    # Plot all locations first
    depot_lats, depot_lons = [], []
    pickup_lats, pickup_lons = [], []
    dropoff_lats, dropoff_lons = [], []
    service_lats, service_lons = [], []
    
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
        elif any(word in loc_id for word in ['service', 'area', 'stop']):
            service_lats.append(loc.lat)
            service_lons.append(loc.lon)
    
    # Plot locations
    if depot_lats:
        ax.scatter(depot_lons, depot_lats, c='black', s=200, marker='s', 
                  label='Depots', zorder=5, edgecolors='white', linewidth=2)
    if pickup_lats:
        ax.scatter(pickup_lons, pickup_lats, c='green', s=60, marker='^', 
                  label='Pickups', alpha=0.7, zorder=3)
    if dropoff_lats:
        ax.scatter(dropoff_lons, dropoff_lats, c='red', s=60, marker='v', 
                  label='Dropoffs', alpha=0.7, zorder=3)
    if service_lats:
        ax.scatter(service_lons, service_lats, c='orange', s=40, marker='o', 
                  label='Service Areas', alpha=0.5, zorder=2)
    
    # Plot routes if result contains routes
    if hasattr(result, 'routes') and result.routes:
        route_count = 0
        active_vehicles = 0
        total_stops = 0
        
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
                       label=f'{vehicle_id} ({len(route)-2} stops)', zorder=4)
                
                # Add arrows to show direction (only for first few routes to avoid clutter)
                if route_count < 5:
                    for i in range(0, len(route_lons)-1, max(1, len(route_lons)//3)):
                        dx = route_lons[i+1] - route_lons[i]
                        dy = route_lats[i+1] - route_lats[i]
                        if abs(dx) > 0.001 or abs(dy) > 0.001:
                            ax.annotate('', xy=(route_lons[i+1], route_lats[i+1]), 
                                       xytext=(route_lons[i], route_lats[i]),
                                       arrowprops=dict(arrowstyle='->', color=color, alpha=0.6, lw=1.5))
                
                route_count += 1
                active_vehicles += 1
                total_stops += len(route) - 2  # Exclude depot start/end
        
        # Customize plot for solved scenario
        distance = getattr(result, 'objective_value', getattr(result.metrics, 'total_distance', 0) if hasattr(result, 'metrics') else 0)
        runtime = getattr(result, 'runtime', 0)
        
        ax.set_title(f'{title} - Solution\n'
                    f'Vehicles Used: {active_vehicles}/{len(instance.vehicles)} | '
                    f'Total Stops: {total_stops}\n'
                    f'Distance: {distance:.2f} | Runtime: {runtime:.0f}ms')
        
        # Show only first few routes in legend to avoid clutter
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 10:
            ax.legend(handles[:10], labels[:10], loc='upper left', bbox_to_anchor=(0, 0.95), fontsize=8)
        else:
            ax.legend(loc='upper left', bbox_to_anchor=(0, 0.95), fontsize=8)
    else:
        ax.set_title(f'{title} - Locations Only')
        ax.legend(loc='upper left', bbox_to_anchor=(0, 0.95), fontsize=9)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')


def create_comprehensive_plots():
    """Create comprehensive comparison plots of MODA scenarios"""
    
    print("🎨 Creating comprehensive MODA scenario visualizations...")
    print("=" * 60)
    
    # Create scenarios
    print("📊 Generating MODA scenarios...")
    moda_small = create_moda_small_scenario()
    moda_first = create_moda_first_scenario()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Overview plots (top row)
    ax1 = plt.subplot(2, 2, 1)
    plot_scenario_overview(moda_small, ax1, "MODA Small Scenario")
    
    ax2 = plt.subplot(2, 2, 2)
    plot_scenario_overview(moda_first, ax2, "MODA First Scenario")
    
    # Try to solve scenarios and plot solutions (bottom row)
    print("🔧 Attempting to solve scenarios with OR-Tools...")
      # Solve MODA_small
    ax3 = plt.subplot(2, 2, 3)
    try:
        optimizer = VRPQuantumOptimizer()
        result_small = optimizer.solve(moda_small, solver_type="ortools", time_limit=30)
        if result_small and hasattr(result_small, 'routes'):
            plot_solved_scenario(moda_small, result_small, ax3, "MODA Small - Solved")
            print("✅ MODA_small solved successfully")
        else:
            plot_scenario_overview(moda_small, ax3, "MODA Small - Could not solve")
            print("⚠️ MODA_small could not be solved")
    except Exception as e:
        plot_scenario_overview(moda_small, ax3, "MODA Small - Solve Error")
        print(f"❌ Error solving MODA_small: {str(e)}")
    
    # Solve MODA_first (with shorter time limit)
    ax4 = plt.subplot(2, 2, 4)
    try:
        optimizer = VRPQuantumOptimizer()
        result_first = optimizer.solve(moda_first, solver_type="ortools", time_limit=10)
        if result_first and hasattr(result_first, 'routes'):
            plot_solved_scenario(moda_first, result_first, ax4, "MODA First - Solved")
            print("✅ MODA_first solved successfully")
        else:
            plot_scenario_overview(moda_first, ax4, "MODA First - Could not solve")
            print("⚠️ MODA_first could not be solved (expected due to size)")
    except Exception as e:
        plot_scenario_overview(moda_first, ax4, "MODA First - Solve Error")
        print(f"⚠️ Error solving MODA_first: {str(e)} (expected for large scenario)")
    
    # Overall title and layout
    fig.suptitle('MODA VRP Scenarios Comparison\n'
                'Northern Italy Trucking Fleet with Driver Regulations & Service Areas', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Create output directory
    os.makedirs('Plots for PPT', exist_ok=True)
    
    # Save plot
    output_path = 'Plots for PPT/moda_scenarios_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"💾 Plot saved as: {output_path}")
    print("🖼️ Displaying plot...")
    
    # Show plot
    plt.show()
    
    # Also create individual plots for better detail
    create_individual_plots(moda_small, moda_first)


def create_individual_plots(moda_small, moda_first):
    """Create individual detailed plots for each scenario"""
    
    print("\n🎨 Creating individual detailed plots...")
    
    # MODA Small detailed plot
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 10))
    plot_scenario_overview(moda_small, ax1, "MODA Small Scenario - Detailed View")
    plt.tight_layout()
    
    # Save individual plot
    small_path = 'Plots for PPT/moda_small_detailed.png'
    plt.savefig(small_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 MODA Small detailed plot saved as: {small_path}")
    plt.show()
    
    # MODA First detailed plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))
    plot_scenario_overview(moda_first, ax2, "MODA First Scenario - Detailed View")
    plt.tight_layout()
    
    # Save individual plot
    first_path = 'Plots for PPT/moda_first_detailed.png'
    plt.savefig(first_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 MODA First detailed plot saved as: {first_path}")
    plt.show()


def create_fleet_comparison_chart():
    """Create a chart comparing fleet compositions between scenarios"""
    
    print("\n📊 Creating fleet composition comparison...")
    
    # Create scenarios
    moda_small = create_moda_small_scenario()
    moda_first = create_moda_first_scenario()
    
    # Extract fleet data
    scenarios = ['MODA Small', 'MODA First']
    heavy_trucks = [
        len([v for v in moda_small.vehicles.values() if getattr(v, 'vehicle_type', '') == 'heavy']),
        len([v for v in moda_first.vehicles.values() if getattr(v, 'vehicle_type', '') == 'heavy'])
    ]
    standard_trucks = [
        len([v for v in moda_small.vehicles.values() if getattr(v, 'vehicle_type', '') == 'standard']),
        len([v for v in moda_first.vehicles.values() if getattr(v, 'vehicle_type', '') == 'standard'])
    ]
      # Extract other statistics
    total_demand = [
        sum(req.passengers for req in moda_small.ride_requests),
        sum(req.passengers for req in moda_first.ride_requests)
    ]
    total_capacity = [
        sum(vehicle.capacity for vehicle in moda_small.vehicles.values()),
        sum(vehicle.capacity for vehicle in moda_first.vehicles.values())
    ]
    total_locations = [len(moda_small.locations), len(moda_first.locations)]
    
    # Create comparison charts
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Fleet composition chart
    x = np.arange(len(scenarios))
    width = 0.35
    
    ax1.bar(x - width/2, heavy_trucks, width, label='24-ton Heavy Trucks', color='darkred', alpha=0.8)
    ax1.bar(x + width/2, standard_trucks, width, label='4-ton Standard Trucks', color='darkblue', alpha=0.8)
    ax1.set_xlabel('Scenario')
    ax1.set_ylabel('Number of Vehicles')
    ax1.set_title('Fleet Composition Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Demand vs capacity chart
    ax2.bar(x - width/2, [d/1000 for d in total_demand], width, label='Total Demand (tons)', color='orange', alpha=0.8)
    ax2.bar(x + width/2, [c/1000 for c in total_capacity], width, label='Total Capacity (tons)', color='green', alpha=0.8)
    ax2.set_xlabel('Scenario')
    ax2.set_ylabel('Weight (tons)')
    ax2.set_title('Demand vs Fleet Capacity')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Location count chart
    ax3.bar(scenarios, total_locations, color=['lightcoral', 'lightsteelblue'], alpha=0.8)
    ax3.set_xlabel('Scenario')
    ax3.set_ylabel('Number of Locations')
    ax3.set_title('Total Locations (including service areas)')
    ax3.grid(axis='y', alpha=0.3)
    
    # Utilization percentage chart
    utilization = [(d/c)*100 for d, c in zip(total_demand, total_capacity)]
    ax4.bar(scenarios, utilization, color=['gold', 'lightgreen'], alpha=0.8)
    ax4.set_xlabel('Scenario')
    ax4.set_ylabel('Utilization (%)')
    ax4.set_title('Fleet Capacity Utilization')
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100% Capacity')
    ax4.legend()
    
    plt.suptitle('MODA Scenarios - Fleet & Operations Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save comparison chart
    comparison_path = 'Plots for PPT/moda_fleet_comparison.png'
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Fleet comparison chart saved as: {comparison_path}")
    plt.show()


if __name__ == "__main__":
    print("🚛 MODA VRP Scenario Visualization Tool")
    print("=" * 60)
    
    try:
        # Import required libraries
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create all visualizations
        create_comprehensive_plots()
        create_fleet_comparison_chart()
        
        print("\n✅ All visualizations completed successfully!")
        print("📁 Check the 'Plots for PPT' directory for saved images")
        
    except ImportError as e:
        print(f"❌ Missing required libraries: {e}")
        print("Please install missing packages: pip install matplotlib numpy")
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
