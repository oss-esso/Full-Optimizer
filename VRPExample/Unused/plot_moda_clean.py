#!/usr/bin/env python3
"""
Simple MODA Scenario Comparison - Clean Overview Plots
Shows both MODA_first and MODA_small scenarios side by side without solving
"""

import matplotlib.pyplot as plt
import numpy as np
from vrp_scenarios import create_moda_first_scenario, create_moda_small_scenario
import os


def plot_scenario_clean(instance, ax, title):
    """Plot clean scenario overview showing locations and fleet"""
    
    # Check if locations have GPS coordinates
    sample_loc = list(instance.locations.values())[0]
    has_gps = hasattr(sample_loc, 'lat') and sample_loc.lat is not None
    
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
        ax.scatter(pickup_lons, pickup_lats, c='green', s=100, marker='^', 
                  label=f'Pickups ({len(pickup_lats)})', alpha=0.8, zorder=3, edgecolors='darkgreen')
    if dropoff_lats:
        ax.scatter(dropoff_lons, dropoff_lats, c='red', s=100, marker='v', 
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


def create_side_by_side_comparison():
    """Create clean side-by-side comparison of both MODA scenarios"""
    
    print("🎨 Creating side-by-side MODA scenario comparison...")
    print("=" * 60)
    
    # Create scenarios
    print("📊 Generating MODA scenarios...")
    moda_small = create_moda_small_scenario()
    moda_first = create_moda_first_scenario()
    
    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot both scenarios
    plot_scenario_clean(moda_small, ax1, "MODA Small Scenario")
    plot_scenario_clean(moda_first, ax2, "MODA First Scenario") 
    
    # Overall title
    fig.suptitle('MODA VRP Scenarios Comparison - Northern Italy Trucking Fleet\n'
                'Mixed Fleet with Driver Regulations & Service Areas for Break Planning', 
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Create output directory
    os.makedirs('Plots for PPT', exist_ok=True)
    
    # Save plot
    output_path = 'Plots for PPT/moda_side_by_side_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"💾 Plot saved as: {output_path}")
    print("🖼️ Displaying plot...")
    
    # Show plot
    plt.show()


def create_location_density_comparison():
    """Create a comparison showing location density differences"""
    
    print("\n🌍 Creating location density comparison...")
    
    # Create scenarios
    moda_small = create_moda_small_scenario()
    moda_first = create_moda_first_scenario()
    
    # Create figure with different visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top row: Location scatter plots
    plot_scenario_clean(moda_small, ax1, "MODA Small - Location Overview")
    plot_scenario_clean(moda_first, ax2, "MODA First - Location Overview")
    
    # Bottom row: Statistics comparison
    scenarios = ['MODA Small', 'MODA First']
    
    # Vehicle counts
    small_heavy = len([v for v in moda_small.vehicles.values() if getattr(v, 'vehicle_type', '') == 'heavy'])
    small_standard = len([v for v in moda_small.vehicles.values() if getattr(v, 'vehicle_type', '') == 'standard'])
    first_heavy = len([v for v in moda_first.vehicles.values() if getattr(v, 'vehicle_type', '') == 'heavy'])
    first_standard = len([v for v in moda_first.vehicles.values() if getattr(v, 'vehicle_type', '') == 'standard'])
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    ax3.bar(x - width/2, [small_heavy, first_heavy], width, label='24-ton Heavy Trucks', color='darkred', alpha=0.8)
    ax3.bar(x + width/2, [small_standard, first_standard], width, label='4-ton Standard Trucks', color='darkblue', alpha=0.8)
    ax3.set_xlabel('Scenario')
    ax3.set_ylabel('Number of Vehicles')
    ax3.set_title('Fleet Composition')
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenarios)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Demand comparison
    small_demand = sum(req.passengers for req in moda_small.ride_requests) / 1000  # Convert to tons
    first_demand = sum(req.passengers for req in moda_first.ride_requests) / 1000
    small_capacity = sum(vehicle.capacity for vehicle in moda_small.vehicles.values()) / 1000
    first_capacity = sum(vehicle.capacity for vehicle in moda_first.vehicles.values()) / 1000
    
    ax4.bar(x - width/2, [small_demand, first_demand], width, label='Total Demand (tons)', color='orange', alpha=0.8)
    ax4.bar(x + width/2, [small_capacity, first_capacity], width, label='Total Capacity (tons)', color='green', alpha=0.8)
    ax4.set_xlabel('Scenario')
    ax4.set_ylabel('Weight (tons)')
    ax4.set_title('Demand vs Fleet Capacity')
    ax4.set_xticks(x)
    ax4.set_xticklabels(scenarios)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('MODA Scenarios - Detailed Comparison\nLocations, Fleet Composition & Capacity Analysis', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save detailed comparison
    detailed_path = 'Plots for PPT/moda_detailed_comparison.png'
    plt.savefig(detailed_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Detailed comparison saved as: {detailed_path}")
    plt.show()


if __name__ == "__main__":
    print("🚛 MODA VRP Scenarios - Clean Comparison Tool")
    print("=" * 60)
    
    try:
        # Create clean comparison plots
        create_side_by_side_comparison()
        create_location_density_comparison()
        
        print("\n✅ All clean visualizations completed successfully!")
        print("📁 Check the 'Plots for PPT' directory for saved images")
        print("\n📊 Generated files:")
        print("  - moda_side_by_side_comparison.png (clean side-by-side)")
        print("  - moda_detailed_comparison.png (with statistics)")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
