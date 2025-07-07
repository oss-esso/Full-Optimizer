#!/usr/bin/env python3
"""
Clean MODA Scenario Visualization Tool
Creates side-by-side comparison plots of MODA_first and MODA_small scenarios
without attempting to solve them.
"""

import matplotlib.pyplot as plt
import numpy as np
from vrp_scenarios import create_moda_small_scenario, create_moda_first_scenario
import os

def plot_scenario_locations(instance, ax, title, max_points=None):
    """Plot scenario locations with different colors for different types."""
    
    # Extract location data
    depots = []
    pickups = []
    dropoffs = []
    service_areas = []
    
    for loc_id, location in instance.locations.items():
        lon, lat = location.x, location.y
        
        if 'depot' in loc_id.lower():
            depots.append((lon, lat, loc_id))
        elif 'pickup' in loc_id.lower():
            pickups.append((lon, lat, loc_id))
        elif 'dropoff' in loc_id.lower():
            dropoffs.append((lon, lat, loc_id))
        elif 'service' in loc_id.lower() or 'area' in loc_id.lower():
            service_areas.append((lon, lat, loc_id))
    
    # Apply point limiting for large scenarios
    if max_points and len(pickups) > max_points:
        import random
        random.seed(42)
        pickups = random.sample(pickups, max_points)
        dropoffs = random.sample(dropoffs, max_points)
    
    # Plot different location types
    if depots:
        depot_lons, depot_lats, _ = zip(*depots)
        ax.scatter(depot_lons, depot_lats, c='red', s=200, marker='s', 
                  label=f'Depots ({len(depots)})', alpha=0.8, edgecolors='black', linewidth=2)
    
    if pickups:
        pickup_lons, pickup_lats, _ = zip(*pickups)
        ax.scatter(pickup_lons, pickup_lats, c='green', s=60, marker='^', 
                  label=f'Pickups ({len(instance.locations)//2 if "pickup" in str(instance.locations) else len(pickups)})', alpha=0.7)
    
    if dropoffs:
        dropoff_lons, dropoff_lats, _ = zip(*dropoffs)
        ax.scatter(dropoff_lons, dropoff_lats, c='blue', s=60, marker='v', 
                  label=f'Dropoffs ({len(instance.locations)//2 if "dropoff" in str(instance.locations) else len(dropoffs)})', alpha=0.7)
    
    if service_areas:
        service_lons, service_lats, _ = zip(*service_areas)
        ax.scatter(service_lons, service_lats, c='orange', s=80, marker='o', 
                  label=f'Service Areas ({len(service_areas)})', alpha=0.6)
    
    # Draw some pickup-dropoff connections for illustration (first few)
    connection_limit = min(5, len(pickups), len(dropoffs))
    for i in range(connection_limit):
        if i < len(pickups) and i < len(dropoffs):
            ax.plot([pickups[i][0], dropoffs[i][0]], 
                   [pickups[i][1], dropoffs[i][1]], 
                   'gray', alpha=0.3, linestyle='--', linewidth=1)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio for geographic accuracy
    ax.set_aspect('equal', adjustable='box')

def create_scenario_info_text(instance):
    """Create informative text about the scenario."""
    info_lines = []
      # Count different location types
    depot_count = len([loc_id for loc_id in instance.locations.keys() if 'depot' in loc_id.lower()])
    pickup_count = len([loc_id for loc_id in instance.locations.keys() if 'pickup' in loc_id.lower()])
    dropoff_count = len([loc_id for loc_id in instance.locations.keys() if 'dropoff' in loc_id.lower()])
    service_count = len([loc_id for loc_id in instance.locations.keys() if 'service' in loc_id.lower() or 'area' in loc_id.lower()])
    
    # Vehicle information
    total_vehicles = len(instance.vehicles)
    heavy_vehicles = len([v for v in instance.vehicles.values() if hasattr(v, 'vehicle_type') and v.vehicle_type == 'heavy'])
    standard_vehicles = total_vehicles - heavy_vehicles
    
    # Cargo information
    total_cargo = sum(req.passengers for req in instance.ride_requests)
    total_capacity = sum(v.capacity for v in instance.vehicles.values())
    
    info_lines.extend([
        f"📍 Locations: {len(instance.locations)} total",
        f"   • Depots: {depot_count}",
        f"   • Pickups: {pickup_count}",
        f"   • Dropoffs: {dropoff_count}",
        f"   • Service Areas: {service_count}",
        "",
        f"🚛 Fleet: {total_vehicles} vehicles",
        f"   • Heavy Trucks (24t): {heavy_vehicles}",
        f"   • Standard Trucks (4t): {standard_vehicles}",
        "",
        f"📦 Cargo: {total_cargo:,} kg total",
        f"   • Fleet Capacity: {total_capacity:,} kg",
        f"   • Utilization: {total_cargo/total_capacity*100:.1f}%",
        "",
        f"🎯 Requests: {len(instance.ride_requests)} shipments",
    ])
    
    return "\n".join(info_lines)

def create_clean_comparison():
    """Create clean side-by-side comparison of MODA scenarios."""
    print("🚛 Creating Clean MODA Scenario Comparison")
    print("=" * 60)
    
    # Generate scenarios
    print("📊 Generating MODA scenarios...")
    moda_small = create_moda_small_scenario()
    moda_first = create_moda_first_scenario()
    
    # Create the comparison plot
    fig = plt.figure(figsize=(20, 10))
    
    # Left plot: MODA Small
    ax1 = plt.subplot(1, 2, 1)
    plot_scenario_locations(moda_small, ax1, "MODA Small Scenario\n(5 vehicles, Northern Italy - Milan Area)")
    
    # Right plot: MODA First (with point limiting for clarity)
    ax2 = plt.subplot(1, 2, 2)
    plot_scenario_locations(moda_first, ax2, "MODA First Scenario\n(60 vehicles, Northern Italy - Full Region)", max_points=30)
    
    # Adjust layout
    plt.tight_layout()
    
    # Add overall title
    fig.suptitle('MODA VRP Scenarios Comparison\nRealistic Northern Italy Trucking Operations', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Save the plot
    os.makedirs("Plots for PPT", exist_ok=True)
    output_path = "Plots for PPT/moda_clean_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Clean comparison plot saved as: {output_path}")
    
    # Show the plot
    plt.show()
    
    return moda_small, moda_first

def create_detailed_info_plots():
    """Create detailed information plots for each scenario."""
    print("\n📊 Creating detailed information plots...")
    
    # Generate scenarios
    moda_small = create_moda_small_scenario()
    moda_first = create_moda_first_scenario()
    
    # Create detailed info plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top row: Location plots
    plot_scenario_locations(moda_small, ax1, "MODA Small - Locations")
    plot_scenario_locations(moda_first, ax2, "MODA First - Locations (Sample)", max_points=25)
    
    # Bottom row: Information text
    ax3.text(0.05, 0.95, create_scenario_info_text(moda_small), 
             transform=ax3.transAxes, fontsize=11, verticalalignment='top',
             fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    ax3.set_title("MODA Small - Details", fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    ax4.text(0.05, 0.95, create_scenario_info_text(moda_first), 
             transform=ax4.transAxes, fontsize=11, verticalalignment='top',
             fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    ax4.set_title("MODA First - Details", fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle('MODA Scenarios - Detailed Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save detailed plot
    output_path = "Plots for PPT/moda_detailed_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Detailed comparison plot saved as: {output_path}")
    
    plt.show()

def create_fleet_composition_chart():
    """Create a chart showing fleet composition comparison."""
    print("\n🚛 Creating fleet composition chart...")
    
    # Generate scenarios
    moda_small = create_moda_small_scenario()
    moda_first = create_moda_first_scenario()
    
    # Extract fleet data
    scenarios = ['MODA Small', 'MODA First']
    heavy_trucks = [
        len([v for v in moda_small.vehicles.values() if hasattr(v, 'vehicle_type') and v.vehicle_type == 'heavy']),
        len([v for v in moda_first.vehicles.values() if hasattr(v, 'vehicle_type') and v.vehicle_type == 'heavy'])
    ]
    standard_trucks = [
        len([v for v in moda_small.vehicles.values() if hasattr(v, 'vehicle_type') and v.vehicle_type == 'standard']),
        len([v for v in moda_first.vehicles.values() if hasattr(v, 'vehicle_type') and v.vehicle_type == 'standard'])
    ]
    
    # Create the chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, standard_trucks, width, label='Standard Trucks (4t)', color='lightblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, heavy_trucks, width, label='Heavy Trucks (24t)', color='orange', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Number of Vehicles', fontsize=12)
    ax.set_title('Fleet Composition Comparison\nMixed Fleet Implementation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
      # Add total vehicle count
    total_small = standard_trucks[0] + heavy_trucks[0]
    total_first = standard_trucks[1] + heavy_trucks[1]
    ax.text(0, max(standard_trucks + heavy_trucks) * 0.9, f'Total: {total_small}', 
            ha='center', fontsize=10, fontweight='bold')
    ax.text(1, max(standard_trucks + heavy_trucks) * 0.9, f'Total: {total_first}', 
            ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save fleet chart
    output_path = "Plots for PPT/moda_fleet_composition.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Fleet composition chart saved as: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    # Create all visualizations
    print("🎨 MODA VRP Scenarios - Clean Visualization Tool")
    print("=" * 60)
    
    # Create main comparison
    moda_small, moda_first = create_clean_comparison()
    
    # Create detailed plots
    create_detailed_info_plots()
    
    # Create fleet composition chart
    create_fleet_composition_chart()
    
    print("\n✅ All clean visualizations completed successfully!")
    print("📁 Check the 'Plots for PPT' directory for saved images:")
    print("   • moda_clean_comparison.png - Side-by-side locations")
    print("   • moda_detailed_comparison.png - Detailed info & locations")
    print("   • moda_fleet_composition.png - Fleet composition chart")
