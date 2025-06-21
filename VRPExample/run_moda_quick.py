#!/usr/bin/env python3
"""
Quick test of MODA Small scenario with the enhanced VRP optimizer.
This will generate both a static map plot and an interactive HTML map with time windows and load info.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from VRPExample.vrp_scenarios import get_all_scenarios
from VRPExample.vrp_optimizer_enhanced_full import VRPOptimizerEnhanced
import matplotlib.pyplot as plt
import folium
import logging

# Set up simple logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_moda_test():
    """Quick test with MODA Small scenario using Level 6 constraints."""
    print("=" * 60)
    print("QUICK MODA SMALL TEST - 70% PRE-LOADED SCENARIO")
    print("=" * 60)
    
    # Load scenarios
    scenarios = get_all_scenarios()
    scenario = scenarios['MODA_small']
    
    print(f"✅ Loaded scenario: {scenario.name}")
    print(f"   📊 {len(scenario.vehicles)} vehicles, {len(scenario.locations)} locations, {len(scenario.ride_requests)} requests")
    
    # Show capacity distribution
    total_capacity = sum(v.capacity for v in scenario.vehicles.values())
    preloaded = sum(getattr(v, 'preloaded_cargo', 0) for v in scenario.vehicles.values())
    pickup_cargo = sum(r.passengers for r in scenario.ride_requests)
    
    print(f"   📦 Capacity: {total_capacity:,}kg total, {preloaded:,}kg pre-loaded (70%), {pickup_cargo:,}kg pickup (30%)")
    
    # Create optimizer and solve with Level 6 constraints
    optimizer = VRPOptimizerEnhanced()
    
    print("\n🚀 Running optimization with Level 6 constraints (full constraints + driver breaks)...")
    result = optimizer._solve_with_constraint_level(scenario, "full", 120)  # 2 minutes, Level 6
    
    if result['success']:
        print(f"✅ SUCCESS!")
        print(f"   Objective: {result.get('objective_value', 'Unknown')}")
        print(f"   Vehicles used: {result['vehicles_used']}")
        print(f"   Solve time: {result['solve_time']:.2f}s")
        print(f"   Constraints: {', '.join(result.get('constraints_applied', []))}")
        
        # Create both static and interactive maps
        print("\n🗺️ Creating static map plot...")
        create_simple_map(scenario, result)
        
        print("\n🌐 Creating interactive HTML map with time windows and load info...")
        create_interactive_html_map(scenario, result)
        
    else:
        print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
        print(f"   Details: {result.get('details', 'No additional details')}")

def create_simple_map(scenario, result):
    """Create a simple scatter plot map of the solution."""
    plt.figure(figsize=(12, 8))
    
    # Plot all locations
    for loc_id, location in scenario.locations.items():
        if 'depot' in loc_id.lower():
            plt.scatter(location.x, location.y, c='red', s=200, marker='s', label='Depot', alpha=0.8)
            plt.annotate(f'{loc_id}\n({getattr(location, "service_time", 0)}min)', 
                        (location.x, location.y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8, ha='left')
        elif 'pickup' in loc_id.lower():
            # Find corresponding request to get cargo weight
            cargo_weight = 0
            for request in scenario.ride_requests:
                if request.pickup_location == loc_id:
                    cargo_weight = request.passengers
                    break
            plt.scatter(location.x, location.y, c='green', s=80, marker='^', alpha=0.7)
            tw_start = getattr(location, 'time_window_start', 0)
            tw_end = getattr(location, 'time_window_end', 1440)
            plt.annotate(f'{loc_id}\n{cargo_weight}kg\n[{tw_start}-{tw_end}min]', 
                        (location.x, location.y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=6, ha='left')
        elif 'dropoff' in loc_id.lower():
            plt.scatter(location.x, location.y, c='blue', s=80, marker='v', alpha=0.7)
            tw_start = getattr(location, 'time_window_start', 0)
            tw_end = getattr(location, 'time_window_end', 1440)
            plt.annotate(f'{loc_id}\n[{tw_start}-{tw_end}min]', 
                        (location.x, location.y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=6, ha='left')
        else:
            plt.scatter(location.x, location.y, c='gray', s=40, marker='o', alpha=0.5)
    
    # Plot routes
    colors = ['orange', 'purple', 'brown', 'pink', 'cyan']
    
    if 'routes' in result:
        for i, (vehicle_id, route) in enumerate(result['routes'].items()):
            color = colors[i % len(colors)]
            
            # Get coordinates for this route
            route_coords = []
            for loc_id in route:
                if loc_id in scenario.locations:
                    location = scenario.locations[loc_id]
                    route_coords.append((location.x, location.y))
            
            # Draw route lines
            if len(route_coords) > 1:
                for j in range(len(route_coords) - 1):
                    x1, y1 = route_coords[j]
                    x2, y2 = route_coords[j + 1]
                    plt.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7, 
                            label=f'{vehicle_id}' if j == 0 else "")
    
    plt.title(f'VRP Solution Map - {scenario.name}\n70% Pre-loaded, 30% Pickup-Delivery')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('moda_small_solution.png', dpi=300, bbox_inches='tight')
    print("📁 Static map saved as: moda_small_solution.png")
    
    # Show the plot
    plt.show()
    
    print("✅ Static map visualization complete!")

def create_interactive_html_map(scenario, result):
    """Create an interactive HTML map with time windows and load information."""
    # Get the center coordinates (Asti area)
    center_lat = 44.9009
    center_lon = 8.2062
    
    # Create the map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, 
                  tiles='OpenStreetMap')
    
    # Add locations with detailed popups
    for loc_id, location in scenario.locations.items():
        lat = getattr(location, 'lat', location.y)
        lon = getattr(location, 'lon', location.x)
        service_time = getattr(location, 'service_time', 0)
        tw_start = getattr(location, 'time_window_start', 0)
        tw_end = getattr(location, 'time_window_end', 1440)
        address = getattr(location, 'address', 'No address')
        
        # Create popup content with time windows and load info
        popup_content = f"""
        <b>{loc_id}</b><br>
        Address: {address}<br>
        Service Time: {service_time} minutes<br>
        Time Window: {tw_start//60:02d}:{tw_start%60:02d} - {tw_end//60:02d}:{tw_end%60:02d}<br>
        """
        
        # Add cargo information for pickup locations
        if 'pickup' in loc_id.lower():
            for request in scenario.ride_requests:
                if request.pickup_location == loc_id:
                    popup_content += f"Cargo to pickup: {request.passengers} kg<br>"
                    popup_content += f"Deliver to: {request.dropoff_location}<br>"
                    break
        
        # Add vehicle pre-load info for depot
        if 'depot' in loc_id.lower():
            total_preloaded = sum(getattr(v, 'preloaded_cargo', 0) for v in scenario.vehicles.values() if v.depot_id == loc_id)
            vehicle_count = sum(1 for v in scenario.vehicles.values() if v.depot_id == loc_id)
            popup_content += f"Vehicles based here: {vehicle_count}<br>"
            popup_content += f"Total pre-loaded cargo: {total_preloaded:,} kg<br>"
        
        # Set marker style based on location type
        if 'depot' in loc_id.lower():
            folium.Marker(
                location=[lat, lon],
                popup=popup_content,
                tooltip=f"Depot: {loc_id}",
                icon=folium.Icon(color='red', icon='home', prefix='fa')
            ).add_to(m)
        elif 'pickup' in loc_id.lower():
            folium.Marker(
                location=[lat, lon],
                popup=popup_content,
                tooltip=f"Pickup: {loc_id}",
                icon=folium.Icon(color='green', icon='arrow-up', prefix='fa')
            ).add_to(m)
        elif 'dropoff' in loc_id.lower():
            folium.Marker(
                location=[lat, lon],
                popup=popup_content,
                tooltip=f"Dropoff: {loc_id}",
                icon=folium.Icon(color='blue', icon='arrow-down', prefix='fa')
            ).add_to(m)
    
    # Add route lines if available
    if 'routes' in result:
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        for i, (vehicle_id, route) in enumerate(result['routes'].items()):
            color = colors[i % len(colors)]
            
            # Get route coordinates
            route_coords = []
            for loc_id in route:
                if loc_id in scenario.locations:
                    location = scenario.locations[loc_id]
                    lat = getattr(location, 'lat', location.y)
                    lon = getattr(location, 'lon', location.x)
                    route_coords.append([lat, lon])
            
            if len(route_coords) > 1:
                # Add route line
                folium.PolyLine(
                    locations=route_coords,
                    color=color,
                    weight=3,
                    opacity=0.8,
                    popup=f"Route: {vehicle_id}"
                ).add_to(m)
                
                # Add route direction arrows
                for j in range(len(route_coords) - 1):
                    mid_lat = (route_coords[j][0] + route_coords[j+1][0]) / 2
                    mid_lon = (route_coords[j][1] + route_coords[j+1][1]) / 2
                    
                    folium.Marker(
                        location=[mid_lat, mid_lon],
                        icon=folium.Icon(color='darkgray', icon='arrow-right', prefix='fa'),
                        tooltip=f"{vehicle_id}: Stop {j+1} → {j+2}"
                    ).add_to(m)
    
    # Add a legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px">
    <h4>Map Legend</h4>
    <i class="fa fa-home" style="color:red"></i> Depot (pre-loaded cargo)<br>
    <i class="fa fa-arrow-up" style="color:green"></i> Pickup Location<br>
    <i class="fa fa-arrow-down" style="color:blue"></i> Dropoff Location<br>
    <hr>
    <b>Scenario:</b> 70% pre-loaded, 30% pickup
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add title
    title_html = '''
    <h3 align="center" style="font-size:16px"><b>MODA Small VRP Solution Map</b><br>
    70% Pre-loaded Capacity + 30% Pickup-Delivery<br>
    Click markers for time windows and cargo details</h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save the map
    map_filename = 'moda_small_interactive_map.html'
    m.save(map_filename)
    
    print(f"📁 Interactive map saved as: {map_filename}")
    print("🌐 Open the HTML file in a web browser to view the interactive map!")
    print("   Click on markers to see time windows and cargo information")
    
    return map_filename

if __name__ == "__main__":
    quick_moda_test()
