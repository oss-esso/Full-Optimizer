#!/usr/bin/env python3
"""
Create HTML GPS map for MODA Small scenario using the main VRP visualization system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from VRPExample.vrp_scenarios import get_all_scenarios
from VRPExample.vrp_optimizer_enhanced_full import VRPOptimizerEnhanced
from VRPExample.vrp_map_visualization import create_all_map_visualizations
import logging

# Set up simple logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_asti_html_map():
    """Create HTML GPS map for MODA Small (Asti-centered) scenario."""
    print("=" * 70)
    print("CREATING HTML GPS MAP FOR MODA SMALL (ASTI-CENTERED)")
    print("=" * 70)
    
    # Load scenarios
    scenarios = get_all_scenarios()
    scenario = scenarios['MODA_small']
    
    print(f"✅ Loaded scenario: {scenario.name}")
    print(f"   📊 {len(scenario.vehicles)} vehicles, {len(scenario.locations)} locations, {len(scenario.ride_requests)} requests")
    
    # Check GPS coordinates
    gps_locations = []
    for loc_id, location in scenario.locations.items():
        if hasattr(location, 'lat') and hasattr(location, 'lon') and location.lat and location.lon:
            gps_locations.append((loc_id, location.lat, location.lon))
    
    print(f"   🗺️ GPS coordinates available: {len(gps_locations)} locations")
    for loc_id, lat, lon in gps_locations[:5]:  # Show first 5
        print(f"      {loc_id}: {lat:.4f}, {lon:.4f}")
    if len(gps_locations) > 5:
        print(f"      ... and {len(gps_locations) - 5} more")
    
    # Create optimizer and solve
    optimizer = VRPOptimizerEnhanced()
    
    print("\n🚀 Running optimization...")
    result = optimizer.solve(scenario, time_limit_seconds=120)  # 2 minutes
    
    if result['success']:
        print(f"✅ SUCCESS!")
        print(f"   Strategy: {result.get('strategy_used', 'Unknown')}")
        print(f"   Vehicles used: {result['vehicles_used']}")
        print(f"   Solve time: {result['solve_time']:.2f}s")
        
        # Convert result to the format expected by map visualization
        from VRPExample.vrp_data_models import VRPResult
        
        # Create VRPResult object compatible with map visualization
        vrp_result = VRPResult()
        vrp_result.routes = result['routes']
        vrp_result.objective_value = result['objective_value']
        vrp_result.runtime = result['solve_time']
        vrp_result.metrics = {
            'total_distance': 0,  # Will be calculated by visualization
            'vehicles_used': result['vehicles_used'],
            'total_time': 0  # Will be calculated by visualization
        }
        
        # Calculate total distance from route analysis if available
        if 'route_analysis' in result:
            total_distance = sum(analysis.get('total_distance', 0) for analysis in result['route_analysis'])
            total_time = sum(analysis.get('total_time', 0) for analysis in result['route_analysis'])
            vrp_result.metrics['total_distance'] = total_distance
            vrp_result.metrics['total_time'] = total_time
        
        # Create map visualizations
        print("\n🗺️ Creating HTML GPS map...")
        
        # Create results directory
        results_dir = "asti_maps"
        os.makedirs(results_dir, exist_ok=True)
        
        try:
            # Use the main VRP visualization system
            map_files = create_all_map_visualizations(scenario, vrp_result, results_dir, "moda_small_asti")
            
            if map_files:
                print(f"✅ Created {len(map_files)} map visualizations:")
                for map_file in map_files:
                    print(f"   📁 {map_file}")
                    
                # Try to open the HTML file in browser
                html_files = [f for f in map_files if f.endswith('.html')]
                if html_files:
                    print(f"\n🌐 Opening HTML map: {html_files[0]}")
                    try:
                        import webbrowser
                        webbrowser.open(f"file://{os.path.abspath(html_files[0])}")
                        print("   ✅ HTML map opened in browser")
                    except Exception as e:
                        print(f"   ⚠️ Could not auto-open browser: {e}")
                        print(f"   💡 Manually open: {os.path.abspath(html_files[0])}")
            else:
                print("❌ No map files were created")
                
        except Exception as e:
            print(f"❌ Error creating maps: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: create a simple HTML map manually
            print("\n🔄 Creating fallback HTML map...")
            create_simple_html_map(scenario, result, results_dir)
        
    else:
        print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
        print("💡 Cannot create map without a solution")

def create_simple_html_map(scenario, result, results_dir):
    """Create a simple HTML map as fallback."""
    try:
        # Check if folium is available
        import folium
        
        # Get GPS coordinates
        lats = []
        lons = []
        for location in scenario.locations.values():
            if hasattr(location, 'lat') and hasattr(location, 'lon') and location.lat and location.lon:
                lats.append(location.lat)
                lons.append(location.lon)
        
        if not lats:
            print("❌ No GPS coordinates available for mapping")
            return
        
        # Create map centered on Asti
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles='OpenStreetMap'
        )
        
        # Add markers for locations
        for loc_id, location in scenario.locations.items():
            if hasattr(location, 'lat') and hasattr(location, 'lon') and location.lat and location.lon:
                if 'depot' in loc_id.lower():
                    folium.Marker(
                        [location.lat, location.lon],
                        popup=f"Depot: {loc_id}",
                        icon=folium.Icon(color='red', icon='home')
                    ).add_to(m)
                elif 'pickup' in loc_id.lower():
                    folium.Marker(
                        [location.lat, location.lon],
                        popup=f"Pickup: {loc_id}",
                        icon=folium.Icon(color='green', icon='arrow-up')
                    ).add_to(m)
                elif 'dropoff' in loc_id.lower():
                    folium.Marker(
                        [location.lat, location.lon],
                        popup=f"Dropoff: {loc_id}",
                        icon=folium.Icon(color='blue', icon='arrow-down')
                    ).add_to(m)
                else:
                    folium.Marker(
                        [location.lat, location.lon],
                        popup=loc_id,
                        icon=folium.Icon(color='gray', icon='info-sign')
                    ).add_to(m)
        
        # Add routes if available
        if 'routes' in result and result['routes']:
            colors = ['orange', 'purple', 'brown', 'pink', 'cyan']
            for i, (vehicle_id, route) in enumerate(result['routes'].items()):
                color = colors[i % len(colors)]
                route_coords = []
                
                for loc_id in route:
                    if loc_id in scenario.locations:
                        location = scenario.locations[loc_id]
                        if hasattr(location, 'lat') and hasattr(location, 'lon') and location.lat and location.lon:
                            route_coords.append([location.lat, location.lon])
                
                if len(route_coords) > 1:
                    folium.PolyLine(
                        route_coords,
                        color=color,
                        weight=4,
                        opacity=0.8,
                        popup=f"Route {vehicle_id}"
                    ).add_to(m)
        
        # Add title
        title_html = f'''
        <h3 align="center" style="font-size:16px"><b>VRP Solution: {scenario.name}</b></h3>
        <p align="center">Vehicles: {result.get('vehicles_used', 0)} | 
        Strategy: {result.get('strategy_used', 'Unknown')} | 
        Time: {result.get('solve_time', 0):.1f}s</p>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Save map
        map_path = os.path.join(results_dir, "moda_small_asti_simple.html")
        m.save(map_path)
        print(f"✅ Simple HTML map created: {map_path}")
        
        # Try to open in browser
        try:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(map_path)}")
            print("   ✅ Map opened in browser")
        except Exception as e:
            print(f"   💡 Manually open: {os.path.abspath(map_path)}")
            
    except ImportError:
        print("❌ folium not available for HTML maps")
        print("💡 Install with: pip install folium")
    except Exception as e:
        print(f"❌ Error creating simple HTML map: {e}")

if __name__ == "__main__":
    create_asti_html_map()
