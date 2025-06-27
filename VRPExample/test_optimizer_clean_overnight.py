"""
Test file to compare the two VRP optimizers on the overnight test scenario.

This test compares:
1. CleanVRPOptimizer (original from vrp_optimizer_clean.py)
2. CleanVRPOptimizer (OSM-enhanced copy from vrp_optimizer_clean_copy.py)

Both are tested on the overnight test scenario to evaluate:
- Distance calculation differences (Manhattan vs OSM)
- Solution quality and feasibility
- Capacity constraint handling
- Performance metrics
"""

import time
import logging
from typing import Dict, Optional, Tuple
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_interactive_vrp_map(scenario, solution, sequential_vrp):
    """Create an interactive HTML map showing the VRP solution with detailed information."""
    
    # Check if folium is available
    try:
        import folium
        from folium import plugins
        import requests
    except ImportError:
        print("❌ Folium not available. Install with: pip install folium")
        return None
    
    # OSRM routing service setup
    osrm_url = "http://router.project-osrm.org"
    routing_session = requests.Session()
    
    def get_street_route(start_coords, end_coords):
        """Get actual street route between two points using OSRM."""
        try:
            # Use OSRM routing API to get actual route
            url = f"{osrm_url}/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}"
            params = {
                'overview': 'full',
                'geometries': 'geojson'
            }
            
            response = routing_session.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] == 'Ok' and 'routes' in data and len(data['routes']) > 0:
                # Extract coordinates from the route geometry
                route_coords = data['routes'][0]['geometry']['coordinates']
                # Convert from [lon, lat] to [lat, lon] for folium
                street_route = [[coord[1], coord[0]] for coord in route_coords]
                return street_route
            else:
                # Fall back to straight line
                return [start_coords, end_coords]
                
        except Exception as e:
            print(f"  ⚠️ OSRM routing failed: {e}, using straight line")
            return [start_coords, end_coords]
    
    # Check if locations have GPS coordinates
    has_gps = False
    print("🔍 Checking GPS coordinates in scenario locations...")
    for loc_id, loc in scenario.locations.items():
        # Check for different GPS coordinate attributes
        lat = getattr(loc, 'lat', None) or getattr(loc, 'latitude', None)
        lon = getattr(loc, 'lon', None) or getattr(loc, 'longitude', None)
        
        if lat is not None and lon is not None:
            has_gps = True
            print(f"  ✅ {loc_id}: lat={lat}, lon={lon}")
            break
        else:
            print(f"  ❌ {loc_id}: lat={getattr(loc, 'lat', 'None')}, lon={getattr(loc, 'lon', 'None')}")
    
    if not has_gps:
        print("❌ No GPS coordinates found in scenario locations")
        # Let's try to find coordinates in x,y format and see if they could be GPS
        print("🔍 Checking if x,y coordinates might be GPS coordinates...")
        for loc_id, loc in list(scenario.locations.items())[:3]:
            x = getattr(loc, 'x', None)
            y = getattr(loc, 'y', None)
            print(f"  - {loc_id}: x={x}, y={y}")
        return None
    
    # Calculate map center from locations with GPS coordinates
    lats = []
    lons = []
    for loc_id, loc in scenario.locations.items():
        lat = getattr(loc, 'lat', None) or getattr(loc, 'latitude', None)
        lon = getattr(loc, 'lon', None) or getattr(loc, 'longitude', None)
        
        if lat is not None and lon is not None:
            lats.append(lat)
            lons.append(lon)
    
    if not lats:
        print("❌ No valid GPS coordinates found")
        return None
    
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    
    print(f"📍 Map center: lat={center_lat:.4f}, lon={center_lon:.4f}")
    print(f"📊 Found {len(lats)} locations with GPS coordinates")
    
    # Create the map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
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
    
    # Get depot coordinates
    depot_coords = None
    depot_location = None
    for loc_id, loc in scenario.locations.items():
        if 'depot' in str(loc_id).lower() and not 'bay' in str(loc_id).lower():
            lat = getattr(loc, 'lat', None) or getattr(loc, 'latitude', None)
            lon = getattr(loc, 'lon', None) or getattr(loc, 'longitude', None)
            if lat is not None and lon is not None:
                depot_coords = (lat, lon)
                depot_location = loc
                print(f"🏭 Found depot at: lat={lat}, lon={lon}")
                break
    
    # Add depot marker
    if depot_coords and depot_location:
        folium.Marker(
            location=depot_coords,
            popup=f"<b>🏭 Main Depot</b><br>ID: {getattr(depot_location, 'id', 'depot')}<br>Address: {getattr(depot_location, 'address', 'N/A')}",
            tooltip="Main Depot",
            icon=folium.Icon(color='red', icon='home', prefix='fa')
        ).add_to(m)
        print("✅ Added depot marker to map")
    
    # Color palette for vehicles
    vehicle_colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'black', 'gray', 'pink', 'lightblue', 'lightgreen']
    
    # Process each vehicle's route
    route_info = {}
    markers_added = 0
    
    print(f"🔍 Solution structure keys: {list(solution.keys())}")
    
    # Check different possible solution structures
    vehicle_routes = solution.get('vehicle_routes', {})
    if not vehicle_routes:
        # Try alternative structure - daily solutions
        daily_solutions = solution.get('daily_solutions', {})
        print(f"🔍 Found daily_solutions with {len(daily_solutions)} days")
        
        # Convert daily solutions to vehicle routes format
        vehicle_routes = {}
        for day_num, day_data in daily_solutions.items():
            routes = day_data.get('routes', {})
            for vehicle_id, route in routes.items():
                if vehicle_id not in vehicle_routes:
                    vehicle_routes[vehicle_id] = {'daily_routes': {}}
                vehicle_routes[vehicle_id]['daily_routes'][day_num] = route
    
    print(f"🔍 Processing {len(vehicle_routes)} vehicle routes")
    
    for vehicle_id, route_data in vehicle_routes.items():
        vehicle_index = list(vehicle_routes.keys()).index(vehicle_id)
        color = vehicle_colors[vehicle_index % len(vehicle_colors)]
        
        route_info[vehicle_id] = {
            'color': color,
            'total_distance': 0,
            'total_time': 0,
            'customers_served': 0,
            'overnight_stays': 0
        }
        
        print(f"🚛 Processing vehicle {vehicle_id} with route_data keys: {list(route_data.keys())}")
        
        # Process daily routes
        daily_routes = route_data.get('daily_routes', {})
        if not daily_routes:
            # Try alternative: full_route structure from sequential solver
            full_route = route_data.get('full_route', {})
            if full_route:
                print(f"  🔍 Found full_route with type: {type(full_route)}")
                if isinstance(full_route, dict):
                    # full_route is a dictionary of day -> route data
                    daily_routes = full_route
                elif isinstance(full_route, list):
                    # full_route is a flat list of stops - process directly without day grouping
                    print(f"  📍 Processing {len(full_route)} stops from flat full_route")
                    for i, stop in enumerate(full_route):
                        if stop.get('is_overnight', False):
                            # Handle overnight stops separately
                            coords = stop.get('coordinates')
                            if coords and isinstance(coords, (list, tuple)) and len(coords) >= 2:
                                lat, lon = coords[1], coords[0]  # coordinates might be [lon, lat]
                                if lat and lon:
                                    # Create overnight marker
                                    popup_text = f"""
                                    <b>🛏️ Overnight Stop</b><br>
                                    <b>Vehicle:</b> {vehicle_id}<br>
                                    <b>Stop #:</b> {i + 1}<br>
                                    <b>Location ID:</b> {stop.get('location_id', 'N/A')}<br>
                                    <b>Coordinates:</b> ({lat:.4f}, {lon:.4f})<br>
                                    <b>Distance from depot:</b> {((lat - depot_coords[0])**2 + (lon - depot_coords[1])**2)**0.5:.2f} km
                                    """
                                    
                                    folium.Marker(
                                        location=[lat, lon],
                                        popup=folium.Popup(popup_text, max_width=300),
                                        tooltip=f"Overnight: {vehicle_id}",
                                        icon=folium.Icon(color='darkblue', icon='bed', prefix='fa')
                                    ).add_to(m)
                                    
                                    markers_added += 1
                                    route_info[vehicle_id]['overnight_stays'] += 1
                                    print(f"    🛏️ Added overnight marker at ({lat:.4f}, {lon:.4f})")
                            continue
                        
                        # Handle regular stops
                        stop_id = stop.get('location_id', f'stop_{i}')
                        
                        # Find location in scenario
                        location = None
                        for loc_id, loc in scenario.locations.items():
                            if str(loc_id) == str(stop_id):
                                location = loc
                                break
                        
                        if location:
                            lat = getattr(location, 'lat', None) or getattr(location, 'latitude', None)
                            lon = getattr(location, 'lon', None) or getattr(location, 'longitude', None)
                            
                            if lat is not None and lon is not None:
                                # Determine marker type and icon
                                if 'depot' in str(stop_id).lower():
                                    if 'bay' in str(stop_id).lower():
                                        icon_type = 'cube'
                                        marker_color = 'orange'
                                        location_type = '📦 Depot Bay'
                                    else:
                                        icon_type = 'home'
                                        marker_color = 'red'
                                        location_type = '🏭 Depot'
                                elif 'pickup' in str(stop_id).lower():
                                    icon_type = 'arrow-up'
                                    marker_color = 'green'
                                    location_type = '📤 Pickup'
                                else:
                                    icon_type = 'map-marker'
                                    marker_color = color
                                    location_type = '📍 Delivery'
                                
                                # Get service time information
                                arrival_time = stop.get('arrival_time', 'N/A')
                                service_time = getattr(location, 'service_time', 15)
                                
                                # Format times if they're numeric (minutes)
                                if isinstance(arrival_time, (int, float)):
                                    hours = int(arrival_time // 60)
                                    mins = int(arrival_time % 60)
                                    arrival_time = f"{hours:02d}:{mins:02d}"
                                
                                # Create popup with detailed information
                                popup_text = f"""
                                <b>{location_type}: {stop_id}</b><br>
                                <b>Vehicle:</b> {vehicle_id}<br>
                                <b>Stop #:</b> {i + 1}<br>
                                <b>Address:</b> {getattr(location, 'address', getattr(location, 'name', 'N/A'))}<br>
                                <b>Arrival:</b> {arrival_time}<br>
                                <b>Service Time:</b> {service_time} min<br>
                                <b>Coordinates:</b> ({lat:.4f}, {lon:.4f})
                                """
                                
                                folium.Marker(
                                    location=[lat, lon],
                                    popup=folium.Popup(popup_text, max_width=300),
                                    tooltip=f"{location_type}: {stop_id}",
                                    icon=folium.Icon(color=marker_color, icon=icon_type, prefix='fa')
                                ).add_to(m)
                                
                                markers_added += 1
                                print(f"    ✅ Added marker for {stop_id} at ({lat:.4f}, {lon:.4f})")
                                
                                if not 'depot' in str(stop_id).lower():
                                    route_info[vehicle_id]['customers_served'] += 1
                        else:
                            print(f"    ❌ Location not found for stop: {stop_id}")
                    
                    # Add OSRM-based route lines for flat route structure
                    print(f"  🗺️ Drawing OSRM routes for {vehicle_id} (flat route with {len(full_route)} stops)")
                    
                    # Collect coordinates for route lines (excluding overnight stops)
                    route_coords = []
                    for stop in full_route:
                        if stop.get('is_overnight', False):
                            continue
                        
                        stop_id = stop.get('location_id', 'unknown')
                        
                        # Find location in scenario
                        location = None
                        for loc_id, loc in scenario.locations.items():
                            if str(loc_id) == str(stop_id):
                                location = loc
                                break
                        
                        if location:
                            lat = getattr(location, 'lat', None) or getattr(location, 'latitude', None)
                            lon = getattr(location, 'lon', None) or getattr(location, 'longitude', None)
                            if lat is not None and lon is not None:
                                route_coords.append([lat, lon])
                    
                    # Draw OSRM-based route lines between consecutive stops
                    if len(route_coords) > 1:
                        all_route_coords = []
                        
                        for i in range(len(route_coords) - 1):
                            start_coord = route_coords[i]
                            end_coord = route_coords[i + 1]
                            
                            # Get street-following route from OSRM
                            street_route = get_street_route(start_coord, end_coord)
                            
                            # Add this segment to the full route (avoid duplicating points)
                            if i == 0:
                                all_route_coords.extend(street_route)
                            else:
                                # Skip the first point to avoid duplication
                                all_route_coords.extend(street_route[1:])
                        
                        # Draw the complete route with OSRM-based paths
                        if len(all_route_coords) > 1:
                            folium.PolyLine(
                                locations=all_route_coords,
                                color=color,
                                weight=4,
                                opacity=0.8,
                                popup=f"{vehicle_id} - Multi-Day Route (OSRM)"
                            ).add_to(m)
                            print(f"    ✅ Added OSRM route with {len(all_route_coords)} points")
                        else:
                            # Fallback to simple line if OSRM fails completely
                            folium.PolyLine(
                                locations=route_coords,
                                color=color,
                                weight=4,
                                opacity=0.8,
                                popup=f"{vehicle_id} - Multi-Day Route (Direct)"
                            ).add_to(m)
                            print(f"    ⚠️ Used direct routes as fallback")
                    
                    # Skip the normal day-by-day processing for flat list
                    continue
            else:
                # Try alternative: direct stops
                stops = route_data.get('stops', [])
                if stops:
                    daily_routes = {'1': {'stops': stops}}
        
        # Process normal day-by-day structure (if we have it)
        for day_num, day_route in daily_routes.items():
            day_stops = day_route.get('stops', [])
            print(f"  📅 Day {day_num}: {len(day_stops)} stops")
                
        for day_num, day_route in daily_routes.items():
            day_stops = day_route.get('stops', [])
            print(f"  📅 Day {day_num}: {len(day_stops)} stops")
            
            # Add markers for regular stops
            for i, stop in enumerate(day_stops):
                if stop.get('is_overnight', False):
                    continue  # Handle overnight stops separately
                
                # Handle different stop formats
                stop_id = stop.get('location_id', stop.get('id', stop.get('location', f'stop_{i}')))
                
                # Find location in scenario
                location = None
                for loc_id, loc in scenario.locations.items():
                    if str(loc_id) == str(stop_id):
                        location = loc
                        break
                
                if location:
                    lat = getattr(location, 'lat', None) or getattr(location, 'latitude', None)
                    lon = getattr(location, 'lon', None) or getattr(location, 'longitude', None)
                    
                    if lat is not None and lon is not None:
                        # Determine marker type and icon
                        if 'depot' in str(stop_id).lower():
                            if 'bay' in str(stop_id).lower():
                                icon_type = 'cube'
                                marker_color = 'orange'
                                location_type = '📦 Depot Bay'
                            else:
                                icon_type = 'home'
                                marker_color = 'red'
                                location_type = '🏭 Depot'
                        elif 'pickup' in str(stop_id).lower():
                            icon_type = 'arrow-up'
                            marker_color = 'green'
                            location_type = '📤 Pickup'
                        else:
                            icon_type = 'map-marker'
                            marker_color = color
                            location_type = '📍 Delivery'
                        
                        # Get service time information
                        arrival_time = stop.get('arrival_time', 'N/A')
                        service_time = stop.get('service_time', getattr(location, 'service_time', 15))
                        departure_time = stop.get('departure_time', 'N/A')
                        
                        # Format times if they're numeric (minutes)
                        if isinstance(arrival_time, (int, float)):
                            hours = int(arrival_time // 60)
                            mins = int(arrival_time % 60)
                            arrival_time = f"{hours:02d}:{mins:02d}"
                        
                        if isinstance(departure_time, (int, float)):
                            hours = int(departure_time // 60)
                            mins = int(departure_time % 60)
                            departure_time = f"{hours:02d}:{mins:02d}"
                        
                        # Create popup with detailed information
                        popup_text = f"""
                        <b>{location_type}: {stop_id}</b><br>
                        <b>Vehicle:</b> {vehicle_id}<br>
                        <b>Day:</b> {day_num}<br>
                        <b>Stop #:</b> {i + 1}<br>
                        <b>Address:</b> {getattr(location, 'address', getattr(location, 'name', 'N/A'))}<br>
                        <b>Arrival:</b> {arrival_time}<br>
                        <b>Service Time:</b> {service_time} min<br>
                        <b>Departure:</b> {departure_time}<br>
                        <b>Coordinates:</b> ({lat:.4f}, {lon:.4f})
                        """
                        
                        folium.Marker(
                            location=[lat, lon],
                            popup=folium.Popup(popup_text, max_width=300),
                            tooltip=f"{location_type}: {stop_id} (Day {day_num})",
                            icon=folium.Icon(color=marker_color, icon=icon_type, prefix='fa')
                        ).add_to(m)
                        
                        markers_added += 1
                        print(f"    ✅ Added marker for {stop_id} at ({lat:.4f}, {lon:.4f})")
                        
                        if not 'depot' in str(stop_id).lower():
                            route_info[vehicle_id]['customers_served'] += 1
                else:
                    print(f"    ❌ Location not found for stop: {stop_id}")
            
            # Add overnight stop if exists
            if day_route.get('overnight_location') or day_route.get('overnight_position'):
                overnight_pos = day_route.get('overnight_position')
                overnight_loc = day_route.get('overnight_location')
                
                if overnight_pos:
                    if isinstance(overnight_pos, tuple) and len(overnight_pos) == 2:
                        lat, lon = overnight_pos
                    elif isinstance(overnight_pos, dict):
                        lat, lon = overnight_pos.get('x', 0), overnight_pos.get('y', 0)
                    else:
                        continue
                    
                    # Create overnight marker
                    popup_text = f"""
                    <b>🛏️ Overnight Stop</b><br>
                    <b>Vehicle:</b> {vehicle_id}<br>
                    <b>Day:</b> {day_num}<br>
                    <b>Type:</b> Road overnight<br>
                    <b>Coordinates:</b> ({lat:.4f}, {lon:.4f})<br>
                    <b>Distance from depot:</b> {((lat - depot_coords[0])**2 + (lon - depot_coords[1])**2)**0.5:.2f} km
                    """
                    
                    folium.Marker(
                        location=[lat, lon],
                        popup=folium.Popup(popup_text, max_width=300),
                        tooltip=f"Overnight: {vehicle_id} (Day {day_num})",
                        icon=folium.Icon(color='darkblue', icon='bed', prefix='fa')
                    ).add_to(m)
                    
                    markers_added += 1
                    route_info[vehicle_id]['overnight_stays'] += 1
                    print(f"    🛏️ Added overnight marker at ({lat:.4f}, {lon:.4f})")
            
            # Add route lines for this day with OSRM-based routing
            if len(day_stops) > 1:
                route_coords = []
                for stop in day_stops:
                    if stop.get('is_overnight', False):
                        continue
                    
                    stop_id = stop.get('location_id', stop.get('id', stop.get('location')))
                    location = None
                    for loc_id, loc in scenario.locations.items():
                        if str(loc_id) == str(stop_id):
                            location = loc
                            break
                    
                    if location:
                        lat = getattr(location, 'lat', None) or getattr(location, 'latitude', None)
                        lon = getattr(location, 'lon', None) or getattr(location, 'longitude', None)
                        if lat is not None and lon is not None:
                            route_coords.append([lat, lon])
                
                # Draw OSRM-based route lines between consecutive stops
                if len(route_coords) > 1:
                    print(f"  🗺️ Drawing OSRM routes for {vehicle_id} - Day {day_num} ({len(route_coords)} stops)")
                    
                    # Get OSRM routes between consecutive stops
                    all_route_coords = []
                    
                    for i in range(len(route_coords) - 1):
                        start_coord = route_coords[i]
                        end_coord = route_coords[i + 1]
                        
                        # Get street-following route from OSRM
                        street_route = get_street_route(start_coord, end_coord)
                        
                        # Add this segment to the full route (avoid duplicating points)
                        if i == 0:
                            all_route_coords.extend(street_route)
                        else:
                            # Skip the first point to avoid duplication
                            all_route_coords.extend(street_route[1:])
                    
                    # Draw the complete route with OSRM-based paths
                    if len(all_route_coords) > 1:
                        folium.PolyLine(
                            locations=all_route_coords,
                            color=color,
                            weight=4,
                            opacity=0.8,
                            popup=f"{vehicle_id} - Day {day_num} (OSRM Routes)"
                        ).add_to(m)
                        print(f"    ✅ Added OSRM route with {len(all_route_coords)} points")
                    else:
                        # Fallback to simple line if OSRM fails completely
                        folium.PolyLine(
                            locations=route_coords,
                            color=color,
                            weight=4,
                            opacity=0.8,
                            popup=f"{vehicle_id} - Day {day_num} (Direct Routes)"
                        ).add_to(m)
                        print(f"    ⚠️ Used direct routes as fallback")
    
    print(f"✅ Added {markers_added} markers to the map")
    
    # Create legend
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 350px; height: auto; max-height: 400px;
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px;
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
                border-radius: 5px;
                overflow-y: auto;">
    <h4 style="margin-top:0; margin-bottom:10px;">VRP Solution - {len(route_info)} Vehicles</h4>
    <div style="max-height: 300px; overflow-y: auto;">
    '''
    
    # Add vehicle information to legend
    for vehicle_id, info in route_info.items():
        legend_html += f'''
        <div style="margin-bottom: 8px; padding: 5px; border-left: 4px solid {info['color']}; background-color: #f9f9f9;">
            <b>{vehicle_id}</b><br>
            <span style="font-size: 11px;">
                📍 {info['customers_served']} customers<br>
                🛏️ {info['overnight_stays']} overnight stays<br>
            </span>
        </div>
        '''
    
    legend_html += '''
    </div>
    <hr style="margin: 10px 0;">
    <div style="font-size: 11px;">
        <b>Legend:</b><br>
        🏭 Red: Main Depot<br>
        📦 Orange: Depot Bays<br>
        📤 Green: Pickup Points<br>
        📍 Colored: Deliveries<br>
        🛏️ Dark Blue: Overnight<br>
        <br>
        <b>Route Lines:</b><br>
        Colored lines show vehicle routes<br>
        using OSRM street routing<br>
    </div>
    </div>
    '''
    
    # Add layer control and legend
    folium.LayerControl().add_to(m)
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add title
    title_html = f'''
    <h3 align="center" style="font-size:16px"><b>Sequential Multi-Day VRP Solution</b></h3>
    <p align="center" style="font-size:12px">MODA Furgoni Scenario - {len(route_info)} Active Vehicles - 7 Days Max</p>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save the map
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_filename = f"vrp_interactive_map_{timestamp}.html"
    html_path = os.path.join(os.path.dirname(__file__), html_filename)
    
    m.save(html_path)
    print(f"📊 Map saved with {markers_added} markers and {len(route_info)} vehicle routes")
    
    return html_path

def test_overnight_scenario_comparison():
    """Compare both optimizers on the overnight test scenario."""
    print("🧪 VRP Optimizers Comparison Test on Overnight Test Scenario")
    print("=" * 80)
    
    # Import scenario creation function
    try:
        from vrp_scenarios import create_overnight_test_scenario
        print("📦 Using overnight test scenario")
    except ImportError:
        print("❌ Error: Could not import create_overnight_test_scenario")
        return
    
    # Create the test scenario
    print("\n📦 Creating overnight test scenario...")
    scenario = create_overnight_test_scenario()
    
    print(f"\n📊 Scenario Overview:")
    print(f"  - Name: {scenario.name}")
    print(f"  - Locations: {len(scenario.locations)}")
    print(f"  - Vehicles: {len(scenario.vehicles)}")
    print(f"  - Ride requests: {len(scenario.ride_requests)}")
    print(f"  - Is realistic: {getattr(scenario, 'is_realistic', False)}")
    
    # Convert vehicles to dict format for both optimizers
    vehicles_dicts = [{
        'id': v.id,
        'capacity': v.capacity,
        'volume_capacity': getattr(v, 'volume_capacity', v.capacity * 0.001),  # Estimate volume
        'cost_per_km': getattr(v, 'cost_per_km', 1.0),
        'start_location': v.depot_id,
        'end_location': v.depot_id,
        'max_time': getattr(v, 'max_time', 24 * 60)
    } for v in scenario.vehicles.values()]
    
    print(f"\n🚛 Vehicle Configuration:")
    for vehicle in vehicles_dicts:
        print(f"  - {vehicle['id']}: {vehicle['capacity']}kg, {vehicle['volume_capacity']:.2f}m³, "
              f"€{vehicle['cost_per_km']:.2f}/km, {vehicle['max_time']}min max")
    
    # Test constraint levels to evaluate
    constraint_levels = ["none", "capacity", "pickup_delivery", "time_windows", "full"]
    
    results = {}
    
    # Test Original Optimizer (Manhattan distance)
    print("\n" + "="*80)
    print("🔵 Testing Original Optimizer (Manhattan distance)")
    print("="*80)
    
    try:
        from vrp_optimizer_clean import CleanVRPOptimizer as OriginalOptimizer
        
        for level in constraint_levels:
            print(f"\n--- Testing constraint level: {level} ---")
            
            optimizer = OriginalOptimizer(vehicles=vehicles_dicts, locations=None, vrp_instance=scenario)
            optimizer.ride_requests = scenario.ride_requests
            
            start_time = time.time()
            try:
                result = optimizer.solve(constraint_level=level, verbose=False)
                solve_time = time.time() - start_time
                
                if result and len(result) >= 2:
                    solution, status = result[0], result[1]
                    success = solution is not None
                    total_distance = solution.get('total_distance', 0) if solution else 0
                    total_cost = solution.get('total_cost', 0) if solution else 0
                    
                    results[f"original_{level}"] = {
                        'success': success,
                        'status': status,
                        'solve_time': solve_time,
                        'total_distance': total_distance,
                        'total_cost': total_cost,
                        'optimizer': 'original',
                        'level': level
                    }
                    
                    print(f"✅ SUCCESS - Distance: {total_distance}km, Cost: €{total_cost:.2f}, Time: {solve_time:.1f}s")
                else:
                    results[f"original_{level}"] = {
                        'success': False,
                        'status': 'FAILED',
                        'solve_time': solve_time,
                        'optimizer': 'original',
                        'level': level
                    }
                    print(f"❌ FAILED - Time: {solve_time:.1f}s")
                    
            except Exception as e:
                solve_time = time.time() - start_time
                results[f"original_{level}"] = {
                    'success': False,
                    'status': f'ERROR: {str(e)}',
                    'solve_time': solve_time,
                    'optimizer': 'original',
                    'level': level
                }
                print(f"❌ ERROR: {e} - Time: {solve_time:.1f}s")
                
    except ImportError as e:
        print(f"❌ Could not import original optimizer: {e}")
    
    # Test OSM-Enhanced Optimizer 
    print("\n" + "="*80)
    print("🟢 Testing OSM-Enhanced Optimizer (Real routing)")
    print("="*80)
    
    # Print comparison summary
    print_comparison_summary(results, constraint_levels)

def print_comparison_summary(results: Dict, constraint_levels: list):
    """Print a comprehensive comparison summary."""
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Constraint Level':<20} {'Original':<25} {'OSM-Enhanced':<25} {'Winner'}")
    print("-" * 90)
    
    for level in constraint_levels:
        original_key = f"original_{level}"
        osm_key = f"osm_{level}"
        
        # Get results for this level
        orig_result = results.get(original_key, {})
        osm_result = results.get(osm_key, {})
        
        # Format original result
        if orig_result.get('success', False):
            orig_str = f"✅ {orig_result.get('total_distance', 0):.1f}km €{orig_result.get('total_cost', 0):.1f}"
        else:
            orig_str = f"❌ {orig_result.get('status', 'UNKNOWN')}"
        
        # Format OSM result  
        if osm_result.get('success', False):
            osm_str = f"✅ {osm_result.get('total_distance', 0):.1f}km €{osm_result.get('total_cost', 0):.1f}"
        else:
            osm_str = f"❌ {osm_result.get('status', 'UNKNOWN')}"
        
        # Determine winner
        winner = "TIE"
        if orig_result.get('success', False) and osm_result.get('success', False):
            orig_cost = orig_result.get('total_cost', float('inf'))
            osm_cost = osm_result.get('total_cost', float('inf'))
            if orig_cost < osm_cost:
                winner = "Original"
            elif osm_cost < orig_cost:
                winner = "OSM"
        elif orig_result.get('success', False):
            winner = "Original"
        elif osm_result.get('success', False):
            winner = "OSM"
        
        print(f"{level:<20} {orig_str:<25} {osm_str:<25} {winner}")
    
    # Print timing comparison
    print(f"\n⏱️  TIMING COMPARISON")
    print("-" * 50)
    for level in constraint_levels:
        orig_time = results.get(f"original_{level}", {}).get('solve_time', 0)
        osm_time = results.get(f"osm_{level}", {}).get('solve_time', 0)
        
        faster = "TIE"
        if orig_time > 0 and osm_time > 0:
            if orig_time < osm_time:
                faster = "Original"
            elif osm_time < orig_time:
                faster = "OSM"
        elif orig_time > 0:
            faster = "Original"
        elif osm_time > 0:
            faster = "OSM"
        
        print(f"{level:<20} {orig_time:.2f}s vs {osm_time:.2f}s → {faster}")
    
    # Print success rate comparison
    print(f"\n🎯 SUCCESS RATE COMPARISON")
    print("-" * 50)
    
    orig_successes = sum(1 for level in constraint_levels if results.get(f"original_{level}", {}).get('success', False))
    osm_successes = sum(1 for level in constraint_levels if results.get(f"osm_{level}", {}).get('success', False))
    total_tests = len(constraint_levels)
    
    print(f"Original Optimizer:    {orig_successes}/{total_tests} ({orig_successes/total_tests*100:.1f}%)")
    print(f"OSM-Enhanced Optimizer: {osm_successes}/{total_tests} ({osm_successes/total_tests*100:.1f}%)")
    
    # Print key differences
    print(f"\n🔍 KEY DIFFERENCES")
    print("-" * 50)
    print("Original Optimizer:")
    print("  ✅ Fast Manhattan distance calculation")
    print("  ✅ Consistent geometric distance model")
    print("  ❌ Less realistic distance estimates")
    print("  ❌ No real-world routing factors")
    
    print("\nOSM-Enhanced Optimizer:")
    print("  ✅ Real-world OSM routing with actual road networks")
    print("  ✅ Truck-specific speed adjustments")
    print("  ✅ More accurate distance and time estimates")
    print("  ❌ Slower due to API calls and complex calculations")
    print("  ❌ Dependent on external OSRM service")

def test_single_constraint_level():
    """Quick test with just one constraint level for debugging."""
    print("🔬 Quick Single Constraint Test (pickup_delivery)")
    print("=" * 60)
    
    try:
        from vrp_scenarios import create_overnight_test_scenario
    except ImportError:
        print("❌ Error: Could not import create_overnight_test_scenario")
        return
    
    scenario = create_overnight_test_scenario()
    
    # Convert vehicles
    vehicles_dicts = [{
        'id': v.id,
        'capacity': v.capacity,
        'volume_capacity': getattr(v, 'volume_capacity', v.capacity * 0.001),
        'cost_per_km': getattr(v, 'cost_per_km', 1.0),
        'start_location': v.depot_id,
        'end_location': v.depot_id,
        'max_time': getattr(v, 'max_time', 24 * 60)
    } for v in scenario.vehicles.values()]
    
    # Test both optimizers with pickup_delivery constraint
    constraint_level = "pickup_delivery"
    
    print(f"\n🔵 Testing Original Optimizer...")
    try:
        from vrp_optimizer_clean import CleanVRPOptimizer as OriginalOptimizer
        optimizer1 = OriginalOptimizer(vehicles=vehicles_dicts, locations=None, vrp_instance=scenario)
        optimizer1.ride_requests = scenario.ride_requests
        result1 = optimizer1.solve(constraint_level=constraint_level, verbose=True)
        print(f"Original result: {type(result1)} - {result1 is not None}")
    except Exception as e:
        print(f"❌ Original optimizer error: {e}")
    
    print(f"\n🟢 Testing OSM-Enhanced Optimizer...")
    try:
        from vrp_optimizer_clean_copy import CleanVRPOptimizer as OSMOptimizer
        optimizer2 = OSMOptimizer(vehicles=vehicles_dicts, locations=None, vrp_instance=scenario)
        optimizer2.ride_requests = scenario.ride_requests
        result2 = optimizer2.solve(constraint_level=constraint_level, verbose=True)
        print(f"OSM result: {type(result2)} - {result2 is not None}")
    except Exception as e:
        print(f"❌ OSM optimizer error: {e}")

def test_overnight_node_creation():
    """Test the sequential multi-day VRP optimizer on the realistic MODA furgoni scenario.
    
    This tests the sequential multi-day VRP solver on a complex, realistic scenario with:
    - Multiple vehicles with different capacities
    - Real pickup and delivery requests
    - Realistic time constraints and routing
    - Multi-day planning when daily limits are exceeded
    
    The solver should create overnight nodes when needed and ensure all vehicles return to depot.
    """
    print("\n" + "="*80)
    print("🚛 Testing Sequential Multi-Day VRP on MODA Furgoni Scenario")
    print("="*80)
    
    try:
        import sys
        import os
        import importlib.util
        from vrp_scenarios import create_furgoni_scenario
        
        print("📦 Using MODA furgoni scenario for multiday test")
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        return
    
    # Create the test scenario
    print("\n📦 Creating MODA furgoni scenario...")
    scenario = create_furgoni_scenario()
    
    # Analyze the scenario
    print(f"\n📊 Furgoni Scenario Analysis:")
    print(f"  - Total locations: {len(scenario.locations)}")
    print(f"  - Total vehicles: {len(scenario.vehicles)}")
    print(f"  - Total requests: {len(scenario.ride_requests)}")
    
    # Show some key locations
    depot_found = False
    pickup_locations = []
    delivery_locations = []
    
    for loc_id, loc in scenario.locations.items():
        if 'depot' in str(loc_id).lower():
            depot_found = True
            print(f"  ✅ Depot found: {loc_id} at ({loc.x}, {loc.y})")
            if hasattr(loc, 'address'):
                print(f"      Address: {loc.address}")
        elif any(keyword in str(loc_id).lower() for keyword in ['pickup', 'via', 'source']):
            pickup_locations.append(loc_id)
        else:
            delivery_locations.append(loc_id)
    
    print(f"  - Pickup locations: {len(pickup_locations)}")
    print(f"  - Delivery locations: {len(delivery_locations)}")
    
    if not depot_found:
        print("❌ No depot found in scenario locations.")
        print("Available locations:")
        for loc_id, loc in scenario.locations.items():
            print(f" - Location ID: {loc_id}")
            if hasattr(loc, 'address'):
                print(f"   Address: {loc.address}")
        return
    
    # Convert scenario data to format expected by SequentialMultiDayVRP
    
    # Get locations in format needed for SequentialMultiDayVRP
    locations = []
    for loc_id, loc in scenario.locations.items():
        location = {
            'id': str(loc_id),
            'x': float(loc.x),
            'y': float(loc.y),
            'demand': getattr(loc, 'demand', 0),
            'service_time': getattr(loc, 'service_time', 15),
            'address': getattr(loc, 'name', str(loc_id))
        }
        locations.append(location)
    
    # Convert vehicles to format needed for SequentialMultiDayVRP
    vehicles = []
    for v_id, v in scenario.vehicles.items():
        vehicle = {
            'id': str(v_id),
            'capacity': v.capacity,
            'volume_capacity': getattr(v, 'volume_capacity', v.capacity * 0.001),
            'cost_per_km': getattr(v, 'cost_per_km', 1.0),
            'max_daily_km': getattr(v, 'max_daily_km', 600),
            'max_time': getattr(v, 'max_time', 24 * 60)  # Make sure max_time is included
        }
        vehicles.append(vehicle)
    
    print(f"\n📊 Converted Data for Sequential Multi-Day VRP:")
    print(f"  - Locations: {len(locations)}")
    print(f"  - Vehicles: {len(vehicles)}")
    
    # Import and run the sequential multi-day VRP solver
    try:
        spec = importlib.util.spec_from_file_location("vrp_multiday", 
                                                    os.path.join(os.path.dirname(__file__), 
                                                                "vrp_multiday_sequential.py"))
        vrp_multiday = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vrp_multiday)
        
        print("\n📋 Testing Sequential Multi-Day VRP with overnight nodes...")
        
        # Create sequential VRP solver
        sequential_vrp = vrp_multiday.SequentialMultiDayVRP(vehicles, locations)
        
        # Use max_time from scenario vehicles instead of hardcoding
        # Get max_time from the first vehicle (assuming all vehicles use the same max_time)
        if vehicles and 'max_time' in vehicles[0]:
            sequential_vrp.daily_time_limit_minutes = vehicles[0]['max_time']
            print(f"  - Using vehicle max_time: {sequential_vrp.daily_time_limit_minutes} minutes per day")
        else:
            # Fall back to default
            sequential_vrp.daily_time_limit_minutes = 24 * 60  # Default to 24 hours
            print(f"  - Using default max_time: {sequential_vrp.daily_time_limit_minutes} minutes per day")
        
        # Solve the multi-day problem
        print("\n🚀 Solving sequential multi-day VRP...")
        solution = sequential_vrp.solve_sequential_multiday(max_days=7)
        
        if solution:
            print("\n✅ Sequential Multi-Day VRP solution found!")
            
            # Check if there are any overnight stays
            total_overnight_stays = 0
            overnight_locations = []
            
            for vehicle_id, route_data in solution['vehicle_routes'].items():
                if 'daily_routes' in route_data:
                    for day, day_route in route_data['daily_routes'].items():
                        if 'overnight_location' in day_route and day_route['overnight_location']:
                            overnight_location = day_route['overnight_location']
                            overnight_locations.append(overnight_location)
                            total_overnight_stays += 1
                            print(f"  🛏️ Vehicle {vehicle_id} - Day {day}: Overnight at {overnight_location}")
            
            print(f"\n📊 Total overnight stays: {total_overnight_stays}")
            
            # Analyze overnight positions in the solution
            print("\n📊 Analyzing overnight positions and route patterns:")
            
            # Get depot coordinates
            depot_coords = None
            for loc_id, loc in scenario.locations.items():
                if 'depot' in loc_id.lower():
                    depot_coords = (loc.x, loc.y)
                    print(f"  - Depot coordinates: {depot_coords}")
                    break
            
            if depot_coords:
                # Analyze all overnight positions
                all_overnight_positions = 0
                overnight_positions = []
                
                # Extract overnight positions from the solution
                for vehicle_id, route_data in solution['vehicle_routes'].items():
                    # Look for overnight positions in various formats
                    if 'overnight_positions' in route_data:
                        for day, pos in route_data['overnight_positions'].items():
                            x, y = pos
                            overnight_positions.append((day, vehicle_id, x, y))
                    
                    # Check for daily routes with overnight stops
                    if 'daily_routes' in route_data:
                        for day, day_route in route_data['daily_routes'].items():
                            if 'overnight_position' in day_route and day_route['overnight_position']:
                                pos = day_route['overnight_position']
                                if isinstance(pos, tuple) and len(pos) == 2:
                                    x, y = pos
                                    overnight_positions.append((day, vehicle_id, x, y))
                                elif isinstance(pos, dict) and 'x' in pos and 'y' in pos:
                                    x, y = pos['x'], pos['y']
                                    overnight_positions.append((day, vehicle_id, x, y))
                    
                    # Check for stops that are marked as overnight
                    # This handles the case where overnight positions are stored in the route stops
                    if 'stops' in route_data:
                        for stop in route_data['stops']:
                            if stop.get('is_overnight', False) and 'coordinates' in stop:
                                x, y = stop['coordinates']
                                day = stop.get('day', 'unknown')
                                overnight_positions.append((day, vehicle_id, x, y))
                                
                # Additionally, check all day solutions for overnight stops
                for day_num, day_solution in solution.get('daily_solutions', {}).items():
                    for vehicle_id, route in day_solution.get('routes', {}).items():
                        for stop in route.get('stops', []):
                            if stop.get('is_overnight', False) and 'coordinates' in stop:
                                x, y = stop['coordinates']
                                overnight_positions.append((day_num, vehicle_id, x, y))
                
                # Check if any overnight positions exist
                print("\n🛏️ Overnight positions found:")
                for day, vehicle_id, x, y in overnight_positions:
                    # Calculate distance from depot
                    if depot_coords:
                        distance_from_depot = ((x - depot_coords[0])**2 + 
                                             (y - depot_coords[1])**2)**0.5
                        
                        all_overnight_positions += 1
                        print(f"  • Day {day}, {vehicle_id}: ({x:.4f}, {y:.4f})")
                        print(f"    Distance from depot: {distance_from_depot:.2f} km")
                
                print(f"\n📊 Found {all_overnight_positions} overnight positions across all days")
            else:
                print("❌ Could not perform route analysis: missing depot coordinates")
                
            # Plot the solution
            try:
                plot_filename = sequential_vrp.plot_sequential_solution(solution, 
                                                               "Sequential Multi-Day VRP - MODA Furgoni Scenario")
                print(f"\n📊 Solution plotted and saved as: {plot_filename}")
            except Exception as plot_error:
                print(f"❌ Error plotting solution: {plot_error}")
            
            # Create interactive HTML map visualization
            try:
                print("\n🗺️ Creating interactive HTML map visualization...")
                html_map_path = create_interactive_vrp_map(scenario, solution, sequential_vrp)
                if html_map_path:
                    print(f"📊 Interactive map saved as: {html_map_path}")
                else:
                    print("❌ Could not create interactive map")
            except Exception as map_error:
                print(f"❌ Error creating interactive map: {map_error}")
                import traceback
                traceback.print_exc()
                
            return solution
        else:
            print("❌ Failed to solve sequential multi-day VRP")
            return None
        
    except ImportError as e:
        print(f"❌ Could not import Sequential Multi-Day VRP solver: {e}")
    except Exception as e:
        print(f"❌ Error in Sequential Multi-Day VRP test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Choose which test to run
    test_type = "overnight"  # Options: "full", "quick", "overnight"
    
    if test_type == "full":
        test_overnight_scenario_comparison()
    elif test_type == "quick":
        test_single_constraint_level()
    elif test_type == "overnight":
        test_overnight_node_creation()
    else:
        print("Available tests: 'full', 'quick', 'overnight'")
        print("Set test_type variable to choose which test to run.")
