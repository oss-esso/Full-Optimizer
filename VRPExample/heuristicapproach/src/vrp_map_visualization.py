"""
Enhanced visualization for VRP solutions with real GPS maps.
Supports both interactive maps (folium) and static maps with real backgrounds.
"""

import os
import folium
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# Optional dependencies for map backgrounds
try:
    import contextily as ctx
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    import pandas as pd
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False
    print("Optional map dependencies not available. Install with: pip install contextily geopandas")

from vrp_data_models import VRPResult

logger = logging.getLogger(__name__)

class VRPMapVisualizer:
    """Enhanced VRP visualizer with real GPS map backgrounds."""
    
    def __init__(self):
        # Generate colors for up to 100 vehicles using HSV color space
        self.colors = self._generate_vehicle_colors(100)
        
        # Try to import routing service for street-following routes
        try:
            import requests
            self.has_routing = True
            self.routing_session = requests.Session()
            # Test routing service availability with a simple request
            try:
                test_url = "http://router.project-osrm.org/route/v1/driving/-87.63,41.88;-87.64,41.89"
                response = self.routing_session.get(test_url, timeout=3)
                if response.status_code == 200:
                    logger.info("OSRM routing service is available")  # Remove Unicode check mark
                    self.osrm_url = "http://router.project-osrm.org"
                else:
                    logger.warning(f"OSRM returned status {response.status_code}")
                    # Try the development server as fallback
                    self.osrm_url = "http://dev.router.project-osrm.org"
                    logger.info(f"Using development OSRM server: {self.osrm_url}")
            except Exception as e:
                logger.warning(f"OSRM service test failed: {e}")
                self.osrm_url = "http://router.project-osrm.org"  # Use default anyway
                
        except ImportError:
            self.has_routing = False
            logger.warning("Requests not available - using straight lines for routes")
          # Check for folium plugins availability
        try:
            import folium.plugins
            self.has_folium_plugins = True
        except (ImportError, AttributeError):
            self.has_folium_plugins = False
            logger.warning("Folium plugins not available - using simplified routes")
    
    def _generate_vehicle_colors(self, n):
        """Generate n visually distinct colors using matplotlib's tab20, tab20b, and tab20c colormaps."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        # Try to use tab20, tab20b, tab20c for up to 60, then fall back to hsv
        base_maps = [plt.get_cmap('tab20'), plt.get_cmap('tab20b'), plt.get_cmap('tab20c')]
        colors = []
        for cmap in base_maps:
            for i in range(cmap.N):
                colors.append(mcolors.rgb2hex(cmap(i)))
                if len(colors) >= n:
                    return colors[:n]
        # If more colors needed, use HSV
        if len(colors) < n:
            hsv_colors = [mcolors.rgb2hex(plt.cm.hsv(i / n)) for i in range(n - len(colors))]
            colors.extend(hsv_colors)
        return colors[:n]
    
    def _get_street_route(self, start_coords, end_coords):
        """Get actual street route between two points using OSRM."""
        if not self.has_routing:
            return [start_coords, end_coords]
        
        try:
            # Use OSRM routing API to get actual route
            url = f"{self.osrm_url}/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}"
            params = {
                'overview': 'full',
                'geometries': 'geojson',
                'steps': 'true'  # Add step information for better routes
            }
            
            logger.info(f"Requesting route from {start_coords} to {end_coords}")
            response = self.routing_session.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] == 'Ok' and 'routes' in data and len(data['routes']) > 0:
                # Extract coordinates from the route geometry
                route_coords = data['routes'][0]['geometry']['coordinates']
                # Convert from [lon, lat] to [lat, lon] for folium
                street_route = [[coord[1], coord[0]] for coord in route_coords]
                logger.info(f"Route found with {len(street_route)} points")
                return street_route
            else:
                logger.warning(f"OSRM routing failed: {data.get('message', 'Unknown error')}")
                # Fall back to direct path but with multiple points for better visualization
                return self._create_dotted_line(start_coords, end_coords, 5)
                
        except Exception as e:
            logger.warning(f"Error getting street route: {e}")
            return self._create_dotted_line(start_coords, end_coords, 5)
    
    def _create_dotted_line(self, start, end, num_points=5):
        """Create a dotted line with multiple points between start and end."""
        points = []
        for i in range(num_points + 1):
            t = i / num_points
            lat = start[0] + t * (end[0] - start[0])
            lon = start[1] + t * (end[1] - start[1])
            points.append([lat, lon])
        return points
    
    def create_interactive_map(self, instance, result: VRPResult, save_path: Optional[str] = None) -> str:
        """Create an interactive folium map with the VRP solution."""
        
        # Check if this is a realistic scenario with GPS coordinates
        has_gps = any(hasattr(loc, 'lat') and hasattr(loc, 'lon') and 
                     loc.lat is not None and loc.lon is not None 
                     for loc in instance.locations.values())
        
        if not has_gps:
            logger.warning("No GPS coordinates found. Cannot create GPS map.")
            return None
        
        # Calculate map center
        lats = [loc.lat for loc in instance.locations.values() 
                if hasattr(loc, 'lat') and loc.lat is not None]
        lons = [loc.lon for loc in instance.locations.values() 
                if hasattr(loc, 'lon') and loc.lon is not None]
        
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        # Create folium map with proper attribution
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add alternative tile layers with proper attribution
        folium.TileLayer(
            tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            attr='OpenStreetMap',
            name='OpenStreetMap',
            overlay=False,
            control=True
        ).add_to(m)
        
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Add depot markers
        for loc_id, location in instance.locations.items():
            if not (hasattr(location, 'lat') and hasattr(location, 'lon')):
                continue
                
            if loc_id.startswith("depot"):
                popup_text = f"<b>Depot: {loc_id}</b><br>"
                if hasattr(location, 'address') and location.address:
                    popup_text += f"Address: {location.address}<br>"
                popup_text += f"Coordinates: ({location.lat:.4f}, {location.lon:.4f})"
                
                folium.Marker(
                    location=[location.lat, location.lon],
                    popup=popup_text,
                    tooltip=f"Depot: {loc_id}",
                    icon=folium.Icon(color='red', icon='home', prefix='fa')                ).add_to(m)
        
        # Add customer markers and routes with street following
        route_colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'black', 'gray', 'pink', 'lightblue', 'lightgreen']
          # Store route information for legend
        route_info = {}
        
        for i, (vehicle_id, route) in enumerate(result.routes.items()):
            if len(route) <= 2:  # Skip empty routes
                continue
                
            color = route_colors[i % len(route_colors)]
            
            # Calculate route metrics for legend
            route_distance = 0
            route_time = 0
            customer_count = len([loc for loc in route if not loc.startswith("depot")])
            
            # Calculate distance and time if available
            if hasattr(result, 'route_metrics') and vehicle_id in result.route_metrics:
                route_metrics = result.route_metrics[vehicle_id]
                route_distance = route_metrics.get('distance', 0)
                route_time = route_metrics.get('time', 0)
            else:
                # Fallback: estimate from locations if coordinates available
                for j in range(len(route) - 1):
                    start_loc = instance.locations.get(route[j])
                    end_loc = instance.locations.get(route[j + 1])
                    if (start_loc and end_loc and 
                        hasattr(start_loc, 'lat') and hasattr(start_loc, 'lon') and
                        hasattr(end_loc, 'lat') and hasattr(end_loc, 'lon')):
                        # Simple Haversine distance approximation
                        import math
                        lat1, lon1 = math.radians(start_loc.lat), math.radians(start_loc.lon)
                        lat2, lon2 = math.radians(end_loc.lat), math.radians(end_loc.lon)
                        dlat, dlon = lat2 - lat1, lon2 - lon1
                        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                        distance = 6371 * 2 * math.asin(math.sqrt(a))  # Earth radius in km
                        route_distance += distance
                        route_time += distance / 50 * 3600  # Assume 50 km/h average speed for time estimate
            
            # Store route information for legend
            route_info[vehicle_id] = {
                'color': color,
                'distance': route_distance,
                'time': route_time,
                'customers': customer_count,
                'route_sequence': route
            }
            
            # Add customer markers first
            for j, loc_id in enumerate(route):
                location = instance.locations.get(loc_id)
                if not location or not hasattr(location, 'lat') or not hasattr(location, 'lon'):
                    continue
                
                # Add customer markers (skip depot)
                if not loc_id.startswith("depot"):
                    popup_text = f"<b>{loc_id}</b><br>"
                    if hasattr(location, 'address') and location.address:
                        popup_text += f"Address: {location.address}<br>"
                    popup_text += f"Vehicle: {vehicle_id}<br>"
                    popup_text += f"Stop #{j}<br>"
                    if hasattr(location, 'demand'):
                        popup_text += f"Demand: {location.demand}<br>"
                    popup_text += f"Coordinates: ({location.lat:.4f}, {location.lon:.4f})"
                    
                    # Different icons for different location types
                    if loc_id.startswith("pickup"):
                        icon = folium.Icon(color=color, icon='arrow-up', prefix='fa')
                        tooltip_text = f"Pickup: {loc_id}"
                    elif loc_id.startswith("dropoff"):
                        icon = folium.Icon(color=color, icon='arrow-down', prefix='fa')
                        tooltip_text = f"Dropoff: {loc_id}"
                    else:
                        icon = folium.Icon(color=color, icon='shopping-cart', prefix='fa')
                        tooltip_text = f"Customer: {loc_id}"
                    
                    folium.Marker(
                        location=[location.lat, location.lon],
                        popup=popup_text,
                        tooltip=tooltip_text,
                        icon=icon
                    ).add_to(m)
            
            # Generate route coordinates for each leg
            all_route_segments = []
            print(f"Creating street-following route for {vehicle_id}...")
            
            for j in range(len(route) - 1):
                start_loc = instance.locations.get(route[j])
                end_loc = instance.locations.get(route[j + 1])
                
                if (start_loc and end_loc and 
                    hasattr(start_loc, 'lat') and hasattr(start_loc, 'lon') and
                    hasattr(end_loc, 'lat') and hasattr(end_loc, 'lon')):
                    
                    start_coords = [start_loc.lat, start_loc.lon]
                    end_coords = [end_loc.lat, end_loc.lon]
                    
                    # Get street route between points
                    leg_route = self._get_street_route(start_coords, end_coords)
                    all_route_segments.append((leg_route, f"Route: {route[j]} → {route[j+1]}"))
            
            # Add route segments to map with unique IDs for highlighting
            for segment, popup_text in all_route_segments:
                try:
                    # Each segment is its own polyline with vehicle-specific class
                    folium.PolyLine(
                        locations=segment,
                        color=color,
                        weight=4,
                        opacity=0.8,
                        popup=popup_text,
                        className=f'route-{vehicle_id}'  # Add class for JavaScript targeting
                    ).add_to(m)
                    
                    # Add direction arrow at the end of each segment if plugins available
                    if self.has_folium_plugins and len(segment) >= 2:
                        try:
                            # Calculate bearing for arrow direction
                            start_point = segment[-2]
                            end_point = segment[-1]
                            
                            # Add simple arrow marker instead of PolyLineTextPath
                            folium.RegularPolygonMarker(
                                location=end_point,
                                number_of_sides=3,
                                radius=6,
                                rotation=45,
                                color=color,
                                fill_color=color,
                                fill_opacity=0.8
                            ).add_to(m)
                            
                        except Exception as e:
                            logger.warning(f"Could not add direction arrow: {e}")
                
                except Exception as e:
                    logger.warning(f"Error adding route segment: {e}")
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Create color-coded legend for vehicles with interactive functionality
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 300px; height: auto; max-height: 400px;
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.2);
                    border-radius: 5px;
                    overflow-y: auto; overflow-x: hidden;
                    "><h4 style="margin-top:0; margin-bottom:10px;">Vehicle Routes Legend</h4>
        <div style="max-height: 320px; overflow-y: auto; overflow-x: hidden; padding-right: 5px;">
        '''
        
        # Add each vehicle's information to the legend with click handlers
        for vehicle_id, info in route_info.items():
            time_hours = info['time'] / 3600 if info['time'] > 0 else 0
            hours = int(time_hours)
            minutes = int((time_hours - hours) * 60)
            time_str = f"{hours}h {minutes}m" if time_hours > 0 else "N/A"
            
            legend_html += f'''
            <div class="legend-item" data-vehicle="{vehicle_id}" 
                 style="margin-bottom: 8px; padding: 5px; border-radius: 3px; cursor: pointer; 
                        transition: background-color 0.3s;" 
                 onmouseover="this.style.backgroundColor='#f0f0f0'" 
                 onmouseout="this.style.backgroundColor='white'"
                 onclick="toggleRouteHighlight('{vehicle_id}', '{info['color']}')">
                <span style="display: inline-block; width: 20px; height: 20px; 
                           background-color: {info['color']}; margin-right: 8px; 
                           border: 1px solid #000; vertical-align: middle;"></span>
                <strong>{vehicle_id}</strong><br>
                <span style="margin-left: 28px; font-size: 12px;">
                    📍 {info['customers']} customers<br>
                    📏 {info['distance']:.1f} km<br>
                    ⏱️ {time_str}<br>
                </span>
            </div>
            '''
        
        # Close the scrollable content div and add total runtime
        legend_html += '''
        </div>
        <hr style="margin: 10px 0;">
        <div style="font-size: 12px; text-align: center;">
            <strong>Total Runtime: ''' + f"{result.runtime * 1000:.1f}ms" + '''</strong>
        </div>
        <div style="font-size: 11px; text-align: center; color: #666; margin-top: 5px;">
            <em>Click vehicle names to highlight routes</em>
        </div>
        </div>
        '''
        
        # Add JavaScript for route highlighting
        javascript_code = '''
        <script>
        var highlightedVehicle = null;
        var originalStyles = {};
        
        function toggleRouteHighlight(vehicleId, vehicleColor) {
            // Get all route elements
            var allRoutes = document.querySelectorAll('path[stroke]');
            
            // If clicking the same vehicle, toggle off
            if (highlightedVehicle === vehicleId) {
                resetAllRoutes();
                highlightedVehicle = null;
                updateLegendStyles();
                return;
            }
            
            // Store original styles if first time
            if (Object.keys(originalStyles).length === 0) {
                allRoutes.forEach(function(route) {
                    var stroke = route.getAttribute('stroke');
                    var strokeWidth = route.getAttribute('stroke-width');
                    var strokeOpacity = route.getAttribute('stroke-opacity');
                    originalStyles[route] = {
                        stroke: stroke,
                        strokeWidth: strokeWidth,
                        strokeOpacity: strokeOpacity
                    };
                });
            }
            
            // Reset all routes to dimmed state
            allRoutes.forEach(function(route) {
                route.setAttribute('stroke-opacity', '0.2');
                route.setAttribute('stroke-width', '2');
            });
            
            // Highlight selected vehicle's routes
            allRoutes.forEach(function(route) {
                var stroke = route.getAttribute('stroke');
                if (stroke === vehicleColor) {
                    route.setAttribute('stroke-opacity', '1.0');
                    route.setAttribute('stroke-width', '6');
                    route.style.zIndex = '1000';
                }
            });
            
            highlightedVehicle = vehicleId;
            updateLegendStyles();
        }
        
        function resetAllRoutes() {
            var allRoutes = document.querySelectorAll('path[stroke]');
            allRoutes.forEach(function(route) {
                if (originalStyles[route]) {
                    route.setAttribute('stroke', originalStyles[route].stroke);
                    route.setAttribute('stroke-width', originalStyles[route].strokeWidth);
                    route.setAttribute('stroke-opacity', originalStyles[route].strokeOpacity);
                    route.style.zIndex = '';
                }
            });
        }
        
        function updateLegendStyles() {
            var legendItems = document.querySelectorAll('.legend-item');
            legendItems.forEach(function(item) {
                var vehicleId = item.getAttribute('data-vehicle');
                if (highlightedVehicle === vehicleId) {
                    item.style.backgroundColor = '#e6f3ff';
                    item.style.border = '2px solid #007acc';
                    item.style.fontWeight = 'bold';
                } else if (highlightedVehicle !== null) {
                    item.style.backgroundColor = '#f5f5f5';
                    item.style.border = '1px solid #ccc';
                    item.style.opacity = '0.6';
                    item.style.fontWeight = 'normal';
                } else {
                    item.style.backgroundColor = 'white';
                    item.style.border = 'none';
                    item.style.opacity = '1.0';
                    item.style.fontWeight = 'normal';
                }
            });
        }
        
        // Add double-click to reset
        document.addEventListener('dblclick', function() {
            if (highlightedVehicle !== null) {
                resetAllRoutes();
                highlightedVehicle = null;
                updateLegendStyles();
            }
        });
        </script>
        '''
        
        # Add legend and JavaScript to map
        m.get_root().html.add_child(folium.Element(legend_html))
        m.get_root().html.add_child(folium.Element(javascript_code))
        
        # Add map title with routing info and time estimates
        routing_info = "Routes follow actual streets using OSRM routing" if self.has_routing else "Routes use straight lines"
        
        # Calculate total time from route info
        total_route_time = sum(info['time'] for info in route_info.values())
        total_route_distance = sum(info['distance'] for info in route_info.values())
        
        # Format total route time
        time_info = ""
        if total_route_time > 0:
            time_hours = total_route_time / 3600
            hours = int(time_hours)
            minutes = int((time_hours - hours) * 60)
            time_info = f" | Total Travel Time: {hours}h {minutes}m"
        elif 'total_time' in result.metrics and result.metrics['total_time'] > 0:
            time_hours = result.metrics['total_time'] / 3600
            hours = int(time_hours)
            minutes = int((time_hours - hours) * 60)
            time_info = f" | Time-on-Route: {hours}h {minutes}m"
        
        # Use calculated distance if available, otherwise fall back to metrics
        display_distance = total_route_distance if total_route_distance > 0 else result.metrics.get("total_distance", 0)
        
        title_html = f'''
        <h3 align="center" style="font-size:16px"><b>VRP Solution: {instance.name}</b></h3>
        <p align="center">Total Distance: {display_distance:.2f} km | 
        Active Vehicles: {len(route_info)}{time_info} | 
        Runtime: {result.runtime:.2f}s</p>
        <p align="center" style="font-size:12px">
        <i>{routing_info}</i></p>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Save the map
        if save_path:
            m.save(save_path)
            logger.info(f"Interactive map with street routing saved to {save_path}")
        
        return save_path
    
    def create_static_map_with_background(self, instance, result: VRPResult, 
                                        save_path: Optional[str] = None) -> Optional[str]:
        """Create a static matplotlib plot with real map background using contextily."""
        
        if not HAS_CONTEXTILY:
            logger.warning("Contextily not available. Creating fallback GPS plot...")
            return self._create_fallback_gps_plot(instance, result, save_path)
        
        # Check if this is a realistic scenario with GPS coordinates
        has_gps = any(hasattr(loc, 'lat') and hasattr(loc, 'lon') and 
                     loc.lat is not None and loc.lon is not None 
                     for loc in instance.locations.values())
        
        if not has_gps:
            logger.warning("No GPS coordinates found. Cannot create GPS map.")
            return None
        
        try:
            # Create GeoDataFrame for locations
            location_data = []
            for loc_id, location in instance.locations.items():
                if hasattr(location, 'lat') and hasattr(location, 'lon'):
                    location_data.append({
                        'id': loc_id,
                        'geometry': Point(location.lon, location.lat),  # Note: lon, lat order for Point
                        'type': 'depot' if loc_id.startswith('depot') else 
                               'pickup' if loc_id.startswith('pickup') else
                               'dropoff' if loc_id.startswith('dropoff') else 'customer',
                        'address': getattr(location, 'address', ''),
                        'demand': getattr(location, 'demand', 0)
                    })
            
            if not location_data:
                logger.warning("No valid GPS coordinates found.")
                return None
            
            # Create GeoDataFrame
            gdf_locations = gpd.GeoDataFrame(location_data, crs='EPSG:4326')
            
            # Convert to Web Mercator for contextily
            gdf_locations = gdf_locations.to_crs('EPSG:3857')
            
            # Create figure
            fig, ax = plt.subplots(figsize=(15, 12))
            
            # Plot locations
            depot_mask = gdf_locations['type'] == 'depot'
            pickup_mask = gdf_locations['type'] == 'pickup'
            dropoff_mask = gdf_locations['type'] == 'dropoff'
            customer_mask = gdf_locations['type'] == 'customer'
            
            # Plot different location types
            if depot_mask.any():
                gdf_locations[depot_mask].plot(ax=ax, color='red', markersize=200, 
                                              marker='s', label='Depot', zorder=5)
            if pickup_mask.any():
                gdf_locations[pickup_mask].plot(ax=ax, color='green', markersize=100, 
                                               marker='^', label='Pickup', zorder=4)
            if dropoff_mask.any():
                gdf_locations[dropoff_mask].plot(ax=ax, color='blue', markersize=100, 
                                                marker='v', label='Dropoff', zorder=4)
            if customer_mask.any():
                gdf_locations[customer_mask].plot(ax=ax, color='orange', markersize=100, 
                                                 marker='o', label='Customer', zorder=4)
            
            # Plot routes
            route_colors = ['purple', 'brown', 'pink', 'gray', 'olive']
            
            for i, (vehicle_id, route) in enumerate(result.routes.items()):
                if len(route) <= 2:  # Skip empty routes
                    continue
                
                color = route_colors[i % len(route_colors)]
                
                # Create route line
                route_coords = []
                for loc_id in route:
                    location = instance.locations.get(loc_id)
                    if location and hasattr(location, 'lat') and hasattr(location, 'lon'):
                        # Convert to Web Mercator
                        point = Point(location.lon, location.lat)
                        point_gdf = gpd.GeoDataFrame([1], geometry=[point], crs='EPSG:4326')
                        point_gdf = point_gdf.to_crs('EPSG:3857')
                        coords = point_gdf.geometry.iloc[0].coords[0]
                        route_coords.append(coords)
                
                if len(route_coords) > 1:
                    route_line = LineString(route_coords)
                    route_gdf = gpd.GeoDataFrame([1], geometry=[route_line], crs='EPSG:3857')
                    route_gdf.plot(ax=ax, color=color, linewidth=3, alpha=0.7, 
                                  label=vehicle_id, zorder=3)
            
            # Add map background with proper attribution and fallbacks
            basemap_added = False
            
            # Try different basemap sources in order of preference
            basemap_sources = [
                (ctx.providers.OpenStreetMap.Mapnik, "OpenStreetMap"),
                (ctx.providers.CartoDB.Positron, "CartoDB"),
                (ctx.providers.Stamen.TonerLite, "Stamen Toner"),
                ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}", "Esri")
            ]
            
            for source, name in basemap_sources:
                try:
                    if isinstance(source, str):
                        # Custom URL with proper attribution
                        ctx.add_basemap(ax, crs=gdf_locations.crs, source=source, 
                                      attribution=f"© {name}", alpha=0.8)
                    else:
                        # Provider object
                        ctx.add_basemap(ax, crs=gdf_locations.crs, source=source, alpha=0.8)
                    
                    basemap_added = True
                    logger.info(f"Successfully added {name} basemap")
                    break
                    
                except Exception as e:
                    logger.warning(f"Could not add {name} basemap: {e}")
                    continue
            
            if not basemap_added:
                logger.warning("Could not add any basemap. Using plain background.")
                ax.set_facecolor('#f0f0f0')  # Light gray background as fallback
            
            # Customize plot
            ax.set_title(f'VRP Solution on Real Map - {instance.name}\n'
                        f'Distance: {result.metrics.get("total_distance", 0):.2f} | '
                        f'Vehicles: {result.metrics.get("vehicles_used", 0)} | '
                        f'Runtime: {result.runtime:.2f}s', fontsize=14)
            
            # Remove axis ticks (they're in Web Mercator coordinates)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add legend
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Static map saved to {save_path}")
            
            plt.close()
            return save_path
            
        except Exception as e:
            logger.error(f"Error creating static map: {e}")
            # Create a fallback plot without map background
            return self._create_fallback_gps_plot(instance, result, save_path)
    
    def _create_fallback_gps_plot(self, instance, result: VRPResult, save_path: Optional[str] = None) -> Optional[str]:
        """Create a GPS coordinate plot without map background as fallback."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot locations using GPS coordinates
            for loc_id, location in instance.locations.items():
                if not (hasattr(location, 'lat') and hasattr(location, 'lon')):
                    continue
                
                if loc_id.startswith("depot"):
                    plt.scatter(location.lon, location.lat, c='red', s=200, marker='s', 
                               label='Depot' if 'Depot' not in plt.gca().get_legend_handles_labels()[1] else "")
                elif loc_id.startswith("pickup"):
                    plt.scatter(location.lon, location.lat, c='green', s=100, marker='^', 
                               label='Pickup' if 'Pickup' not in plt.gca().get_legend_handles_labels()[1] else "")
                elif loc_id.startswith("dropoff"):
                    plt.scatter(location.lon, location.lat, c='blue', s=100, marker='v', 
                               label='Dropoff' if 'Dropoff' not in plt.gca().get_legend_handles_labels()[1] else "")
                else:
                    plt.scatter(location.lon, location.lat, c='orange', s=100, marker='o', 
                               label='Customer' if 'Customer' not in plt.gca().get_legend_handles_labels()[1] else "")
                
                # Add location labels
                label_text = loc_id
                if hasattr(location, 'address') and location.address:
                    label_text += f"\n{location.address[:20]}..."
                plt.annotate(label_text, (location.lon, location.lat), xytext=(5, 5), 
                            textcoords='offset points', fontsize=7)
            
            # Plot routes
            route_colors = ['purple', 'brown', 'pink', 'gray', 'olive']
            for i, (vehicle_id, route) in enumerate(result.routes.items()):
                if len(route) <= 2:
                    continue
                
                color = route_colors[i % len(route_colors)]
                
                # Try to get improved routes with multiple points
                improved_route_segments = []
                
                for j in range(len(route) - 1):
                    start_loc = instance.locations.get(route[j])
                    end_loc = instance.locations.get(route[j + 1])
                    
                    if (start_loc and end_loc and 
                        hasattr(start_loc, 'lat') and hasattr(start_loc, 'lon') and
                        hasattr(end_loc, 'lat') and hasattr(end_loc, 'lon')):
                        
                        start_coords = [start_loc.lat, start_loc.lon]
                        end_coords = [end_loc.lat, end_loc.lon]
                        
                        # Create a dotted line with multiple points
                        dotted_line = self._create_dotted_line(start_coords, end_coords, 5)
                        improved_route_segments.append(dotted_line)
                
                # Plot each segment with multiple points for smoother curves
                for segment in improved_route_segments:
                    lons = [point[1] for point in segment]
                    lats = [point[0] for point in segment]
                    plt.plot(lons, lats, color=color, linewidth=2, alpha=0.7)
                
                # Add vehicle label only for the first segment
                if improved_route_segments:
                    first_segment = improved_route_segments[0]
                    plt.text(first_segment[0][1], first_segment[0][0], 
                            f"{vehicle_id}", color=color, fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
            
            plt.title(f'VRP Solution - GPS Coordinates - {instance.name}\n'
                     f'Distance: {result.metrics.get("total_distance", 0):.2f} | '
                     f'Vehicles: {result.metrics.get("vehicles_used", 0)} | '
                     f'Runtime: {result.runtime:.2f}s')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # Add note about map background
            plt.figtext(0.5, 0.01, 
                       "Note: Map background unavailable. Install contextily and geopandas for real map tiles.",
                       ha="center", fontsize=9, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
            
            plt.tight_layout()
            
            if save_path:
                fallback_path = save_path.replace('.png', '_gps_fallback.png')
                plt.savefig(fallback_path, dpi=300, bbox_inches='tight')
                logger.info(f"Fallback GPS plot saved to {fallback_path}")
                plt.close()
                return fallback_path
            
            plt.close()
            return None
            
        except Exception as e:
            logger.error(f"Error creating fallback GPS plot: {e}")
            plt.close()
            return None

def create_all_map_visualizations(instance, result: VRPResult, results_dir: str, scenario_name: str):
    """Create both interactive and static map visualizations."""
    
    # Check if this scenario has GPS coordinates
    has_gps = any(hasattr(loc, 'lat') and hasattr(loc, 'lon') and 
                 loc.lat is not None and loc.lon is not None 
                 for loc in instance.locations.values())
    
    if not has_gps:
        logger.info(f"Scenario {scenario_name} has no GPS coordinates. Skipping map visualizations.")
        return []
    
    visualizer = VRPMapVisualizer()
    created_files = []
    
    # Create interactive map
    try:
        interactive_path = os.path.join(results_dir, f"{scenario_name}_interactive_map.html")
        result_path = visualizer.create_interactive_map(instance, result, interactive_path)
        if result_path:
            created_files.append(result_path)
            print(f"  -> Interactive map: {result_path}")  # Use ASCII instead of Unicode
    except Exception as e:
        logger.warning(f"Could not create interactive map: {e}")
        logger.info("This is often due to missing folium.plugins or network issues")
    
    # Create static map with background
    try:
        static_path = os.path.join(results_dir, f"{scenario_name}_map_background.png")
        created_path = visualizer.create_static_map_with_background(instance, result, static_path)
        if created_path:
            created_files.append(created_path)
            print(f"  -> Static map: {created_path}")
    except Exception as e:
        logger.warning(f"Could not create static map: {e}")
        logger.info("This is often due to missing contextily/geopandas dependencies")
    
    # Always create a fallback GPS plot
    if not created_files:
        try:
            fallback_path = os.path.join(results_dir, f"{scenario_name}_gps_fallback.png")
            fallback_result = visualizer._create_fallback_gps_plot(instance, result, fallback_path)
            if fallback_result:
                created_files.append(fallback_result)
                print(f"  -> GPS fallback plot: {fallback_result}")
        except Exception as e:
            logger.warning(f"Could not create fallback GPS plot: {e}")
    
    return created_files

def enhance_existing_plot_with_gps_info(instance, save_path: str):
    """Add GPS coordinate information to existing plots."""
    
    has_gps = any(hasattr(loc, 'lat') and hasattr(loc, 'lon') and 
                 loc.lat is not None and loc.lon is not None 
                 for loc in instance.locations.values())
    
    if has_gps:
        # Add GPS info text file
        gps_info_path = save_path.replace('.png', '_gps_info.txt')
        with open(gps_info_path, 'w') as f:
            f.write(f"GPS Coordinates for {instance.name}\n")
            f.write("=" * 50 + "\n\n")
            
            for loc_id, location in instance.locations.items():
                if hasattr(location, 'lat') and hasattr(location, 'lon'):
                    f.write(f"{loc_id}:\n")
                    f.write(f"  Latitude: {location.lat:.6f}\n")
                    f.write(f"  Longitude: {location.lon:.6f}\n")
                    if hasattr(location, 'address') and location.address:
                        f.write(f"  Address: {location.address}\n")
                    f.write("\n")
            
            # Add Google Maps links
            f.write("Google Maps Links:\n")
            f.write("-" * 20 + "\n")
            for loc_id, location in instance.locations.items():
                if hasattr(location, 'lat') and hasattr(location, 'lon'):
                    maps_url = f"https://www.google.com/maps?q={location.lat},{location.lon}"
                    f.write(f"{loc_id}: {maps_url}\n")
        
        return gps_info_path
    
    return None
    
    return None
