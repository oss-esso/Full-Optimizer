"""
EPDT Solution Visualization Module

This module provides interactive map visualization functionality for EPDT algorithm solutions.
It creates HTML maps using folium to display routes, tasks, and vehicles in an interactive format.

Adapted from VRPMapVisualizer to work with EPDT data structures.
"""

import os
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import logging
import math
import time
import requests

# Import EPDT data structures
try:
    from epdt_data_structures import Solution, Route, Task, Vehicle, TaskType
except ImportError:
    try:
        from algo.epdt_data_structures import Solution, Route, Task, Vehicle, TaskType
    except ImportError:
        from .epdt_data_structures import Solution, Route, Task, Vehicle, TaskType

logger = logging.getLogger(__name__)


class EPDTMapVisualizer:
    """Enhanced EPDT solution visualizer with interactive GPS maps."""
    
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
                    logger.info("OSRM routing service is available")
                    self.osrm_url = "http://router.project-osrm.org"
                else:
                    logger.warning(f"OSRM returned status {response.status_code}")
                    self.osrm_url = "http://router.project-osrm.org"
            except Exception as e:
                logger.warning(f"OSRM service test failed: {e}")
                self.osrm_url = "http://router.project-osrm.org"
                
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
    
    def _decode_polyline(self, polyline_str: str) -> List[List[float]]:
        """
        Decode a Google polyline string to a list of [lat, lon] coordinates.
        Based on the Google polyline encoding algorithm.
        """
        try:
            coordinates = []
            index = 0
            lat = 0
            lng = 0

            while index < len(polyline_str):
                # Decode latitude
                result = 0
                shift = 0
                while True:
                    b = ord(polyline_str[index]) - 63
                    index += 1
                    result |= (b & 0x1f) << shift
                    shift += 5
                    if b < 0x20:
                        break
                lat += ~(result >> 1) if (result & 1) else (result >> 1)

                # Decode longitude
                result = 0
                shift = 0
                while True:
                    b = ord(polyline_str[index]) - 63
                    index += 1
                    result |= (b & 0x1f) << shift
                    shift += 5
                    if b < 0x20:
                        break
                lng += ~(result >> 1) if (result & 1) else (result >> 1)

                coordinates.append([lat / 1e5, lng / 1e5])

            return coordinates
        except Exception as e:
            logger.warning(f"Error decoding polyline: {e}")
            return []
    
    def _get_street_route(self, start_task, end_task):
        """Get actual street route from cached database or fallback to OSRM."""
        start_coords = [start_task.lat, start_task.lon]
        end_coords = [end_task.lat, end_task.lon]
        
        # First try to get cached route geometry from the route provider database
        try:
            try:
                from route_provider import get_route_provider
            except ImportError:
                try:
                    from algo.route_provider import get_route_provider  
                except ImportError:
                    from .route_provider import get_route_provider
            
            route_provider = get_route_provider()
            
            # Use location_id from tasks as node identifiers (this is how routes are cached)
            start_node_id = getattr(start_task, 'location_id', f"{start_coords[0]:.6f},{start_coords[1]:.6f}")
            end_node_id = getattr(end_task, 'location_id', f"{end_coords[0]:.6f},{end_coords[1]:.6f}")
            
            # Convert coordinates to the format expected by route provider (lon, lat)
            start_coords_lonlat = (start_coords[1], start_coords[0])
            end_coords_lonlat = (end_coords[1], end_coords[0])
            
            # Try to get route details from cache
            route_data = route_provider.get_route_details(
                start_node_id=start_node_id,
                end_node_id=end_node_id,
                start_coords=start_coords_lonlat,
                end_coords=end_coords_lonlat
            )
            
            if route_data and route_data.get('route_geometry'):
                geometry = route_data['route_geometry']
                
                # Handle different geometry formats from the database
                if isinstance(geometry, dict):
                    if 'coordinates' in geometry:
                        # GeoJSON format: coordinates are [lon, lat], convert to [lat, lon] for folium
                        coordinates = geometry['coordinates']
                        street_route = [[coord[1], coord[0]] for coord in coordinates]
                        
                        logger.info(f"✅ Using cached GeoJSON route geometry {start_node_id}->{end_node_id} ({len(street_route)} points)")
                        return street_route
                    elif 'encoded' in geometry:
                        # Encoded polyline - decode it!
                        encoded_polyline = geometry['encoded']
                        street_route = self._decode_polyline(encoded_polyline)
                        
                        if street_route:
                            logger.info(f"✅ Using decoded polyline route geometry {start_node_id}->{end_node_id} ({len(street_route)} points)")
                            return street_route
                        else:
                            logger.warning(f"⚠️  Polyline decode failed for {start_node_id}->{end_node_id}, falling back to OSRM")
                elif isinstance(geometry, list):
                    # Already in [lat, lon] format
                    logger.info(f"✅ Using cached list route geometry {start_node_id}->{end_node_id} ({len(geometry)} points)")
                    return geometry
                    
        except Exception as e:
            logger.warning(f"⚠️  Error getting cached route geometry: {e}")
        
        # Fallback: try OSRM if no cached geometry available
        logger.info(f"🌐 No cached geometry found, falling back to OSRM for route {getattr(start_task, 'location_id', 'unknown')}->{getattr(end_task, 'location_id', 'unknown')}")
        if not self.has_routing:
            logger.info(f"📏 OSRM not available, using dotted line fallback")
            return self._create_dotted_line(start_coords, end_coords, 5)
        
        try:
            # Use OSRM routing API to get actual route
            url = f"{self.osrm_url}/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}"
            params = {
                'overview': 'full', 
                'geometries': 'geojson',
                'steps': 'true'
            }
            
            logger.debug(f"Requesting route from OSRM for {start_coords} to {end_coords}")
            response = self.routing_session.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] == 'Ok' and 'routes' in data and len(data['routes']) > 0:
                # Extract coordinates from the route geometry
                route_coords = data['routes'][0]['geometry']['coordinates']
                # Convert from [lon, lat] to [lat, lon] for folium
                street_route = [[coord[1], coord[0]] for coord in route_coords]
                logger.info(f"🌐 OSRM route found with {len(street_route)} points")
                return street_route
            else:
                logger.warning(f"🌐 OSRM routing failed: {data.get('message', 'Unknown error')}, using dotted line")
                return self._create_dotted_line(start_coords, end_coords, 5)
                
        except Exception as e:
            logger.warning(f"Error getting OSRM route: {e}")
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
    
    def _calculate_task_arrival_times(self, route: Route) -> Dict[str, float]:
        """Calculate arrival times for each task in the route using HoS simulation."""
        task_times = {}
        
        try:
            # Import HoS simulation functions
            try:
                from second_level import _simulate_hos_advanced, _sort_tasks_chronologically, DriverState
            except ImportError:
                try:
                    from algo.second_level import _simulate_hos_advanced, _sort_tasks_chronologically, DriverState
                except ImportError:
                    from .second_level import _simulate_hos_advanced, _sort_tasks_chronologically, DriverState
            
            # Get sorted tasks and simulate the route
            sorted_tasks = _sort_tasks_chronologically(route.tasks)
            driver_state = DriverState()
            
            # Simple simulation to get task completion times
            current_time = 0.0
            
            for i, task in enumerate(sorted_tasks):
                if i == 0:
                    # First task: service time only
                    service_time = getattr(task, 'service_time', 15.0)
                    current_time += service_time
                else:
                    # Subsequent tasks: travel time + waiting + service time
                    prev_task = sorted_tasks[i-1]
                    
                    # Calculate travel time
                    try:
                        try:
                            from route_provider import calculate_travel_time_between_tasks
                        except ImportError:
                            try:
                                from algo.route_provider import calculate_travel_time_between_tasks
                            except ImportError:
                                from .route_provider import calculate_travel_time_between_tasks
                        travel_time = calculate_travel_time_between_tasks(prev_task, task, route.vehicle)
                    except:
                        travel_time = 30.0  # Default fallback
                    
                    # Add travel time
                    current_time += travel_time
                    
                    # Handle time window waiting (simplified)
                    if hasattr(task, 'earliest_time') and task.earliest_time and current_time < task.earliest_time:
                        wait_time = task.earliest_time - current_time
                        current_time += wait_time
                    
                    # Add service time
                    service_time = getattr(task, 'service_time', 15.0)
                    current_time += service_time
                
                task_times[task.id] = current_time
                
        except Exception as e:
            logger.warning(f"Error calculating task arrival times: {e}")
            # Fallback: assign sequential times
            for i, task in enumerate(route.tasks):
                task_times[task.id] = i * 60.0  # 1 hour intervals
        
        return task_times
    
    def _format_time(self, minutes: float) -> str:
        """Format time in minutes to readable format."""
        if minutes < 0:
            return "0m"
        
        total_minutes = int(minutes)
        days = total_minutes // 1440
        remaining_minutes = total_minutes % 1440
        hours = remaining_minutes // 60
        mins = remaining_minutes % 60
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if mins > 0 or len(parts) == 0:
            parts.append(f"{mins}m")
        
        return " ".join(parts)
    
    def _format_time_window(self, earliest_time: Optional[float], latest_time: Optional[float]) -> str:
        """Format time window for display."""
        if earliest_time is None or latest_time is None:
            return "No time window"
        
        # Convert multi-day time windows to readable format
        earliest_day = int(earliest_time // 1440)
        earliest_time_of_day = int(earliest_time % 1440)
        earliest_hours = earliest_time_of_day // 60
        earliest_minutes = earliest_time_of_day % 60
        
        latest_day = int(latest_time // 1440)
        latest_time_of_day = int(latest_time % 1440)
        latest_hours = latest_time_of_day // 60
        latest_mins = latest_time_of_day % 60
        
        if earliest_day == latest_day:
            return f"Day {earliest_day}: {earliest_hours:02d}:{earliest_minutes:02d}-{latest_hours:02d}:{latest_mins:02d}"
        else:
            return f"Day {earliest_day} {earliest_hours:02d}:{earliest_minutes:02d} - Day {latest_day} {latest_hours:02d}:{latest_mins:02d}"
    
    def create_interactive_map(self, solution: Solution, save_path: str) -> str:
        """
        Create an interactive folium map with the EPDT solution.
        
        Args:
            solution: EPDT Solution object containing routes and assignments
            save_path: Path where to save the HTML map file
            
        Returns:
            str: Path to the created HTML file
        """
        logger.info("Creating interactive map for EPDT solution...")
        
        # Check if solution has any routes with tasks
        has_tasks = any(route.tasks for route in solution.routes.values())
        if not has_tasks:
            logger.warning("No tasks found in solution. Cannot create map.")
            return None
        
        # Calculate map center from all task locations
        all_lats = []
        all_lons = []
        
        for route in solution.routes.values():
            for task in route.tasks:
                if hasattr(task, 'lat') and hasattr(task, 'lon'):
                    all_lats.append(task.lat)
                    all_lons.append(task.lon)
        
        if not all_lats:
            logger.warning("No GPS coordinates found in tasks. Cannot create GPS map.")
            return None
        
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
        
        # Create folium map with proper attribution
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add alternative tile layers
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
        
        # Add depot markers (look for depot in vehicle data or task locations)
        depot_locations = set()
        for route in solution.routes.values():
            if hasattr(route.vehicle, 'depot_lat') and hasattr(route.vehicle, 'depot_lon'):
                depot_locations.add((route.vehicle.depot_lat, route.vehicle.depot_lon, 
                                   getattr(route.vehicle, 'depot_location_id', 'depot')))
        
        for depot_lat, depot_lon, depot_id in depot_locations:
            popup_text = f"<b>Depot: {depot_id}</b><br>"
            popup_text += f"Coordinates: ({depot_lat:.4f}, {depot_lon:.4f})"
            
            folium.Marker(
                location=[depot_lat, depot_lon],
                popup=popup_text,
                tooltip=f"Depot: {depot_id}",
                icon=folium.Icon(color='red', icon='home', prefix='fa')
            ).add_to(m)
        
        # Store route information for legend
        route_info = {}
        vehicle_groups = {}  # Store FeatureGroup for each vehicle
        vehicle_layer_js_refs = []  # Store JavaScript references to layers
        
        # Process each vehicle route
        for i, (vehicle_id, route) in enumerate(solution.routes.items()):
            if not route.tasks:  # Skip empty routes
                continue
            
            color = self.colors[i % len(self.colors)]
            
            # Create a FeatureGroup for this vehicle's elements
            vehicle_group = folium.FeatureGroup(name=f"Vehicle {vehicle_id}")
            vehicle_groups[vehicle_id] = vehicle_group
            
            # Add JavaScript reference for direct layer access
            vehicle_layer_js_refs.append(f"'{vehicle_id}': null")  # Will be set after map creation
        
        # Process each vehicle route
        for i, (vehicle_id, route) in enumerate(solution.routes.items()):
            if not route.tasks:  # Skip empty routes
                continue
            
            color = self.colors[i % len(self.colors)]
            
            # Create a FeatureGroup for this vehicle's elements
            vehicle_group = folium.FeatureGroup(name=f"Vehicle {vehicle_id}")
            vehicle_groups[vehicle_id] = vehicle_group
            
            # Calculate task arrival times
            task_arrival_times = self._calculate_task_arrival_times(route)
            
            # Calculate route metrics
            route_distance = route.get_total_distance()
            total_orders = len(route.get_orders())
            total_tasks = len(route.tasks)
            
            # Store route information for legend
            driver_name = "Unassigned"
            if hasattr(route, 'driver') and route.driver:
                driver_name = route.driver.name if hasattr(route.driver, 'name') else str(route.driver)
            
            route_info[vehicle_id] = {
                'color': color,
                'distance': route_distance,
                'orders': total_orders,
                'tasks': total_tasks,
                'vehicle_type': route.vehicle.vehicle_type if hasattr(route.vehicle, 'vehicle_type') else 'Unknown',
                'driver': driver_name
            }
            
            # Add task markers
            for j, task in enumerate(route.tasks):
                if not hasattr(task, 'lat') or not hasattr(task, 'lon'):
                    continue
                
                # Build task popup information
                popup_text = f"<b>Task: {task.id}</b><br>"
                popup_text += f"Order: {task.order_id}<br>"
                popup_text += f"Location: {task.location_id}<br>"
                popup_text += f"Vehicle: {vehicle_id}<br>"
                popup_text += f"Sequence: #{j+1} of {len(route.tasks)}<br>"
                popup_text += f"Type: {task.task_type.value}<br>"
                
                # Load information
                if task.demand != 0 or task.volume != 0:
                    popup_text += f"Load: {task.demand:+.0f}kg, {task.volume:+.1f}m³<br>"
                
                # Time window information
                time_window_str = self._format_time_window(task.earliest_time, task.latest_time)
                popup_text += f"Time Window: {time_window_str}<br>"
                
                # Actual arrival time
                if task.id in task_arrival_times:
                    arrival_time_str = self._format_time(task_arrival_times[task.id])
                    popup_text += f"Est. Arrival: {arrival_time_str}<br>"
                
                popup_text += f"Service Time: {task.service_time:.0f}min<br>"
                popup_text += f"Coordinates: ({task.lat:.4f}, {task.lon:.4f})"
                
                # Choose icon based on task type
                if task.is_pickup():
                    icon = folium.Icon(color=color, icon='arrow-up', prefix='fa')
                    tooltip_text = f"Pickup: {task.location_id}"
                elif task.is_delivery():
                    icon = folium.Icon(color=color, icon='arrow-down', prefix='fa')
                    tooltip_text = f"Delivery: {task.location_id}"
                elif task.is_depot_return():
                    icon = folium.Icon(color=color, icon='home', prefix='fa')
                    tooltip_text = f"Return: {task.location_id}"
                elif hasattr(task, 'is_depot_start') and task.is_depot_start():
                    icon = folium.Icon(color=color, icon='play', prefix='fa')
                    tooltip_text = f"Start: {task.location_id}"
                else:
                    icon = folium.Icon(color=color, icon='map-marker', prefix='fa')
                    tooltip_text = f"Task: {task.location_id}"
                
                folium.Marker(
                    location=[task.lat, task.lon],
                    popup=popup_text,
                    tooltip=tooltip_text,
                    icon=icon
                ).add_to(vehicle_group)  # Add to vehicle's FeatureGroup instead of map
            
            # Generate route lines
            logger.info(f"Creating route for {vehicle_id} using cached geometries...")
            
            # Start from depot if available
            route_segments = []
            
            if hasattr(route.vehicle, 'depot_lat') and hasattr(route.vehicle, 'depot_lon'):
                if route.tasks:
                    # Create a mock depot task for route calculation
                    depot_task = type('DepotTask', (), {
                        'lat': route.vehicle.depot_lat,
                        'lon': route.vehicle.depot_lon,
                        'location_id': getattr(route.vehicle, 'depot_location_id', 'depot')
                    })()
                    
                    leg_route = self._get_street_route(depot_task, route.tasks[0])
                    route_segments.append((leg_route, f"Depot -> {route.tasks[0].location_id}"))
            
            # Add segments between consecutive tasks
            for j in range(len(route.tasks) - 1):
                current_task = route.tasks[j]
                next_task = route.tasks[j + 1]
                
                leg_route = self._get_street_route(current_task, next_task)
                route_segments.append((leg_route, f"{current_task.location_id} -> {next_task.location_id}"))
            
            # Add route segments to map
            for segment, popup_text in route_segments:
                try:
                    folium.PolyLine(
                        locations=segment,
                        color=color,
                        weight=4,
                        opacity=0.8,
                        popup=popup_text,
                        className=f'route-{vehicle_id}'
                    ).add_to(vehicle_group)  # Add to vehicle's FeatureGroup instead of map
                    
                    # Add direction arrow at the end of each segment
                    if self.has_folium_plugins and len(segment) >= 2:
                        try:
                            end_point = segment[-1]
                            folium.RegularPolygonMarker(
                                location=end_point,
                                number_of_sides=3,
                                radius=6,
                                rotation=45,
                                color=color,
                                fill_color=color,
                                fill_opacity=0.8
                            ).add_to(vehicle_group)  # Add to vehicle's FeatureGroup
                        except Exception as e:
                            logger.warning(f"Could not add direction arrow: {e}")
                
                except Exception as e:
                    logger.warning(f"Error adding route segment: {e}")
            
            # Add the vehicle's FeatureGroup to the map
            vehicle_group.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Create interactive legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 350px; height: auto; max-height: 400px;
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.2);
                    border-radius: 5px;
                    overflow-y: auto; overflow-x: hidden;
                    "><h4 style="margin-top:0; margin-bottom:10px;">EPDT Solution Legend</h4>
        <div style="max-height: 320px; overflow-y: auto; overflow-x: hidden; padding-right: 5px;">
        '''
        
        # Add summary statistics
        total_vehicles_used = solution.get_total_vehicles_used()
        total_assigned_orders = len(solution.get_assigned_orders())
        total_unassigned_orders = len(solution.unassigned_orders)
        total_distance = sum(route.get_total_distance() for route in solution.routes.values())
        
        legend_html += f'''
        <div style="margin-bottom: 10px; padding: 8px; background-color: #f0f0f0; border-radius: 3px;">
            <strong>Solution Summary:</strong><br>
            🚛 Vehicles used: {total_vehicles_used}<br>
            📦 Orders assigned: {total_assigned_orders}<br>
            ❌ Orders unassigned: {total_unassigned_orders}<br>
            🛣️ Total distance: {total_distance:.1f} km
        </div>
        '''
        
        # Add "Show All" button
        legend_html += f'''
        <div id="vehicle-show-all" style="margin-bottom: 8px; padding: 6px; background-color: #e6f3ff; border: 2px solid #0066cc; border-radius: 4px; cursor: pointer; text-align: center; font-weight: bold; color: #0066cc;" onclick="showAllVehicles()" onmouseover="this.style.backgroundColor='#cce6ff'" onmouseout="this.style.backgroundColor='#e6f3ff'">
            🚛 Show All Vehicles
        </div>
        '''
        
        # Add each vehicle's information to the legend as clickable entries
        for vehicle_id, info in route_info.items():
            legend_html += f'''
            <div id="vehicle-{vehicle_id}" class="vehicle-legend-entry" style="margin-bottom: 8px; padding: 6px; border-left: 4px solid {info['color']}; background-color: #f9f9f9; cursor: pointer; border-radius: 3px; transition: all 0.3s ease;" onclick="toggleVehicle('{vehicle_id}')" onmouseover="if(!this.classList.contains('selected')) this.style.backgroundColor='#e6f7ff'" onmouseout="if(!this.classList.contains('selected')) this.style.backgroundColor='#f9f9f9'">
                <div style="font-weight: bold; color: {info['color']};">🚚 {vehicle_id}</div>
                <div style="font-size: 11px; margin-top: 1px; color: #555;">👤 {info['driver']}</div>
                <div style="font-size: 12px; margin-top: 2px;">
                    Type: {info['vehicle_type']}<br>
                    📦 Orders: {info['orders']} | Tasks: {info['tasks']}<br>
                    🛣️ Distance: {info['distance']:.1f} km
                </div>
            </div>
            '''
        
        legend_html += '''
        </div>
        <hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">
        <div style="margin-top: 10px; font-size: 12px; color: #666;">
            💡 <strong>Tips:</strong><br>
            • Click vehicle entries to isolate routes<br>
            • Double-click map to show all routes<br>
            • Press Escape key to reset selection<br>
            🗺️ Click markers for task details<br>
            🎛️ Use layer control (top right) for map types
        </div>
        </div>
        '''
        
        # Add the legend to the map
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add JavaScript for interactive legend functionality
        vehicle_ids_js = list(route_info.keys())
        js_code = f'''
        <script>
        // Vehicle control functionality - Direct layer reference approach
        const vehicleIds = {vehicle_ids_js};
        let selectedVehicle = null;
        let mapInstance = null;
        let vehicleLayers = {{}};
        
        // Store direct references to vehicle layer elements
        let vehicleLayerElements = {{}};
        
        // Wait for map to be fully loaded
        document.addEventListener('DOMContentLoaded', function() {{
            setTimeout(function() {{
                initializeMapAndLayers();
            }}, 2000); // Increased timeout to ensure all layers are loaded
        }});
        
        function initializeMapAndLayers() {{
            console.log("=== ENHANCED LAYER DETECTION DEBUG ===");
            
            // Get the map instance
            const mapElements = document.querySelectorAll('[id*="map"]');
            for (let element of mapElements) {{
                const mapId = element.id;
                if (window[mapId] && window[mapId]._container) {{
                    mapInstance = window[mapId];
                    console.log("Found map instance:", mapId);
                    break;
                }}
            }}
            
            if (!mapInstance) {{
                console.error("Could not find map instance");
                return;
            }}
            
            console.log("Map instance found. Available methods:", Object.getOwnPropertyNames(mapInstance).filter(name => typeof mapInstance[name] === 'function').slice(0, 10));
            
            // NEW APPROACH: Instead of looking for layers in _layers, let's find them by CSS class
            console.log("=== TRYING CSS-based approach ===");
            vehicleIds.forEach(function(vehicleId) {{
                // Look for elements with route-specific classes
                const routeElements = document.querySelectorAll('.route-' + vehicleId);
                console.log("Found route elements for", vehicleId, ":", routeElements.length);
                
                if (routeElements.length > 0) {{
                    vehicleLayerElements[vehicleId] = routeElements;
                    console.log("✅ Stored CSS elements for vehicle:", vehicleId);
                }}
            }});
            
            // FALLBACK: Try to find layers through Leaflet's layer control
            if (mapInstance._controlLayers) {{
                console.log("Found layer control, examining overlays...");
                const layerControl = mapInstance._controlLayers;
                if (layerControl._layers) {{
                    layerControl._layers.forEach(function(layerInfo, index) {{
                        if (layerInfo.name) {{
                            console.log("Layer control entry", index, ":", layerInfo.name);
                            vehicleIds.forEach(function(vehicleId) {{
                                if (layerInfo.name === "Vehicle " + vehicleId) {{
                                    vehicleLayers[vehicleId] = layerInfo.layer;
                                    console.log("✅ Found layer via control:", vehicleId);
                                }}
                            }});
                        }}
                    }});
                }}
            }}
            
            console.log("=== FINAL DETECTION RESULTS ===");
            console.log("Vehicle layers found:", Object.keys(vehicleLayers));
            console.log("Vehicle CSS elements found:", Object.keys(vehicleLayerElements));
            console.log("Expected vehicle IDs:", vehicleIds);
            
            // Use whichever method found layers
            if (Object.keys(vehicleLayers).length > 0) {{
                console.log("✅ Using Leaflet layer approach");
            }} else if (Object.keys(vehicleLayerElements).length > 0) {{
                console.log("✅ Using CSS element approach");
            }} else {{
                console.error("❌ No vehicle layers found with any method");
            }}
        }}
        
        function showAllVehicles() {{
            console.log("Showing all vehicles...");
            
            // Method 1: Show all vehicle layers (if found)
            vehicleIds.forEach(function(vehicleId) {{
                const layer = vehicleLayers[vehicleId];
                if (layer && mapInstance && !mapInstance.hasLayer(layer)) {{
                    mapInstance.addLayer(layer);
                    console.log("Added layer for vehicle:", vehicleId);
                }}
            }});
            
            // Method 2: Show all CSS elements (fallback)
            vehicleIds.forEach(function(vehicleId) {{
                const elements = vehicleLayerElements[vehicleId];
                if (elements) {{
                    elements.forEach(function(element) {{
                        element.style.display = '';
                        element.style.opacity = '0.8';
                    }});
                    console.log("Restored CSS elements for vehicle:", vehicleId);
                }}
            }});
            
            // Reset all legend entries
            updateLegendStyles(null);
            
            // Update show all button
            const showAllBtn = document.getElementById('vehicle-show-all');
            if (showAllBtn) {{
                showAllBtn.style.backgroundColor = '#e6f3ff';
                showAllBtn.style.border = '2px solid #0066cc';
                showAllBtn.style.fontWeight = 'bold';
            }}
            
            selectedVehicle = null;
            console.log("All vehicles restored");
        }}
        
        function toggleVehicle(vehicleId) {{
            console.log("Toggling vehicle:", vehicleId);
            
            if (selectedVehicle === vehicleId) {{
                // If clicking the same vehicle, show all
                showAllVehicles();
                return;
            }}
            
            // Method 1: Hide/show vehicle layers (if available)
            if (Object.keys(vehicleLayers).length > 0) {{
                console.log("Using layer-based approach");
                vehicleIds.forEach(function(id) {{
                    const layer = vehicleLayers[id];
                    if (layer && mapInstance) {{
                        if (id === vehicleId) {{
                            if (!mapInstance.hasLayer(layer)) {{
                                mapInstance.addLayer(layer);
                            }}
                            console.log("Showed layer for selected vehicle:", id);
                        }} else {{
                            if (mapInstance.hasLayer(layer)) {{
                                mapInstance.removeLayer(layer);
                            }}
                            console.log("Hidden layer for vehicle:", id);
                        }}
                    }}
                }});
            }}
            
            // Method 2: Hide/show CSS elements (fallback)
            if (Object.keys(vehicleLayerElements).length > 0) {{
                console.log("Using CSS-based approach");
                vehicleIds.forEach(function(id) {{
                    const elements = vehicleLayerElements[id];
                    if (elements) {{
                        elements.forEach(function(element) {{
                            if (id === vehicleId) {{
                                element.style.display = '';
                                element.style.opacity = '0.8';
                            }} else {{
                                element.style.display = 'none';
                            }}
                        }});
                        console.log("Updated CSS for vehicle:", id, id === vehicleId ? "shown" : "hidden");
                    }}
                }});
            }}
            
            // Update legend styling
            updateLegendStyles(vehicleId);
            
            // Update show all button
            const showAllBtn = document.getElementById('vehicle-show-all');
            if (showAllBtn) {{
                showAllBtn.style.backgroundColor = '#f8f9fa';
                showAllBtn.style.border = '2px solid #6c757d';
                showAllBtn.style.fontWeight = 'normal';
            }}
            
            selectedVehicle = vehicleId;
            console.log("Vehicle selection complete:", vehicleId);
        }}
        
        function updateLegendStyles(highlightedVehicleId) {{
            vehicleIds.forEach(function(vehicleId) {{
                const legendEntry = document.getElementById('vehicle-' + vehicleId);
                if (legendEntry) {{
                    if (highlightedVehicleId === vehicleId) {{
                        // Highlight selected vehicle
                        legendEntry.classList.add('selected');
                        legendEntry.style.backgroundColor = '#d4edda';
                        legendEntry.style.border = '2px solid #28a745';
                        legendEntry.style.fontWeight = 'bold';
                        legendEntry.style.opacity = '1.0';
                    }} else if (highlightedVehicleId !== null) {{
                        // Dim other vehicles when one is selected
                        legendEntry.classList.remove('selected');
                        legendEntry.style.backgroundColor = '#f5f5f5';
                        legendEntry.style.border = '1px solid #ccc';
                        legendEntry.style.fontWeight = 'normal';
                        legendEntry.style.opacity = '0.6';
                    }} else {{
                        // Reset to normal when showing all
                        legendEntry.classList.remove('selected');
                        legendEntry.style.backgroundColor = '#f9f9f9';
                        legendEntry.style.border = '';
                        legendEntry.style.fontWeight = 'normal';
                        legendEntry.style.opacity = '1.0';
                    }}
                }}
            }});
        }}
        
        // Add double-click to reset functionality
        document.addEventListener('dblclick', function(e) {{
            // Only reset if we're not double-clicking on legend items
            if (!e.target.closest('.vehicle-legend-entry') && selectedVehicle !== null) {{
                showAllVehicles();
            }}
        }});
        
        // Add keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape' && selectedVehicle !== null) {{
                showAllVehicles();
            }}
        }});
        
        console.log("EPDT Interactive Map with Enhanced Vehicle Controls Loaded");
        </script>
        '''
        m.get_root().html.add_child(folium.Element(js_code))
        
        # Save the map
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            m.save(save_path)
            logger.info(f"Interactive map saved to: {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Error saving map to {save_path}: {e}")
            return None


# Convenience function for easy access
def create_interactive_map(solution: Solution, save_path: str) -> str:
    """
    Create an interactive map for an EPDT solution.
    
    Args:
        solution: EPDT Solution object
        save_path: Path where to save the HTML map file
        
    Returns:
        str: Path to the created HTML file
    """
    visualizer = EPDTMapVisualizer()
    return visualizer.create_interactive_map(solution, save_path)
