"""
Clean VRP Optimizer - Built step by step to debug constraint issues
"""
from typing import Dict, List, Tuple, Optional
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import logging
import matplotlib.pyplot as plt
import numpy as np
import requests
import time

class OSMDistanceCalculator:
    """OSM-based calculator for distances and travel times using actual routes."""
    
    def __init__(self, locations: List[Dict], osrm_url: str = "https://router.project-osrm.org",
                 truck_speed_ratios: Optional[Dict] = None, use_truck_speeds: bool = False):
        """
        Initialize the calculator with locations and OSRM service.
        
        Args:
            locations: List of location dictionaries with 'id', 'x', 'y' keys
            osrm_url: OSRM service URL for routing
            truck_speed_ratios: Dictionary of road type to speed ratio adjustments
            use_truck_speeds: Whether to apply truck speed adjustments
        """
        self.locations = locations
        self.osrm_url = osrm_url
        self.truck_speed_ratios = truck_speed_ratios or {}
        self.use_truck_speeds = use_truck_speeds
        self.location_index = {loc['id']: i for i, loc in enumerate(locations)}
        self.logger = logging.getLogger(__name__)
        
        # Cache for route data to avoid repeated API calls
        self.route_cache = {}
        
        # Pre-calculate distance and time matrices using actual OSRM routes
        self.distance_matrix = None
        self.time_matrix = None
        self._calculate_osm_matrices()
    
    def _calculate_osm_matrices(self):
        """Calculate distance and time matrices using OSRM routing."""
        n = len(self.locations)
        self.distance_matrix = np.zeros((n, n))
        self.time_matrix = np.zeros((n, n))
        
        print(f"🗺️ Calculating OSM route matrices for {n} locations...")
        
        # Try to use OSRM table service for bulk calculation (much faster)
        if self._calculate_bulk_matrix():
            print("✅ Successfully retrieved OSM route matrix")
            return
        
        # Fallback to individual route calculations
        print("⚠️ Bulk matrix failed, calculating individual routes...")
        successful_routes = 0
        total_routes = n * (n - 1)  # No self-routes
        
        for i, loc1 in enumerate(self.locations):
            for j, loc2 in enumerate(self.locations):
                if i != j:
                    try:
                        distance_km, time_minutes = self._get_osm_route(loc1, loc2)
                        self.distance_matrix[i, j] = distance_km
                        self.time_matrix[i, j] = time_minutes
                        successful_routes += 1
                        
                        # Add small delay to avoid overwhelming OSRM
                        time.sleep(0.1)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to get route {loc1['id']} → {loc2['id']}: {e}")
                        # Fallback to Haversine distance
                        distance_km = self._haversine_distance(loc1, loc2)
                        time_minutes = distance_km * 60 / 50  # 50 km/h average speed
                        
                        self.distance_matrix[i, j] = distance_km
                        self.time_matrix[i, j] = time_minutes
        
        print(f"✅ OSM routing completed: {successful_routes}/{total_routes} routes successful")
    
    def _calculate_bulk_matrix(self) -> bool:
        """Try to calculate matrix using OSRM table service (bulk calculation)."""
        try:
            # Build coordinate string for OSRM table service
            coordinates = []
            for loc in self.locations:
                # Use lat/lon if available, otherwise convert x/y
                if 'lat' in loc and 'lon' in loc:
                    coordinates.append(f"{loc['lon']},{loc['lat']}")
                else:
                    # Assume x/y are already in lon/lat format
                    coordinates.append(f"{loc['x']},{loc['y']}")
            
            coordinate_string = ";".join(coordinates)
            
            # Call OSRM table service
            url = f"{self.osrm_url}/table/v1/driving/{coordinate_string}"
            params = {
                'annotations': 'distance,duration'
            }
            
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] == 'Ok':
                # Extract distance matrix (convert from meters to kilometers)
                if 'distances' in data:
                    distance_matrix_m = np.array(data['distances'])
                    self.distance_matrix = distance_matrix_m / 1000.0  # Convert to km
                
                # Extract duration matrix (convert from seconds to minutes)
                if 'durations' in data:
                    duration_matrix_s = np.array(data['durations'])
                    self.time_matrix = duration_matrix_s / 60.0  # Convert to minutes
                    
                    # Apply truck speed adjustment if enabled (using average ratio)
                    if self.use_truck_speeds and self.truck_speed_ratios:
                        # Use a conservative average truck speed ratio for bulk calculations
                        avg_truck_ratio = sum(self.truck_speed_ratios.values()) / len(self.truck_speed_ratios)
                        self.time_matrix = self.time_matrix / avg_truck_ratio
                        print(f"   🚛 Applied average truck speed adjustment factor: {1/avg_truck_ratio:.3f}")
                
                # Add service times to time matrix
                for i, loc in enumerate(self.locations):
                    service_time = loc.get('service_time', 0)
                    if service_time > 0:
                        # Add service time to all outgoing routes from this location
                        self.time_matrix[i, :] += service_time
                
                return True
            else:
                self.logger.error(f"OSRM table service error: {data.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error calling OSRM table service: {str(e)}")
            return False
    
    def _get_osm_route(self, loc1: Dict, loc2: Dict, verbose: bool = False) -> Tuple[float, float]:
        """Get actual route distance and time between two locations using OSRM."""
        # Create cache key
        cache_key = f"{loc1['id']}→{loc2['id']}"
        if cache_key in self.route_cache:
            return self.route_cache[cache_key]
        
        # Get coordinates
        if 'lat' in loc1 and 'lon' in loc1:
            from_coords = f"{loc1['lon']},{loc1['lat']}"
            to_coords = f"{loc2['lon']},{loc2['lat']}"
            from_lat_lon = (loc1['lat'], loc1['lon'])
            to_lat_lon = (loc2['lat'], loc2['lon'])
        else:
            # Assume x/y are in lon/lat format
            from_coords = f"{loc1['x']},{loc1['y']}"
            to_coords = f"{loc2['x']},{loc2['y']}"
            from_lat_lon = (loc1['y'], loc1['x'])
            to_lat_lon = (loc2['y'], loc2['x'])
        
        # Call OSRM route service
        url = f"{self.osrm_url}/route/v1/driving/{from_coords};{to_coords}"
        params = {
            'overview': 'full',  # Get geometry for point count
            'geometries': 'geojson',
            'alternatives': 'false',
            'steps': 'true' if self.use_truck_speeds else 'false'  # Get step info for truck speed analysis
        }
        
        if verbose:
            print(f"🌐 OSM Route Request: {loc1['id']} → {loc2['id']}")
            print(f"   From: {from_lat_lon} ({loc1['id']})")
            print(f"   To: {to_lat_lon} ({loc2['id']})")
            print(f"   URL: {url}")
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data['code'] == 'Ok' and 'routes' in data and len(data['routes']) > 0:
            route = data['routes'][0]
            
            # Extract distance (convert from meters to kilometers)
            distance_km = route['distance'] / 1000.0
            
            # Extract duration (convert from seconds to minutes)
            duration_minutes = route['duration'] / 60.0
            
            # Count route points if geometry is available
            route_points = 0
            if 'geometry' in route and 'coordinates' in route['geometry']:
                route_points = len(route['geometry']['coordinates'])
            
            # Analyze road types and apply truck speed adjustments if enabled
            adjusted_duration_minutes = duration_minutes
            
            if self.use_truck_speeds and 'legs' in route and len(route['legs']) > 0:
                road_analysis, adjusted_duration_minutes = self._analyze_and_adjust_for_truck_speeds(
                    route, duration_minutes, verbose)
            
            # Add service time at origin
            service_time = loc1.get('service_time', 0)
            total_time_minutes = adjusted_duration_minutes + service_time
            
            if verbose:
                print(f"   ✅ Route found with {route_points} points")
                print(f"   Distance: {distance_km:.2f} km")
                if self.use_truck_speeds and 'legs' in route:
                    print(f"   Car time: {duration_minutes:.1f} min → Truck time: {adjusted_duration_minutes:.1f} min")
                    print(f"   Truck adjustment factor: {adjusted_duration_minutes/duration_minutes:.3f}")
                else:
                    print(f"   Duration: {adjusted_duration_minutes:.1f} min")
                print(f"   Total time: {adjusted_duration_minutes:.1f} min + {service_time} min service = {total_time_minutes:.1f} min")
            
            # Cache the result
            self.route_cache[cache_key] = (distance_km, total_time_minutes)
            
            return distance_km, total_time_minutes
        else:
            error_msg = f"OSRM route error: {data.get('message', 'Unknown error')}"
            if verbose:
                print(f"   ❌ {error_msg}")
            raise Exception(error_msg)
    
    def _haversine_distance(self, loc1: Dict, loc2: Dict) -> float:
        """Fallback Haversine distance calculation in kilometers."""
        import math
        
        # Get coordinates (prefer lat/lon, fallback to x/y)
        if 'lat' in loc1 and 'lon' in loc1:
            lat1, lon1 = loc1['lat'], loc1['lon']
            lat2, lon2 = loc2['lat'], loc2['lon']
        else:
            lat1, lon1 = loc1['y'], loc1['x']  # Assume y=lat, x=lon
            lat2, lon2 = loc2['y'], loc2['x']
        
        # Haversine formula
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        distance_km = 6371 * 2 * math.asin(math.sqrt(a))  # Earth radius = 6371 km
        
        return distance_km
    
    def get_distance(self, from_location_id: str, to_location_id: str) -> int:
        """Get distance between two locations for OR-Tools (integer required)."""
        try:
            from_idx = self.location_index[from_location_id]
            to_idx = self.location_index[to_location_id]
            
            distance_km = self.distance_matrix[from_idx, to_idx]
            
            # Return as integer scaled for OR-Tools
            # Scale to make constraint penalties dominate over distance costs
            return int(distance_km * 100)  # 100x scale factor
            
        except KeyError as e:
            self.logger.warning(f"Location not found in distance calculation: {e}")
            return 999999  # High penalty for missing locations
    
    def get_time(self, from_location_id: str, to_location_id: str) -> int:
        """Get travel time between two locations for OR-Tools (integer required)."""
        try:
            from_idx = self.location_index[from_location_id]
            to_idx = self.location_index[to_location_id]
            
            time_minutes = self.time_matrix[from_idx, to_idx]
            
            # Return as integer minutes
            return int(time_minutes)
            
        except KeyError as e:
            self.logger.warning(f"Location not found in time calculation: {e}")
            return 9999  # High penalty for missing locations
    
    def get_distance_km(self, from_location_id: str, to_location_id: str) -> float:
        """Get actual distance in kilometers (for reporting/validation)."""
        try:
            from_idx = self.location_index[from_location_id]
            to_idx = self.location_index[to_location_id]
            return self.distance_matrix[from_idx, to_idx]
        except KeyError:
            return 0.0
    
    def get_time_minutes(self, from_location_id: str, to_location_id: str) -> float:
        """Get actual travel time in minutes (for reporting/validation)."""
        try:
            from_idx = self.location_index[from_location_id]
            to_idx = self.location_index[to_location_id]
            return self.time_matrix[from_idx, to_idx]
        except KeyError:
            return 0.0

    def get_route_details(self, from_location_id: str, to_location_id: str, verbose: bool = True) -> Dict:
        """Get detailed route information for testing/debugging purposes."""
        try:
            from_idx = self.location_index[from_location_id]
            to_idx = self.location_index[to_location_id]
            
            loc1 = self.locations[from_idx]
            loc2 = self.locations[to_idx]
            
            # Get fresh route data with detailed logging
            distance_km, time_minutes = self._get_osm_route(loc1, loc2, verbose=verbose)
            
            return {
                'from_location': from_location_id,
                'to_location': to_location_id,
                'distance_km': distance_km,
                'time_minutes': time_minutes,
                'success': True
            }
            
        except Exception as e:
            if verbose:
                print(f"   ❌ Route request failed: {e}")
            return {
                'from_location': from_location_id,
                'to_location': to_location_id,
                'distance_km': 0.0,
                'time_minutes': 0.0,
                'success': False,
                'error': str(e)
            }

    def _analyze_and_adjust_for_truck_speeds(self, route: Dict, original_duration_minutes: float, 
                                           verbose: bool = False) -> Tuple[Dict, float]:
        """
        Analyze road types in the route and adjust travel time for truck speeds.
        
        Args:
            route: OSRM route response with steps
            original_duration_minutes: Original car travel time
            verbose: Whether to print detailed analysis
            
        Returns:
            Tuple of (road_analysis_dict, adjusted_duration_minutes)
        """
        road_type_distances = {}
        total_distance = 0
        
        # Analyze each leg and step
        for leg in route.get('legs', []):
            for step in leg.get('steps', []):
                step_distance = step.get('distance', 0) / 1000.0  # Convert to km
                total_distance += step_distance
                
                # Extract road type from step - try multiple methods
                road_type = self._extract_road_type(step)
                
                # Aggregate by road type
                if road_type not in road_type_distances:
                    road_type_distances[road_type] = 0
                road_type_distances[road_type] += step_distance
        
        # If we couldn't identify specific road types, use heuristics based on route characteristics
        if len(road_type_distances) == 1 and 'default' in road_type_distances:
            road_type_distances = self._estimate_road_types_from_route(route)
            total_distance = sum(road_type_distances.values())
        
        # Calculate percentages and weighted speed adjustment
        road_analysis = {}
        total_speed_adjustment = 0
        
        for road_type, distance in road_type_distances.items():
            percentage = (distance / total_distance * 100) if total_distance > 0 else 0
            
            # Get speed ratio for this road type
            speed_ratio = self.truck_speed_ratios.get(road_type, self.truck_speed_ratios.get('default', 1.0))
            
            road_analysis[road_type] = {
                'distance': distance,
                'percentage': percentage,
                'speed_ratio': speed_ratio
            }
            
            # Weight the speed adjustment by distance percentage
            total_speed_adjustment += (percentage / 100.0) * speed_ratio
        
        # Apply the weighted speed adjustment (slower speed = longer time)
        # time_truck = time_car / speed_ratio (since speed_ratio = speed_truck / speed_car)
        adjusted_duration_minutes = original_duration_minutes / total_speed_adjustment if total_speed_adjustment > 0 else original_duration_minutes
        
        return road_analysis, adjusted_duration_minutes
    
    def _extract_road_type(self, step: Dict) -> str:
        """Extract road type from a single OSRM step."""
        # Try to get road type from intersections
        if 'intersections' in step and len(step['intersections']) > 0:
            intersection = step['intersections'][0]
            if 'classes' in intersection and len(intersection['classes']) > 0:
                road_class = intersection['classes'][0]
                # Map OSRM classes to our road types
                if 'motorway' in road_class or 'highway' in road_class:
                    return 'motorway'
                elif 'trunk' in road_class:
                    return 'trunk'
                elif 'primary' in road_class:
                    return 'primary'
                elif 'secondary' in road_class:
                    return 'secondary'
                elif 'tertiary' in road_class:
                    return 'tertiary'
                elif 'residential' in road_class:
                    return 'residential'
                elif 'service' in road_class:
                    return 'service'
        
        # Try to get road type from step name or maneuver
        if 'name' in step:
            name = step['name'].lower()
            if any(highway in name for highway in ['highway', 'freeway', 'interstate', 'motorway']):
                return 'motorway'
            elif any(road in name for road in ['trunk', 'expressway']):
                return 'trunk'
            elif any(road in name for road in ['primary', 'main', 'major']):
                return 'primary'
            elif any(road in name for road in ['secondary', 'avenue', 'boulevard']):
                return 'secondary'
            elif any(road in name for road in ['street', 'road', 'drive']):
                return 'tertiary'
            elif any(road in name for road in ['lane', 'court', 'place']):
                return 'residential'
        
        return 'default'
    
    def _estimate_road_types_from_route(self, route: Dict) -> Dict[str, float]:
        """Estimate road type distribution based on route characteristics."""
        total_distance = route.get('distance', 0) / 1000.0  # Convert to km
        total_duration = route.get('duration', 0) / 60.0    # Convert to minutes
        
        if total_distance == 0:
            return {'default': 0}
        
        # Calculate average speed
        avg_speed_kmh = (total_distance / total_duration * 60) if total_duration > 0 else 50
        
        # Estimate road type distribution based on average speed and distance
        road_dist = {}
        
        if avg_speed_kmh > 80:  # High speed suggests motorway
            road_dist['motorway'] = total_distance * 0.6
            road_dist['primary'] = total_distance * 0.3
            road_dist['secondary'] = total_distance * 0.1
        elif avg_speed_kmh > 60:  # Medium-high speed suggests primary roads
            road_dist['primary'] = total_distance * 0.5
            road_dist['secondary'] = total_distance * 0.3
            road_dist['tertiary'] = total_distance * 0.2
        elif avg_speed_kmh > 40:  # Medium speed suggests secondary/tertiary
            road_dist['secondary'] = total_distance * 0.4
            road_dist['tertiary'] = total_distance * 0.4
            road_dist['residential'] = total_distance * 0.2
        else:  # Low speed suggests residential/service roads
            road_dist['tertiary'] = total_distance * 0.3
            road_dist['residential'] = total_distance * 0.5
            road_dist['service'] = total_distance * 0.2
        
        return road_dist


class CleanVRPOptimizer:
    """A clean VRP optimizer built step by step to debug constraint issues."""
    
    def __init__(self, vehicles=None, locations=None, vrp_instance=None, osrm_url: str = "https://router.project-osrm.org"):
        """
        Initializes the optimizer with vehicle and location data or VRPInstance.
        
        Args:
            vehicles: List of vehicle dictionaries
            locations: List of location dictionaries
            vrp_instance: VRPInstance object (alternative to vehicles/locations)
            osrm_url: OSRM service URL for OSM routing
        """
        self.vrp_instance = vrp_instance  # Store instance for truck speed profile access
        
        if vrp_instance is not None:
            # Convert VRPInstance to the expected format
            self.vehicles = self._convert_vehicles_from_instance(vrp_instance)
            self.locations = self._convert_locations_from_instance(vrp_instance)
            self.ride_requests = vrp_instance.ride_requests
            
            # Extract truck speed profile from VRPInstance
            truck_speed_ratios = vrp_instance.get_truck_speed_profile()
            use_truck_speeds = vrp_instance.should_use_truck_speeds()
            
            print(f"🚛 Truck speed profile extracted from scenario:")
            print(f"   Use truck speeds: {use_truck_speeds}")
            if use_truck_speeds:
                print(f"   Fleet composition speed ratios:")
                for road_type, ratio in truck_speed_ratios.items():
                    print(f"      {road_type}: {ratio:.3f}x")
        else:
            self.vehicles = vehicles or []
            self.locations = locations or []
            self.ride_requests = {}
            
            # Default truck speed settings when no VRPInstance provided
            truck_speed_ratios = {
                'motorway': 0.75, 'trunk': 0.70, 'primary': 0.67,
                'secondary': 0.71, 'tertiary': 0.75, 'residential': 0.60,
                'service': 0.67, 'default': 0.75
            }
            use_truck_speeds = False
        
        self.osrm_url = osrm_url
        self.logger = logging.getLogger(__name__)
        self._active_ride_requests = []  # Track only ride requests actually added as pickup-delivery pairs
        
        # Initialize OSM distance calculator with truck speed profile from scenario
        self.distance_calculator = OSMDistanceCalculator(
            self.locations, 
            osrm_url=osrm_url,
            truck_speed_ratios=truck_speed_ratios,
            use_truck_speeds=use_truck_speeds
        )
        
        print(f"🔧 Initialized with OSM routing via: {osrm_url}")
        if use_truck_speeds:
            print(f"🚛 Truck speed adjustments enabled based on scenario vehicle specifications")
    
    def _convert_vehicles_from_instance(self, instance):
        """Convert VRPInstance vehicles to dict format (no capacity logic, just copy attributes)."""
        vehicles = []
        for vehicle in instance.vehicles.values():
            vehicles.append({
                'id': vehicle.id,
                'capacity': getattr(vehicle, 'capacity', 0),  # Weight capacity in kg
                'volume_capacity': getattr(vehicle, 'volume_capacity', 0.0),  # Volume capacity in m³
                'cost_per_km': getattr(vehicle, 'cost_per_km', 1.0),  # Cost per kilometer
                'start_location': getattr(vehicle, 'depot_id', None),
                'end_location': getattr(vehicle, 'depot_id', None),
                'max_time': getattr(vehicle, 'max_time', 24 * 60)
            })
        return vehicles
    
    def _convert_locations_from_instance(self, instance):
        """Convert VRPInstance locations to dict format."""
        locations = []
        
        # First pass: find the depot time window
        depot_time_window = None
        print("  Looking for depot time window...")
        
        for location in instance.locations.values():
            print(f"    Checking location: {location.id}")

            if 'depot' in location.id.lower() and 'bay' not in location.id.lower():
                if hasattr(location, 'time_window_start') and hasattr(location, 'time_window_end'):
                    if location.time_window_start is not None and location.time_window_end is not None:
                        depot_time_window = (location.time_window_start, location.time_window_end)
                        print(f"    Found depot time window: {depot_time_window} from {location.id}")
                        break
        
        if depot_time_window is None:
            print("    ⚠️ No depot time window found, depot bays will have no time windows")
          
        # Second pass: convert all locations
        for location in instance.locations.values():
            loc_dict = {
                'id': location.id,
                'x': location.x,  # longitude
                'y': location.y,  # latitude
                'demand': location.demand,
                'address': getattr(location, 'address', location.id)
            }
            
            # Add service time - 30 minutes for all locations except depot
            if 'depot' in location.id.lower():
                loc_dict['service_time'] = 0  # No service time at depot
            else:
                loc_dict['service_time'] = 30  # 30 minutes service time for all other locations

            if hasattr(location, 'time_window_start') and hasattr(location, 'time_window_end'):
                if location.time_window_start is not None and location.time_window_end is not None:
                    loc_dict['time_window'] = (location.time_window_start, location.time_window_end)
                else:
                    loc_dict['time_window'] = (0, 1440)  # default full day
            
            locations.append(loc_dict)
        return locations

    def solve(self, constraint_level: str = "none", verbose: bool = True, use_hybrid_calculator: bool = False):
        """
        Solves the VRP with the specified level of constraints.
        
        Constraint levels:
        - "none": Just basic VRP
        - "capacity": VRP with dual capacity constraints (weight and volume)
        - "pickup_delivery": VRP with pickup-delivery constraints
        - "time_windows": VRP with time window constraints
        - "full": All constraints
        """
        print(f"🚀 Solving VRP with constraint level: {constraint_level}")
        print(f"   Vehicles: {len(self.vehicles)}")
        print(f"   Locations: {len(self.locations)}")
        
        # Analyze dual capacity constraints
        self._analyze_dual_capacity_constraints()
        
        # Import OR-Tools for actual solving
        from ortools.constraint_solver import routing_enums_pb2
        from ortools.constraint_solver import pywrapcp
        
        # Create the routing model
        manager = pywrapcp.RoutingIndexManager(len(self.locations), len(self.vehicles), 0)
        routing = pywrapcp.RoutingModel(manager)
        
        # Add distance callback with vehicle-specific costs
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            distance_km = self.distance_calculator.distance_matrix[from_node][to_node]
            # Return distance in meters for OR-Tools (base cost without vehicle-specific multiplier)
            return int(distance_km * 1000)
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        
        # Set vehicle-specific arc cost evaluators based on cost_per_km
        print(f"\n💰 Setting vehicle-specific costs:")
        for vehicle_id in range(len(self.vehicles)):
            vehicle = self.vehicles[vehicle_id]
            cost_per_km = vehicle.get('cost_per_km', 1.0)
            
            # Create vehicle-specific cost callback
            def make_vehicle_cost_callback(v_id, cost_multiplier):
                def vehicle_cost_callback(from_index, to_index):
                    from_node = manager.IndexToNode(from_index)
                    to_node = manager.IndexToNode(to_index)
                    distance_km = self.distance_calculator.distance_matrix[from_node][to_node]
                    # Apply vehicle-specific cost per km (scaled for OR-Tools)
                    return int(distance_km * cost_multiplier * 100)  # 100x scale for precision
                return vehicle_cost_callback
            
            vehicle_callback_index = routing.RegisterTransitCallback(
                make_vehicle_cost_callback(vehicle_id, cost_per_km))
            routing.SetArcCostEvaluatorOfVehicle(vehicle_callback_index, vehicle_id)
            
            print(f"   {vehicle['id']}: €{cost_per_km:.2f}/km")
        
        # Apply constraints based on level
        applied_constraints = ["distance"]
        
        if constraint_level in ["capacity", "full"]:
            # Add dual capacity constraints (weight and volume)
            self._add_dual_capacity_constraints(routing, manager)
            applied_constraints.append("dual_capacity")
        
        if constraint_level in ["pickup_delivery", "full"]:
            # Add pickup-delivery constraints
            self._add_pickup_delivery_constraints(routing, manager)
            applied_constraints.append("pickup_delivery")
        
        if constraint_level in ["time_windows", "full"]:
            # Add time window constraints
            self._add_time_window_constraints(routing, manager)
            applied_constraints.append("time_windows")
        
        print(f"✅ Constraints applied: {applied_constraints}")
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.time_limit.seconds = 30
        
        # Set high cost for unassigned nodes to allow feasible solutions even with tight constraints
        # This is critical for handling complex constraint combinations
        penalty_cost = 10000000  # Very high penalty for unassigned nodes
        unassigned_nodes_count = 0
        for node in range(routing.Size()):
            if not routing.IsStart(node) and not routing.IsEnd(node):
                routing.AddDisjunction([node], penalty_cost)
                unassigned_nodes_count += 1
        
        print(f"🔧 Node-dropping penalties added:")
        print(f"   - Nodes with disjunctions: {unassigned_nodes_count}")
        print(f"   - Penalty cost per unassigned node: {penalty_cost:,}")
        
        # Solve the problem
        print(f"\n🔍 Solving with OR-Tools...")
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            print(f"✅ Solution found!")
            solution_data = self._extract_solution(routing, manager, solution)
            solution_data['status'] = 'SUCCESS'
            solution_data['truck_speed_enabled'] = self.distance_calculator.use_truck_speeds
            solution_data['applied_constraints'] = applied_constraints
            return solution_data
        else:
            print(f"❌ No solution found")
            return {
                'status': 'NO_SOLUTION',
                'message': 'OR-Tools could not find a solution',
                'applied_constraints': applied_constraints
            }
    
    def _analyze_dual_capacity_constraints(self):
        """Analyze dual capacity constraints (weight and volume) for the fleet."""
        print(f"\n📦 Analyzing Dual Capacity Constraints:")
        print("-" * 50)
        
        # Analyze vehicle capacities and costs
        total_weight_capacity = 0
        total_volume_capacity = 0
        
        for vehicle in self.vehicles:
            weight_cap = vehicle.get('capacity', 0)
            volume_cap = vehicle.get('volume_capacity', 0.0)
            cost_per_km = vehicle.get('cost_per_km', 1.0)
            total_weight_capacity += weight_cap
            total_volume_capacity += volume_cap
            
            print(f"   {vehicle['id']}: {weight_cap}kg, {volume_cap:.1f}m³, €{cost_per_km:.2f}/km")
        
        print(f"\n   Total fleet capacity: {total_weight_capacity}kg, {total_volume_capacity:.1f}m³")
        
        # Analyze cargo demands
        if hasattr(self, 'ride_requests') and self.ride_requests:
            total_weight_demand = 0
            total_volume_demand = 0
            
            # Handle both dict and list formats
            requests_to_analyze = []
            if isinstance(self.ride_requests, dict):
                requests_to_analyze = list(self.ride_requests.values())
            elif isinstance(self.ride_requests, list):
                requests_to_analyze = self.ride_requests
            
            for request in requests_to_analyze:
                weight = getattr(request, 'passengers', 0)
                volume = getattr(request, 'volume', 0.0)
                total_weight_demand += weight
                total_volume_demand += volume
            
            print(f"   Total cargo demand: {total_weight_demand}kg, {total_volume_demand:.1f}m³")
            
            # Calculate utilization ratios
            weight_ratio = (total_weight_demand / total_weight_capacity * 100) if total_weight_capacity > 0 else 0
            volume_ratio = (total_volume_demand / total_volume_capacity * 100) if total_volume_capacity > 0 else 0
            
            print(f"   Weight utilization: {weight_ratio:.1f}%")
            print(f"   Volume utilization: {volume_ratio:.1f}%")
            
            # Determine limiting constraint
            if weight_ratio > volume_ratio:
                print(f"   🔴 Weight is the limiting constraint ({weight_ratio:.1f}% vs {volume_ratio:.1f}%)")
            elif volume_ratio > weight_ratio:
                print(f"   📦 Volume is the limiting constraint ({volume_ratio:.1f}% vs {weight_ratio:.1f}%)")
            else:
                print(f"   ⚖️  Weight and volume constraints are balanced")
                
            # Check if any constraint is exceeded
            if weight_ratio > 100:
                print(f"   ⚠️  WARNING: Total weight demand exceeds fleet capacity!")
            if volume_ratio > 100:
                print(f"   ⚠️  WARNING: Total volume demand exceeds fleet capacity!")
                
        else:
            print(f"   No ride requests to analyze")
        
        print("-" * 50)
    
    def _add_dual_capacity_constraints(self, routing, manager):
        """Add dual capacity constraints for both weight and volume."""
        print("\n📦 Adding dual capacity constraints (weight and volume)...")
        
        # Weight capacity constraint
        print("  🔴 Adding weight capacity constraint...")
        def weight_demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            location = self.locations[from_node]
            
            weight_demand = 0
            # Handle ride requests for weight demand
            if hasattr(self, 'ride_requests') and self.ride_requests:
                requests_to_check = []
                if isinstance(self.ride_requests, dict):
                    requests_to_check = list(self.ride_requests.values())
                elif isinstance(self.ride_requests, list):
                    requests_to_check = self.ride_requests
                
                for request in requests_to_check:
                    # Check if this location is a pickup or dropoff
                    if hasattr(request, 'pickup_location') and hasattr(request, 'dropoff_location'):
                        if request.pickup_location == location['id']:
                            weight_demand += getattr(request, 'passengers', 0)  # pickup increases weight
                        elif request.dropoff_location == location['id']:
                            weight_demand -= getattr(request, 'passengers', 0)  # dropoff decreases weight
            
            return int(weight_demand)
        
        weight_callback_index = routing.RegisterUnaryTransitCallback(weight_demand_callback)
        weight_capacities = [int(vehicle.get('capacity', 0)) for vehicle in self.vehicles]
        
        routing.AddDimensionWithVehicleCapacity(
            weight_callback_index,
            0,  # No slack
            weight_capacities,
            True,  # Capacity cumulates from start
            'Weight'
        )
        print(f"    Weight capacities: {weight_capacities}")
        
        # Volume capacity constraint
        print("  📦 Adding volume capacity constraint...")
        def volume_demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            location = self.locations[from_node]
            
            volume_demand = 0.0
            # Handle ride requests for volume demand
            if hasattr(self, 'ride_requests') and self.ride_requests:
                requests_to_check = []
                if isinstance(self.ride_requests, dict):
                    requests_to_check = list(self.ride_requests.values())
                elif isinstance(self.ride_requests, list):
                    requests_to_check = self.ride_requests
                
                for request in requests_to_check:
                    # Check if this location is a pickup or dropoff
                    if hasattr(request, 'pickup_location') and hasattr(request, 'dropoff_location'):
                        if request.pickup_location == location['id']:
                            volume_demand += getattr(request, 'volume', 0.0)  # pickup increases volume
                        elif request.dropoff_location == location['id']:
                            volume_demand -= getattr(request, 'volume', 0.0)  # dropoff decreases volume
            
            # Convert to integer (multiply by 1000 to preserve precision)
            return int(volume_demand * 1000)
        
        volume_callback_index = routing.RegisterUnaryTransitCallback(volume_demand_callback)
        # Convert volume capacities to integers (multiply by 1000)
        volume_capacities = [int(vehicle.get('volume_capacity', 0.0) * 1000) for vehicle in self.vehicles]
        
        routing.AddDimensionWithVehicleCapacity(
            volume_callback_index,
            0,  # No slack
            volume_capacities,
            True,  # Capacity cumulates from start
            'Volume'
        )
        print(f"    Volume capacities: {[cap/1000 for cap in volume_capacities]}m³")
        
        print("✅ Dual capacity constraints added")
    
    def _add_pickup_delivery_constraints(self, routing, manager):
        """Add pickup and delivery constraints using ride requests."""
        print("\n🔄 Adding pickup-delivery constraints...")
        
        pickup_delivery_count = 0
        processed_pairs = set()  # Track processed pairs to avoid duplicates
        
        # Process ride requests to create pickup-delivery pairs
        if hasattr(self, 'ride_requests') and self.ride_requests:
            requests_to_process = []
            if isinstance(self.ride_requests, dict):
                requests_to_process = list(self.ride_requests.items())
            elif isinstance(self.ride_requests, list):
                requests_to_process = [(i, req) for i, req in enumerate(self.ride_requests)]
            
            for request_id, request in requests_to_process:
                if hasattr(request, 'pickup_location') and hasattr(request, 'dropoff_location'):
                    pickup_loc = request.pickup_location
                    dropoff_loc = request.dropoff_location
                    
                    # Skip if already processed
                    pair_key = f"{pickup_loc}->{dropoff_loc}"
                    if pair_key in processed_pairs:
                        continue
                    processed_pairs.add(pair_key)
                    
                    # Find location indices
                    pickup_idx = None
                    dropoff_idx = None
                    for i, loc in enumerate(self.locations):
                        if loc['id'] == pickup_loc:
                            pickup_idx = i
                        elif loc['id'] == dropoff_loc:
                            dropoff_idx = i
                    
                    if pickup_idx is not None and dropoff_idx is not None:
                        # Add pickup-delivery constraint
                        routing.AddPickupAndDelivery(pickup_idx, dropoff_idx)
                        
                        # Ensure same vehicle handles both
                        routing.solver().Add(
                            routing.VehicleVar(pickup_idx) == routing.VehicleVar(dropoff_idx)
                        )
                        
                        pickup_delivery_count += 1
                        print(f"    Added P-D pair: {pickup_loc} -> {dropoff_loc}")
        
        print(f"✅ Added {pickup_delivery_count} pickup-delivery constraints")
    
    def _add_time_window_constraints(self, routing, manager):
        """Add time window constraints."""
        print("\n⏰ Adding time window constraints...")
        
        # Time callback
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            # Use actual time from OSM calculator
            time_minutes = self.distance_calculator.time_matrix[from_node][to_node]
            return int(time_minutes)
        
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        
        # Add time dimension
        max_time_per_vehicle = 480  # 8 hours in minutes
        routing.AddDimension(
            time_callback_index,
            max_time_per_vehicle,  # Allow waiting
            max_time_per_vehicle,  # Maximum time per vehicle
            False,  # Don't force start cumul to zero
            'Time'
        )
        
        time_dimension = routing.GetDimensionOrDie('Time')
        
        # Add time window constraints for each location
        for i, location in enumerate(self.locations):
            if 'time_window' in location:
                start_time, end_time = location['time_window']
                time_dimension.CumulVar(i).SetRange(start_time, end_time)
                print(f"    {location['id']}: [{start_time}, {end_time}]")
        
        # Add time window constraints for vehicle start times
        for vehicle_id in range(len(self.vehicles)):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(0, 1440)  # 24 hours
        
        print("✅ Time window constraints added")
        
        print("✅ Time window constraints added")
    
    def _extract_solution(self, routing, manager, solution):
        """Extract solution from OR-Tools and analyze capacity usage per stop."""
        print(f"\n📊 Extracting solution with dual capacity analysis...")
        
        solution_data = {
            'routes': [],
            'total_distance': 0,
            'total_cost': 0,
            'total_time': 0,
            'capacity_analysis': []
        }
        
        # Get dimensions for capacity analysis
        weight_dimension = routing.GetDimensionOrDie('Weight')
        volume_dimension = routing.GetDimensionOrDie('Volume')
        
        total_distance = 0
        total_cost = 0
        
        for vehicle_id in range(len(self.vehicles)):
            vehicle = self.vehicles[vehicle_id]
            cost_per_km = vehicle.get('cost_per_km', 1.0)
            route = {
                'vehicle_id': vehicle['id'],
                'vehicle_capacity_kg': vehicle.get('capacity', 0),
                'vehicle_capacity_m3': vehicle.get('volume_capacity', 0.0),
                'vehicle_cost_per_km': cost_per_km,
                'stops': [],
                'total_distance': 0,
                'total_cost': 0,
                'capacity_usage': []
            }
            
            index = routing.Start(vehicle_id)
            route_distance_km = 0
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                location = self.locations[node_index]
                
                # Get capacity usage at this stop
                weight_var = weight_dimension.CumulVar(index)
                volume_var = volume_dimension.CumulVar(index)
                weight_usage = solution.Value(weight_var)
                volume_usage = solution.Value(volume_var) / 1000.0  # Convert back from integer
                
                # Calculate utilization percentages
                weight_utilization = (weight_usage / vehicle.get('capacity', 1)) * 100 if vehicle.get('capacity', 0) > 0 else 0
                volume_utilization = (volume_usage / vehicle.get('volume_capacity', 1)) * 100 if vehicle.get('volume_capacity', 0) > 0 else 0
                
                # Determine limiting constraint at this stop
                limiting_constraint = "weight" if weight_utilization > volume_utilization else "volume"
                if weight_utilization == volume_utilization:
                    limiting_constraint = "balanced"
                
                stop_info = {
                    'location_id': location['id'],
                    'weight_usage_kg': weight_usage,
                    'volume_usage_m3': round(volume_usage, 2),
                    'weight_utilization_pct': round(weight_utilization, 1),
                    'volume_utilization_pct': round(volume_utilization, 1),
                    'limiting_constraint': limiting_constraint
                }
                
                route['stops'].append(stop_info)
                route['capacity_usage'].append(stop_info)
                
                # Move to next stop and calculate distance
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                if not routing.IsEnd(index):
                    # Calculate actual distance for this segment
                    from_node = manager.IndexToNode(previous_index)
                    to_node = manager.IndexToNode(index)
                    segment_distance_km = self.distance_calculator.distance_matrix[from_node][to_node]
                    route_distance_km += segment_distance_km
            
            # Calculate route cost
            route_cost = route_distance_km * cost_per_km
            route['total_distance'] = round(route_distance_km, 2)
            route['total_cost'] = round(route_cost, 2)
            
            total_distance += route_distance_km
            total_cost += route_cost
            solution_data['routes'].append(route)
        
        solution_data['total_distance'] = round(total_distance, 2)
        solution_data['total_cost'] = round(total_cost, 2)
        
        # Print detailed capacity analysis
        self._print_capacity_analysis(solution_data)
        
        return solution_data
        
    def _print_capacity_analysis(self, solution_data):
        """Print detailed capacity analysis for each route and stop."""
        print(f"\n📊 Detailed Capacity Analysis:")
        print("=" * 80)
        
        for route in solution_data['routes']:
            print(f"\n🚛 Vehicle: {route['vehicle_id']}")
            print(f"   Capacity: {route['vehicle_capacity_kg']}kg, {route['vehicle_capacity_m3']}m³")
            print(f"   Cost rate: €{route['vehicle_cost_per_km']:.2f}/km")
            print(f"   Route: {route['total_distance']}km, €{route['total_cost']:.2f}")
            
            if route['stops']:
                print(f"   Stop-by-stop capacity usage:")
                for i, stop in enumerate(route['stops']):
                    constraint_icon = "🔴" if stop['limiting_constraint'] == "weight" else "📦" if stop['limiting_constraint'] == "volume" else "⚖️"
                    print(f"     {i+1}. {stop['location_id']}: "
                          f"{stop['weight_usage_kg']}kg ({stop['weight_utilization_pct']}%), "
                          f"{stop['volume_usage_m3']}m³ ({stop['volume_utilization_pct']}%) "
                          f"{constraint_icon} {stop['limiting_constraint']}")
                
                # Find peak utilization
                max_weight_stop = max(route['stops'], key=lambda x: x['weight_utilization_pct'])
                max_volume_stop = max(route['stops'], key=lambda x: x['volume_utilization_pct'])
                
                print(f"   Peak weight utilization: {max_weight_stop['weight_utilization_pct']}% at {max_weight_stop['location_id']}")
                print(f"   Peak volume utilization: {max_volume_stop['volume_utilization_pct']}% at {max_volume_stop['location_id']}")
                
                # Check for constraint violations
                weight_violations = [s for s in route['stops'] if s['weight_utilization_pct'] > 100]
                volume_violations = [s for s in route['stops'] if s['volume_utilization_pct'] > 100]
                
                if weight_violations:
                    print(f"   ⚠️  WARNING: Weight capacity exceeded at {len(weight_violations)} stops!")
                if volume_violations:
                    print(f"   ⚠️  WARNING: Volume capacity exceeded at {len(volume_violations)} stops!")
        
        # Print total summary
        print(f"\n💰 Solution Summary:")
        print(f"   Total distance: {solution_data['total_distance']}km")
        print(f"   Total cost: €{solution_data['total_cost']:.2f}")
        print("=" * 80)
