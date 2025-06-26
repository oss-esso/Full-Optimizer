#!/usr/bin/env python3
"""
OSM Route Testing Demo
Demonstrates detailed OSM/OSRM route request and response logging
for VRP optimization using real route data.
"""

from typing import Dict, List, Tuple, Optional
import logging
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
            locations: List of location dictionaries with id, lat, lon
            osrm_url: OSRM service URL
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
                    if self.use_truck_speeds:
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
            'steps': 'true'  # Get step information for road type analysis
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
            road_analysis = {}
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
                if self.use_truck_speeds and road_analysis:
                    print(f"   🚛 Road Type Analysis:")
                    for road_type, data in road_analysis.items():
                        print(f"      {road_type}: {data['percentage']:.1f}% ({data['distance']:.2f}km)")
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

    # ...existing methods...
    

# ---
# To use a custom speed profile (e.g., for trucks), run your own OSRM server with a custom Lua profile (mytruck.lua).
# Example: osrm-extract -p mytruck.lua planet-latest.osm.pbf
#          osrm-contract planet-latest.osrm
#          osrm-routed planet-latest.osrm
# Then set the osrm_url below to your server (e.g., 'http://localhost:5000')
# The public OSRM server only supports the default car profile.
# ---

# Set this to your OSRM server URL (with custom profile) if needed:
osrm_url = "https://router.project-osrm.org"  # Default: public car profile
# osrm_url = "http://localhost:5000"  # Example: your own truck profile

# Truck speed profile adjustments (truck speed / car speed ratios)
# These ratios will be applied to renormalize travel times based on road type analysis
truck_speed_ratios = {
    'motorway': 80 / 130,      # Trucks: 80 km/h, Cars: ~130 km/h on motorways
    'trunk': 70 / 100,         # Trucks: 70 km/h, Cars: ~100 km/h on trunk roads
    'primary': 60 / 90,        # Trucks: 60 km/h, Cars: ~90 km/h on primary roads
    'secondary': 50 / 70,      # Trucks: 50 km/h, Cars: ~70 km/h on secondary roads
    'tertiary': 45 / 60,       # Trucks: 45 km/h, Cars: ~60 km/h on tertiary roads
    'residential': 30 / 50,    # Trucks: 30 km/h, Cars: ~50 km/h in residential areas
    'service': 20 / 30,        # Trucks: 20 km/h, Cars: ~30 km/h on service roads
    'default': 0.75            # Default ratio for unknown road types (truck 25% slower)
}

# Enable truck speed adjustments
use_truck_speeds = True  # Set to False to use original car speeds


def demo_osm_route_logging():
    """Demonstrate detailed OSM route logging for VRP optimization."""
    print("🧪 OSM Route Logging Demo for VRP Optimization")
    print("=" * 80)
    print(f"Using OSRM server: {osrm_url}")
    
    if use_truck_speeds:
        print("🚛 Truck speed adjustments ENABLED:")
        print("   Analyzing road types and applying truck speed ratios")
        for road_type, ratio in truck_speed_ratios.items():
            print(f"   {road_type}: {ratio:.3f}x speed ratio")
    else:
        print("🚗 Using original car speeds (truck adjustments disabled)")
        
    print("Note: For custom profiles, run your own OSRM server with a Lua profile (see comments at top).\n")
    
    # Create test locations with real coordinates (Chicago area)
    test_locations = [
        {"id": "depot", "lat": 41.8781, "lon": -87.6298, "service_time": 0},       # Chicago downtown
        {"id": "customer_1", "lat": 41.9000, "lon": -87.6500, "service_time": 30}, # North Chicago
        {"id": "customer_2", "lat": 41.8500, "lon": -87.6000, "service_time": 30}, # East Chicago
        {"id": "customer_3", "lat": 41.8400, "lon": -87.6800, "service_time": 45}, # West Chicago
    ]
    
    print(f"Testing with {len(test_locations)} locations:")
    for loc in test_locations:
        print(f"  {loc['id']}: ({loc['lat']:.4f}, {loc['lon']:.4f}) [service: {loc['service_time']}min]")
    print()
    
    try:
        print("🗺️  Initializing OSM Distance Calculator...")
        calculator = OSMDistanceCalculator(
            test_locations, 
            osrm_url=osrm_url,
            truck_speed_ratios=truck_speed_ratios,
            use_truck_speeds=use_truck_speeds
        )
        print("✅ Calculator initialized successfully")
        print()
        
        print("🌐 Testing Individual OSM Route Requests:")
        print("-" * 80)
        
        # Test individual route requests with detailed logging
        route_count = 0
        successful_routes = 0
        
        for i, loc1 in enumerate(test_locations):
            for j, loc2 in enumerate(test_locations):
                if i != j:  # Skip self-routes
                    route_count += 1
                    print(f"\n📍 Route #{route_count}: {loc1['id']} → {loc2['id']}")
                    
                    # Get detailed route information
                    route_info = calculator.get_route_details(loc1['id'], loc2['id'], verbose=True)
                    
                    if route_info['success']:
                        successful_routes += 1
                        print(f"   📊 Summary: {route_info['distance_km']:.2f} km, {route_info['time_minutes']:.1f} min")
                    else:
                        print(f"   ⚠️  Fallback used due to: {route_info.get('error', 'Unknown error')}")
                    
                    # Add small delay to avoid overwhelming OSRM
                    time.sleep(0.2)
        
        print(f"\n" + "=" * 80)
        print(f"🎯 OSM Route Testing Summary:")
        print(f"   Total routes tested: {route_count}")
        print(f"   Successful OSM routes: {successful_routes}")
        print(f"   Success rate: {(successful_routes/route_count)*100:.1f}%")
        
        print(f"\n📋 Distance Matrix (km):")
        print("     ", end="")
        for loc in test_locations:
            print(f"{loc['id'][:8]:>10}", end="")
        print()
        
        for i, loc1 in enumerate(test_locations):
            print(f"{loc1['id'][:8]:>5}", end="")
            for j, loc2 in enumerate(test_locations):
                if i == j:
                    print(f"{'0.00':>10}", end="")
                else:
                    distance = calculator.get_distance_km(loc1['id'], loc2['id'])
                    print(f"{distance:>10.2f}", end="")
            print()
        
        print(f"\n⏱️  Time Matrix (min):")
        print("     ", end="")
        for loc in test_locations:
            print(f"{loc['id'][:8]:>10}", end="")
        print()
        
        for i, loc1 in enumerate(test_locations):
            print(f"{loc1['id'][:8]:>5}", end="")
            for j, loc2 in enumerate(test_locations):
                if i == j:
                    print(f"{'0.0':>10}", end="")
                else:
                    time_min = calculator.get_time_minutes(loc1['id'], loc2['id'])
                    print(f"{time_min:>10.1f}", end="")
            print()
        
        print(f"\n✅ OSM route logging demo completed successfully")
        
    except Exception as e:
        print(f"❌ OSM route logging demo failed: {e}")
        print("   This might be due to network connectivity or OSRM service unavailability")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_osm_route_logging()
