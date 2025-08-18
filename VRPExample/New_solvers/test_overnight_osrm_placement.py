#!/usr/bin/env python3
"""
Test to fix overnight node placement on OSRM routes instead of straight lines.

This test demonstrates:
1. How to get the actual OSRM route geometry between two points
2. How to place overnight nodes on the OSRM route path at the correct distance
3. How to integrate this into the sequential VRP solver

The fix addresses the issue where overnight nodes are placed on straight lines
between locations, causing distance calculation mismatches.
"""

import os
import sys
import importlib.util
import math
import requests
import json
from typing import List, Tuple, Optional

def interpolate_osrm_route(start_coords: Tuple[float, float], 
                          end_coords: Tuple[float, float], 
                          distance_ratio: float,
                          osrm_url: str = "http://router.project-osrm.org") -> Tuple[float, float]:
    """
    Find a point on the OSRM route at a specific distance ratio between start and end.
    
    Args:
        start_coords: (lon, lat) of start point
        end_coords: (lon, lat) of end point  
        distance_ratio: Ratio of distance along route (0.0 = start, 1.0 = end)
        osrm_url: OSRM server URL
        
    Returns:
        (lon, lat) of the interpolated point on the OSRM route
    """
    try:
        # Get OSRM route with geometry
        url = f"{osrm_url}/route/v1/driving/{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
        params = {
            'geometries': 'geojson',
            'overview': 'full',
            'steps': 'false'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['code'] != 'Ok' or not data['routes']:
            print(f"⚠️ OSRM route failed: {data.get('code', 'Unknown error')}")
            return interpolate_straight_line(start_coords, end_coords, distance_ratio)
        
        # Get route coordinates
        route_coords = data['routes'][0]['geometry']['coordinates']
        
        if len(route_coords) < 2:
            print(f"⚠️ OSRM returned insufficient coordinates")
            return interpolate_straight_line(start_coords, end_coords, distance_ratio)
        
        # Calculate cumulative distances along the route
        cumulative_distances = [0.0]
        total_distance = 0.0
        
        for i in range(1, len(route_coords)):
            prev_coord = route_coords[i-1]
            curr_coord = route_coords[i]
            
            # Calculate distance between consecutive points using Haversine formula
            segment_distance = haversine_distance(prev_coord[1], prev_coord[0], 
                                                curr_coord[1], curr_coord[0])
            total_distance += segment_distance
            cumulative_distances.append(total_distance)
        
        # Find target distance
        target_distance = total_distance * distance_ratio
        
        # Find the segment that contains the target distance
        for i in range(len(cumulative_distances) - 1):
            if cumulative_distances[i] <= target_distance <= cumulative_distances[i + 1]:
                # Interpolate within this segment
                segment_start_dist = cumulative_distances[i]
                segment_end_dist = cumulative_distances[i + 1]
                segment_length = segment_end_dist - segment_start_dist
                
                if segment_length == 0:
                    return tuple(route_coords[i])
                
                # Calculate ratio within this segment
                segment_ratio = (target_distance - segment_start_dist) / segment_length
                
                # Interpolate coordinates within the segment
                start_coord = route_coords[i]
                end_coord = route_coords[i + 1]
                
                interpolated_lon = start_coord[0] + segment_ratio * (end_coord[0] - start_coord[0])
                interpolated_lat = start_coord[1] + segment_ratio * (end_coord[1] - start_coord[1])
                
                return (interpolated_lon, interpolated_lat)
        
        # If we reach here, use the end point
        return tuple(route_coords[-1])
        
    except Exception as e:
        print(f"⚠️ Error interpolating OSRM route: {e}")
        return interpolate_straight_line(start_coords, end_coords, distance_ratio)

def interpolate_straight_line(start_coords: Tuple[float, float], 
                            end_coords: Tuple[float, float], 
                            distance_ratio: float) -> Tuple[float, float]:
    """Fallback: interpolate on straight line between two points."""
    lon = start_coords[0] + distance_ratio * (end_coords[0] - start_coords[0])
    lat = start_coords[1] + distance_ratio * (end_coords[1] - start_coords[1])
    return (lon, lat)

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on Earth in kilometers."""
    R = 6371  # Earth's radius in kilometers
    
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def get_osrm_route_geometry(start_coords: Tuple[float, float], 
                           end_coords: Tuple[float, float],
                           osrm_url: str = "http://router.project-osrm.org") -> List[Tuple[float, float]]:
    """
    Get the complete OSRM route geometry between two points.
    
    Returns:
        List of (lon, lat) coordinates along the route
    """
    try:
        url = f"{osrm_url}/route/v1/driving/{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
        params = {
            'geometries': 'geojson',
            'overview': 'full',
            'steps': 'false'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['code'] != 'Ok' or not data['routes']:
            return [start_coords, end_coords]
        
        # Convert coordinates to (lon, lat) tuples
        route_coords = [(coord[0], coord[1]) for coord in data['routes'][0]['geometry']['coordinates']]
        return route_coords
        
    except Exception as e:
        print(f"⚠️ Error getting OSRM route geometry: {e}")
        return [start_coords, end_coords]

def test_overnight_osrm_placement():
    """Test placing overnight nodes on OSRM routes instead of straight lines."""
    print("🧪 Testing Overnight Node Placement on OSRM Routes")
    print("=" * 60)
    
    # Test coordinates (from our previous test)
    depot_coords = (9.1896, 45.4642)  # depot
    schonaich_coords = (9.0000, 48.6667)  # schonaich_de
    malmoe_coords = (13.0038, 55.6050)  # malmoe_sweden
    
    print(f"📍 Test coordinates:")
    print(f"  Depot: {depot_coords}")
    print(f"  Schonaich: {schonaich_coords}")
    print(f"  Malmoe: {malmoe_coords}")
    
    # Test 1: Route from depot to Schonaich
    print(f"\n🚗 Test 1: Depot -> Schonaich")
    route_coords = get_osrm_route_geometry(depot_coords, schonaich_coords)
    print(f"  OSRM route has {len(route_coords)} coordinate points")
    
    # Calculate distances
    straight_line_dist = haversine_distance(depot_coords[1], depot_coords[0], 
                                          schonaich_coords[1], schonaich_coords[0])
    
    # Calculate OSRM route distance
    osrm_distance = 0.0
    for i in range(1, len(route_coords)):
        osrm_distance += haversine_distance(route_coords[i-1][1], route_coords[i-1][0],
                                          route_coords[i][1], route_coords[i][0])
    
    print(f"  Straight line distance: {straight_line_dist:.2f} km")
    print(f"  OSRM route distance: {osrm_distance:.2f} km")
    print(f"  Route factor: {osrm_distance/straight_line_dist:.2f}x")
    
    # Test placing overnight nodes at different positions along the route
    test_ratios = [0.25, 0.5, 0.75]
    
    print(f"\n🛏️ Testing overnight node placement:")
    for ratio in test_ratios:
        # Old method: straight line interpolation
        straight_overnight = interpolate_straight_line(depot_coords, schonaich_coords, ratio)
        
        # New method: OSRM route interpolation
        osrm_overnight = interpolate_osrm_route(depot_coords, schonaich_coords, ratio)
        
        # Calculate distances from depot for both methods
        straight_dist_from_depot = haversine_distance(depot_coords[1], depot_coords[0],
                                                    straight_overnight[1], straight_overnight[0])
        osrm_dist_from_depot = haversine_distance(depot_coords[1], depot_coords[0],
                                                osrm_overnight[1], osrm_overnight[0])
        
        print(f"  At {ratio*100}% along route:")
        print(f"    Straight line overnight: {straight_overnight} ({straight_dist_from_depot:.2f}km from depot)")
        print(f"    OSRM route overnight: {osrm_overnight} ({osrm_dist_from_depot:.2f}km from depot)")
        print(f"    Position difference: {haversine_distance(straight_overnight[1], straight_overnight[0], osrm_overnight[1], osrm_overnight[0]):.2f}km")
    
    # Test 2: Longer route to show more dramatic differences
    print(f"\n🚗 Test 2: Schonaich -> Malmoe (longer route)")
    route_coords_long = get_osrm_route_geometry(schonaich_coords, malmoe_coords)
    print(f"  OSRM route has {len(route_coords_long)} coordinate points")
    
    straight_line_dist_long = haversine_distance(schonaich_coords[1], schonaich_coords[0],
                                               malmoe_coords[1], malmoe_coords[0])
    
    osrm_distance_long = 0.0
    for i in range(1, len(route_coords_long)):
        osrm_distance_long += haversine_distance(route_coords_long[i-1][1], route_coords_long[i-1][0],
                                                route_coords_long[i][1], route_coords_long[i][0])
    
    print(f"  Straight line distance: {straight_line_dist_long:.2f} km")
    print(f"  OSRM route distance: {osrm_distance_long:.2f} km")
    print(f"  Route factor: {osrm_distance_long/straight_line_dist_long:.2f}x")
    
    # Test overnight placement on the longer route
    print(f"\n🛏️ Overnight placement comparison (longer route):")
    for ratio in [0.33, 0.66]:
        straight_overnight_long = interpolate_straight_line(schonaich_coords, malmoe_coords, ratio)
        osrm_overnight_long = interpolate_osrm_route(schonaich_coords, malmoe_coords, ratio)
        
        position_diff = haversine_distance(straight_overnight_long[1], straight_overnight_long[0],
                                         osrm_overnight_long[1], osrm_overnight_long[0])
        
        print(f"  At {ratio*100}% along route:")
        print(f"    Straight line: {straight_overnight_long}")
        print(f"    OSRM route: {osrm_overnight_long}")
        print(f"    Position difference: {position_diff:.2f}km")

def create_improved_overnight_placement_method():
    """
    Demonstrate how to integrate OSRM-based overnight placement into the sequential VRP.
    """
    print(f"\n" + "="*60)
    print("💡 IMPROVED OVERNIGHT PLACEMENT METHOD")
    print("="*60)
    
    improvement_code = '''
def create_overnight_stop_on_osrm_route(self, vehicle_id: str, 
                                       current_position: Tuple[float, float],
                                       target_position: Tuple[float, float],
                                       remaining_time: float,
                                       travel_speed_kmh: float) -> Tuple[str, Tuple[float, float]]:
    """
    Create an overnight stop on the actual OSRM route instead of straight line.
    
    Args:
        vehicle_id: ID of the vehicle
        current_position: (x, y) current position  
        target_position: (x, y) target position
        remaining_time: Remaining driving time in minutes
        travel_speed_kmh: Vehicle travel speed in km/h
        
    Returns:
        (overnight_location_id, (x, y) overnight coordinates)
    """
    # Convert remaining time to distance
    remaining_distance_km = (remaining_time / 60.0) * travel_speed_kmh
    
    # Get total OSRM route distance
    try:
        # Use cached distance if available
        current_idx = self.get_location_index(current_position)
        target_idx = self.get_location_index(target_position)
        
        if (current_idx is not None and target_idx is not None and 
            hasattr(self.distance_calculator, 'distance_matrix')):
            total_route_distance = self.distance_calculator.distance_matrix[current_idx][target_idx]
        else:
            # Fallback to direct OSRM call
            total_route_distance = self.get_osrm_distance(current_position, target_position)
        
        # Calculate how far along the route we can travel
        distance_ratio = min(remaining_distance_km / total_route_distance, 1.0)
        
        # Get overnight position on OSRM route
        overnight_coords = interpolate_osrm_route(
            (current_position[0], current_position[1]),  # Convert to (lon, lat)
            (target_position[0], target_position[1]),
            distance_ratio,
            self.osrm_url if hasattr(self, 'osrm_url') else "http://router.project-osrm.org"
        )
        
        # Create overnight location
        overnight_id = f"osrm_overnight_{vehicle_id}_day{self.current_day}"
        
        # Add to locations list with proper OSRM distance matrix integration
        self.add_dynamic_location(overnight_id, overnight_coords[0], overnight_coords[1])
        
        return overnight_id, (overnight_coords[0], overnight_coords[1])
        
    except Exception as e:
        self.logger.warning(f"OSRM overnight placement failed: {e}, using straight line")
        # Fallback to straight line method
        return self.create_overnight_stop_straight_line(vehicle_id, current_position, 
                                                       target_position, remaining_time, travel_speed_kmh)

def add_dynamic_location(self, location_id: str, x: float, y: float):
    """
    Add a dynamically created location (like overnight stops) to the distance matrix.
    """
    # Add to locations list
    new_location = {
        'id': location_id,
        'x': x, 'y': y,
        'lat': y, 'lon': x,  # Assuming x,y are lon,lat
        'address': f'Overnight stop {location_id}',
        'service_time': 0,
        'is_overnight': True
    }
    
    self.locations.append(new_location)
    new_location_idx = len(self.locations) - 1
    
    # Extend distance matrix to include new location
    if hasattr(self.distance_calculator, 'distance_matrix'):
        self.extend_distance_matrix_for_location(new_location_idx, x, y)

def extend_distance_matrix_for_location(self, new_idx: int, x: float, y: float):
    """
    Extend the distance matrix to include a new dynamically added location.
    """
    n = len(self.distance_calculator.distance_matrix)
    
    # Add new row
    new_row = []
    for i in range(n):
        existing_location = self.locations[i]
        distance = self.get_osrm_distance_direct(
            (x, y), (existing_location['x'], existing_location['y'])
        )
        new_row.append(distance)
    
    # Add distance to self (0.0)
    new_row.append(0.0)
    self.distance_calculator.distance_matrix.append(new_row)
    
    # Add new column to existing rows
    for i in range(n):
        existing_location = self.locations[i]  
        distance = self.get_osrm_distance_direct(
            (existing_location['x'], existing_location['y']), (x, y)
        )
        self.distance_calculator.distance_matrix[i].append(distance)
    '''
    
    print("📋 Key improvements:")
    print("1. Use interpolate_osrm_route() instead of straight line interpolation")
    print("2. Dynamically extend distance matrix when adding overnight locations")
    print("3. Use actual OSRM distances for new overnight locations")
    print("4. Fallback to straight line if OSRM fails")
    
    print(f"\n💻 Implementation code:")
    print(improvement_code)

if __name__ == "__main__":
    test_overnight_osrm_placement()
    create_improved_overnight_placement_method()
