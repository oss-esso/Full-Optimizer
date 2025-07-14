"""
Time Details Module for tsp_multiple_days1.py Integration

This module provides the interface that tsp_multiple_days1.py expects,
implementing the required functions using the same logic as vrp_multiday_sequential.py
"""

import math
from typing import List, Dict, Tuple, Optional

# Global variables to store scenario data
_scenario_locations = []
_distance_calculator = None
_service_times = {}
_node_count = 0

def initialize_scenario_data(locations: List[Dict], distance_calculator, service_times: Dict = None):
    """Initialize the module with scenario data"""
    global _scenario_locations, _distance_calculator, _service_times, _node_count
    
    _scenario_locations = locations
    _distance_calculator = distance_calculator
    _service_times = service_times or {}
    _node_count = len(locations)

def num_nodes() -> int:
    """Return the number of nodes in the scenario"""
    return _node_count

def transit_callback(manager, day_end: int, night_nodes: List[int], morning_nodes: List[int], from_index: int, to_index: int) -> int:
    """
    Calculate transit cost between two nodes
    
    Args:
        manager: OR-Tools routing index manager
        day_end: End of day time in seconds
        night_nodes: List of night node indices
        morning_nodes: List of morning node indices
        from_index: Starting node index
        to_index: Destination node index
        
    Returns:
        Transit cost (travel time in seconds)
    """
    global _distance_calculator, _scenario_locations
    
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    
    # Handle special cases for night and morning nodes
    if from_node in night_nodes or to_node in night_nodes:
        # Night nodes have minimal cost
        return 1
    
    if from_node in morning_nodes or to_node in morning_nodes:
        # Morning nodes have minimal cost
        return 1
    
    # Regular nodes - use distance calculator if available
    if _distance_calculator and from_node < len(_scenario_locations) and to_node < len(_scenario_locations):
        try:
            from_loc = _scenario_locations[from_node]
            to_loc = _scenario_locations[to_node]
            
            from_coords = (from_loc['lat'], from_loc['lon'])
            to_coords = (to_loc['lat'], to_loc['lon'])
            
            # Get travel time in minutes and convert to seconds
            travel_time_min = _distance_calculator.get_travel_time(from_coords, to_coords)
            return int(travel_time_min * 60)
        except:
            # Fallback to haversine distance
            return _haversine_time(from_node, to_node)
    
    # Fallback to haversine distance calculation
    return _haversine_time(from_node, to_node)

def time_callback(manager, node_service_time: int, overnight_time: int, night_nodes: List[int], morning_nodes: List[int], from_index: int, to_index: int) -> int:
    """
    Calculate time dimension callback including service times
    
    Args:
        manager: OR-Tools routing index manager
        node_service_time: Default service time in seconds
        overnight_time: Overnight time in seconds
        night_nodes: List of night node indices
        morning_nodes: List of morning node indices
        from_index: Starting node index
        to_index: Destination node index
        
    Returns:
        Time including travel and service time (in seconds)
    """
    global _service_times, _scenario_locations
    
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    
    # Get base transit time
    transit_time = transit_callback(manager, 0, night_nodes, morning_nodes, from_index, to_index)
    
    # Add service time for the destination node
    service_time = 0
    if to_node < len(_scenario_locations):
        location_id = _scenario_locations[to_node].get('id', '')
        service_time = _service_times.get(location_id, node_service_time)
    else:
        # For special nodes (night/morning), use default service time
        service_time = node_service_time
    
    # Handle overnight nodes
    if to_node in night_nodes:
        return transit_time + overnight_time
    
    return transit_time + service_time

def _haversine_time(from_node: int, to_node: int) -> int:
    """
    Calculate travel time using haversine distance as fallback
    
    Args:
        from_node: Starting node index
        to_node: Destination node index
        
    Returns:
        Travel time in seconds
    """
    global _scenario_locations
    
    if from_node >= len(_scenario_locations) or to_node >= len(_scenario_locations):
        return 3600  # 1 hour default for special nodes
    
    from_loc = _scenario_locations[from_node]
    to_loc = _scenario_locations[to_node]
    
    # Haversine distance calculation
    lat1, lon1 = math.radians(from_loc['lat']), math.radians(from_loc['lon'])
    lat2, lon2 = math.radians(to_loc['lat']), math.radians(to_loc['lon'])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth radius in km
    r = 6371
    distance_km = c * r
    
    # Assume average speed of 50 km/h
    travel_time_hours = distance_km / 50
    travel_time_seconds = int(travel_time_hours * 3600)
    
    return travel_time_seconds

def get_location_coordinates(node_index: int) -> Tuple[float, float]:
    """
    Get coordinates for a node
    
    Args:
        node_index: Node index
        
    Returns:
        Tuple of (latitude, longitude)
    """
    global _scenario_locations
    
    if node_index < len(_scenario_locations):
        loc = _scenario_locations[node_index]
        return (loc['lat'], loc['lon'])
    
    # Default coordinates for special nodes
    return (45.4642, 9.1896)  # Milan coordinates as default

def get_location_info(node_index: int) -> Dict:
    """
    Get location information for a node
    
    Args:
        node_index: Node index
        
    Returns:
        Dictionary with location information
    """
    global _scenario_locations
    
    if node_index < len(_scenario_locations):
        return _scenario_locations[node_index].copy()
    
    # Default info for special nodes
    return {
        'id': f'special_node_{node_index}',
        'lat': 45.4642,
        'lon': 9.1896,
        'demand': 0,
        'service_time': 0
    }
