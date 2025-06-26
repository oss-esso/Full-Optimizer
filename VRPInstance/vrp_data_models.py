from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import numpy as np

class VRPObjective(Enum):
    """Types of VRP optimization objectives."""
    MINIMIZE_DISTANCE = "minimize_distance"
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_VEHICLES = "minimize_vehicles"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"

@dataclass
class Location:
    """Represents a location in the VRP problem."""
    id: str
    x: float
    y: float
    demand: int = 0  # For CVRP
    time_window_start: Optional[float] = None  # For VRPTW
    time_window_end: Optional[float] = None
    service_time: float = 0  # Time required to service this location
    address: Optional[str] = None  # Real-world address from OSM
    lat: Optional[float] = None  # Latitude for OSM integration
    lon: Optional[float] = None  # Longitude for OSM integration

@dataclass
class Vehicle:
    """Represents a vehicle in the VRP fleet with driver regulations."""
    id: str
    capacity: int
    depot_id: str
    max_distance: Optional[float] = None
    max_time: Optional[float] = None
    cost_per_km: float = 1.0
    
    # Driver regulation fields for mixed fleet
    vehicle_type: str = "standard"  # "standard" or "heavy"
    max_driving_time: Optional[float] = None  # Maximum continuous driving time (minutes)
    required_break_time: Optional[float] = None  # Required break duration (minutes)
    max_total_work_time: Optional[float] = None  # Maximum total work time per day (minutes)
    break_frequency: Optional[float] = None  # How often breaks are required (minutes)
    
    # Truck speed profile settings for OSM routing
    truck_speed_ratios: Optional[Dict[str, float]] = None  # Custom speed ratios by road type
    use_truck_speeds: bool = True  # Whether to apply truck speed adjustments in OSM routing

@dataclass
class RideRequest:
    """Represents a cargo transport request for Vehicle Routing Problem with Pickup and Delivery (VRPPD)."""
    id: str
    pickup_location: str
    dropoff_location: str
    passengers: float = 1.0  # Represents cargo load size (can be fractional)
    earliest_pickup: Optional[float] = None
    latest_dropoff: Optional[float] = None

@dataclass
class VRPResult:
    """Results from VRP optimization."""
    status: str
    objective_value: float
    routes: Dict[str, List[str]]  # vehicle_id -> list of location_ids
    solution_variables: Optional[Dict] = None
    metrics: Optional[Dict[str, float]] = None
    runtime: float = 0.0
    quantum_metrics: Optional[Dict] = None

class VRPInstance:
    """Complete VRP instance definition."""
    
    def __init__(self, name: str):
        self.name = name
        self.locations: Dict[str, Location] = {}
        self.vehicles: Dict[str, Vehicle] = {}
        self.ride_requests: List[RideRequest] = []
        self.distance_matrix: Optional[np.ndarray] = None
        self.duration_matrix: Optional[np.ndarray] = None  # Travel time matrix
        self.location_ids: List[str] = []
        self.is_realistic: bool = False  # Flag for OSM-based scenarios
        
    def add_location(self, location: Location):
        """Add a location to the instance."""
        self.locations[location.id] = location
        if location.id not in self.location_ids:
            self.location_ids.append(location.id)
    
    def add_vehicle(self, vehicle: Vehicle):
        """Add a vehicle to the instance."""
        self.vehicles[vehicle.id] = vehicle
    
    def add_ride_request(self, request: RideRequest):
        """Add a ride request to the instance."""
        self.ride_requests.append(request)
        # Ensure pickup and dropoff locations are in the location list
        if request.pickup_location not in self.location_ids:
            self.location_ids.append(request.pickup_location)
        if request.dropoff_location not in self.location_ids:
            self.location_ids.append(request.dropoff_location)
    
    def calculate_distance_matrix(self, distance_method: str = "euclidean"):
        """
        Calculate distance matrix between all locations.
        
        Args:
            distance_method: "euclidean" or "manhattan"
        """
        n_locations = len(self.location_ids)
        self.distance_matrix = np.zeros((n_locations, n_locations))
        
        for i, loc1_id in enumerate(self.location_ids):
            for j, loc2_id in enumerate(self.location_ids):
                if i != j:
                    loc1 = self.locations[loc1_id]
                    loc2 = self.locations[loc2_id]
                    
                    if distance_method == "manhattan":
                        # Manhattan distance: |x1-x2| + |y1-y2|
                        distance = abs(loc1.x - loc2.x) + abs(loc1.y - loc2.y)
                    else:
                        # Euclidean distance: sqrt((x1-x2)² + (y1-y2)²)
                        distance = np.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)
                    
                    self.distance_matrix[i, j] = distance
                else:
                    self.distance_matrix[i, j] = 0

    def get_distance(self, loc1_id: str, loc2_id: str) -> float:
        """Get distance between two locations."""
        if self.distance_matrix is None:
            self.calculate_distance_matrix()
        
        i = self.location_ids.index(loc1_id)
        j = self.location_ids.index(loc2_id)
        return self.distance_matrix[i, j]
    
    def get_duration(self, loc1_id: str, loc2_id: str) -> float:
        """Get travel time between two locations in seconds."""
        if self.duration_matrix is not None:
            i = self.location_ids.index(loc1_id)
            j = self.location_ids.index(loc2_id)
            return self.duration_matrix[i, j]
        
        # If no duration matrix exists, estimate from distance
        distance = self.get_distance(loc1_id, loc2_id)
        
        # Set fixed speed of 50 km/h for all scenarios as requested
        speed_kph = 50.0
        
        # Convert from km/h to m/s and calculate duration
        speed_mps = speed_kph / 3.6
        return distance / speed_mps  # Returns seconds
    
    def export_for_visualization(self) -> Dict[str, Any]:
        """Export instance data for web-based visualization."""
        export_data = {
            'name': self.name,
            'locations': [],
            'vehicles': [],
            'distance_matrix': self.distance_matrix.tolist() if self.distance_matrix is not None else None,
            'is_realistic': self.is_realistic
        }
        
        for loc_id, location in self.locations.items():
            loc_data = {
                'id': location.id,
                'x': location.x,
                'y': location.y,
                'demand': location.demand,
                'address': getattr(location, 'address', None),
                'lat': getattr(location, 'lat', None),
                'lon': getattr(location, 'lon', None)
            }
            export_data['locations'].append(loc_data)
        
        for vehicle_id, vehicle in self.vehicles.items():
            vehicle_data = {
                'id': vehicle.id,
                'capacity': vehicle.capacity,
                'depot_id': vehicle.depot_id
            }
            export_data['vehicles'].append(vehicle_data)
        
        return export_data

    def get_truck_speed_profile(self) -> Dict[str, float]:
        """
        Get truck speed profile for OSM routing based on vehicle fleet composition.
        
        Returns:
            Dictionary of road type to speed ratio adjustments based on fleet composition.
        """
        if not self.vehicles:
            # Default to standard truck profile if no vehicles defined
            from vrp_scenarios import DEFAULT_TRUCK_SPEED_RATIOS
            return DEFAULT_TRUCK_SPEED_RATIOS['standard'].copy()
        
        # Analyze fleet composition to determine appropriate speed profile
        vehicle_types = []
        total_capacity = 0
        
        for vehicle in self.vehicles.values():
            vehicle_type = getattr(vehicle, 'vehicle_type', 'standard')
            capacity = getattr(vehicle, 'capacity', 0)
            
            vehicle_types.append(vehicle_type)
            total_capacity += capacity
        
        # Calculate weighted average speed profile based on fleet composition
        from vrp_scenarios import DEFAULT_TRUCK_SPEED_RATIOS
        
        if not vehicle_types:
            return DEFAULT_TRUCK_SPEED_RATIOS['standard'].copy()
        
        # Count vehicle types
        type_counts = {}
        type_capacities = {}
        
        for vehicle in self.vehicles.values():
            vehicle_type = getattr(vehicle, 'vehicle_type', 'standard')
            capacity = getattr(vehicle, 'capacity', 0)
            
            if vehicle_type not in type_counts:
                type_counts[vehicle_type] = 0
                type_capacities[vehicle_type] = 0
            
            type_counts[vehicle_type] += 1
            type_capacities[vehicle_type] += capacity
        
        # Create weighted speed profile based on capacity distribution
        combined_speed_ratios = {}
        total_weight = sum(type_capacities.values())
        
        if total_weight == 0:
            return DEFAULT_TRUCK_SPEED_RATIOS['standard'].copy()
        
        # Get all road types from any profile
        all_road_types = set()
        for profile in DEFAULT_TRUCK_SPEED_RATIOS.values():
            all_road_types.update(profile.keys())
        
        # Calculate weighted average for each road type
        for road_type in all_road_types:
            weighted_ratio = 0.0
            
            for vehicle_type, capacity in type_capacities.items():
                if vehicle_type in DEFAULT_TRUCK_SPEED_RATIOS:
                    weight = capacity / total_weight
                    ratio = DEFAULT_TRUCK_SPEED_RATIOS[vehicle_type].get(road_type, 
                                                                        DEFAULT_TRUCK_SPEED_RATIOS[vehicle_type].get('default', 0.75))
                    weighted_ratio += weight * ratio
                else:
                    # Unknown vehicle type, use standard profile
                    weight = capacity / total_weight
                    ratio = DEFAULT_TRUCK_SPEED_RATIOS['standard'].get(road_type,
                                                                      DEFAULT_TRUCK_SPEED_RATIOS['standard'].get('default', 0.80))
                    weighted_ratio += weight * ratio
            
            combined_speed_ratios[road_type] = weighted_ratio
        
        return combined_speed_ratios
    
    def should_use_truck_speeds(self) -> bool:
        """
        Determine if truck speed adjustments should be used based on vehicle specifications.
        
        Returns:
            True if any vehicle in the fleet is configured to use truck speeds.
        """
        if not self.vehicles:
            return False
        
        # Check if any vehicle explicitly enables truck speeds
        for vehicle in self.vehicles.values():
            if getattr(vehicle, 'use_truck_speeds', True):  # Default to True for realistic scenarios
                return True
        
        return False
    
    def get_vehicle_speed_specs(self) -> Dict[str, Dict]:
        """
        Get detailed speed specifications for all vehicles in the fleet.
        
        Returns:
            Dictionary mapping vehicle IDs to their speed specifications.
        """
        vehicle_specs = {}
        
        for vehicle_id, vehicle in self.vehicles.items():
            specs = {
                'vehicle_type': getattr(vehicle, 'vehicle_type', 'standard'),
                'capacity': getattr(vehicle, 'capacity', 0),
                'use_truck_speeds': getattr(vehicle, 'use_truck_speeds', True),
                'truck_speed_ratios': getattr(vehicle, 'truck_speed_ratios', None)
            }
            
            # If vehicle has custom speed ratios, use those; otherwise use default for type
            if specs['truck_speed_ratios'] is None:
                from vrp_scenarios import DEFAULT_TRUCK_SPEED_RATIOS
                vehicle_type = specs['vehicle_type']
                if vehicle_type in DEFAULT_TRUCK_SPEED_RATIOS:
                    specs['truck_speed_ratios'] = DEFAULT_TRUCK_SPEED_RATIOS[vehicle_type].copy()
                else:
                    specs['truck_speed_ratios'] = DEFAULT_TRUCK_SPEED_RATIOS['standard'].copy()
            
            vehicle_specs[vehicle_id] = specs
        
        return vehicle_specs
