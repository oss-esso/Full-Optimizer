import os
import sys
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import threading
import queue

# Add parent directory to path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vrp_data_models import VRPInstance, VRPResult, VRPObjective

# Import advanced optimization components
from initial_solution_generator import InitialSolutionGenerator
from two_opt_optimizer import TwoOptLocalSearch

# Try to import OR-Tools for industrial-strength optimization benchmarking
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
    print("OR-Tools successfully imported for industrial benchmarking")
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("Warning: OR-Tools not available. Install with 'pip install ortools' for industry-standard benchmarking")

class TimeoutError(Exception):
    """Custom timeout exception for cross-platform compatibility."""
    pass

def run_with_timeout(func, args=(), kwargs=None, timeout_duration=60):
    """Run a function with a timeout using threading (cross-platform)."""
    if kwargs is None:
        kwargs = {}
    
    result_queue = queue.Queue()
    exception_queue = queue.Queue()
    
    def target():
        try:
            result = func(*args, **kwargs)
            result_queue.put(result)
        except Exception as e:
            exception_queue.put(e)
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    
    thread.join(timeout_duration)
    
    if thread.is_alive():
        raise TimeoutError(f"Function timed out after {timeout_duration} seconds")
    
    if not exception_queue.empty():
        raise exception_queue.get()
    
    if not result_queue.empty():
        return result_queue.get()
    else:
        raise RuntimeError("Function completed but no result was returned")

class VRPOptimizerRollingWindow:
    """VRP optimizer with rolling 10-hour window for realistic trucking operations."""
    
    def __init__(self, instance: VRPInstance, objective: VRPObjective = VRPObjective.MINIMIZE_DISTANCE):
        self.instance = instance
        self.objective = objective
        self.logger = logging.getLogger(__name__)
        
        # Set up logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def optimize_with_rolling_window(self, timeout_duration: int = 30, time_limit_seconds: int = None) -> VRPResult:
        """
        Optimize VRP using OR-Tools with rolling 10-hour window logic.
        
        Args:
            timeout_duration: Global timeout for the optimization process (seconds)
            time_limit_seconds: OR-Tools search time limit (seconds). If None, uses adaptive logic.
        """
        self.logger.info(f"Starting VRP optimization with rolling 10-hour window (timeout: {timeout_duration}s, search limit: {time_limit_seconds or 'adaptive'}s)")
        start_time = time.time()
        
        if not ORTOOLS_AVAILABLE:
            self.logger.error("OR-Tools not available for rolling window optimization")
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": "OR-Tools not available"},
                runtime=0
            )
        
        try:
            def _optimize_internal():
                return self._solve_with_ortools_rolling_window(time_limit_seconds)
            
            result = run_with_timeout(_optimize_internal, timeout_duration=timeout_duration)
            return result
            
        except TimeoutError:
            self.logger.error(f"Rolling window optimization timed out after {timeout_duration}s")
            runtime = (time.time() - start_time) * 1000
            return VRPResult(
                status="timeout",
                objective_value=0.0,
                routes={},
                metrics={"total_distance": 0.0, "vehicles_used": 0},
                runtime=runtime
            )
        except Exception as e:
            self.logger.error(f"Error in rolling window optimization: {str(e)}")
            runtime = (time.time() - start_time) * 1000
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": str(e)},
                runtime=runtime
            )

    def _solve_with_ortools_rolling_window(self, time_limit_seconds: int = None) -> VRPResult:
        """Solve VRP with OR-Tools using rolling 10-hour window approach."""
        start_time = time.time()
        
        try:
            # Determine if this is a ride pooling (pickup-delivery) problem
            is_ride_pooling = bool(self.instance.ride_requests)
            
            if is_ride_pooling:
                # Setup for pickup-delivery problem
                depot_locations = [loc_id for loc_id in self.instance.location_ids if loc_id.startswith("depot")]
                
                if len(depot_locations) > 1:
                    # Multi-depot VRPPD
                    location_list = depot_locations.copy()
                    location_to_index = {loc: idx for idx, loc in enumerate(depot_locations)}
                    
                    # Add pickup and dropoff locations
                    for request in self.instance.ride_requests:
                        if request.pickup_location not in location_to_index:
                            location_to_index[request.pickup_location] = len(location_list)
                            location_list.append(request.pickup_location)
                        if request.dropoff_location not in location_to_index:
                            location_to_index[request.dropoff_location] = len(location_list)
                            location_list.append(request.dropoff_location)
                    
                    # Create vehicle start/end depot mappings
                    vehicle_starts = []
                    vehicle_ends = []
                    depot_indices = {depot: location_to_index[depot] for depot in depot_locations}
                    
                    for vehicle_id, vehicle in self.instance.vehicles.items():
                        depot_idx = depot_indices.get(vehicle.depot_id, 0)
                        vehicle_starts.append(depot_idx)
                        vehicle_ends.append(depot_idx)
                    
                    num_locations = len(location_list)
                    num_vehicles = len(self.instance.vehicles)
                    
                    # Create routing index manager with multi-depot
                    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, vehicle_starts, vehicle_ends)
                    
                else:
                    # Single depot VRPPD
                    depot_id = depot_locations[0] if depot_locations else self.instance.location_ids[0]
                    
                    location_list = [depot_id]
                    location_to_index = {depot_id: 0}
                    
                    # Add pickup and dropoff locations
                    for request in self.instance.ride_requests:
                        if request.pickup_location not in location_to_index:
                            location_to_index[request.pickup_location] = len(location_list)
                            location_list.append(request.pickup_location)
                        if request.dropoff_location not in location_to_index:
                            location_to_index[request.dropoff_location] = len(location_list)
                            location_list.append(request.dropoff_location)
                            
                    num_locations = len(location_list)
                    num_vehicles = len(self.instance.vehicles)
                    depot_index = 0
                    
                    # Create routing index manager with single depot
                    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_index)
                
                # Create pickup-delivery pairs
                pickup_deliveries = []
                for request in self.instance.ride_requests:
                    pickup_idx = location_to_index[request.pickup_location]
                    dropoff_idx = location_to_index[request.dropoff_location]
                    pickup_deliveries.append([pickup_idx, dropoff_idx])
            else:
                # Standard VRP: all non-depot locations are customers
                depot_locations = [loc_id for loc_id in self.instance.location_ids if loc_id.startswith("depot")]
                depot_id = depot_locations[0] if depot_locations else self.instance.location_ids[0]
                
                customer_locations = [loc_id for loc_id in self.instance.location_ids if not loc_id.startswith("depot")]
                location_list = [depot_id] + customer_locations
                location_to_index = {loc: idx for idx, loc in enumerate(location_list)}
                num_locations = len(location_list)
                num_vehicles = len(self.instance.vehicles)
                depot_index = 0
                pickup_deliveries = []
                
                # Create routing index manager
                manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_index)
            
            # Create routing model
            routing = pywrapcp.RoutingModel(manager)
            
            # Create distance callback
            def distance_callback(from_index, to_index):
                """Return the distance between two points."""
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                
                from_location_id = location_list[from_node]
                to_location_id = location_list[to_node]
                
                return int(self.instance.get_distance(from_location_id, to_location_id))
            
            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            
            # Add Distance dimension to encourage distribution across vehicles
            dimension_name = "Distance"
            routing.AddDimension(
                transit_callback_index,
                0,  # no slack
                100000,  # vehicle maximum travel distance (high limit)
                True,  # start cumul to zero
                dimension_name,
            )
            distance_dimension = routing.GetDimensionOrDie(dimension_name)
            distance_dimension.SetGlobalSpanCostCoefficient(1000)
            
            # Add capacity constraints if applicable
            vehicle_capacities = []
            has_capacity_constraints = False
            
            for vehicle_id in sorted(self.instance.vehicles.keys()):
                vehicle = self.instance.vehicles[vehicle_id]
                capacity = getattr(vehicle, 'capacity', 100)
                vehicle_capacities.append(capacity)
                if capacity > 0:
                    has_capacity_constraints = True
            
            if has_capacity_constraints:
                def demand_callback(from_index):
                    """Return the demand of the node."""
                    from_node = manager.IndexToNode(from_index)
                    location_id = location_list[from_node]
                    
                    demand = 0
                    if location_id.startswith("depot"):
                        demand = 0
                    elif is_ride_pooling:
                        # For ride pooling: pickups add cargo weight, dropoffs remove them
                        if location_id.startswith("pickup"):
                            for request in self.instance.ride_requests:
                                if request.pickup_location == location_id:
                                    demand = request.passengers
                                    break
                        elif location_id.startswith("dropoff"):
                            for request in self.instance.ride_requests:
                                if request.dropoff_location == location_id:
                                    demand = -request.passengers
                                    break
                    else:
                        # Standard VRP: use location demand
                        location = self.instance.locations[location_id]
                        demand = getattr(location, 'demand', 1)
                    
                    return demand
                
                demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
                
                routing.AddDimensionWithVehicleCapacity(
                    demand_callback_index,
                    0,  # null capacity slack
                    vehicle_capacities,  # vehicle maximum capacities
                    True,  # start cumul to zero
                    'Capacity'
                )
                
                capacity_dimension = routing.GetDimensionOrDie('Capacity')
                
                # Add pickup and delivery constraints for ride pooling
                if is_ride_pooling and pickup_deliveries:
                    for pickup_idx, delivery_idx in pickup_deliveries:
                        pickup_node = manager.NodeToIndex(pickup_idx)
                        delivery_node = manager.NodeToIndex(delivery_idx)
                        routing.AddPickupAndDelivery(pickup_node, delivery_node)
                        
                        # Ensure pickup and delivery are on the same route
                        routing.solver().Add(
                            routing.VehicleVar(pickup_node) == routing.VehicleVar(delivery_node)
                        )
                        
                        # Ensure pickup comes before delivery
                        routing.solver().Add(
                            capacity_dimension.CumulVar(pickup_node) <= 
                            capacity_dimension.CumulVar(delivery_node)
                        )
            
            # *** KEY CHANGE: Rolling 10-hour window time constraints ***
            # Instead of a global time window, we create route duration constraints
            vehicle_time_constraints = {}
            has_time_constraints = False
            for vid, vehicle in enumerate(self.instance.vehicles.values()):
                max_time = None
                if hasattr(vehicle, 'max_total_work_time') and vehicle.max_total_work_time is not None:
                    max_time = vehicle.max_total_work_time  # in minutes
                elif hasattr(vehicle, 'max_time') and vehicle.max_time is not None:
                    max_time = vehicle.max_time  # in minutes
                else:
                    max_time = 600  # Default: 10 hours = 600 minutes
                
                vehicle_time_constraints[vid] = max_time
                has_time_constraints = True
            
            if has_time_constraints:
                # Log the time constraints
                for vid, max_time in vehicle_time_constraints.items():
                    vehicle_list = list(self.instance.vehicles.values())
                    if vid < len(vehicle_list):
                        vehicle = vehicle_list[vid]
                        vehicle_type = getattr(vehicle, 'vehicle_type', 'unknown')
                        self.logger.info(f"Vehicle {vid} ({vehicle_type}): max route duration {max_time} minutes ({max_time/60:.1f} hours)")
                
                # Create a route duration callback (not absolute time)
                def route_duration_callback(from_index, to_index):
                    """Return the time spent traveling from one location to another plus service time."""
                    from_node = manager.IndexToNode(from_index)
                    to_node = manager.IndexToNode(to_index)
                    from_location_id = location_list[from_node]
                    to_location_id = location_list[to_node]
                    
                    # Get travel distance
                    distance_units = self.instance.get_distance(from_location_id, to_location_id)
                    
                    # Convert distance to realistic travel time
                    # For GPS coordinates, 1 degree ≈ 111 km at latitude ~45° (Northern Italy)
                    distance_km = distance_units * 111
                    
                    # Calculate travel time assuming realistic speed
                    avg_speed_kmh = 50  # Urban/highway mixed driving
                    travel_time_hours = distance_km / avg_speed_kmh
                    travel_time_minutes = travel_time_hours * 60
                    
                    # Add service time at the 'from' location
                    service_time_minutes = 0
                    if from_location_id in self.instance.locations:
                        location = self.instance.locations[from_location_id]
                        service_time_minutes = getattr(location, 'service_time', 0)
                    
                    total_time_minutes = travel_time_minutes + service_time_minutes
                    # Convert to seconds for OR-Tools (but keep it as route duration, not absolute time)
                    total_time_seconds = int(total_time_minutes * 60)
                    
                    return total_time_seconds
                
                route_duration_callback_index = routing.RegisterTransitCallback(route_duration_callback)
                
                # Use the maximum route duration constraint
                max_overall_duration = max(vehicle_time_constraints.values())
                max_duration_seconds = int(max_overall_duration * 60)
                
                self.logger.info(f"Adding route duration dimension with max {max_overall_duration} minutes ({max_overall_duration/60:.1f} hours) per vehicle")
                  # Key difference: This is ROUTE DURATION, not absolute time
                # Each vehicle can start at any time but cannot exceed the route duration
                routing.AddDimension(
                    route_duration_callback_index,
                    max_duration_seconds // 10,  # small slack: 10% of max duration for flexibility
                    max_duration_seconds,  # maximum route duration per vehicle
                    True,  # *** FIXED: start_cumul_to_zero = True ***
                    'RouteDuration'        # This tracks cumulative route duration from start
                )
                  # *** ROLLING WINDOW APPROACH ***
                # Use traditional time dimension but allow flexible start times for each vehicle
                # The key insight: each vehicle can start at any time within the day,
                # but the route duration must not exceed max_duration
                
                routing.AddDimension(
                    route_duration_callback_index,
                    max_duration_seconds // 10,  # small slack for flexibility
                    max_duration_seconds,  # maximum route duration per vehicle
                    True,  # start cumul to zero (traditional approach)
                    'RouteDuration'
                )
                
                # Apply individual vehicle route duration constraints
                route_duration_dimension = routing.GetDimensionOrDie('RouteDuration')
                for vid, max_duration in vehicle_time_constraints.items():
                    max_duration_seconds_vehicle = int(max_duration * 60)
                    
                    # Set the maximum cumulative time at the end node
                    end_node = routing.End(vid)
                    route_duration_dimension.CumulVar(end_node).SetMax(max_duration_seconds_vehicle)
                    
                    # For rolling window: allow each vehicle to start at any time
                    # by setting flexible start time windows 
                    start_node = routing.Start(vid)
                    # Allow start between 0 (early morning) and late enough to complete route
                    latest_start = max(0, (24 * 60 * 60) - max_duration_seconds_vehicle)  # 24 hours - route duration
                    route_duration_dimension.CumulVar(start_node).SetRange(0, latest_start)
                    
                    self.logger.debug(f"Vehicle {vid}: route duration max {max_duration} min, start window 0-{latest_start//3600}h")
              # Note: No arbitrary stop count constraint - only time constraint matters
            # Vehicles can do as many stops as they can fit within the 10-hour time limit
            self.logger.info("No stop count constraint - vehicles limited only by route duration (10 hours max)")
            
            # Handle pickup-delivery constraints for non-capacity scenarios
            if not has_capacity_constraints and is_ride_pooling and pickup_deliveries:
                for pickup_idx, delivery_idx in pickup_deliveries:
                    pickup_node = manager.NodeToIndex(pickup_idx)
                    delivery_node = manager.NodeToIndex(delivery_idx)
                    routing.AddPickupAndDelivery(pickup_node, delivery_node)
                    
                    # Ensure pickup and delivery are on the same route
                    routing.solver().Add(
                        routing.VehicleVar(pickup_node) == routing.VehicleVar(delivery_node)
                    )
            
            # Set search parameters
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
            )
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH            )
            
            # Set time limit (custom or adaptive)
            if time_limit_seconds is not None:
                time_limit = time_limit_seconds
                self.logger.info(f"Using custom time limit: {time_limit} seconds")
            else:
                # Adaptive time limit
                num_locations = len(self.instance.location_ids)
                if num_locations <= 10:
                    time_limit = 3
                elif num_locations <= 50:
                    time_limit = 8
                elif num_locations <= 100:
                    time_limit = 15
                else:
                    time_limit = 30
                self.logger.info(f"Using adaptive time limit: {time_limit} seconds for {num_locations} locations")
                
            search_parameters.time_limit.FromSeconds(time_limit)
            
            # Solve the problem
            self.logger.info("Solving with OR-Tools rolling window approach...")
            solution = routing.SolveWithParameters(search_parameters)
            runtime = (time.time() - start_time) * 1000
            
            if solution:
                # Extract solution
                routes = {}
                total_distance = 0
                vehicle_ids = sorted(self.instance.vehicles.keys())
                
                self.logger.info(f"OR-Tools found solution with objective value: {solution.ObjectiveValue()}")
                
                for vehicle_idx in range(num_vehicles):
                    if vehicle_idx < len(vehicle_ids):
                        vehicle_id = vehicle_ids[vehicle_idx]
                        vehicle = self.instance.vehicles[vehicle_id]
                        
                        # Get vehicle's start depot
                        if is_ride_pooling and len(depot_locations) > 1:
                            start_depot = vehicle.depot_id
                        else:
                            start_depot = depot_locations[0] if depot_locations else self.instance.location_ids[0]
                        
                        route = [start_depot]
                        
                        index = routing.Start(vehicle_idx)
                        route_distance = 0
                        
                        while not routing.IsEnd(index):
                            node_index = manager.IndexToNode(index)
                            
                            # Add non-depot locations to route
                            if is_ride_pooling and len(depot_locations) > 1:
                                if node_index < len(depot_locations):
                                    depot_location_id = location_list[node_index]
                                    if depot_location_id != start_depot:
                                        route.append(depot_location_id)
                                else:
                                    route.append(location_list[node_index])
                            else:
                                if (not is_ride_pooling and node_index != 0) or (is_ride_pooling and node_index != 0):
                                    route.append(location_list[node_index])
                            
                            previous_index = index
                            index = solution.Value(routing.NextVar(index))
                            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_idx)
                        
                        # Only include routes that visit locations
                        if len(route) > 1:
                            route.append(start_depot)  # Return to depot
                            routes[vehicle_id] = route
                            total_distance += route_distance
                            
                            # Log route duration if available
                            if has_time_constraints:
                                start_node = routing.Start(vehicle_idx)
                                end_node = routing.End(vehicle_idx)
                                route_duration_dim = routing.GetDimensionOrDie('RouteDuration')
                                route_duration_seconds = (
                                    solution.Value(route_duration_dim.CumulVar(end_node)) - 
                                    solution.Value(route_duration_dim.CumulVar(start_node))
                                )
                                route_duration_minutes = route_duration_seconds / 60
                                self.logger.info(f"Vehicle {vehicle_id} route duration: {route_duration_minutes:.1f} minutes ({route_duration_minutes/60:.1f} hours)")
                            
                            self.logger.debug(f"Vehicle {vehicle_id} route: {route}, distance: {route_distance}")
                        else:
                            routes[vehicle_id] = [start_depot]
                
                # Calculate metrics
                metrics = self._calculate_vrp_metrics(routes)
                actual_total_distance = metrics.get('total_distance', total_distance)
                
                metrics.update({
                    'ortools_distance': actual_total_distance,
                    'ortools_objective': solution.ObjectiveValue(),
                    'ortools_runtime': runtime,
                    'ortools_status': 'optimal',
                    'rolling_window': True,
                    'max_route_duration_minutes': max_overall_duration if has_time_constraints else None
                })
                
                if is_ride_pooling:
                    metrics['ortools_note'] = "OR-Tools with pickup-delivery constraints and rolling 10-hour window"
                else:
                    metrics['ortools_note'] = "OR-Tools with rolling 10-hour window"
                
                self.logger.info(f"Rolling window solution: distance={actual_total_distance:.2f}, vehicles={len([r for r in routes.values() if len(r) > 2])}, runtime={runtime:.2f}ms")
                
                return VRPResult(
                    status="optimal",
                    objective_value=-actual_total_distance,
                    routes=routes,
                    metrics=metrics,
                    runtime=runtime
                )
            else:
                self.logger.warning("OR-Tools could not find solution with rolling window approach")
                return VRPResult(
                    status="infeasible",
                    objective_value=0.0,
                    routes={},
                    metrics={"error": "No feasible solution found with rolling window"},
                    runtime=runtime
                )
                
        except Exception as e:
            self.logger.error(f"Error in OR-Tools rolling window optimization: {str(e)}")
            runtime = (time.time() - start_time) * 1000
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": str(e)},
                runtime=runtime
            )
    
    def _calculate_vrp_metrics(self, routes: Dict[str, List[str]]) -> Dict[str, Any]:
        """Calculate VRP solution metrics."""
        metrics = {
            'total_distance': 0.0,
            'vehicles_used': 0,
            'total_locations_served': 0,
            'average_route_length': 0.0
        }
        
        total_distance = 0.0
        vehicles_used = 0
        total_locations = 0
        
        for vehicle_id, route in routes.items():
            if len(route) > 2:  # More than just depot-depot
                vehicles_used += 1
                locations_in_route = len(route) - 2  # Exclude start and end depot
                total_locations += locations_in_route
                
                # Calculate route distance
                route_distance = 0.0
                for i in range(len(route) - 1):
                    from_loc = route[i]
                    to_loc = route[i + 1]
                    distance = self.instance.get_distance(from_loc, to_loc)
                    route_distance += distance
                
                total_distance += route_distance
        
        metrics['total_distance'] = total_distance
        metrics['vehicles_used'] = vehicles_used
        metrics['total_locations_served'] = total_locations
        metrics['average_route_length'] = total_locations / vehicles_used if vehicles_used > 0 else 0
        
        return metrics
