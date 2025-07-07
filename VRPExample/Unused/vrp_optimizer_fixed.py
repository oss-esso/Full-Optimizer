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

# Try to import Advanced Initial Solution methods from recent literature
try:
    from advanced_initial_solution_optimizer import AdvancedInitialSolutionOptimizer
    ADVANCED_METHODS_AVAILABLE = True
except ImportError:
    ADVANCED_METHODS_AVAILABLE = False
    print("Warning: Advanced methods not available")

# Import new implementation methods from recent literature
try:
    from initial_solution_generator import InitialSolutionGenerator
    from two_opt_optimizer import TwoOptLocalSearch
    NEW_METHODS_AVAILABLE = True
    print("New implementation methods (InitialSolutionGenerator, TwoOptLocalSearch) imported successfully")
except ImportError as e:
    NEW_METHODS_AVAILABLE = False
    print(f"Warning: New implementation methods not available: {e}")
    print("Advanced Initial Solution methods successfully imported (Firefly Algorithm, etc.)")
except ImportError:
    ADVANCED_METHODS_AVAILABLE = False
    print("Warning: Advanced Initial Solution methods not available")

# Legacy PyVRP support removed
PYVRP_AVAILABLE = False

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

class VRPQuantumOptimizer:
    """VRP optimizer using quantum-inspired methods - fixed version matching debug scenario."""
    
    def __init__(self, instance: VRPInstance, objective: VRPObjective = VRPObjective.MINIMIZE_DISTANCE):
        self.instance = instance
        self.objective = objective
        self.logger = logging.getLogger(__name__)
        
        # Quantum configuration
        self.quantum_metrics = {}
          # Set up logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def optimize_with_quantum_benders(self) -> VRPResult:
        """Optimize VRP using quantum-enhanced approach."""
        self.logger.info("Starting VRP optimization with quantum-enhanced methods")
        start_time = time.time()
        
        try:
            def _optimize_internal():
                routes = self._quantum_inspired_heuristic()
                # Apply driver regulations for heavy trucks
                routes = self._check_driver_regulations(routes)
                return routes
            
            routes = run_with_timeout(_optimize_internal, timeout_duration=15)
            
            runtime = (time.time() - start_time) * 1000  # Convert to milliseconds
            metrics = self._calculate_vrp_metrics(routes)
            obj_value = -metrics.get("total_distance", 0.0)
            
            self.quantum_metrics = {
                'problem_size': len(self.instance.location_ids),
                'num_vehicles': len(self.instance.vehicles),
                'algorithm': 'quantum_inspired_heuristic'
            }
            return VRPResult(
                status="optimal",
                objective_value=obj_value,
                routes=routes,
                metrics=metrics,
                runtime=runtime,
                quantum_metrics=self.quantum_metrics
            )
            
        except TimeoutError:
            self.logger.error("Quantum optimization timed out")
            runtime = (time.time() - start_time) * 1000  # Convert to milliseconds
            return VRPResult(
                status="timeout",
                objective_value=0.0,
                routes={},
                metrics={"total_distance": 0.0, "vehicles_used": 0},
                runtime=runtime
            )
        except Exception as e:
            self.logger.error(f"Error in quantum optimization: {str(e)}")
            runtime = (time.time() - start_time) * 1000  # Convert to milliseconds
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"total_distance": 0.0, "vehicles_used": 0},
                runtime=runtime
            )

    def _quantum_inspired_heuristic(self) -> Dict[str, List[str]]:
        """Quantum-inspired heuristic for VRP - optimized version."""
        routes = {}
        vehicle_ids = list(self.instance.vehicles.keys())
        
        # Initialize routes with depot starts
        for vehicle_id in vehicle_ids:
            depot_id = self.instance.vehicles[vehicle_id].depot_id
            routes[vehicle_id] = [depot_id]
        
        # Handle ride pooling scenario
        if self.instance.ride_requests:
            return self._handle_ride_pooling_quantum(routes)
        
        # Handle delivery scenario
        non_depot_locations = [loc for loc in self.instance.location_ids if not loc.startswith("depot")]
        remaining_locations = set(non_depot_locations)
        vehicle_loads = {vehicle_id: 0 for vehicle_id in vehicle_ids}
        
        max_iterations = min(len(remaining_locations) * 2, 50)
        iteration_count = 0
        
        while remaining_locations and iteration_count < max_iterations:
            iteration_count += 1
            locations_assigned_this_round = False
            
            for vehicle_id in vehicle_ids:
                if not remaining_locations:
                    break
                
                current_location = routes[vehicle_id][-1]
                vehicle_capacity = self.instance.vehicles[vehicle_id].capacity
                current_load = vehicle_loads[vehicle_id]
                
                # Find valid locations
                valid_locations = []
                distances = []
                
                for loc_id in list(remaining_locations)[:10]:  # Limit candidates
                    if loc_id in self.instance.locations:
                        demand = self.instance.locations[loc_id].demand
                        if current_load + demand <= vehicle_capacity:
                            try:
                                distance = self.instance.get_distance(current_location, loc_id)
                                valid_locations.append(loc_id)
                                distances.append(distance)
                            except (ValueError, KeyError, IndexError):
                                continue
                
                if valid_locations:
                    # Quantum-inspired probabilistic selection
                    inv_distances = [1.0 / (d + 0.1) for d in distances]
                    total_prob = sum(inv_distances)
                    
                    if total_prob > 0:
                        probabilities = [p / total_prob for p in inv_distances]
                        
                        try:
                            selected_idx = np.random.choice(len(valid_locations), p=probabilities)
                            selected_location = valid_locations[selected_idx]
                        except (ValueError, IndexError):
                            # Fallback to greedy
                            min_dist_idx = distances.index(min(distances))
                            selected_location = valid_locations[min_dist_idx]
                        
                        # Add to route
                        routes[vehicle_id].append(selected_location)
                        vehicle_loads[vehicle_id] += self.instance.locations[selected_location].demand
                        remaining_locations.remove(selected_location)
                        locations_assigned_this_round = True
                        break
            
            if not locations_assigned_this_round:
                self.logger.warning(f"Could not assign {len(remaining_locations)} remaining locations due to capacity constraints")
                break
        
        # Return vehicles to depot
        for vehicle_id in vehicle_ids:
            if len(routes[vehicle_id]) > 1:
                depot_id = self.instance.vehicles[vehicle_id].depot_id
                routes[vehicle_id].append(depot_id)
        
        return routes

    def _handle_ride_pooling_quantum(self, routes: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Handle ride pooling scenario with quantum-inspired assignment."""
        vehicle_ids = list(self.instance.vehicles.keys())
        
        for i, request in enumerate(self.instance.ride_requests):
            vehicle_id = vehicle_ids[i % len(vehicle_ids)]            
            # Simple capacity check for cargo weight
            current_cargo_weight = sum(1 for loc in routes[vehicle_id] if loc.startswith("pickup")) - \
                               sum(1 for loc in routes[vehicle_id] if loc.startswith("dropoff"))
            if current_cargo_weight + request.passengers <= self.instance.vehicles[vehicle_id].capacity:
                routes[vehicle_id].append(request.pickup_location)
                routes[vehicle_id].append(request.dropoff_location)
            else:
                # Try next available vehicle
                for alt_vehicle_id in vehicle_ids:
                    alt_cargo_weight = sum(1 for loc in routes[alt_vehicle_id] if loc.startswith("pickup")) - \
                                   sum(1 for loc in routes[alt_vehicle_id] if loc.startswith("dropoff"))
                    if alt_cargo_weight + request.passengers <= self.instance.vehicles[alt_vehicle_id].capacity:
                        routes[alt_vehicle_id].append(request.pickup_location)
                        routes[alt_vehicle_id].append(request.dropoff_location)
                        break
        return routes
    
    def optimize_with_ortools(self) -> VRPResult:
        """Optimize VRP using Google OR-Tools - industry standard solver."""
        if not ORTOOLS_AVAILABLE:
            self.logger.error("OR-Tools not available for benchmarking")
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"total_distance": 0.0, "vehicles_used": 0, "error": "OR-Tools not installed"},
                runtime=0.0
            )
        
        self.logger.info("Starting VRP optimization with OR-Tools")
        start_time = time.time()
        
        try:
            # Check if this is a ride pooling scenario
            is_ride_pooling = bool(self.instance.ride_requests)
            if is_ride_pooling:
                self.logger.warning("OR-Tools configured for ride pooling with pickup-delivery constraints")
              # Create the routing index manager
            # For ride pooling, we need to include pickup and dropoff locations
            if is_ride_pooling:
                # Handle multi-depot VRPPD scenario
                depot_locations = [loc_id for loc_id in self.instance.location_ids if loc_id.startswith("depot")]
                
                # For multi-depot, we need to create depot start/end indices for each vehicle
                if len(depot_locations) > 1:
                    self.logger.info(f"Multi-depot VRPPD with {len(depot_locations)} depots: {depot_locations}")
                    
                    # Create location list: all depots + pickup/dropoff locations
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
                    # depot_indices should map to actual indices in location_list, not enumeration
                    depot_indices = {depot: location_to_index[depot] for depot in depot_locations}
                    
                    self.logger.debug(f"Depot locations: {depot_locations}")
                    self.logger.debug(f"Depot indices in location_list: {depot_indices}")
                    
                    for vehicle_id, vehicle in self.instance.vehicles.items():
                        depot_idx = depot_indices.get(vehicle.depot_id, 0)
                        vehicle_starts.append(depot_idx)
                        vehicle_ends.append(depot_idx)
                        self.logger.debug(f"Vehicle {vehicle_id} assigned to depot {vehicle.depot_id} (index {depot_idx})")
                    
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
                    depot_index = 0  # Single depot at index 0
                    
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
                depot_index = 0  # Depot is always at index 0
                pickup_deliveries = []
                
                # Create routing index manager
                manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_index)
            
            # Create routing model
            routing = pywrapcp.RoutingModel(manager)
              # Create distance callback
            def distance_callback(from_index, to_index):
                """Return the Manhattan distance between two points."""
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
                dimension_name,            )
            distance_dimension = routing.GetDimensionOrDie(dimension_name)
            # This is CRITICAL: penalize long routes to encourage using multiple vehicles
            # Increase coefficient for stronger penalty
            distance_dimension.SetGlobalSpanCostCoefficient(1000)
            
            # Add capacity constraints if applicable
            vehicle_capacities = []
            has_capacity_constraints = False
            
            for vehicle_id in sorted(self.instance.vehicles.keys()):
                vehicle = self.instance.vehicles[vehicle_id]
                capacity = getattr(vehicle, 'capacity', 100)  # Default capacity
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
                            # Find the request for this pickup
                            for request in self.instance.ride_requests:
                                if request.pickup_location == location_id:
                                    demand = request.passengers  # Contains cargo weight in kg
                                    break
                        elif location_id.startswith("dropoff"):
                            # Dropoffs have negative demand (cargo unloaded)
                            for request in self.instance.ride_requests:
                                if request.dropoff_location == location_id:
                                    demand = -request.passengers  # Negative cargo weight
                                    break
                    else:
                        # Standard VRP: use location demand
                        location = self.instance.locations[location_id]
                        demand = getattr(location, 'demand', 1)
                    
                    # Debug: Log demand for first few calls to verify callback
                    if from_index < 5:
                        self.logger.debug(f"Demand callback: index={from_index}, node={from_node}, location={location_id}, demand={demand}")
                    
                    return demand
                
                demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
                
                # Debug: Print vehicle capacities being used
                self.logger.info(f"Vehicle capacities being set: {vehicle_capacities}")
                
                routing.AddDimensionWithVehicleCapacity(
                    demand_callback_index,
                    0,  # null capacity slack
                    vehicle_capacities,  # vehicle maximum capacities
                    True,  # start cumul to zero
                    'Capacity'
                )
                  # Get the capacity dimension for use in constraints
                capacity_dimension = routing.GetDimensionOrDie('Capacity')
                
                # Add pickup and delivery constraints for ride pooling
                if is_ride_pooling and pickup_deliveries:
                    for pickup_idx, delivery_idx in pickup_deliveries:
                        pickup_node = manager.NodeToIndex(pickup_idx)
                        delivery_node = manager.NodeToIndex(delivery_idx)
                        routing.AddPickupAndDelivery(pickup_node, delivery_node)
                        
                        # Ensure pickup and delivery are on the same route
                        routing.solver().Add(
                            routing.VehicleVar(pickup_node) == routing.VehicleVar(delivery_node)                        )
                        
                        # Ensure pickup comes before delivery
                        routing.solver().Add(
                            capacity_dimension.CumulVar(pickup_node) <= 
                            capacity_dimension.CumulVar(delivery_node)
                        )
                  # Add time constraints based on vehicle max_total_work_time settings
                # Check if any vehicle has time constraints and collect them
                vehicle_time_constraints = {}
                has_time_constraints = False
                for vid, vehicle in enumerate(self.instance.vehicles.values()):
                    max_time = None
                    if hasattr(vehicle, 'max_total_work_time') and vehicle.max_total_work_time is not None:
                        max_time = vehicle.max_total_work_time  # in minutes
                    elif hasattr(vehicle, 'max_time') and vehicle.max_time is not None:
                        max_time = vehicle.max_time  # in minutes
                    
                    if max_time is not None:
                        vehicle_time_constraints[vid] = max_time
                        has_time_constraints = True
                
                if has_time_constraints:
                    # Log the different time constraints
                    for vid, max_time in vehicle_time_constraints.items():
                        vehicle_list = list(self.instance.vehicles.values())
                        vehicle = vehicle_list[vid]
                        vehicle_type = getattr(vehicle, 'vehicle_type', 'unknown')
                        self.logger.info(f"Vehicle {vid} ({vehicle_type}): max {max_time} minutes ({max_time/60:.1f} hours)")
                    
                    # Create a proper time callback that includes travel time + service time
                    def time_callback(from_index, to_index):
                        from_node = manager.IndexToNode(from_index)
                        to_node = manager.IndexToNode(to_index)
                        from_location_id = location_list[from_node]
                        to_location_id = location_list[to_node]
                        
                        # Get travel distance (in coordinate units, likely lat/lon degrees)
                        distance_units = self.instance.get_distance(from_location_id, to_location_id)
                        
                        # Convert distance units to approximate km for realistic travel time
                        # For GPS coordinates, 1 degree ≈ 111 km at latitude ~45° (Northern Italy)
                        distance_km = distance_units * 111  # approximate conversion
                        
                        # Calculate travel time assuming realistic urban delivery speed
                        avg_speed_kmh = 50  # Higher speed for faster urban/highway travel
                        travel_time_hours = distance_km / avg_speed_kmh
                        travel_time_minutes = travel_time_hours * 60
                        
                        # Add service time at the 'from' location
                        service_time_minutes = 0
                        if from_location_id in self.instance.locations:
                            location = self.instance.locations[from_location_id]
                            service_time_minutes = getattr(location, 'service_time', 0)
                        
                        total_time_minutes = travel_time_minutes + service_time_minutes
                        # Convert to seconds for OR-Tools
                        total_time_seconds = int(total_time_minutes * 60)
                        
                        return total_time_seconds
                    
                    time_callback_index = routing.RegisterTransitCallback(time_callback)
                      # Use the maximum time constraint for the dimension (OR-Tools requires a single max)
                    max_overall_time = max(vehicle_time_constraints.values())
                    max_time_seconds = int(max_overall_time * 60)
                    
                    self.logger.info(f"Adding time dimension with max {max_overall_time} minutes ({max_overall_time/60:.1f} hours)")
                    
                    # Add time dimension with proper time constraints
                    routing.AddDimension(
                        time_callback_index,
                        max_time_seconds // 5,  # slack: 20% of max time for flexibility
                        max_time_seconds,  # maximum total time per vehicle
                        True,  # start cumul to zero
                        'TotalTime'
                    )
                    
                    # Apply individual vehicle time constraints
                    time_dimension = routing.GetDimensionOrDie('TotalTime')
                    for vid, max_time in vehicle_time_constraints.items():
                        max_time_seconds_vehicle = int(max_time * 60)
                        end_node = routing.End(vid)
                        time_dimension.CumulVar(end_node).SetMax(max_time_seconds_vehicle)                
                # Note: No arbitrary stop count constraint - only time constraint matters
                # Vehicles can do as many stops as they can fit within the time limit
                self.logger.info("No stop count constraint - vehicles limited only by time constraints")
            else:
                # Even without capacity constraints, we still need pickup-delivery constraints
                if is_ride_pooling and pickup_deliveries:
                    for pickup_idx, delivery_idx in pickup_deliveries:
                        pickup_node = manager.NodeToIndex(pickup_idx)
                        delivery_node = manager.NodeToIndex(delivery_idx)
                        routing.AddPickupAndDelivery(pickup_node, delivery_node)
                          # Ensure pickup and delivery are on the same route
                        routing.solver().Add(
                            routing.VehicleVar(pickup_node) == routing.VehicleVar(delivery_node)
                        )
            
            # Set search parameters with optimized time limit
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
            )
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )
            
            # Adaptive time limit based on problem size
            num_locations = len(self.instance.location_ids)
            if num_locations <= 10:
                time_limit = 2  # Small problems: 2 seconds
            elif num_locations <= 50:
                time_limit = 5  # Medium problems: 5 seconds  
            elif num_locations <= 100:
                time_limit = 10  # Large problems: 10 seconds
            else:
                time_limit = 20  # Very large problems: 20 seconds
                
            search_parameters.time_limit.FromSeconds(time_limit)
            
            # Solve the problem
            self.logger.info("Solving with OR-Tools...")
            solution = routing.SolveWithParameters(search_parameters)
            runtime = (time.time() - start_time) * 1000  # Convert to milliseconds
            
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
                        
                        # Get vehicle's start depot for multi-depot scenarios
                        if is_ride_pooling and len(depot_locations) > 1:
                            start_depot = vehicle.depot_id
                        else:
                            start_depot = depot_locations[0] if depot_locations else self.instance.location_ids[0]
                        
                        route = [start_depot]  # Start at vehicle's depot
                        
                        index = routing.Start(vehicle_idx)
                        route_distance = 0
                        
                        while not routing.IsEnd(index):
                            node_index = manager.IndexToNode(index)
                            
                            # For multi-depot, skip depot nodes unless they're different from start
                            if is_ride_pooling and len(depot_locations) > 1:
                                if node_index < len(depot_locations):
                                    # This is a depot node - only add if different from start
                                    depot_location_id = location_list[node_index]
                                    if depot_location_id != start_depot:
                                        route.append(depot_location_id)
                                else:
                                    # This is a pickup/dropoff location
                                    route.append(location_list[node_index])
                            else:
                                # Single depot case - skip depot nodes in middle
                                if (not is_ride_pooling and node_index != 0) or (is_ride_pooling and node_index != 0):
                                    route.append(location_list[node_index])
                            
                            previous_index = index
                            index = solution.Value(routing.NextVar(index))
                            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_idx)
                        
                        # Only include routes that actually visit locations
                        if len(route) > 1:  # More than just starting depot
                            route.append(start_depot)  # Return to depot
                            routes[vehicle_id] = route
                            total_distance += route_distance
                            self.logger.debug(f"Vehicle {vehicle_id} route: {route}, distance: {route_distance}")
                        else:                            # Empty route - vehicle not used
                            routes[vehicle_id] = [start_depot]
                
                # Apply driver regulations for heavy trucks
                routes = self._check_driver_regulations(routes)
                
                # Calculate metrics
                metrics = self._calculate_vrp_metrics(routes)
                
                # Use actual total distance calculated from routes
                actual_total_distance = metrics.get('total_distance', total_distance)
                
                metrics.update({
                    'ortools_distance': actual_total_distance,
                    'ortools_objective': solution.ObjectiveValue(),
                    'ortools_runtime': runtime,
                    'ortools_status': 'optimal' if solution else 'failed'
                })  
                              
                if is_ride_pooling:
                    metrics['ortools_note'] = "OR-Tools with pickup-delivery constraints (VRPPD)"
                
                self.logger.info(f"OR-Tools solution: distance={actual_total_distance:.2f}, vehicles={len([r for r in routes.values() if len(r) > 2])}, runtime={runtime:.2f}ms")
                
                return VRPResult(
                    status="optimal",
                    objective_value=actual_total_distance,  # Use actual distance, not negative
                    routes=routes,
                    metrics=metrics,
                    runtime=runtime
                )
            else:
                self.logger.warning("OR-Tools could not find a solution")
                # Create empty routes for each vehicle using their respective depots
                empty_routes = {}
                for vehicle_id, vehicle in self.instance.vehicles.items():
                    if is_ride_pooling and len(depot_locations) > 1:
                        empty_routes[vehicle_id] = [vehicle.depot_id]
                    else:
                        empty_routes[vehicle_id] = [depot_locations[0] if depot_locations else self.instance.location_ids[0]]
                
                return VRPResult(
                    status="infeasible",
                    objective_value=0.0,
                    routes=empty_routes,
                    metrics={"total_distance": 0.0, "vehicles_used": 0, "error": "no solution found"},
                    runtime=runtime
                )
                
        except Exception as e:
            self.logger.error(f"Error in OR-Tools optimization: {str(e)}")
            runtime = (time.time() - start_time) * 1000  # Convert to milliseconds
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"total_distance": 0.0, "vehicles_used": 0, "error": str(e)},
                runtime=runtime
            )

    def optimize_with_classical_benders(self) -> VRPResult:
        """Optimize VRP using classical greedy approach."""
        self.logger.info("Starting VRP optimization with classical methods")
        start_time = time.time()
        
        try:
            def _optimize_classical():
                return self._greedy_vrp_solution()
            routes = run_with_timeout(_optimize_classical, timeout_duration=15)
            metrics = self._calculate_vrp_metrics(routes)
            
            runtime = (time.time() - start_time) * 1000  # Convert to milliseconds
            obj_value = -metrics.get("total_distance", 0.0)
            
            return VRPResult(
                status="optimal",
                objective_value=obj_value,
                routes=routes,
                metrics=metrics,
                runtime=runtime
            )
        except TimeoutError:
            self.logger.error("Classical optimization timed out")
            runtime = (time.time() - start_time) * 1000  # Convert to milliseconds
            return VRPResult(
                status="timeout",
                objective_value=0.0,
                routes={},
                metrics={"total_distance": 0.0, "vehicles_used": 0},
                runtime=runtime)
        except Exception as e:
            self.logger.error(f"Error in classical optimization: {str(e)}")
            runtime = (time.time() - start_time) * 1000  # Convert to milliseconds
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"total_distance": 0.0, "vehicles_used": 0},
                runtime=runtime
            )

    def _greedy_vrp_solution(self) -> Dict[str, List[str]]:
        """Generate a greedy VRP solution as baseline."""
        routes = {}
        vehicle_ids = list(self.instance.vehicles.keys())
        
        # Initialize routes with depot starts
        for vehicle_id in vehicle_ids:
            depot_id = self.instance.vehicles[vehicle_id].depot_id
            routes[vehicle_id] = [depot_id]
        
        # Handle ride pooling
        if self.instance.ride_requests:
            for i, request in enumerate(self.instance.ride_requests):
                vehicle_id = vehicle_ids[i % len(vehicle_ids)]
                routes[vehicle_id].append(request.pickup_location)
                routes[vehicle_id].append(request.dropoff_location)
        
        # Handle delivery with nearest neighbor
        else:
            unvisited = set([loc_id for loc_id in self.instance.location_ids 
                           if not loc_id.startswith("depot")])
            
            for vehicle_id, vehicle in self.instance.vehicles.items():
                current_location = vehicle.depot_id
                vehicle_load = 0
                local_iterations = 0
                max_local_iterations = min(len(unvisited), 20)
                
                while unvisited and vehicle_load < vehicle.capacity and local_iterations < max_local_iterations:
                    local_iterations += 1
                    
                    if not unvisited:
                        break
                    
                    # Find nearest unvisited customer
                    search_candidates = list(unvisited)[:min(10, len(unvisited))]
                    
                    try:
                        nearest_customer = min(search_candidates, 
                                             key=lambda loc: self.instance.get_distance(current_location, loc))
                    except (ValueError, KeyError):
                        nearest_customer = search_candidates[0] if search_candidates else None
                    
                    if nearest_customer is None:
                        break
                    
                    if nearest_customer in self.instance.locations:
                        customer_demand = self.instance.locations[nearest_customer].demand
                        if vehicle_load + customer_demand <= vehicle.capacity:
                            routes[vehicle_id].append(nearest_customer)
                            current_location = nearest_customer
                            vehicle_load += customer_demand
                            unvisited.remove(nearest_customer)
                        else:
                            break
                    else:
                        routes[vehicle_id].append(nearest_customer)
                        current_location = nearest_customer
                        unvisited.remove(nearest_customer)                # Return to depot if visited any customers
                if len(routes[vehicle_id]) > 1:
                    routes[vehicle_id].append(vehicle.depot_id)
        
        return routes
    
    # PyVRP plotting method removed - no longer using pyVRP for benchmarking
    
    def _calculate_vrp_metrics(self, routes: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate VRP-specific metrics."""
        metrics = {}
        
        total_distance = 0.0
        total_vehicles_used = 0
        
        for vehicle_id, route in routes.items():
            if len(route) > 1:
                total_vehicles_used += 1
                
                route_distance = 0.0
                for i in range(len(route) - 1):
                    try:
                        route_distance += self.instance.get_distance(route[i], route[i + 1])
                    except (ValueError, KeyError):
                        route_distance += 1.0
                
                total_distance += route_distance
                metrics[f"{vehicle_id}_distance"] = route_distance
                metrics[f"{vehicle_id}_stops"] = len(route) - 2
        
        metrics["total_distance"] = total_distance
        metrics["vehicles_used"] = total_vehicles_used
        metrics["avg_distance_per_vehicle"] = total_distance / max(total_vehicles_used, 1)
        
        # Calculate service metrics for ride pooling
        if self.instance.ride_requests:
            served_requests = 0
            for request in self.instance.ride_requests:
                pickup_served = False
                dropoff_served = False
                
                for route in routes.values():
                    if request.pickup_location in route:
                        pickup_served = True
                    if request.dropoff_location in route:
                        dropoff_served = True
                
                if pickup_served and dropoff_served:
                    served_requests += 1
            
            metrics["service_rate"] = served_requests / len(self.instance.ride_requests)
            metrics["requests_served"] = served_requests
        
        if total_distance > 0:
            metrics["solution_quality"] = 1000.0 / total_distance
        else:
            metrics["solution_quality"] = 0.0
        
        return metrics
    
    def _calculate_route_metrics(self, route, include_time=True):
        """Calculate metrics for a single route including distance and time."""
        metrics = {}
        total_distance = 0
        total_time = 0
        current_time = 0
        
        for i in range(len(route) - 1):
            loc1_id = route[i]
            loc2_id = route[i + 1]
            
            # Calculate distance
            distance = self.instance.get_distance(loc1_id, loc2_id)
            total_distance += distance
            
            # Calculate time if requested
            if include_time:
                # Get travel time between locations (or estimate from distance)
                travel_time = self.instance.get_duration(loc1_id, loc2_id)
                current_time += travel_time
                
                # Add service time if applicable
                if i > 0:  # Skip service time at depot start
                    loc1 = self.instance.locations[loc1_id]
                    service_time = getattr(loc1, 'service_time', 0)
                    current_time += service_time
        
        metrics['distance'] = total_distance
        
        if include_time:
            metrics['time'] = current_time
            
        return metrics

    def _create_result_from_routes(self, routes, status="optimal", runtime=0):
        """Create VRPResult object from routes with enhanced metrics."""
        result = VRPResult(
            status=status,
            objective_value=0,  # Will be updated below
            routes=routes,
            runtime=runtime
        )
        
        # Initialize metrics
        metrics = {
            'total_distance': 0,
            'total_time': 0,
            'vehicles_used': len([r for r in routes.values() if len(r) > 2]),
            'avg_distance_per_vehicle': 0,
            'avg_time_per_vehicle': 0,
        }
        
        # Calculate metrics for each route
        active_vehicles = 0
        for vehicle_id, route in routes.items():
            if len(route) > 2:  # Vehicle is used
                active_vehicles += 1
                route_metrics = self._calculate_route_metrics(route, include_time=True)
                
                vehicle_distance = route_metrics['distance']
                metrics['total_distance'] += vehicle_distance
                metrics[f"{vehicle_id}_distance"] = vehicle_distance
                
                # Add time metrics
                if 'time' in route_metrics:
                    vehicle_time = route_metrics['time']
                    metrics['total_time'] += vehicle_time
                    metrics[f"{vehicle_id}_time"] = vehicle_time
        
        # Calculate averages
        if active_vehicles > 0:
            metrics['avg_distance_per_vehicle'] = metrics['total_distance'] / active_vehicles
            if metrics['total_time'] > 0:
                metrics['avg_time_per_vehicle'] = metrics['total_time'] / active_vehicles
        
        # Use total distance as objective value
        result.objective_value = metrics['total_distance']
        result.metrics = metrics
        return result
    
    # Advanced Initial Solution Methods from Recent Literature
    
    def optimize_with_advanced_heuristics(self, method='best_of_multiple') -> VRPResult:
        """
        Optimize VRP using advanced construction heuristics from recent literature.
        
        Implements multiple state-of-the-art construction algorithms:
        - Nearest Neighbor with perturbation
        - Clarke-Wright Savings Algorithm  
        - Firefly Algorithm initialization (CVRP-FA)
        - Greedy insertion with demand-distance prioritization
        
        Args:
            method: Which heuristic to use ('nearest_neighbor', 'savings', 'firefly', 
                   'greedy_insertion', 'best_of_multiple')
        """
        self.logger.info(f"Starting VRP optimization with advanced heuristics: {method}")
        start_time = time.time()
        try:
            # Skip ride pooling scenarios for now (incompatible with current generator)
            if self.instance.ride_requests:
                self.logger.warning("Advanced heuristics not yet implemented for ride pooling scenarios")
                return self._create_empty_result("not_implemented", time.time() - start_time)
            
            # Prepare data for InitialSolutionGenerator
            customers = []
            vehicles = []
            depot_location = None
              # Convert instance data for the generator
            for i, loc_id in enumerate(self.instance.location_ids):
                location = self.instance.locations[loc_id]
                
                # Robust coordinate extraction
                if (hasattr(location, 'lat') and hasattr(location, 'lon') and 
                    location.lat is not None and location.lon is not None):
                    x, y = float(location.lat), float(location.lon)
                else:
                    # Use index-based coordinates for synthetic data
                    x, y = float(i * 10), float(i * 10)
                  # Set depot as first location with 0 demand
                if loc_id.startswith('depot') or i == 0:
                    if depot_location is None:
                        depot_location = (float(x), float(y))
                else:
                    from initial_solution_generator import Customer
                    demand = float(getattr(location, 'demand', 1))
                    customers.append(Customer(id=loc_id, x=float(x), y=float(y), demand=demand))
              # Prepare vehicles
            for v_id, vehicle in self.instance.vehicles.items():
                from initial_solution_generator import Vehicle
                capacity = float(getattr(vehicle, 'capacity', 100))
                vehicles.append(Vehicle(id=v_id, capacity=capacity))
            
            # Set default depot if not found
            if depot_location is None:
                depot_location = (0.0, 0.0)
            
            # Create initial solution generator
            generator = InitialSolutionGenerator(customers, vehicles, depot_location)
              # Generate solution using the specified method
            if method == 'best_of_multiple':
                # Try all methods and select the best
                methods = ['nearest_neighbor', 'savings', 'firefly', 'greedy_insertion']
                best_routes = None
                best_distance = float('inf')
                best_method_name = None
                
                for test_method in methods:
                    try:
                        if test_method == 'nearest_neighbor':
                            routes = generator.generate_nearest_neighbor_solution()
                        elif test_method == 'savings':
                            routes = generator.generate_savings_algorithm_solution()
                        elif test_method == 'firefly':
                            routes = generator.generate_firefly_algorithm_solution()
                        elif test_method == 'greedy_insertion':
                            routes = generator.generate_greedy_insertion_solution()
                        
                        # Calculate total distance for comparison
                        total_distance = sum(route.total_distance for route in routes)
                        if total_distance < best_distance:
                            best_distance = total_distance
                            best_routes = routes
                            best_method_name = test_method
                    except Exception as e:
                        self.logger.warning(f"Method {test_method} failed: {e}")
                        continue
                
                if best_routes is None:
                    self.logger.error("All heuristic methods failed")
                    return self._create_empty_result("error", time.time() - start_time)
                
                routes = best_routes
                self.logger.info(f"Best method: {best_method_name} with distance {best_distance:.2f}")
            else:
                # Use specific method
                if method == 'nearest_neighbor':
                    routes = generator.generate_nearest_neighbor_solution()
                elif method == 'savings':
                    routes = generator.generate_savings_algorithm_solution()
                elif method == 'firefly':
                    routes = generator.generate_firefly_algorithm_solution()
                elif method == 'greedy_insertion':
                    routes = generator.generate_greedy_insertion_solution()
                else:
                    self.logger.error(f"Unknown method: {method}")
                    return self._create_empty_result("error", time.time() - start_time)            
            if not routes:
                self.logger.error("No routes generated")
                return self._create_empty_result("infeasible", time.time() - start_time)
            
            # Convert solution to our route format
            vrp_routes = {}
            for vehicle_id in self.instance.vehicles.keys():
                vrp_routes[vehicle_id] = [list(self.instance.location_ids)[0]]  # Start with depot
            
            # Add routes from solution
            for i, route in enumerate(routes):
                if route.customers:
                    vehicle_id = list(self.instance.vehicles.keys())[i % len(self.instance.vehicles)]
                    vrp_routes[vehicle_id] = [list(self.instance.location_ids)[0]]  # Depot
                    vrp_routes[vehicle_id].extend(route.customers)
                    vrp_routes[vehicle_id].append(list(self.instance.location_ids)[0])  # Return to depot
            
            runtime = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Create result with metrics
            result = self._create_result_from_routes(vrp_routes, "optimal", runtime)
            result.metrics.update({
                'method': method,
                'algorithm': 'advanced_heuristics',
                'feasible': True,
                'vehicles_used': len([r for r in vrp_routes.values() if len(r) > 2])
            })
            
            self.logger.info(f"Advanced heuristics solution: distance={result.objective_value:.2f}, "
                           f"vehicles={result.metrics.get('vehicles_used', 0)}, runtime={runtime:.2f}ms")            
            return result
        except Exception as e:
            self.logger.error(f"Error in advanced heuristics optimization: {e}")
            return self._create_empty_result("error", (time.time() - start_time) * 1000)
    
    def optimize_with_2opt_improvement(self, initial_method='greedy_insertion',
                                     improvement_type='best') -> VRPResult:
        """
        Optimize VRP using advanced initial solution + 2-opt local search improvement.
        
        Implements Croes' 1958 2-opt framework with modern enhancements:
        - Complete search mechanism with restart on improvement
        - First-improvement and best-improvement variants
        - VRP-specific depot boundary constraints
        - Efficient edge-crossing detection
        
        Args:
            initial_method: Initial solution method ('greedy_insertion', 'firefly', etc.)
            improvement_type: '2opt_first' or '2opt_best' improvement strategy
        """
        self.logger.info(f"Starting VRP optimization with {initial_method} + 2-opt {improvement_type}")
        start_time = time.time()
        
        try:
            # Skip ride pooling scenarios for now
            if self.instance.ride_requests:
                self.logger.warning("2-opt improvement not yet implemented for ride pooling scenarios")
                return self._create_empty_result("not_implemented", time.time() - start_time)
            
            # Get initial solution using advanced heuristics
            initial_result = self.optimize_with_advanced_heuristics(initial_method)
            if initial_result.status != "optimal":
                return initial_result
              # Create distance matrix first
            distance_matrix = self._create_distance_matrix()
            
            # Create 2-opt optimizer with distance matrix
            depot_nodes = [loc_id for loc_id in self.instance.location_ids if 'depot' in loc_id.lower()]
            optimizer = TwoOptLocalSearch(distance_matrix, depot_nodes)
              # Create location to index mapping
            location_to_index = {loc_id: i for i, loc_id in enumerate(self.instance.location_ids)}
            
            # Improve each route with 2-opt
            improved_routes = {}
            total_improvement = 0
            
            for vehicle_id, route in initial_result.routes.items():
                if len(route) <= 3:  # Skip routes with only depot or 1 customer
                    improved_routes[vehicle_id] = route
                    continue
                
                # Apply 2-opt improvement directly on route (location IDs)
                original_distance = optimizer.calculate_route_distance(route, location_to_index)
                
                if improvement_type == 'best':
                    improved_route, improved_distance, stats = optimizer.two_opt_best_improvement(route, location_to_index)
                else:  # first improvement
                    improved_route, improved_distance, stats = optimizer.two_opt_first_improvement(route, location_to_index)
                
                improvement = original_distance - improved_distance
                total_improvement += improvement
                
                improved_routes[vehicle_id] = improved_route
                
                self.logger.debug(f"Vehicle {vehicle_id}: {original_distance:.2f} -> {improved_distance:.2f} "
                                f"(improvement: {improvement:.2f})")
            
            runtime = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Create result with improved routes
            result = self._create_result_from_routes(improved_routes, "optimal", runtime)
            result.metrics.update({
                'method': f"{initial_method}+{improvement_type}",
                'algorithm': '2opt_local_search',
                'initial_distance': initial_result.objective_value,
                'total_improvement': total_improvement,
                'improvement_percentage': (total_improvement / initial_result.objective_value) * 100 
                                        if initial_result.objective_value > 0 else 0,
                'vehicles_used': len([r for r in improved_routes.values() if len(r) > 2])
            })
            
            self.logger.info(f"2-opt improvement: {initial_result.objective_value:.2f} -> {result.objective_value:.2f} "
                           f"(improvement: {total_improvement:.2f}, {result.metrics['improvement_percentage']:.1f}%), "
                           f"runtime={runtime:.2f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in 2-opt optimization: {str(e)}")
            runtime = (time.time() - start_time) * 1000
            return self._create_empty_result("error", runtime)
    
    def _convert_advanced_solution_to_vrp_routes(self, solution):
        """Convert advanced heuristic solution format to VRP routes format."""
        routes = {}
        
        for i, route_data in enumerate(solution['routes']):
            vehicle_ids = list(self.instance.vehicles.keys())
            if i < len(vehicle_ids):
                vehicle_id = vehicle_ids[i]
                
                # Convert customer indices back to location IDs
                route = []
                for customer_idx in route_data['customers']:
                    if customer_idx < len(self.instance.location_ids):
                        route.append(self.instance.location_ids[customer_idx])
                
                # Add depot at start and end if not already present
                depot_locations = [loc_id for loc_id in self.instance.location_ids if loc_id.startswith('depot')]
                depot_id = depot_locations[0] if depot_locations else self.instance.location_ids[0]
                
                if not route or route[0] != depot_id:
                    route.insert(0, depot_id)
                if not route or route[-1] != depot_id:
                    route.append(depot_id)
                
                routes[vehicle_id] = route
        
        return routes
    
    def _create_distance_matrix(self):
        """Create distance matrix for all locations."""
        n = len(self.instance.location_ids)
        distance_matrix = np.zeros((n, n))
        
        for i, loc1_id in enumerate(self.instance.location_ids):
            for j, loc2_id in enumerate(self.instance.location_ids):
                if i != j:
                    distance = self.instance.get_distance(loc1_id, loc2_id)
                    distance_matrix[i][j] = distance
        
        return distance_matrix
    
    def _calculate_route_distance(self, route_indices, distance_matrix):
        """Calculate total distance for a route given as indices."""
        total_distance = 0
        for i in range(len(route_indices) - 1):
            total_distance += distance_matrix[route_indices[i]][route_indices[i + 1]]
        return total_distance
    
    def _create_empty_result(self, status, runtime):
        """Create an empty result for failed optimizations."""
        empty_routes = {vehicle_id: [] for vehicle_id in self.instance.vehicles.keys()}
        return VRPResult(
            status=status,
            objective_value=0.0,
            routes=empty_routes,
            metrics={"total_distance": 0.0, "vehicles_used": 0, "error": status},
            runtime=runtime
        )
    
    def optimize_with_nearest_neighbor_heuristic(self) -> VRPResult:
        """Optimize using Nearest Neighbor construction heuristic with perturbation."""
        if not ADVANCED_METHODS_AVAILABLE:
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": "Advanced methods not available"},
                runtime=0.0
            )
        
        try:
            advanced_optimizer = AdvancedInitialSolutionOptimizer(self.instance, self.objective)
            result = advanced_optimizer.optimize_nearest_neighbor(perturbation=0.1)
            
            if result.status == "optimal":
                self.logger.info(f"Nearest Neighbor heuristic: distance={result.objective_value:.2f}, "
                               f"vehicles={result.metrics.get('vehicles_used', 0)}, "
                               f"runtime={result.runtime:.2f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Nearest Neighbor optimization: {str(e)}")
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": str(e)},
                runtime=0.0
            )
    
    def optimize_with_savings_algorithm(self) -> VRPResult:
        """Optimize using Clarke-Wright Savings Algorithm."""
        if not ADVANCED_METHODS_AVAILABLE:
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": "Advanced methods not available"},
                runtime=0.0
            )
        
        try:
            advanced_optimizer = AdvancedInitialSolutionOptimizer(self.instance, self.objective)
            result = advanced_optimizer.optimize_savings_algorithm(perturbation=0.12)
            
            if result.status == "optimal":
                self.logger.info(f"Savings Algorithm: distance={result.objective_value:.2f}, "
                               f"vehicles={result.metrics.get('vehicles_used', 0)}, "
                               f"runtime={result.runtime:.2f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Savings Algorithm optimization: {str(e)}")
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": str(e)},
                runtime=0.0
            )
    
    def optimize_with_firefly_algorithm(self) -> VRPResult:
        """
        Optimize using Firefly Algorithm (CVRP-FA) from recent literature.
        
        Implements:
        - Firefly Algorithm initialization for CVRP
        - Position vectors encoding customer-to-vehicle assignments
        - Random perturbation factors (10-15% of customers)
        - Large Neighbourhood Search feasibility principles
        """
        if not ADVANCED_METHODS_AVAILABLE:
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": "Advanced methods not available"},
                runtime=0.0
            )
        
        try:
            advanced_optimizer = AdvancedInitialSolutionOptimizer(self.instance, self.objective)
            
            # Adaptive parameters based on problem size
            num_customers = len(advanced_optimizer.customers) if advanced_optimizer.customers else 0
            if num_customers <= 10:
                num_fireflies = 15
                perturbation = 0.12
            elif num_customers <= 50:
                num_fireflies = 20
                perturbation = 0.15
            else:
                num_fireflies = 25
                perturbation = 0.12
            
            result = advanced_optimizer.optimize_firefly_algorithm(num_fireflies, perturbation)
            
            if result.status == "optimal":
                self.logger.info(f"Firefly Algorithm (CVRP-FA): distance={result.objective_value:.2f}, "
                               f"vehicles={result.metrics.get('vehicles_used', 0)}, "
                               f"fireflies={num_fireflies}, perturbation={perturbation}, "
                               f"runtime={result.runtime:.2f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Firefly Algorithm optimization: {str(e)}")
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": str(e)},
                runtime=0.0
            )
    
    def optimize_with_greedy_insertion(self) -> VRPResult:
        """
        Optimize using Greedy Insertion with demand-to-distance ratio priority.
        
        Implements:
        - Demand-to-distance ratio prioritization
        - Greedy insertion with capacity constraints
        - Random perturbation for solution diversity
        """
        if not ADVANCED_METHODS_AVAILABLE:
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": "Advanced methods not available"},
                runtime=0.0
            )
        
        try:
            advanced_optimizer = AdvancedInitialSolutionOptimizer(self.instance, self.objective)
            result = advanced_optimizer.optimize_greedy_insertion(
                demand_priority=True, 
                perturbation=0.1
            )
            
            if result.status == "optimal":
                self.logger.info(f"Greedy Insertion: distance={result.objective_value:.2f}, "
                               f"vehicles={result.metrics.get('vehicles_used', 0)}, "
                               f"capacity_util={result.metrics.get('avg_capacity_utilization', 0):.1%}, "
                               f"runtime={result.runtime:.2f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Greedy Insertion optimization: {str(e)}")
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": str(e)},
                runtime=0.0
            )
    
    def optimize_with_best_construction_heuristic(self) -> VRPResult:
        """
        Optimize using the best of multiple construction heuristics.
        
        Combines:
        - Nearest Neighbor with perturbation
        - Clarke-Wright Savings Algorithm  
        - Firefly Algorithm (CVRP-FA)
        - Greedy Insertion with demand-distance priority
        - Automatic best selection based on objective value
        """
        if not ADVANCED_METHODS_AVAILABLE:
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": "Advanced methods not available"},
                runtime=0.0
            )
        
        try:
            advanced_optimizer = AdvancedInitialSolutionOptimizer(self.instance, self.objective)
            result = advanced_optimizer.optimize_best_of_multiple(num_solutions=4)
            
            if result.status == "optimal":
                self.logger.info(f"Best Construction Heuristic: distance={result.objective_value:.2f}, "
                               f"best_method={result.metrics.get('best_method', 'unknown')}, "
                               f"vehicles={result.metrics.get('vehicles_used', 0)}, "
                               f"runtime={result.runtime:.2f}ms")
                
                # Log method comparison if available
                if "method_comparison" in result.metrics:
                    comparison = result.metrics["method_comparison"]
                    self.logger.debug(f"Method comparison: {comparison}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Best Construction Heuristic optimization: {str(e)}")
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": str(e)},
                runtime=0.0
            )
    
    def optimize_with_initial_solution_generator(self) -> VRPResult:
        """
        Optimize using InitialSolutionGenerator with multiple heuristics.
        
        Returns:
            VRPResult: Optimization result with the best solution from multiple heuristics
        """
        start_time = time.time()
        
        if not NEW_METHODS_AVAILABLE:
            self.logger.warning("InitialSolutionGenerator not available")
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": "InitialSolutionGenerator not available"},
                runtime=0.0
            )
        
        try:
            # Create distance matrix from instance
            distance_matrix = self._create_distance_matrix()
            
            # Initialize the generator
            generator = InitialSolutionGenerator(self.instance, distance_matrix, self.logger)
            
            # Try all available heuristics and select the best
            best_result = None
            best_distance = float('inf')
            method_results = {}
            
            # Available heuristics in the generator
            heuristics = ['nearest_neighbor', 'savings', 'greedy_insertion', 'firefly']
            
            for heuristic in heuristics:
                try:
                    self.logger.info(f"Running {heuristic} heuristic...")
                    
                    if heuristic == 'nearest_neighbor':
                        routes = generator.nearest_neighbor_heuristic()
                    elif heuristic == 'savings':
                        routes = generator.savings_algorithm()
                    elif heuristic == 'greedy_insertion':
                        routes = generator.greedy_insertion_heuristic()
                    elif heuristic == 'firefly':
                        routes = generator.firefly_algorithm_heuristic()
                    
                    # Calculate metrics for this solution
                    metrics = self._calculate_vrp_metrics(routes)
                    total_distance = metrics.get("total_distance", float('inf'))
                    
                    method_results[heuristic] = {
                        'distance': total_distance,
                        'vehicles_used': metrics.get("vehicles_used", 0),
                        'routes': routes
                    }
                    
                    if total_distance < best_distance:
                        best_distance = total_distance
                        best_result = {
                            'routes': routes,
                            'metrics': metrics,
                            'method': heuristic
                        }
                    
                    self.logger.info(f"{heuristic}: distance={total_distance:.2f}, vehicles={metrics.get('vehicles_used', 0)}")
                    
                except Exception as e:
                    self.logger.warning(f"Error in {heuristic} heuristic: {str(e)}")
                    method_results[heuristic] = {'error': str(e)}
            
            runtime = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            if best_result:
                # Add method comparison to metrics
                best_result['metrics']['best_method'] = best_result['method']
                best_result['metrics']['method_comparison'] = method_results
                
                self.logger.info(f"InitialSolutionGenerator best: {best_result['method']} with distance={best_distance:.2f}")
                
                return VRPResult(
                    status="optimal",
                    objective_value=-best_distance,  # Negative for maximization format
                    routes=best_result['routes'],
                    metrics=best_result['metrics'],
                    runtime=runtime
                )
            else:
                return VRPResult(
                    status="no_solution",
                    objective_value=0.0,
                    routes={},
                    metrics={"error": "No feasible solution found", "method_comparison": method_results},
                    runtime=runtime
                )
                
        except Exception as e:
            runtime = (time.time() - start_time) * 1000
            self.logger.error(f"Error in InitialSolutionGenerator optimization: {str(e)}")
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": str(e)},
                runtime=runtime
            )

    def optimize_with_two_opt_local_search(self) -> VRPResult:
        """
        Optimize using 2-opt local search algorithm (Croes' 1958 framework with modern enhancements).
        
        This method first generates an initial solution using the nearest neighbor heuristic,
        then applies 2-opt local search to improve it.
        
        Returns:
            VRPResult: Optimization result after 2-opt improvement
        """
        start_time = time.time()
        
        if not NEW_METHODS_AVAILABLE:
            self.logger.warning("TwoOptLocalSearch not available")
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": "TwoOptLocalSearch not available"},
                runtime=0.0
            )
        
        try:
            # Create distance matrix from instance
            distance_matrix = self._create_distance_matrix()
            
            # Step 1: Generate initial solution using nearest neighbor
            generator = InitialSolutionGenerator(self.instance, distance_matrix, self.logger)
            initial_routes = generator.nearest_neighbor_heuristic()
            
            if not initial_routes:
                return VRPResult(
                    status="no_solution",
                    objective_value=0.0,
                    routes={},
                    metrics={"error": "Could not generate initial solution"},
                    runtime=(time.time() - start_time) * 1000
                )
            
            initial_metrics = self._calculate_vrp_metrics(initial_routes)
            initial_distance = initial_metrics.get("total_distance", 0.0)
            
            self.logger.info(f"Initial solution distance: {initial_distance:.2f}")
            
            # Step 2: Apply 2-opt local search
            # Get depot nodes from instance
            depot_nodes = [vehicle_data.get('depot', 'depot') for vehicle_data in self.instance.vehicles.values()]
            depot_nodes = list(set(depot_nodes))  # Remove duplicates
            
            # Initialize 2-opt optimizer
            two_opt = TwoOptLocalSearch(distance_matrix, depot_nodes)
            
            # Create location to index mapping
            location_to_index = {loc: idx for idx, loc in enumerate(self.instance.location_ids)}
            
            # Apply 2-opt to each route
            improved_routes = {}
            total_improvement = 0.0
            
            for vehicle_id, route in initial_routes.items():
                if len(route) > 3:  # Only apply 2-opt to routes with more than 3 nodes (depot + 2+ customers)
                    self.logger.debug(f"Applying 2-opt to vehicle {vehicle_id} with {len(route)} nodes")
                    
                    # Apply 2-opt improvement
                    improved_route, improvement = two_opt.improve_route(
                        route, location_to_index, strategy='first_improvement', max_iterations=100
                    )
                    
                    improved_routes[vehicle_id] = improved_route
                    total_improvement += improvement
                    
                    if improvement > 0:
                        self.logger.debug(f"Vehicle {vehicle_id}: improved by {improvement:.2f}")
                else:
                    improved_routes[vehicle_id] = route
            
            # Calculate final metrics
            final_metrics = self._calculate_vrp_metrics(improved_routes)
            final_distance = final_metrics.get("total_distance", 0.0)
            
            runtime = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Add 2-opt specific metrics
            final_metrics.update({
                'initial_distance': initial_distance,
                'final_distance': final_distance,
                'total_improvement': initial_distance - final_distance,
                'improvement_percentage': ((initial_distance - final_distance) / initial_distance * 100) if initial_distance > 0 else 0,
                'two_opt_stats': two_opt.improvement_stats
            })
            
            self.logger.info(f"2-opt Local Search: initial={initial_distance:.2f}, "
                           f"final={final_distance:.2f}, improvement={initial_distance - final_distance:.2f} "
                           f"({final_metrics['improvement_percentage']:.1f}%)")
            
            return VRPResult(
                status="optimal",
                objective_value=-final_distance,  # Negative for maximization format
                routes=improved_routes,
                metrics=final_metrics,
                runtime=runtime
            )
            
        except Exception as e:
            runtime = (time.time() - start_time) * 1000
            self.logger.error(f"Error in 2-opt Local Search optimization: {str(e)}")
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": str(e)},
                runtime=runtime
            )

    def optimize_with_hybrid_construction_and_improvement(self) -> VRPResult:
        """
        Hybrid method: Best initial solution from multiple heuristics + 2-opt improvement.
        
        This method combines the InitialSolutionGenerator (multiple heuristics) with
        2-opt local search for a comprehensive approach.
        
        Returns:
            VRPResult: Optimization result from hybrid approach
        """
        start_time = time.time()
        
        if not NEW_METHODS_AVAILABLE:
            self.logger.warning("Hybrid method not available - missing new implementation methods")
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": "New implementation methods not available"},
                runtime=0.0
            )
        
        try:
            # Step 1: Get best initial solution from multiple heuristics
            initial_result = self.optimize_with_initial_solution_generator()
            
            if initial_result.status != "optimal":
                self.logger.warning("Could not generate initial solution for hybrid method")
                return initial_result
            
            initial_routes = initial_result.routes
            initial_distance = initial_result.metrics.get("total_distance", 0.0)
            best_initial_method = initial_result.metrics.get("best_method", "unknown")
            
            self.logger.info(f"Hybrid: Best initial method '{best_initial_method}' with distance {initial_distance:.2f}")
            
            # Step 2: Apply 2-opt improvement
            distance_matrix = self._create_distance_matrix()
            depot_nodes = [vehicle_data.get('depot', 'depot') for vehicle_data in self.instance.vehicles.values()]
            depot_nodes = list(set(depot_nodes))
            
            two_opt = TwoOptLocalSearch(distance_matrix, depot_nodes)
            location_to_index = {loc: idx for idx, loc in enumerate(self.instance.location_ids)}
            
            # Apply 2-opt to each route
            improved_routes = {}
            total_improvement = 0.0
            routes_improved_by_2opt = 0
            
            for vehicle_id, route in initial_routes.items():
                if len(route) > 3:  # Only apply 2-opt to routes with more than 3 nodes
                    improved_route, improvement = two_opt.improve_route(
                        route, location_to_index, strategy='best_improvement', max_iterations=200
                    )
                    
                    improved_routes[vehicle_id] = improved_route
                    total_improvement += improvement
                    
                    if improvement > 0:
                        routes_improved_by_2opt += 1
                        self.logger.debug(f"Vehicle {vehicle_id}: 2-opt improved by {improvement:.2f}")
                else:
                    improved_routes[vehicle_id] = route
            
            # Calculate final metrics
            final_metrics = self._calculate_vrp_metrics(improved_routes)
            final_distance = final_metrics.get("total_distance", 0.0)
            
            runtime = (time.time() - start_time) * 1000
            
            # Combine metrics from both phases
            final_metrics.update({
                'initial_method': best_initial_method,
                'initial_distance': initial_distance,
                'final_distance': final_distance,
                'total_improvement': initial_distance - final_distance,
                'improvement_percentage': ((initial_distance - final_distance) / initial_distance * 100) if initial_distance > 0 else 0,
                'routes_improved_by_2opt': routes_improved_by_2opt,
                'two_opt_stats': two_opt.improvement_stats,
                'method_comparison': initial_result.metrics.get('method_comparison', {})
            })
            
            self.logger.info(f"Hybrid Construction+2-opt: {best_initial_method} -> 2-opt: "
                           f"{initial_distance:.2f} -> {final_distance:.2f} "
                           f"(improvement: {initial_distance - final_distance:.2f}, "
                           f"{final_metrics['improvement_percentage']:.1f}%)")
            
            return VRPResult(
                status="optimal",
                objective_value=-final_distance,
                routes=improved_routes,
                metrics=final_metrics,
                runtime=runtime
            )
            
        except Exception as e:
            runtime = (time.time() - start_time) * 1000
            self.logger.error(f"Error in Hybrid Construction+2-opt optimization: {str(e)}")
            return VRPResult(                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": str(e)},
                runtime=runtime
            )
    
    def _check_driver_regulations(self, routes: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Check and enforce driver regulations for heavy trucks (24-ton vehicles).
        
        Heavy truck regulations:
        - Maximum 9 hours total work time per day
        - Maximum 4.5 hours continuous driving time
        - Mandatory 45-minute break after 4.5 hours driving (can be split: 15 + 30 min)
        - Break time not counted in driving time or service time
        
        Args:
            routes: Dictionary of vehicle routes
            
        Returns:
            Updated routes with break locations inserted where needed
        """
        updated_routes = {}
        
        for vehicle_id, route in routes.items():
            if vehicle_id not in self.instance.vehicles:
                updated_routes[vehicle_id] = route
                continue
                
            vehicle = self.instance.vehicles[vehicle_id]
            
            # Only check heavy trucks (24-ton vehicles)
            if not hasattr(vehicle, 'vehicle_type') or vehicle.vehicle_type != 'heavy':
                updated_routes[vehicle_id] = route
                continue
            
            # Check if route needs break enforcement
            if len(route) <= 2:  # Just depot start/end
                updated_routes[vehicle_id] = route
                continue
            
            updated_route = self._optimize_route_with_breaks(route, vehicle)
            updated_routes[vehicle_id] = updated_route
            
        return updated_routes
    
    def _optimize_route_with_breaks(self, route: List[str], vehicle) -> List[str]:
        """
        Optimize a single route to comply with driver break regulations.
        
        For heavy trucks, insert break locations after 4.5 hours of driving.
        Break locations can be:
        1. Service areas along the route (preferred)
        2. Depot returns for break (fallback)
        
        Args:
            route: Original route for the vehicle
            vehicle: Vehicle object with regulations
            
        Returns:
            Optimized route with break locations inserted
        """
        if not hasattr(vehicle, 'max_driving_time') or vehicle.max_driving_time is None:
            return route
            
        max_driving_minutes = vehicle.max_driving_time  # 270 minutes (4.5 hours)
        required_break_minutes = vehicle.required_break_time  # 45 minutes
        
        # Track driving time
        current_driving_time = 0.0
        new_route = [route[0]]  # Start with depot
        
        for i in range(1, len(route)):
            prev_location = route[i-1]
            current_location = route[i]
            
            # Calculate travel time between locations (assume 50 km/h average speed)
            try:
                distance = self.instance.get_distance(prev_location, current_location)
                travel_time = distance * 1.2  # minutes (assuming 50 km/h = 0.83 km/min)
            except:
                travel_time = 30  # fallback: 30 minutes between locations
            
            # Check if adding this travel would exceed driving time limit
            if current_driving_time + travel_time > max_driving_minutes:
                # Insert break location
                break_location = self._find_break_location(prev_location, current_location)
                if break_location and break_location not in new_route:
                    new_route.append(break_location)
                    self.logger.info(f"Inserted break location {break_location} for vehicle {vehicle.id}")
                
                # Reset driving time after break
                current_driving_time = 0.0
            
            # Add the current location
            new_route.append(current_location)
            current_driving_time += travel_time
              # Add service time (loading/unloading doesn't count toward driving time)
            if current_location in self.instance.locations:
                location = self.instance.locations[current_location]
                if hasattr(location, 'service_time') and location.service_time:
                    # Service time doesn't count toward driving time limit
                    pass
        
        return new_route
    
    def _find_break_location(self, from_location: str, to_location: str) -> Optional[str]:
        """
        Find an appropriate break location between two route points.
        
        Preference order:
        1. Service areas along the route (if available)
        2. Return to depot for break (fallback)
        
        Args:
            from_location: Starting location
            to_location: Destination location
            
        Returns:
            Break location ID or None if no suitable location found
        """
        # Try to find a service area along the route
        try:
            # Import service areas database if available
            from service_areas_db import service_areas_db
            
            # Get coordinates of from/to locations
            from_coords = self._get_location_coords(from_location)
            to_coords = self._get_location_coords(to_location)
            
            if from_coords and to_coords:
                # Find the best service area for a break between these points
                best_service_area = service_areas_db.find_break_location_between_points(
                    from_coords[0], from_coords[1], to_coords[0], to_coords[1]
                )
                
                if best_service_area:
                    # Check if this service area is already in our instance locations
                    if best_service_area.id in self.instance.locations:
                        return best_service_area.id
                    else:
                        # Add the service area to our instance for this route
                        from service_areas_db import service_area_to_location
                        self.instance.locations[best_service_area.id] = service_area_to_location(best_service_area)
                        return best_service_area.id
                    
        except ImportError:
            # Service areas database not available
            self.logger.warning("Service areas database not available for break planning")
        except Exception as e:
            self.logger.warning(f"Error finding service area for break: {e}")
        
        # Fallback: suggest depot return for break (simplified)
        # In a real implementation, this would return to the nearest depot
        if from_location.startswith('depot'):
            return from_location
        
        # Look for any depot in the instance
        for location_id in self.instance.location_ids:
            if location_id.startswith('depot'):
                return location_id
        
        return None
    
    def _get_location_coords(self, location_id: str) -> Optional[Tuple[float, float]]:
        """Get latitude and longitude coordinates for a location."""
        if location_id not in self.instance.locations:
            return None
            
        location = self.instance.locations[location_id]
        
        # Try to get real GPS coordinates
        if (hasattr(location, 'lat') and hasattr(location, 'lon') and 
            location.lat is not None and location.lon is not None):
            return (float(location.lat), float(location.lon))
        
        # Fallback to synthetic coordinates
        if hasattr(location, 'x') and hasattr(location, 'y'):
            return (float(location.x), float(location.y))
        
        return None
