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

# Try to import pyVRP for classical benchmarking
try:
    from pyvrp import Model, ProblemData, Client, Depot, VehicleType
    from pyvrp.stop import MaxRuntime
    from pyvrp.plotting import plot_result
    import matplotlib.pyplot as plt
    PYVRP_AVAILABLE = True
    print("pyVRP successfully imported for classical benchmarking")
except ImportError:
    PYVRP_AVAILABLE = False
    print("Warning: pyVRP not available. Install with 'pip install pyvrp' for classical benchmarking")

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
                return routes
            
            routes = run_with_timeout(_optimize_internal, timeout_duration=15)
            
            runtime = time.time() - start_time
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
            runtime = time.time() - start_time
            return VRPResult(
                status="timeout",
                objective_value=0.0,
                routes={},
                metrics={"total_distance": 0.0, "vehicles_used": 0},
                runtime=runtime
            )
        except Exception as e:
            self.logger.error(f"Error in quantum optimization: {str(e)}")
            runtime = time.time() - start_time
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
            
            # Simple capacity check
            current_passengers = sum(1 for loc in routes[vehicle_id] if loc.startswith("pickup")) - \
                               sum(1 for loc in routes[vehicle_id] if loc.startswith("dropoff"))
            
            if current_passengers + request.passengers <= self.instance.vehicles[vehicle_id].capacity:
                routes[vehicle_id].append(request.pickup_location)
                routes[vehicle_id].append(request.dropoff_location)
            else:
                # Try next available vehicle
                for alt_vehicle_id in vehicle_ids:
                    alt_passengers = sum(1 for loc in routes[alt_vehicle_id] if loc.startswith("pickup")) - \
                                   sum(1 for loc in routes[alt_vehicle_id] if loc.startswith("dropoff"))
                    if alt_passengers + request.passengers <= self.instance.vehicles[alt_vehicle_id].capacity:
                        routes[alt_vehicle_id].append(request.pickup_location)
                        routes[alt_vehicle_id].append(request.dropoff_location)
                        break
        
        return routes

    def optimize_with_pyvrp_classical(self) -> VRPResult:
        """Optimize VRP using pyVRP - exactly matching debug scenario approach."""
        if not PYVRP_AVAILABLE:
            self.logger.error("pyVRP not available for classical benchmarking")
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"total_distance": 0.0, "vehicles_used": 0, "error": "pyVRP not installed"},
                runtime=0.0
            )
        
        self.logger.info("Starting VRP optimization with pyVRP classical solver")
        start_time = time.time()
        
        try:
            # Use the direct Model API approach exactly like debug scenario
            m = Model()
            
            # Get vehicle capacity from the first vehicle
            vehicle_capacity = list(self.instance.vehicles.values())[0].capacity
            
            # Add vehicle type
            m.add_vehicle_type(len(self.instance.vehicles), capacity=vehicle_capacity)
            
            # Add depot
            depot_locations = [loc_id for loc_id in self.instance.location_ids if loc_id.startswith("depot")]
            depot_id = depot_locations[0] if depot_locations else self.instance.location_ids[0]
            depot_location = self.instance.locations[depot_id]
            depot = m.add_depot(x=int(depot_location.x), y=int(depot_location.y))
            
            # Add clients (non-depot locations)
            client_locations = [loc_id for loc_id in self.instance.location_ids if not loc_id.startswith("depot")]
            clients = []
            for loc_id in client_locations:
                location = self.instance.locations[loc_id]
                demand = getattr(location, 'demand', 1)
                client = m.add_client(
                    x=int(location.x),
                    y=int(location.y),
                    delivery=demand if demand > 0 else 0
                )
                clients.append(client)
            
            # Add edges with Manhattan distance
            for frm in m.locations:
                for to in m.locations:
                    distance = abs(frm.x - to.x) + abs(frm.y - to.y)  # Manhattan
                    m.add_edge(frm, to, distance=distance)
            
            # Solve with the same parameters as debug scenario
            result = m.solve(stop=MaxRuntime(10), seed=42, display=False)
            
            runtime = time.time() - start_time
            
            # Check if solution is feasible
            if not result.is_feasible():
                self.logger.warning("pyVRP found infeasible solution")
                return VRPResult(
                    status="infeasible",
                    objective_value=0.0,
                    routes={},
                    metrics={"total_distance": 0.0, "vehicles_used": 0, "error": "infeasible solution"},
                    runtime=runtime
                )
            
            # Extract objective value
            pyvrp_objective = result.cost()
            pyvrp_num_routes = len([route for route in result.best.routes() if len(route) > 0])
            
            # Convert to our format
            routes = self._convert_model_solution_to_routes(result, client_locations, depot_id)
            metrics = self._calculate_vrp_metrics(routes)
            
            # Add pyVRP specific metrics
            metrics.update({
                'pyvrp_cost': pyvrp_objective,
                'pyvrp_num_routes': pyvrp_num_routes,
                'pyvrp_runtime': runtime,
                'pyvrp_distance': pyvrp_objective,
            })
            
            obj_value = -pyvrp_objective
            
            self.logger.info(f"pyVRP solution: cost={pyvrp_objective}, routes={pyvrp_num_routes}, runtime={runtime:.2f}s")
            
            return VRPResult(
                status="optimal",
                objective_value=obj_value,
                routes=routes,
                metrics=metrics,
                runtime=runtime
            )
            
        except Exception as e:
            self.logger.error(f"Error in pyVRP optimization: {str(e)}")
            runtime = time.time() - start_time
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"total_distance": 0.0, "vehicles_used": 0, "error": str(e)},
                runtime=runtime
            )

    def _convert_model_solution_to_routes(self, result, client_locations, depot_id) -> Dict[str, List[str]]:
        """Convert pyVRP solution from Model API back to our route format."""
        routes = {}
        vehicle_ids = list(self.instance.vehicles.keys())
        
        try:
            solution_routes = list(result.best.routes())
            self.logger.info(f"Converting {len(solution_routes)} routes from pyVRP solution")
            
            for route_idx, route in enumerate(solution_routes):
                if route_idx < len(vehicle_ids):
                    vehicle_id = vehicle_ids[route_idx]
                    route_locations = [depot_id]  # Start at depot
                    
                    for client_idx in route:
                        client_idx_0based = client_idx - 1  # Convert from 1-based to 0-based
                        if 0 <= client_idx_0based < len(client_locations):
                            client_location = client_locations[client_idx_0based]
                            route_locations.append(client_location)
                    
                    route_locations.append(depot_id)  # Return to depot
                    routes[vehicle_id] = route_locations
            
            # Add empty routes for unused vehicles
            for vehicle_id in vehicle_ids:
                if vehicle_id not in routes:
                    routes[vehicle_id] = [depot_id]
                    
        except Exception as e:
            self.logger.error(f"Error converting pyVRP solution: {str(e)}")
            for vehicle_id in vehicle_ids:
                routes[vehicle_id] = [depot_id]
        
        return routes

    def optimize_with_classical_benders(self) -> VRPResult:
        """Optimize VRP using classical greedy approach."""
        self.logger.info("Starting VRP optimization with classical methods")
        start_time = time.time()
        
        try:
            def _optimize_classical():
                return self._greedy_vrp_solution()
            
            routes = run_with_timeout(_optimize_classical, timeout_duration=15)
            metrics = self._calculate_vrp_metrics(routes)
            
            runtime = time.time() - start_time
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
            runtime = time.time() - start_time
            return VRPResult(
                status="timeout",
                objective_value=0.0,
                routes={},
                metrics={"total_distance": 0.0, "vehicles_used": 0},
                runtime=runtime
            )
        except Exception as e:
            self.logger.error(f"Error in classical optimization: {str(e)}")
            runtime = time.time() - start_time
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
                        unvisited.remove(nearest_customer)
                
                # Return to depot if visited any customers
                if len(routes[vehicle_id]) > 1:
                    routes[vehicle_id].append(vehicle.depot_id)
        
        return routes
    
    def plot_pyvrp_solution(self, save_path: Optional[str] = None) -> bool:
        """Plot the pyVRP solution using its native plotting function."""
        if not PYVRP_AVAILABLE:
            return False
        
        try:
            # Create ProblemData object for plotting
            depot_locations = [loc_id for loc_id in self.instance.location_ids if loc_id.startswith("depot")]
            depot_id = depot_locations[0] if depot_locations else self.instance.location_ids[0]
            depot_location = self.instance.locations[depot_id]
            
            # Create depot
            depot = Depot(x=int(depot_location.x), y=int(depot_location.y))
            depots = [depot]
            
            # Create clients
            client_locations = [loc_id for loc_id in self.instance.location_ids if not loc_id.startswith("depot")]
            clients = []
            for loc_id in client_locations:
                location = self.instance.locations[loc_id]
                demand = getattr(location, 'demand', 1)
                client = Client(
                    x=int(location.x),
                    y=int(location.y),
                    delivery=[demand if demand > 0 else 0],
                    pickup=[0],
                    service_duration=0,
                    tw_early=0,
                    tw_late=1000
                )
                clients.append(client)
            
            # Create distance matrix
            n_clients = len(clients)
            distance_matrix = np.zeros((n_clients + 1, n_clients + 1), dtype=int)
            
            all_locations = [depot] + clients
            
            for i in range(len(all_locations)):
                for j in range(len(all_locations)):
                    if i != j:
                        loc1 = all_locations[i]
                        loc2 = all_locations[j]
                        manhattan_dist = abs(loc1.x - loc2.x) + abs(loc1.y - loc2.y)
                        distance_matrix[i, j] = manhattan_dist
                    else:
                        distance_matrix[i, j] = 0
            
            # Create vehicle type
            vehicle_capacity = list(self.instance.vehicles.values())[0].capacity
            vehicle_type = VehicleType(
                num_available=len(self.instance.vehicles),
                capacity=[vehicle_capacity, vehicle_capacity],  # Two dimensions
                start_depot=0,
                end_depot=0,
                fixed_cost=0,
                unit_distance_cost=1
            )
            vehicle_types = [vehicle_type]
            
            # Create ProblemData
            problem_data = ProblemData(
                clients=clients,
                depots=depots,
                vehicle_types=vehicle_types,
                distance_matrices=[distance_matrix],
                duration_matrices=[distance_matrix]
            )
            
            # Solve and plot
            model = Model.from_data(problem_data)
            result = model.solve(stop=MaxRuntime(10), seed=42, display=False)
            
            if not result.is_feasible():
                self.logger.warning("pyVRP found infeasible solution for plotting")
                return False
            
            plot_result(result, problem_data)
            plt.title(f'pyVRP Solution - {self.instance.name}\nCost: {result.cost()}')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"pyVRP plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Error plotting pyVRP solution: {str(e)}")
            return False
    
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
