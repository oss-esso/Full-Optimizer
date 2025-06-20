#!/usr/bin/env python3
"""
Advanced Initial Solution Generator for VRP
Integrates multiple construction heuristics into the main VRP optimizer
"""
import logging
from typing import List, Dict, Optional
from initial_solution_generator import InitialSolutionGenerator, Customer, Vehicle, Route
from vrp_data_models import VRPResult

class AdvancedInitialSolutionOptimizer:
    """
    Advanced optimizer using multiple initial solution construction heuristics.
    
    Implements recent literature approaches:
    - Firefly Algorithm (FA) initialization for CVRP
    - Large Neighbourhood Search principles
    - Multiple construction heuristics with diversity
    - Greedy insertion with demand-distance ratios
    """
    
    def __init__(self, instance, objective):
        """Initialize with VRP instance and objective."""
        self.instance = instance
        self.objective = objective
        self.logger = logging.getLogger(__name__)
        
        # Convert instance to generator format
        self.customers, self.vehicles, self.depot_location = self._convert_instance()
        
        if self.customers:
            self.generator = InitialSolutionGenerator(
                self.customers, self.vehicles, self.depot_location
            )
        else:
            self.generator = None
    
    def _convert_instance(self):
        """Convert VRP instance to InitialSolutionGenerator format."""
        customers = []
        
        # Handle both standard VRP and ride pooling scenarios
        if hasattr(self.instance, 'ride_requests') and self.instance.ride_requests:
            # For ride pooling, treat pickup locations as customers
            for request in self.instance.ride_requests:
                pickup_loc = self.instance.locations[request.pickup_location]
                customer = Customer(
                    id=request.pickup_location,
                    x=pickup_loc.lat if hasattr(pickup_loc, 'lat') else hash(request.pickup_location) % 100,
                    y=pickup_loc.lon if hasattr(pickup_loc, 'lon') else hash(request.pickup_location) % 100,
                    demand=request.passengers
                )
                customers.append(customer)
        else:
            # Standard VRP: use customer locations
            depot_locations = [loc_id for loc_id in self.instance.location_ids if loc_id.startswith("depot")]
            customer_locations = [loc_id for loc_id in self.instance.location_ids if not loc_id.startswith("depot")]
            
            for loc_id in customer_locations:
                location = self.instance.locations[loc_id]
                customer = Customer(
                    id=loc_id,
                    x=location.lat if hasattr(location, 'lat') else hash(loc_id) % 100,
                    y=location.lon if hasattr(location, 'lon') else hash(loc_id) % 100,
                    demand=getattr(location, 'demand', 1)
                )
                customers.append(customer)
        
        # Convert vehicles
        vehicles = []
        for vehicle_id, vehicle in self.instance.vehicles.items():
            vehicles.append(Vehicle(
                id=vehicle_id,
                capacity=getattr(vehicle, 'capacity', 100)
            ))
        
        # Get depot location
        depot_locations = [loc_id for loc_id in self.instance.location_ids if loc_id.startswith("depot")]
        if depot_locations:
            depot_loc = self.instance.locations[depot_locations[0]]
            depot_x = depot_loc.lat if hasattr(depot_loc, 'lat') else 0.0
            depot_y = depot_loc.lon if hasattr(depot_loc, 'lon') else 0.0
        else:
            depot_x, depot_y = 0.0, 0.0
        
        return customers, vehicles, (depot_x, depot_y)
    
    def _convert_routes_to_vrp_format(self, routes: List[Route]) -> Dict[str, List[str]]:
        """Convert generator routes to VRP format."""
        vrp_routes = {}
        
        # Add depot information
        depot_locations = [loc_id for loc_id in self.instance.location_ids if loc_id.startswith("depot")]
        depot_id = depot_locations[0] if depot_locations else "depot"
        
        for route in routes:
            vrp_route = [depot_id]  # Start at depot
            vrp_route.extend(route.customers)  # Add customers
            vrp_route.append(depot_id)  # Return to depot
            vrp_routes[route.vehicle_id] = vrp_route
        
        return vrp_routes
    
    def optimize_nearest_neighbor(self, perturbation: float = 0.1) -> VRPResult:
        """Optimize using Nearest Neighbor construction heuristic."""
        import time
        start_time = time.time()
        
        if not self.generator:
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": "No customers to optimize"},
                runtime=0.0
            )
        
        try:
            routes = self.generator.generate_nearest_neighbor_solution(perturbation)
            runtime = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            if not routes:
                return VRPResult(
                    status="infeasible",
                    objective_value=0.0,
                    routes={},
                    metrics={"total_distance": 0.0, "vehicles_used": 0},
                    runtime=runtime
                )
            
            vrp_routes = self._convert_routes_to_vrp_format(routes)
            total_distance = sum(r.total_distance for r in routes)
            
            metrics = {
                "total_distance": total_distance,
                "vehicles_used": len(routes),
                "customers_served": sum(len(r.customers) for r in routes),
                "avg_capacity_utilization": sum(r.total_demand for r in routes) / sum(v.capacity for v in self.vehicles[:len(routes)]) if routes else 0,
                "method": "Nearest Neighbor",
                "perturbation": perturbation
            }
            
            return VRPResult(
                status="optimal",
                objective_value=total_distance,
                routes=vrp_routes,
                metrics=metrics,
                runtime=runtime
            )
            
        except Exception as e:
            runtime = (time.time() - start_time) * 1000
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": str(e), "method": "Nearest Neighbor"},
                runtime=runtime
            )
    
    def optimize_savings_algorithm(self, perturbation: float = 0.12) -> VRPResult:
        """Optimize using Clarke-Wright Savings Algorithm."""
        import time
        start_time = time.time()
        
        if not self.generator:
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": "No customers to optimize"},
                runtime=0.0
            )
        
        try:
            routes = self.generator.generate_savings_algorithm_solution(perturbation)
            runtime = (time.time() - start_time) * 1000
            
            if not routes:
                return VRPResult(
                    status="infeasible",
                    objective_value=0.0,
                    routes={},
                    metrics={"total_distance": 0.0, "vehicles_used": 0},
                    runtime=runtime
                )
            
            vrp_routes = self._convert_routes_to_vrp_format(routes)
            total_distance = sum(r.total_distance for r in routes)
            
            metrics = {
                "total_distance": total_distance,
                "vehicles_used": len(routes),
                "customers_served": sum(len(r.customers) for r in routes),
                "avg_capacity_utilization": sum(r.total_demand for r in routes) / sum(v.capacity for v in self.vehicles[:len(routes)]) if routes else 0,
                "method": "Savings Algorithm",
                "perturbation": perturbation
            }
            
            return VRPResult(
                status="optimal",
                objective_value=total_distance,
                routes=vrp_routes,
                metrics=metrics,
                runtime=runtime
            )
            
        except Exception as e:
            runtime = (time.time() - start_time) * 1000
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": str(e), "method": "Savings Algorithm"},
                runtime=runtime
            )
    
    def optimize_firefly_algorithm(self, num_fireflies: int = 20, perturbation: float = 0.12) -> VRPResult:
        """Optimize using Firefly Algorithm (CVRP-FA)."""
        import time
        start_time = time.time()
        
        if not self.generator:
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": "No customers to optimize"},
                runtime=0.0
            )
        
        try:
            routes = self.generator.generate_firefly_algorithm_solution(num_fireflies, perturbation)
            runtime = (time.time() - start_time) * 1000
            
            if not routes:
                return VRPResult(
                    status="infeasible",
                    objective_value=0.0,
                    routes={},
                    metrics={"total_distance": 0.0, "vehicles_used": 0},
                    runtime=runtime
                )
            
            vrp_routes = self._convert_routes_to_vrp_format(routes)
            total_distance = sum(r.total_distance for r in routes)
            
            metrics = {
                "total_distance": total_distance,
                "vehicles_used": len(routes),
                "customers_served": sum(len(r.customers) for r in routes),
                "avg_capacity_utilization": sum(r.total_demand for r in routes) / sum(v.capacity for v in self.vehicles[:len(routes)]) if routes else 0,
                "method": "Firefly Algorithm",
                "num_fireflies": num_fireflies,
                "perturbation": perturbation
            }
            
            return VRPResult(
                status="optimal",
                objective_value=total_distance,
                routes=vrp_routes,
                metrics=metrics,
                runtime=runtime
            )
            
        except Exception as e:
            runtime = (time.time() - start_time) * 1000
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": str(e), "method": "Firefly Algorithm"},
                runtime=runtime
            )
    
    def optimize_greedy_insertion(self, demand_priority: bool = True, perturbation: float = 0.1) -> VRPResult:
        """Optimize using Greedy Insertion with demand-distance priority."""
        import time
        start_time = time.time()
        
        if not self.generator:
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": "No customers to optimize"},
                runtime=0.0
            )
        
        try:
            routes = self.generator.generate_greedy_insertion_solution(demand_priority, perturbation)
            runtime = (time.time() - start_time) * 1000
            
            if not routes:
                return VRPResult(
                    status="infeasible",
                    objective_value=0.0,
                    routes={},
                    metrics={"total_distance": 0.0, "vehicles_used": 0},
                    runtime=runtime
                )
            
            vrp_routes = self._convert_routes_to_vrp_format(routes)
            total_distance = sum(r.total_distance for r in routes)
            
            metrics = {
                "total_distance": total_distance,
                "vehicles_used": len(routes),
                "customers_served": sum(len(r.customers) for r in routes),
                "avg_capacity_utilization": sum(r.total_demand for r in routes) / sum(v.capacity for v in self.vehicles[:len(routes)]) if routes else 0,
                "method": "Greedy Insertion",
                "demand_priority": demand_priority,
                "perturbation": perturbation
            }
            
            return VRPResult(
                status="optimal",
                objective_value=total_distance,
                routes=vrp_routes,
                metrics=metrics,
                runtime=runtime
            )
            
        except Exception as e:
            runtime = (time.time() - start_time) * 1000
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": str(e), "method": "Greedy Insertion"},
                runtime=runtime
            )
    
    def optimize_best_of_multiple(self, num_solutions: int = 5) -> VRPResult:
        """Generate multiple diverse solutions and return the best one."""
        import time
        start_time = time.time()
        
        if not self.generator:
            return VRPResult(
                status="error",
                objective_value=0.0,
                routes={},
                metrics={"error": "No customers to optimize"},
                runtime=0.0
            )
        
        self.logger.info(f"Generating {num_solutions} diverse solutions to find the best")
        
        best_result = None
        best_distance = float('inf')
        all_results = []
        
        # Try different methods
        methods = [
            ("nearest_neighbor", lambda: self.optimize_nearest_neighbor(0.1)),
            ("savings_algorithm", lambda: self.optimize_savings_algorithm(0.12)),
            ("firefly_algorithm", lambda: self.optimize_firefly_algorithm(20, 0.12)),
            ("greedy_insertion", lambda: self.optimize_greedy_insertion(True, 0.1)),
            ("firefly_large", lambda: self.optimize_firefly_algorithm(30, 0.15)),
        ]
        
        for i, (method_name, method_func) in enumerate(methods[:num_solutions]):
            try:
                result = method_func()
                all_results.append((method_name, result))
                
                if result.status == "optimal" and result.objective_value < best_distance:
                    best_distance = result.objective_value
                    best_result = result
                    best_result.metrics["best_method"] = method_name
                    
            except Exception as e:
                self.logger.warning(f"Method {method_name} failed: {str(e)}")
        
        runtime = (time.time() - start_time) * 1000
        
        if best_result:
            best_result.runtime = runtime
            best_result.metrics["total_methods_tried"] = len(all_results)
            best_result.metrics["method_comparison"] = {
                name: result.objective_value for name, result in all_results 
                if result.status == "optimal"
            }
            
            self.logger.info(f"Best method: {best_result.metrics.get('best_method', 'unknown')} "
                           f"with distance: {best_result.objective_value:.2f}")
            
            return best_result
        else:
            return VRPResult(
                status="infeasible",
                objective_value=0.0,
                routes={},
                metrics={"error": "No feasible solution found", "methods_tried": len(all_results)},
                runtime=runtime
            )
