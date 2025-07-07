#!/usr/bin/env python3
"""
Initial Solution Generator for Capacitated Vehicle Routing Problem (CVRP)
Implements multiple construction heuristics including firefly algorithm initialization.

Based on recent literature combining:
- Firefly Algorithm (FA) for CVRP optimization
- Large Neighbourhood Search (LNS) principles
- Greedy construction heuristics
"""

import numpy as np
import random
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

@dataclass
class Customer:
    """Represents a customer with location and demand."""
    id: str
    x: float
    y: float
    demand: float

@dataclass
class Vehicle:
    """Represents a vehicle with capacity and depot."""
    id: str
    capacity: float
    depot_x: float = 0.0
    depot_y: float = 0.0

@dataclass
class Route:
    """Represents a route with customers and total metrics."""
    vehicle_id: str
    customers: List[str]
    total_distance: float
    total_demand: float
    total_time: float = 0.0

class Firefly:
    """Represents a firefly in the firefly algorithm for CVRP."""
    
    def __init__(self, num_customers: int, num_vehicles: int):
        """Initialize firefly with random position vector."""
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        # Position vector: each element represents customer-to-vehicle assignment
        self.position = np.random.rand(num_customers)
        self.brightness = float('inf')  # Will be calculated as objective value
        self.routes = []
        self.total_distance = float('inf')
        self.feasible = False
    
    def decode_to_routes(self, customers: List[Customer], vehicles: List[Vehicle]) -> List[Route]:
        """Decode firefly position to actual routes using discrete mapping."""
        # Map continuous position values to discrete vehicle assignments
        vehicle_assignments = []
        for i, pos_val in enumerate(self.position):
            # Map position value [0,1] to vehicle index
            vehicle_idx = int(pos_val * len(vehicles)) % len(vehicles)
            vehicle_assignments.append(vehicle_idx)
        
        # Group customers by assigned vehicle
        vehicle_customer_groups = [[] for _ in vehicles]
        for customer_idx, vehicle_idx in enumerate(vehicle_assignments):
            vehicle_customer_groups[vehicle_idx].append(customers[customer_idx])
        
        routes = []
        for vehicle_idx, vehicle in enumerate(vehicles):
            if vehicle_customer_groups[vehicle_idx]:
                # Create route for this vehicle
                route_customers = vehicle_customer_groups[vehicle_idx]
                # Sort customers by nearest neighbor within the route
                sorted_customers = self._sort_by_nearest_neighbor(
                    route_customers, vehicle.depot_x, vehicle.depot_y
                )
                
                # Calculate route metrics
                total_distance = self._calculate_route_distance(
                    sorted_customers, vehicle.depot_x, vehicle.depot_y
                )
                total_demand = sum(c.demand for c in sorted_customers)
                
                route = Route(
                    vehicle_id=vehicle.id,
                    customers=[c.id for c in sorted_customers],
                    total_distance=total_distance,
                    total_demand=total_demand
                )
                routes.append(route)
        
        self.routes = routes
        self.total_distance = sum(r.total_distance for r in routes)
        self.brightness = 1.0 / (1.0 + self.total_distance)  # Higher brightness = better solution        # Check feasibility (capacity constraints)
        self.feasible = True
        for i, route in enumerate(routes):
            if route.customers and i < len(vehicles):
                if route.total_demand > vehicles[i].capacity:
                    self.feasible = False
                    break
        
        return routes
    
    def _sort_by_nearest_neighbor(self, customers: List[Customer], depot_x: float, depot_y: float) -> List[Customer]:
        """Sort customers in a route using nearest neighbor heuristic."""
        if not customers:
            return []
        
        sorted_customers = []
        remaining = customers.copy()
        current_x, current_y = depot_x, depot_y
        
        while remaining:
            # Find nearest customer
            nearest_idx = 0
            min_distance = self._euclidean_distance(
                current_x, current_y, remaining[0].x, remaining[0].y
            )
            
            for i, customer in enumerate(remaining[1:], 1):
                distance = self._euclidean_distance(
                    current_x, current_y, customer.x, customer.y
                )
                if distance < min_distance:
                    min_distance = distance
                    nearest_idx = i
            
            # Add nearest customer to route
            nearest_customer = remaining.pop(nearest_idx)
            sorted_customers.append(nearest_customer)
            current_x, current_y = nearest_customer.x, nearest_customer.y
        
        return sorted_customers
    
    def _calculate_route_distance(self, customers: List[Customer], depot_x: float, depot_y: float) -> float:
        """Calculate total distance for a route including depot returns."""
        if not customers:
            return 0.0
        
        total_distance = 0.0
        current_x, current_y = depot_x, depot_y
        
        # Distance from depot to first customer
        if customers:
            total_distance += self._euclidean_distance(
                current_x, current_y, customers[0].x, customers[0].y
            )
            current_x, current_y = customers[0].x, customers[0].y
        
        # Distance between customers
        for i in range(1, len(customers)):
            distance = self._euclidean_distance(
                current_x, current_y, customers[i].x, customers[i].y
            )
            total_distance += distance
            current_x, current_y = customers[i].x, customers[i].y
        
        # Distance back to depot
        total_distance += self._euclidean_distance(
            current_x, current_y, depot_x, depot_y
        )
        
        return total_distance
    
    def _euclidean_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

class InitialSolutionGenerator:
    """
    Generates initial feasible solutions for CVRP using multiple heuristics.
    
    Implements:
    - Nearest Neighbor construction
    - Savings Algorithm (Clarke-Wright)
    - Firefly Algorithm initialization
    - Large Neighbourhood Search principles
    - Random perturbation for diversity
    """
    
    def __init__(self, customers: List[Customer], vehicles: List[Vehicle], depot_location: Tuple[float, float]):
        """
        Initialize the solution generator.
        
        Args:
            customers: List of customers with locations and demands
            vehicles: List of vehicles with capacities
            depot_location: (x, y) coordinates of the depot
        """
        self.customers = customers
        self.vehicles = vehicles
        self.depot_x, self.depot_y = depot_location
        self.logger = logging.getLogger(__name__)
        
        # Update vehicle depot locations
        for vehicle in self.vehicles:
            vehicle.depot_x = self.depot_x
            vehicle.depot_y = self.depot_y
    
    def generate_nearest_neighbor_solution(self, perturbation_factor: float = 0.0) -> List[Route]:
        """
        Generate initial solution using nearest neighbor heuristic.
        
        Args:
            perturbation_factor: Percentage of customers to randomly perturb (0.1-0.15 recommended)
        
        Returns:
            List of routes for each vehicle
        """
        self.logger.info(f"Generating nearest neighbor solution with {perturbation_factor*100}% perturbation")
        
        routes = []
        remaining_customers = self.customers.copy()
          # Apply random perturbation
        if perturbation_factor > 0:
            num_perturb = max(1, int(len(remaining_customers) * perturbation_factor))
            num_perturb = min(num_perturb, len(remaining_customers))  # Ensure sample size <= population
            if num_perturb < len(remaining_customers):
                perturb_indices = random.sample(range(len(remaining_customers)), num_perturb)
                # Shuffle perturbed customers
                for _ in range(len(perturb_indices)):
                    if len(perturb_indices) >= 2:
                        i, j = random.sample(perturb_indices, 2)
                        remaining_customers[i], remaining_customers[j] = remaining_customers[j], remaining_customers[i]
        
        vehicle_idx = 0
        while remaining_customers and vehicle_idx < len(self.vehicles):
            vehicle = self.vehicles[vehicle_idx]
            route_customers = []
            current_capacity = 0
            current_x, current_y = self.depot_x, self.depot_y
            
            while remaining_customers:
                # Find nearest feasible customer
                best_customer = None
                best_idx = -1
                min_distance = float('inf')
                
                for i, customer in enumerate(remaining_customers):
                    if current_capacity + customer.demand <= vehicle.capacity:
                        distance = self._euclidean_distance(current_x, current_y, customer.x, customer.y)
                        if distance < min_distance:
                            min_distance = distance
                            best_customer = customer
                            best_idx = i
                
                if best_customer is None:
                    break  # No more feasible customers for this vehicle
                
                # Add customer to route
                route_customers.append(best_customer)
                current_capacity += best_customer.demand
                current_x, current_y = best_customer.x, best_customer.y
                remaining_customers.pop(best_idx)
            
            if route_customers:
                total_distance = self._calculate_route_distance(route_customers)
                total_demand = sum(c.demand for c in route_customers)
                
                route = Route(
                    vehicle_id=vehicle.id,
                    customers=[c.id for c in route_customers],
                    total_distance=total_distance,
                    total_demand=total_demand
                )
                routes.append(route)
            
            vehicle_idx += 1
        
        # Check if all customers are served
        if remaining_customers:
            self.logger.warning(f"Could not assign {len(remaining_customers)} customers due to capacity constraints")
        
        self.logger.info(f"Generated nearest neighbor solution with {len(routes)} routes")
        return routes
    
    def generate_savings_algorithm_solution(self, perturbation_factor: float = 0.0) -> List[Route]:
        """
        Generate initial solution using Clarke-Wright Savings Algorithm.
        
        Args:
            perturbation_factor: Percentage of savings to randomly perturb
        
        Returns:
            List of routes for each vehicle
        """
        self.logger.info(f"Generating savings algorithm solution with {perturbation_factor*100}% perturbation")
        
        # Calculate savings for all customer pairs
        savings = []
        for i in range(len(self.customers)):
            for j in range(i + 1, len(self.customers)):
                customer_i = self.customers[i]
                customer_j = self.customers[j]
                
                dist_depot_i = self._euclidean_distance(self.depot_x, self.depot_y, customer_i.x, customer_i.y)
                dist_depot_j = self._euclidean_distance(self.depot_x, self.depot_y, customer_j.x, customer_j.y)
                dist_i_j = self._euclidean_distance(customer_i.x, customer_i.y, customer_j.x, customer_j.y)
                
                saving = dist_depot_i + dist_depot_j - dist_i_j
                savings.append((saving, i, j))
        
        # Sort savings in descending order
        savings.sort(reverse=True)
          # Apply perturbation to savings list
        if perturbation_factor > 0:
            num_perturb = max(1, int(len(savings) * perturbation_factor))
            num_perturb = min(num_perturb, len(savings))  # Ensure sample size <= population
            if num_perturb < len(savings):
                perturb_indices = random.sample(range(len(savings)), num_perturb)
                perturbed_items = [savings[i] for i in perturb_indices]
                random.shuffle(perturbed_items)
                for i, item in enumerate(perturbed_items):
                    savings[perturb_indices[i]] = item
        
        # Initialize routes: each customer starts as a separate route
        routes = [[customer] for customer in self.customers]
        customer_to_route = {i: i for i in range(len(self.customers))}
        
        # Merge routes based on savings
        for saving, i, j in savings:
            route_i = customer_to_route[i]
            route_j = customer_to_route[j]
            
            if route_i != route_j:  # Customers are in different routes
                # Check if merging is feasible
                combined_demand = sum(c.demand for c in routes[route_i]) + sum(c.demand for c in routes[route_j])
                
                if combined_demand <= self.vehicles[0].capacity:  # Assuming homogeneous fleet
                    # Merge routes
                    if routes[route_i][-1] == self.customers[i] and routes[route_j][0] == self.customers[j]:
                        # Merge j's route to end of i's route
                        routes[route_i].extend(routes[route_j])
                        for customer_idx in [self.customers.index(c) for c in routes[route_j]]:
                            customer_to_route[customer_idx] = route_i
                        routes[route_j] = []
                    elif routes[route_i][0] == self.customers[i] and routes[route_j][-1] == self.customers[j]:
                        # Merge i's route to end of j's route
                        routes[route_j].extend(routes[route_i])
                        for customer_idx in [self.customers.index(c) for c in routes[route_i]]:
                            customer_to_route[customer_idx] = route_j
                        routes[route_i] = []
        
        # Convert to Route objects
        final_routes = []
        vehicle_idx = 0
        for route_customers in routes:
            if route_customers and vehicle_idx < len(self.vehicles):
                total_distance = self._calculate_route_distance(route_customers)
                total_demand = sum(c.demand for c in route_customers)
                
                route = Route(
                    vehicle_id=self.vehicles[vehicle_idx].id,
                    customers=[c.id for c in route_customers],
                    total_distance=total_distance,
                    total_demand=total_demand
                )
                final_routes.append(route)
                vehicle_idx += 1
        
        self.logger.info(f"Generated savings algorithm solution with {len(final_routes)} routes")
        return final_routes
    
    def generate_firefly_algorithm_solution(self, num_fireflies: int = 20, perturbation_factor: float = 0.12) -> List[Route]:
        """
        Generate initial solution using Firefly Algorithm (CVRP-FA).
        
        Args:
            num_fireflies: Number of fireflies in the population
            perturbation_factor: Random perturbation factor for diversity (10-15% recommended)
        
        Returns:
            Best routes found by firefly algorithm
        """
        self.logger.info(f"Generating firefly algorithm solution with {num_fireflies} fireflies and {perturbation_factor*100}% perturbation")
        
        # Initialize firefly population
        fireflies = []
        for _ in range(num_fireflies):
            firefly = Firefly(len(self.customers), len(self.vehicles))
            # Apply perturbation
            if random.random() < perturbation_factor:
                perturbation = np.random.normal(0, 0.1, len(self.customers))
                firefly.position = np.clip(firefly.position + perturbation, 0, 1)
            
            firefly.decode_to_routes(self.customers, self.vehicles)
            fireflies.append(firefly)
        
        # Select best feasible firefly or best infeasible one
        feasible_fireflies = [f for f in fireflies if f.feasible]
        if feasible_fireflies:
            best_firefly = min(feasible_fireflies, key=lambda f: f.total_distance)
        else:
            best_firefly = max(fireflies, key=lambda f: f.brightness)
        
        self.logger.info(f"Generated firefly algorithm solution with {len(best_firefly.routes)} routes, "
                        f"feasible: {best_firefly.feasible}, distance: {best_firefly.total_distance:.2f}")
        
        return best_firefly.routes
    
    def generate_greedy_insertion_solution(self, demand_distance_priority: bool = True, 
                                         perturbation_factor: float = 0.1) -> List[Route]:
        """
        Generate solution using greedy insertion with demand-to-distance ratio priority.
        
        Args:
            demand_distance_priority: If True, prioritize customers with high demand-to-distance ratio
            perturbation_factor: Percentage of customers to randomly prioritize
        
        Returns:
            List of routes generated by greedy insertion
        """
        self.logger.info(f"Generating greedy insertion solution with demand-distance priority: {demand_distance_priority}")
        
        # Calculate priority scores for customers
        customer_priorities = []
        for customer in self.customers:
            distance_to_depot = self._euclidean_distance(
                self.depot_x, self.depot_y, customer.x, customer.y
            )
            if demand_distance_priority:
                # Higher priority for high demand, low distance customers
                priority = customer.demand / (distance_to_depot + 1e-6)
            else:
                # Simple distance-based priority
                priority = 1.0 / (distance_to_depot + 1e-6)
            
            customer_priorities.append((priority, customer))
        
        # Sort by priority (descending)
        customer_priorities.sort(reverse=True)
          # Apply random perturbation
        if perturbation_factor > 0:
            num_perturb = max(1, int(len(customer_priorities) * perturbation_factor))
            num_perturb = min(num_perturb, len(customer_priorities))  # Ensure sample size <= population
            if num_perturb < len(customer_priorities):
                perturb_indices = random.sample(range(len(customer_priorities)), num_perturb)
                perturbed_items = [customer_priorities[i] for i in perturb_indices]
                random.shuffle(perturbed_items)
                for i, item in enumerate(perturbed_items):
                    customer_priorities[perturb_indices[i]] = item
        
        # Greedy insertion
        routes = []
        remaining_customers = [cp[1] for cp in customer_priorities]
        
        vehicle_idx = 0
        while remaining_customers and vehicle_idx < len(self.vehicles):
            vehicle = self.vehicles[vehicle_idx]
            route_customers = []
            current_capacity = 0
            
            i = 0
            while i < len(remaining_customers):
                customer = remaining_customers[i]
                if current_capacity + customer.demand <= vehicle.capacity:
                    route_customers.append(customer)
                    current_capacity += customer.demand
                    remaining_customers.pop(i)
                else:
                    i += 1
            
            if route_customers:
                # Optimize route order using nearest neighbor
                optimized_customers = self._optimize_route_order(route_customers)
                total_distance = self._calculate_route_distance(optimized_customers)
                total_demand = sum(c.demand for c in optimized_customers)
                
                route = Route(
                    vehicle_id=vehicle.id,
                    customers=[c.id for c in optimized_customers],
                    total_distance=total_distance,
                    total_demand=total_demand
                )
                routes.append(route)
            
            vehicle_idx += 1
        
        # Handle remaining customers if any
        if remaining_customers:
            self.logger.warning(f"Could not assign {len(remaining_customers)} customers due to capacity constraints")
        
        self.logger.info(f"Generated greedy insertion solution with {len(routes)} routes")
        return routes
    
    def _optimize_route_order(self, customers: List[Customer]) -> List[Customer]:
        """Optimize the order of customers in a route using nearest neighbor."""
        if len(customers) <= 1:
            return customers
        
        optimized = []
        remaining = customers.copy()
        current_x, current_y = self.depot_x, self.depot_y
        
        while remaining:
            nearest_idx = 0
            min_distance = self._euclidean_distance(current_x, current_y, remaining[0].x, remaining[0].y)
            
            for i, customer in enumerate(remaining[1:], 1):
                distance = self._euclidean_distance(current_x, current_y, customer.x, customer.y)
                if distance < min_distance:
                    min_distance = distance
                    nearest_idx = i
            
            nearest_customer = remaining.pop(nearest_idx)
            optimized.append(nearest_customer)
            current_x, current_y = nearest_customer.x, nearest_customer.y
        
        return optimized
    
    def _calculate_route_distance(self, customers: List[Customer]) -> float:
        """Calculate total distance for a route including depot returns."""
        if not customers:
            return 0.0
        
        total_distance = 0.0
        
        # Depot to first customer
        total_distance += self._euclidean_distance(
            self.depot_x, self.depot_y, customers[0].x, customers[0].y
        )
        
        # Between customers
        for i in range(1, len(customers)):
            total_distance += self._euclidean_distance(
                customers[i-1].x, customers[i-1].y, customers[i].x, customers[i].y
            )
        
        # Last customer back to depot
        total_distance += self._euclidean_distance(
            customers[-1].x, customers[-1].y, self.depot_x, self.depot_y
        )
        
        return total_distance
    
    def _euclidean_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def generate_diverse_solutions(self, num_solutions: int = 5) -> List[List[Route]]:
        """
        Generate multiple diverse initial solutions using different heuristics.
        
        Args:
            num_solutions: Number of diverse solutions to generate
        
        Returns:
            List of solution sets, each containing routes
        """
        self.logger.info(f"Generating {num_solutions} diverse initial solutions")
        
        solutions = []
        
        # Generate solutions with different methods and perturbation levels
        methods = [
            ("nearest_neighbor", 0.0),
            ("nearest_neighbor", 0.1),
            ("nearest_neighbor", 0.15),
            ("savings_algorithm", 0.0),
            ("savings_algorithm", 0.12),
            ("firefly_algorithm", 0.1),
            ("firefly_algorithm", 0.15),
            ("greedy_insertion", 0.1),
            ("greedy_insertion", 0.12),
        ]
        
        for i in range(min(num_solutions, len(methods))):
            method, perturbation = methods[i]
            
            try:
                if method == "nearest_neighbor":
                    solution = self.generate_nearest_neighbor_solution(perturbation)
                elif method == "savings_algorithm":
                    solution = self.generate_savings_algorithm_solution(perturbation)
                elif method == "firefly_algorithm":
                    solution = self.generate_firefly_algorithm_solution(perturbation_factor=perturbation)
                elif method == "greedy_insertion":
                    solution = self.generate_greedy_insertion_solution(perturbation_factor=perturbation)
                
                if solution:  # Only add non-empty solutions
                    solutions.append(solution)
                    
            except Exception as e:
                self.logger.warning(f"Failed to generate solution with {method}: {str(e)}")
        
        # Fill remaining slots with random variations
        while len(solutions) < num_solutions:
            try:
                # Use random method with random perturbation
                method = random.choice(["nearest_neighbor", "greedy_insertion", "firefly_algorithm"])
                perturbation = random.uniform(0.1, 0.2)
                
                if method == "nearest_neighbor":
                    solution = self.generate_nearest_neighbor_solution(perturbation)
                elif method == "greedy_insertion":
                    solution = self.generate_greedy_insertion_solution(perturbation_factor=perturbation)
                else:  # firefly_algorithm
                    solution = self.generate_firefly_algorithm_solution(perturbation_factor=perturbation)
                
                if solution:
                    solutions.append(solution)
                    
            except Exception as e:
                self.logger.warning(f"Failed to generate additional solution: {str(e)}")
                break
        
        self.logger.info(f"Generated {len(solutions)} diverse initial solutions")
        return solutions
