#!/usr/bin/env python3
"""
Improved 2-opt Local Search Algorithm for VRP
Following Croes' 1958 framework with modern enhancements
"""

import time
import copy
from typing import List, Tuple, Dict, Optional
import numpy as np


class TwoOptLocalSearch:
    """
    Improved 2-opt local search algorithm with VRP-specific constraints.
    
    Implements Croes' 1958 framework with modern enhancements:
    - Complete search mechanism with nested loops
    - First-improvement and best-improvement variants
    - Depot constraint handling for VRP
    - Goto restart mechanism for immediate improvement acceptance
    """
    
    def __init__(self, distance_matrix: np.ndarray, depot_nodes: Optional[List[str]] = None):
        """
        Initialize the 2-opt optimizer.
        
        Args:
            distance_matrix: NxN distance matrix between locations
            depot_nodes: List of depot node identifiers (default: None for single depot)
        """
        self.distance_matrix = distance_matrix
        self.depot_nodes = depot_nodes or ['depot']
        self.improvement_stats = {
            'total_swaps': 0,
            'successful_swaps': 0,
            'total_iterations': 0,
            'total_improvement': 0.0
        }
    
    def is_depot_node(self, node: str) -> bool:
        """Check if a node is a depot node."""
        return any(depot in str(node) for depot in self.depot_nodes)
    
    def calculate_route_distance(self, route: List[str], location_to_index: Dict[str, int]) -> float:
        """Calculate total distance for a route."""
        if len(route) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(route) - 1):
            from_idx = location_to_index[route[i]]
            to_idx = location_to_index[route[i + 1]]
            total_distance += self.distance_matrix[from_idx][to_idx]
        
        return total_distance
    
    def get_eligible_nodes(self, route: List[str]) -> List[int]:
        """
        Get indices of nodes eligible for 2-opt swapping.
        Excludes depot nodes to maintain route validity.
        """
        eligible = []
        for i, node in enumerate(route):
            if not self.is_depot_node(node):
                eligible.append(i)
        return eligible
    
    def perform_2opt_swap(self, route: List[str], i: int, j: int) -> List[str]:
        """
        Perform 2-opt swap on route between positions i and j.
        
        For route A→B→C→D→E→F→A, swapping (B,E) yields:
        A→B→E→D→C→F→A (reversing segment C→D→E)
        
        Args:
            route: Current route
            i, j: Swap positions (i < j)
            
        Returns:
            New route after 2-opt swap
        """
        if i >= j or i < 0 or j >= len(route):
            return route.copy()
        
        # Create new route by reversing segment between i and j
        new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
        return new_route
    
    def validate_route_constraints(self, route: List[str]) -> bool:
        """
        Validate that route maintains VRP constraints.
        Depot must remain at route boundaries.
        """
        if len(route) < 2:
            return True
        
        # Check depot constraints - depot should be at start and end
        if not self.is_depot_node(route[0]) or not self.is_depot_node(route[-1]):
            return False
        
        # Check that no depot appears in the middle (for single depot scenarios)
        if len(self.depot_nodes) == 1:
            for node in route[1:-1]:
                if self.is_depot_node(node):
                    return False
        
        return True
    
    def two_opt_first_improvement(self, route: List[str], location_to_index: Dict[str, int]) -> Tuple[List[str], float, Dict]:
        """
        2-opt with first-improvement strategy.
        Accepts the first swap that improves the solution (goto restart mechanism).
        
        Args:
            route: Initial route
            location_to_index: Mapping from location names to matrix indices
            
        Returns:
            Tuple of (improved_route, final_distance, stats)
        """
        current_route = route.copy()
        current_distance = self.calculate_route_distance(current_route, location_to_index)
        initial_distance = current_distance
        
        stats = {
            'iterations': 0,
            'improvements': 0,
            'total_swaps_evaluated': 0,
            'improvement_restarts': 0
        }
        
        improved = True
        while improved:  # Repeat until no improvement
            improved = False
            stats['iterations'] += 1
            
            eligible_nodes = self.get_eligible_nodes(current_route)
            
            # Complete search mechanism with nested loops
            for i in range(len(eligible_nodes)):
                if improved:  # Restart immediately when improvement found
                    stats['improvement_restarts'] += 1
                    break
                    
                for j in range(i + 1, len(eligible_nodes)):
                    stats['total_swaps_evaluated'] += 1
                    
                    # Get actual route indices
                    route_i = eligible_nodes[i]
                    route_j = eligible_nodes[j]
                    
                    # Perform 2-opt swap
                    new_route = self.perform_2opt_swap(current_route, route_i, route_j)
                    
                    # Validate VRP constraints
                    if not self.validate_route_constraints(new_route):
                        continue
                    
                    # Calculate new distance
                    new_distance = self.calculate_route_distance(new_route, location_to_index)
                    
                    # First improvement: accept immediately if better
                    if new_distance < current_distance:
                        improvement = current_distance - new_distance
                        current_route = new_route
                        current_distance = new_distance
                        stats['improvements'] += 1
                        improved = True  # Trigger restart
                        
                        # Update global stats
                        self.improvement_stats['total_swaps'] += 1
                        self.improvement_stats['successful_swaps'] += 1
                        self.improvement_stats['total_improvement'] += improvement
                        
                        break  # Goto restart mechanism
        
        stats['final_improvement'] = initial_distance - current_distance
        stats['improvement_percentage'] = (stats['final_improvement'] / initial_distance) * 100 if initial_distance > 0 else 0
        
        return current_route, current_distance, stats
    
    def two_opt_best_improvement(self, route: List[str], location_to_index: Dict[str, int]) -> Tuple[List[str], float, Dict]:
        """
        2-opt with best-improvement strategy.
        Evaluates all possible swaps and selects the best improvement per iteration.
        
        Args:
            route: Initial route
            location_to_index: Mapping from location names to matrix indices
            
        Returns:
            Tuple of (improved_route, final_distance, stats)
        """
        current_route = route.copy()
        current_distance = self.calculate_route_distance(current_route, location_to_index)
        initial_distance = current_distance
        
        stats = {
            'iterations': 0,
            'improvements': 0,
            'total_swaps_evaluated': 0,
            'best_improvements_found': 0
        }
        
        improved = True
        while improved:  # Repeat until no improvement
            improved = False
            stats['iterations'] += 1
            
            best_route = None
            best_distance = current_distance
            best_improvement = 0
            
            eligible_nodes = self.get_eligible_nodes(current_route)
            
            # Complete search mechanism - evaluate all possible swaps
            for i in range(len(eligible_nodes)):
                for j in range(i + 1, len(eligible_nodes)):
                    stats['total_swaps_evaluated'] += 1
                    
                    # Get actual route indices
                    route_i = eligible_nodes[i]
                    route_j = eligible_nodes[j]
                    
                    # Perform 2-opt swap
                    new_route = self.perform_2opt_swap(current_route, route_i, route_j)
                    
                    # Validate VRP constraints
                    if not self.validate_route_constraints(new_route):
                        continue
                    
                    # Calculate new distance
                    new_distance = self.calculate_route_distance(new_route, location_to_index)
                    
                    # Track best improvement in this iteration
                    if new_distance < best_distance:
                        improvement = best_distance - new_distance
                        if improvement > best_improvement:
                            best_route = new_route
                            best_distance = new_distance
                            best_improvement = improvement
                            improved = True
            
            # Apply best improvement found (if any)
            if improved and best_route is not None:
                current_route = best_route
                current_distance = best_distance
                stats['improvements'] += 1
                stats['best_improvements_found'] += 1
                
                # Update global stats
                self.improvement_stats['total_swaps'] += 1
                self.improvement_stats['successful_swaps'] += 1
                self.improvement_stats['total_improvement'] += best_improvement
        
        stats['final_improvement'] = initial_distance - current_distance
        stats['improvement_percentage'] = (stats['final_improvement'] / initial_distance) * 100 if initial_distance > 0 else 0
        
        return current_route, current_distance, stats
    
    def optimize_route(self, route: List[str], location_to_index: Dict[str, int], 
                      strategy: str = 'first_improvement') -> Tuple[List[str], float, Dict]:
        """
        Optimize a single route using 2-opt local search.
        
        Args:
            route: Route to optimize
            location_to_index: Mapping from location names to matrix indices
            strategy: 'first_improvement' or 'best_improvement'
            
        Returns:
            Tuple of (optimized_route, final_distance, optimization_stats)
        """
        if len(route) < 4:  # Need at least 4 nodes for meaningful 2-opt
            distance = self.calculate_route_distance(route, location_to_index)
            return route, distance, {'message': 'Route too short for 2-opt'}
        
        start_time = time.time()
        
        if strategy == 'first_improvement':
            optimized_route, final_distance, stats = self.two_opt_first_improvement(route, location_to_index)
        elif strategy == 'best_improvement':
            optimized_route, final_distance, stats = self.two_opt_best_improvement(route, location_to_index)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'first_improvement' or 'best_improvement'")
        
        stats['optimization_time'] = (time.time() - start_time) * 1000  # milliseconds
        stats['strategy'] = strategy
        
        self.improvement_stats['total_iterations'] += stats['iterations']
        
        return optimized_route, final_distance, stats
    
    def optimize_multiple_routes(self, routes: Dict[str, List[str]], location_to_index: Dict[str, int], 
                               strategy: str = 'first_improvement') -> Tuple[Dict[str, List[str]], float, Dict]:
        """
        Optimize multiple routes using 2-opt local search.
        
        Args:
            routes: Dictionary of vehicle_id -> route
            location_to_index: Mapping from location names to matrix indices
            strategy: 'first_improvement' or 'best_improvement'
            
        Returns:
            Tuple of (optimized_routes, total_distance, comprehensive_stats)
        """
        optimized_routes = {}
        total_distance = 0.0
        all_stats = {
            'strategy': strategy,
            'routes_optimized': 0,
            'total_improvements': 0,
            'total_optimization_time': 0.0,
            'route_details': {}
        }
        
        start_time = time.time()
        
        for vehicle_id, route in routes.items():
            if len(route) > 2:  # Only optimize non-empty routes
                optimized_route, route_distance, route_stats = self.optimize_route(
                    route, location_to_index, strategy
                )
                optimized_routes[vehicle_id] = optimized_route
                total_distance += route_distance
                
                all_stats['routes_optimized'] += 1
                all_stats['total_improvements'] += route_stats.get('improvements', 0)
                all_stats['route_details'][vehicle_id] = route_stats
            else:
                # Keep empty or single-stop routes as-is
                optimized_routes[vehicle_id] = route
                if len(route) > 1:
                    total_distance += self.calculate_route_distance(route, location_to_index)
        
        all_stats['total_optimization_time'] = (time.time() - start_time) * 1000
        all_stats['total_distance'] = total_distance
        all_stats['average_improvement_per_route'] = (
            all_stats['total_improvements'] / max(1, all_stats['routes_optimized'])
        )
        
        return optimized_routes, total_distance, all_stats
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics."""
        return {
            'global_stats': self.improvement_stats.copy(),
            'success_rate': (
                self.improvement_stats['successful_swaps'] / max(1, self.improvement_stats['total_swaps'])
            ) * 100,
            'average_improvement_per_swap': (
                self.improvement_stats['total_improvement'] / max(1, self.improvement_stats['successful_swaps'])
            )
        }


def create_test_scenario():
    """Create a test scenario for 2-opt optimization."""
    # Simple test case with depot and 4 customers
    locations = {
        'depot': (0, 0),
        'customer_1': (1, 0),
        'customer_2': (2, 0),
        'customer_3': (2, 1),
        'customer_4': (1, 1)
    }
    
    # Create distance matrix
    location_names = list(locations.keys())
    n = len(location_names)
    distance_matrix = np.zeros((n, n))
    
    for i, loc1 in enumerate(location_names):
        for j, loc2 in enumerate(location_names):
            if i != j:
                x1, y1 = locations[loc1]
                x2, y2 = locations[loc2]
                distance_matrix[i][j] = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    location_to_index = {name: i for i, name in enumerate(location_names)}
    
    # Create a suboptimal route (deliberately inefficient)
    test_route = ['depot', 'customer_3', 'customer_1', 'customer_4', 'customer_2', 'depot']
    
    return distance_matrix, location_to_index, test_route, locations


if __name__ == "__main__":
    print("Testing 2-opt Local Search Algorithm")
    print("=" * 50)
    
    # Create test scenario
    distance_matrix, location_to_index, test_route, locations = create_test_scenario()
    
    # Initialize optimizer
    optimizer = TwoOptLocalSearch(distance_matrix, depot_nodes=['depot'])
    
    print(f"Initial route: {' -> '.join(test_route)}")
    initial_distance = optimizer.calculate_route_distance(test_route, location_to_index)
    print(f"Initial distance: {initial_distance:.3f}")
    print()
    
    # Test first-improvement strategy
    print("Testing First-Improvement Strategy:")
    print("-" * 40)
    optimized_route_first, final_distance_first, stats_first = optimizer.optimize_route(
        test_route, location_to_index, 'first_improvement'
    )
    print(f"Optimized route: {' -> '.join(optimized_route_first)}")
    print(f"Final distance: {final_distance_first:.3f}")
    print(f"Improvement: {initial_distance - final_distance_first:.3f} ({stats_first['improvement_percentage']:.1f}%)")
    print(f"Iterations: {stats_first['iterations']}")
    print(f"Improvements made: {stats_first['improvements']}")
    print(f"Optimization time: {stats_first['optimization_time']:.2f}ms")
    print()
    
    # Reset for fair comparison
    optimizer = TwoOptLocalSearch(distance_matrix, depot_nodes=['depot'])
    
    # Test best-improvement strategy
    print("Testing Best-Improvement Strategy:")
    print("-" * 40)
    optimized_route_best, final_distance_best, stats_best = optimizer.optimize_route(
        test_route, location_to_index, 'best_improvement'
    )
    print(f"Optimized route: {' -> '.join(optimized_route_best)}")
    print(f"Final distance: {final_distance_best:.3f}")
    print(f"Improvement: {initial_distance - final_distance_best:.3f} ({stats_best['improvement_percentage']:.1f}%)")
    print(f"Iterations: {stats_best['iterations']}")
    print(f"Improvements made: {stats_best['improvements']}")
    print(f"Optimization time: {stats_best['optimization_time']:.2f}ms")
    print()
    
    # Test multiple routes optimization
    print("Testing Multiple Routes Optimization:")
    print("-" * 40)
    
    test_routes = {
        'vehicle_1': ['depot', 'customer_1', 'customer_2', 'depot'],
        'vehicle_2': ['depot', 'customer_3', 'customer_4', 'depot']
    }
    
    optimizer = TwoOptLocalSearch(distance_matrix, depot_nodes=['depot'])
    optimized_routes, total_distance, multi_stats = optimizer.optimize_multiple_routes(
        test_routes, location_to_index, 'first_improvement'
    )
    
    print("Original routes:")
    for vehicle_id, route in test_routes.items():
        print(f"  {vehicle_id}: {' -> '.join(route)}")
    
    print("\nOptimized routes:")
    for vehicle_id, route in optimized_routes.items():
        print(f"  {vehicle_id}: {' -> '.join(route)}")
    
    print(f"\nTotal distance: {total_distance:.3f}")
    print(f"Routes optimized: {multi_stats['routes_optimized']}")
    print(f"Total improvements: {multi_stats['total_improvements']}")
    print(f"Total optimization time: {multi_stats['total_optimization_time']:.2f}ms")
    
    # Performance statistics
    print("\nPerformance Statistics:")
    print("-" * 40)
    perf_stats = optimizer.get_performance_stats()
    print(f"Total swaps attempted: {perf_stats['global_stats']['total_swaps']}")
    print(f"Successful swaps: {perf_stats['global_stats']['successful_swaps']}")
    print(f"Success rate: {perf_stats['success_rate']:.1f}%")
    print(f"Average improvement per successful swap: {perf_stats['average_improvement_per_swap']:.3f}")
