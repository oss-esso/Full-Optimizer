
from typing import List, Optional, Iterator, Callable, TYPE_CHECKING
from itertools import combinations
import copy

# NetworkX for proximity graph and clique finding (Granular Tabu Search)
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: NetworkX not available. Granular Tabu Search will be disabled.")

if TYPE_CHECKING:
    from route import Route
    from order import Order
    from solution import Solution
    from vehicle import Vehicle

from second_level import l2_heuristic
from first_level import multiple_order_relocation_neighborhood

def granular_multiple_order_relocation_neighborhood(solution: 'Solution', max_orders: int = 3, 
                                                   proximity_threshold: float = 0.1) -> Iterator['Solution']:
    """
    Generate neighborhood using Granular Tabu Search for multiple order relocations (Section 5.8).
    
    This implements granular search by building proximity graphs and only considering
    moves of order subsets that form cliques in the proximity graph.
    
    Args:
        solution: Current solution to generate neighbors from
        max_orders: Maximum number of orders to relocate in a single move
        proximity_threshold: Threshold for considering orders as proximate
        
    Yields:
        New solutions with multiple proximate orders relocated between routes
    """
    if not solution or not solution.routes or not NETWORKX_AVAILABLE:
        # Fallback to regular multiple order relocation if NetworkX unavailable
        for neighbor in multiple_order_relocation_neighborhood(solution, max_orders):
            yield neighbor
        return
    
    # For each source vehicle route
    for from_idx, from_route in enumerate(solution.routes):
        if not from_route or not from_route.tasks:
            continue
            
        # Build proximity graph for orders in this route
        proximity_graph = _build_proximity_graph(from_route, proximity_threshold)
        
        if not proximity_graph.nodes():
            continue
        
        # Find all cliques in the proximity graph (up to max_orders size)
        cliques = _find_cliques_up_to_size(proximity_graph, max_orders)
        
        # For each clique, try relocating those orders together
        for clique in cliques:
            if len(clique) < 1:
                continue
                
            # For each destination vehicle route (different from source)
            for to_idx, to_route in enumerate(solution.routes):
                if from_idx == to_idx:
                    continue  # Skip same route
                
                # Create a new solution with the clique orders relocated
                new_solution = copy.deepcopy(solution)
                
                # Remove tasks from source route
                source_route = new_solution.routes[from_idx]
                orders_in_route = _get_orders_from_route(from_route)
                
                for order_id in clique:
                    if order_id in orders_in_route:
                        for task in orders_in_route[order_id]:
                            source_route.remove_task(task)
                
                # Add all orders to destination route using L2 heuristic
                dest_route = new_solution.routes[to_idx]
                all_inserted = True
                
                for order_id in clique:
                    # Get the order object
                    order = next((o for o in solution.orders if o.id == order_id), None)
                    if order:
                        # Use L2 heuristic to optimally insert the order
                        optimized_route = l2_heuristic(dest_route, order)
                        if optimized_route:
                            dest_route = optimized_route
                        else:
                            all_inserted = False
                            break
                
                if all_inserted:
                    new_solution.routes[to_idx] = dest_route
                    yield new_solution


def _build_proximity_graph(route: 'Route', threshold: float) -> 'nx.Graph':
    """
    Build a proximity graph for orders in a route (Definition 5.2).
    
    Orders are connected if they are geographically close or have similar
    time windows, indicating they should be moved together.
    
    Args:
        route: Route to build proximity graph for
        threshold: Proximity threshold for connecting orders
        
    Returns:
        NetworkX graph with orders as nodes and proximity edges
    """
    if not NETWORKX_AVAILABLE:
        return None
    
    graph = nx.Graph()
    
    # Get all orders in the route
    orders_in_route = _get_orders_from_route(route)
    order_ids = list(orders_in_route.keys())
    
    # Add all orders as nodes
    graph.add_nodes_from(order_ids)
    
    # Add edges between proximate orders
    for i, order1_id in enumerate(order_ids):
        for j, order2_id in enumerate(order_ids[i+1:], i+1):
            proximity_score = _calculate_order_proximity(
                orders_in_route[order1_id], 
                orders_in_route[order2_id]
            )
            
            if proximity_score >= threshold:
                graph.add_edge(order1_id, order2_id, weight=proximity_score)
    
    return graph


def _calculate_order_proximity(tasks1: List, tasks2: List) -> float:
    """
    Calculate proximity score between two orders based on their tasks.
    
    Proximity is based on:
    - Geographic distance between pickup/delivery locations
    - Time window overlap
    - Service time similarity
    
    Args:
        tasks1: Tasks for first order
        tasks2: Tasks for second order
        
    Returns:
        Proximity score between 0 and 1 (higher means more proximate)
    """
    if not tasks1 or not tasks2:
        return 0.0
    
    total_proximity = 0.0
    comparisons = 0
    
    # Compare all task pairs between the two orders
    for task1 in tasks1:
        for task2 in tasks2:
            # Geographic proximity (if location data available)
            geo_proximity = _calculate_geographic_proximity(task1, task2)
            
            # Time window proximity (if time window data available)
            time_proximity = _calculate_time_proximity(task1, task2)
            
            # Service proximity (based on service requirements)
            service_proximity = _calculate_service_proximity(task1, task2)
            
            # Weighted average of proximity factors
            proximity = (0.5 * geo_proximity + 0.3 * time_proximity + 0.2 * service_proximity)
            total_proximity += proximity
            comparisons += 1
    
    return total_proximity / comparisons if comparisons > 0 else 0.0


def _calculate_geographic_proximity(task1, task2) -> float:
    """Calculate geographic proximity between two tasks."""
    # Placeholder implementation - in practice would use actual coordinates
    if hasattr(task1, 'location') and hasattr(task2, 'location'):
        # Calculate distance and convert to proximity score
        # This is a simplified version - real implementation would use actual distances
        loc1 = getattr(task1, 'location', (0, 0))
        loc2 = getattr(task2, 'location', (0, 0))
        
        if isinstance(loc1, (tuple, list)) and isinstance(loc2, (tuple, list)):
            distance = ((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)**0.5
            # Convert distance to proximity (inverse relationship)
            max_distance = 100.0  # Normalize by maximum expected distance
            return max(0.0, 1.0 - (distance / max_distance))
    
    return 0.5  # Default proximity if no location data


def _calculate_time_proximity(task1, task2) -> float:
    """Calculate time window proximity between two tasks."""
    if hasattr(task1, 'time_window') and hasattr(task2, 'time_window'):
        tw1 = getattr(task1, 'time_window', None)
        tw2 = getattr(task2, 'time_window', None)
        
        if tw1 and tw2 and hasattr(tw1, 'start') and hasattr(tw2, 'start'):
            # Calculate overlap and convert to proximity
            overlap_start = max(tw1.start, tw2.start)
            overlap_end = min(getattr(tw1, 'end', tw1.start + 60), 
                            getattr(tw2, 'end', tw2.start + 60))
            
            if overlap_end > overlap_start:
                # There's an overlap
                overlap_duration = overlap_end - overlap_start
                total_duration = max(getattr(tw1, 'end', tw1.start + 60) - tw1.start,
                                   getattr(tw2, 'end', tw2.start + 60) - tw2.start)
                return overlap_duration / total_duration if total_duration > 0 else 0.0
    
    return 0.5  # Default proximity if no time window data


def _calculate_service_proximity(task1, task2) -> float:
    """Calculate service requirement proximity between two tasks."""
    # Compare service types, priorities, or other service-related attributes
    if hasattr(task1, 'service_type') and hasattr(task2, 'service_type'):
        if getattr(task1, 'service_type', None) == getattr(task2, 'service_type', None):
            return 1.0
        else:
            return 0.0
    
    return 0.5  # Default proximity if no service data


def _find_cliques_up_to_size(graph: 'nx.Graph', max_size: int) -> List[List]:
    """
    Find all cliques in the graph up to a maximum size.
    
    Args:
        graph: NetworkX graph to find cliques in
        max_size: Maximum clique size to consider
        
    Returns:
        List of cliques (each clique is a list of node IDs)
    """
    if not NETWORKX_AVAILABLE or not graph:
        return []
    
    all_cliques = []
    
    # Find all maximal cliques
    for clique in nx.find_cliques(graph):
        if len(clique) <= max_size:
            all_cliques.append(list(clique))
        else:
            # If clique is too large, break it into smaller sub-cliques
            for size in range(1, max_size + 1):
                for sub_clique in combinations(clique, size):
                    all_cliques.append(list(sub_clique))
    
    # Remove duplicates and sort by size (smaller first for efficiency)
    unique_cliques = []
    seen = set()
    for clique in sorted(all_cliques, key=len):
        clique_tuple = tuple(sorted(clique))
        if clique_tuple not in seen:
            seen.add(clique_tuple)
            unique_cliques.append(clique)
    
    return unique_cliques


def _get_orders_from_route(route: 'Route') -> dict:
    """
    Extract orders and their tasks from a route.
    
    Args:
        route: Route to extract orders from
        
    Returns:
        Dictionary mapping order_id to list of tasks
    """
    orders_in_route = {}
    
    if route and route.tasks:
        for task in route.tasks:
            order_id = getattr(task, 'order_id', None)
            if order_id is not None:
                if order_id not in orders_in_route:
                    orders_in_route[order_id] = []
                orders_in_route[order_id].append(task)
    
    return orders_in_route
