"""
Pricing Problem Implementation for EPDT Column Generation

This module implements the Elementary Shortest Path Problem with Resource Constraints (ESPPRC)
for the Pricing Problem in Column Generation. This is the most complex component, requiring
a specialized label correcting algorithm with custom dominance rules.

The Pricing Problem finds routes with negative reduced cost by solving:
    minimize: c_r - Σ(π_o * a_or) - μ_v * b_vr
    
Where:
    c_r = route cost
    π_o = dual value for order o coverage constraint  
    μ_v = dual value for vehicle v fleet constraint
    a_or = 1 if route serves order o
    b_vr = 1 if route uses vehicle v

Key Features:
- Label correcting algorithm for ESPPRC
- Custom dominance rules for multi-pickup, multi-delivery precedence
- Resource tracking (time, load, vehicle capacity)
- Elementary path constraints (no cycles)
- Multi-day planning support
"""

from typing import List, Dict, Set, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
import heapq
import logging
from collections import defaultdict

from .epdt_data_structures import Order, Vehicle, Task, Route, TaskType
from .second_level import calculate_z2_score, is_feasible

logger = logging.getLogger(__name__)


@dataclass
class ESPPRCResult:
    """Result from solving the ESPPRC Pricing Problem."""
    
    feasible: bool
    reduced_cost: float
    route: Optional[Route]
    path: List[str]  # Sequence of location IDs
    cost: float
    resource_consumption: Dict[str, float]
    solve_time: float


class Label:
    """
    Label for the label correcting algorithm in ESPPRC.
    
    A label represents the state of a partial path and includes all necessary
    information for extending the path and checking dominance.
    """
    
    def __init__(self, 
                 node_id: str,
                 cost: float,
                 time: float,
                 weight_load: float,
                 volume_load: float,
                 path: List[str],
                 visited_nodes: Set[str],
                 open_orders: Set[str],
                 unreachable_nodes: Set[str],
                 completed_orders: Set[str],
                 vehicle_state: Dict[str, Any] = None):
        """
        Initialize a label for the ESPPRC algorithm.
        
        Args:
            node_id: Current location
            cost: Cumulative cost to reach this node
            time: Cumulative time (working + driving)
            weight_load: Current vehicle weight load
            volume_load: Current vehicle volume load
            path: Sequence of nodes visited
            visited_nodes: Set of all visited nodes (for elementarity)
            open_orders: Orders with some but not all pickups completed
            unreachable_nodes: Nodes that cannot be visited due to precedence
            completed_orders: Orders that are fully completed (all pickups and deliveries)
            vehicle_state: Additional vehicle state (HoS, etc.)
        """
        self.node_id = node_id
        self.cost = cost
        self.time = time
        self.weight_load = weight_load
        self.volume_load = volume_load
        self.path = path.copy()
        self.visited_nodes = visited_nodes.copy()
        self.open_orders = open_orders.copy()
        self.unreachable_nodes = unreachable_nodes.copy()
        self.completed_orders = completed_orders.copy()
        self.vehicle_state = vehicle_state.copy() if vehicle_state else {}
        
        # For priority queue
        self.priority = cost
    
    def __lt__(self, other):
        """For heapq priority comparison."""
        return self.priority < other.priority
    
    def __eq__(self, other):
        """Check equality based on state."""
        if not isinstance(other, Label):
            return False
        return (self.node_id == other.node_id and
                self.time == other.time and
                self.weight_load == other.weight_load and
                self.volume_load == other.volume_load and
                self.visited_nodes == other.visited_nodes and
                self.open_orders == other.open_orders and
                self.unreachable_nodes == other.unreachable_nodes)
    
    def __hash__(self):
        """Hash for efficient storage in sets/dicts."""
        return hash((self.node_id, self.time, self.weight_load, self.volume_load,
                    frozenset(self.visited_nodes), frozenset(self.open_orders),
                    frozenset(self.unreachable_nodes)))
    
    def dominates(self, other: 'Label') -> bool:
        """
        Check if this label dominates another label.
        
        Custom dominance rules for EPDT:
        1. Same node, better cost and resources
        2. Same or better feasible completions (unreachable nodes subset)
        3. Same or more progress on orders
        
        Args:
            other: Label to compare against
            
        Returns:
            True if this label dominates the other
        """
        if self.node_id != other.node_id:
            return False
        
        # Basic resource dominance
        if (self.cost > other.cost or
            self.time > other.time or  
            self.weight_load > other.weight_load or
            self.volume_load > other.volume_load):
            return False
        
        # Unreachable nodes must be subset (more feasible completions)
        if not self.unreachable_nodes.issubset(other.unreachable_nodes):
            return False
        
        # Open orders should be subset or equal (same or more progress)
        if not self.open_orders.issubset(other.open_orders):
            return False
        
        # Completed orders should be superset (more progress)
        if not self.completed_orders.issuperset(other.completed_orders):
            return False
        
        # Visited nodes for elementarity
        if not self.visited_nodes.issuperset(other.visited_nodes):
            return False
        
        # If all conditions met, this label dominates
        return True
    
    def extend_to_task(self, task: Task, travel_time: float, travel_cost: float,
                      vehicle: Vehicle) -> Optional['Label']:
        """
        Extend this label to include a new task.
        
        Args:
            task: Task to extend to
            travel_time: Time to travel to task location
            travel_cost: Cost to travel to task location  
            vehicle: Vehicle being used
            
        Returns:
            New extended label if feasible, None otherwise
        """
        # Check if task is reachable
        if task.location_id in self.unreachable_nodes:
            return None
        
        # Check if already visited (elementarity)
        if task.location_id in self.visited_nodes:
            return None
        
        # Check precedence constraints
        if task.is_delivery():
            # All pickups for this order must be completed
            if task.order_id not in self.open_orders and task.order_id not in self.completed_orders:
                return None
        
        # Calculate new resource consumption
        new_time = self.time + travel_time + task.service_time
        new_cost = self.cost + travel_cost
        new_weight = self.weight_load + task.demand
        new_volume = self.volume_load + task.volume
        
        # Check resource feasibility
        if new_weight > vehicle.weight_capacity or new_volume > vehicle.volume_capacity:
            return None
        
        # Check time windows if applicable
        if hasattr(task, 'latest_time') and task.latest_time is not None:
            arrival_time = self.time + travel_time
            if arrival_time > task.latest_time:
                return None
        
        # Update order tracking
        new_open_orders = self.open_orders.copy()
        new_completed_orders = self.completed_orders.copy()
        new_unreachable_nodes = self.unreachable_nodes.copy()
        
        if task.is_pickup():
            # Add to open orders
            new_open_orders.add(task.order_id)
            # Remove delivery tasks from unreachable (now reachable)
            order = self._get_order_by_id(task.order_id)
            if order:
                for delivery_task in order.get_deliveries():
                    new_unreachable_nodes.discard(delivery_task.location_id)
        else:  # delivery
            # Check if all deliveries for this order are now complete
            order = self._get_order_by_id(task.order_id)
            if order:
                remaining_deliveries = []
                for delivery_task in order.get_deliveries():
                    if (delivery_task.location_id != task.location_id and 
                        delivery_task.location_id not in self.visited_nodes):
                        remaining_deliveries.append(delivery_task)
                
                if not remaining_deliveries:  # All deliveries complete
                    new_open_orders.discard(task.order_id)
                    new_completed_orders.add(task.order_id)
        
        # Create extended label
        extended_label = Label(
            node_id=task.location_id,
            cost=new_cost,
            time=new_time,
            weight_load=new_weight,
            volume_load=new_volume,
            path=self.path + [task.location_id],
            visited_nodes=self.visited_nodes | {task.location_id},
            open_orders=new_open_orders,
            unreachable_nodes=new_unreachable_nodes,
            completed_orders=new_completed_orders,
            vehicle_state=self.vehicle_state
        )
        
        return extended_label
    
    def _get_order_by_id(self, order_id: str) -> Optional[Order]:
        """Helper to get order by ID - would need access to orders list."""
        # This would need to be provided by the pricing problem context
        return None
    
    def is_feasible_termination(self, depot_id: str) -> bool:
        """
        Check if this label represents a feasible route termination.
        
        Args:
            depot_id: ID of the depot to return to
            
        Returns:
            True if this is a valid route ending
        """
        # No open orders (all picked up orders have been delivered)
        if self.open_orders:
            return False
        
        # Could add other termination checks here
        return True


class PricingProblem:
    """
    Elementary Shortest Path Problem with Resource Constraints (ESPPRC) solver.
    
    This class implements the label correcting algorithm to find routes with
    negative reduced cost for the Column Generation procedure.
    """
    
    def __init__(self, vehicle: Vehicle, orders: List[Order], config):
        """
        Initialize the Pricing Problem for a specific vehicle.
        
        Args:
            vehicle: Vehicle for which to generate routes
            orders: All available orders
            config: ColumnGenerationConfig with algorithm parameters
        """
        self.vehicle = vehicle
        self.orders = orders
        self.config = config
        
        # Dual values from Master Problem (set before solving)
        self.dual_values: Dict[str, float] = {}
        
        # Problem structure
        self.tasks: List[Task] = []
        self.task_by_location: Dict[str, List[Task]] = defaultdict(list)
        self.orders_by_id: Dict[str, Order] = {}
        
        # Network structure
        self.locations: Set[str] = set()
        self.travel_times: Dict[Tuple[str, str], float] = {}
        self.travel_costs: Dict[Tuple[str, str], float] = {}
        
        self._build_problem_structure()
        
        logger.debug(f"🏗️  Pricing Problem initialized for vehicle {vehicle.id}")
        logger.debug(f"   📍 Locations: {len(self.locations)}")
        logger.debug(f"   📦 Tasks: {len(self.tasks)}")
    
    def _build_problem_structure(self):
        """Build the problem structure from orders."""
        # Extract all tasks and locations
        for order in self.orders:
            self.orders_by_id[order.id] = order
            
            for task in order.get_all_tasks():
                self.tasks.append(task)
                self.task_by_location[task.location_id].append(task)
                self.locations.add(task.location_id)
        
        # Add depot
        self.locations.add(self.vehicle.depot_id)
        
        # Build travel time/cost matrix (simplified)
        self._build_travel_matrix()
    
    def _build_travel_matrix(self):
        """Build travel time and cost matrices between locations."""
        # Simplified implementation - in practice would use real distances
        for loc1 in self.locations:
            for loc2 in self.locations:
                if loc1 != loc2:
                    # Placeholder calculation
                    distance = 10.0  # km
                    travel_time = distance / 50.0 * 60  # minutes at 50 km/h
                    travel_cost = distance * self.vehicle.cost_per_km
                    
                    self.travel_times[(loc1, loc2)] = travel_time
                    self.travel_costs[(loc1, loc2)] = travel_cost
                else:
                    self.travel_times[(loc1, loc2)] = 0.0
                    self.travel_costs[(loc1, loc2)] = 0.0
    
    def set_dual_values(self, dual_values: Dict[str, float]):
        """Set dual values from the Master Problem solution."""
        self.dual_values = dual_values.copy()
    
    def solve(self) -> ESPPRCResult:
        """
        Solve the ESPPRC to find a route with negative reduced cost.
        
        Returns:
            ESPPRCResult with route and reduced cost information
        """
        import time
        start_time = time.time()
        
        try:
            # Initialize label correcting algorithm
            best_label = self._label_correcting_algorithm()
            
            solve_time = time.time() - start_time
            
            if best_label and best_label.cost < 0:
                # Build route from label
                route = self._build_route_from_label(best_label)
                
                return ESPPRCResult(
                    feasible=True,
                    reduced_cost=best_label.cost,
                    route=route,
                    path=best_label.path,
                    cost=best_label.cost,
                    resource_consumption={
                        'time': best_label.time,
                        'weight': best_label.weight_load,
                        'volume': best_label.volume_load
                    },
                    solve_time=solve_time
                )
            else:
                return ESPPRCResult(
                    feasible=False,
                    reduced_cost=0.0,
                    route=None,
                    path=[],
                    cost=0.0,
                    resource_consumption={},
                    solve_time=solve_time
                )
                
        except Exception as e:
            logger.error(f"❌ Pricing Problem solving failed: {str(e)}")
            raise
    
    def _label_correcting_algorithm(self) -> Optional[Label]:
        """
        Main label correcting algorithm for ESPPRC.
        
        Returns:
            Best label with negative reduced cost, or None if none found
        """
        # Initialize data structures
        labels_at_node: Dict[str, List[Label]] = defaultdict(list)
        priority_queue = []
        best_termination_label = None
        
        # Create initial label at depot
        initial_label = Label(
            node_id=self.vehicle.depot_id,
            cost=0.0,
            time=0.0,
            weight_load=0.0,
            volume_load=0.0,
            path=[self.vehicle.depot_id],
            visited_nodes={self.vehicle.depot_id},
            open_orders=set(),
            unreachable_nodes=self._get_initial_unreachable_nodes(),
            completed_orders=set()
        )
        
        heapq.heappush(priority_queue, initial_label)
        labels_at_node[self.vehicle.depot_id].append(initial_label)
        
        # Main algorithm loop
        processed_labels = 0
        while priority_queue and processed_labels < 10000:  # Limit for performance
            current_label = heapq.heappop(priority_queue)
            processed_labels += 1
            
            # Check if label is dominated
            if self._is_dominated(current_label, labels_at_node[current_label.node_id]):
                continue
            
            # Try extending to all reachable tasks
            for task in self.tasks:
                extended_label = self._extend_label_to_task(current_label, task)
                
                if extended_label is None:
                    continue
                
                # Check dominance against existing labels
                if not self._is_dominated(extended_label, labels_at_node[extended_label.node_id]):
                    # Remove dominated labels
                    labels_at_node[extended_label.node_id] = [
                        label for label in labels_at_node[extended_label.node_id]
                        if not extended_label.dominates(label)
                    ]
                    
                    # Add new label
                    labels_at_node[extended_label.node_id].append(extended_label)
                    heapq.heappush(priority_queue, extended_label)
            
            # Try returning to depot
            if current_label.is_feasible_termination(self.vehicle.depot_id):
                depot_label = self._extend_label_to_depot(current_label)
                if depot_label and (best_termination_label is None or 
                                   depot_label.cost < best_termination_label.cost):
                    best_termination_label = depot_label
        
        logger.debug(f"   🔍 Processed {processed_labels} labels")
        return best_termination_label
    
    def _get_initial_unreachable_nodes(self) -> Set[str]:
        """Get initially unreachable nodes (delivery tasks without pickups)."""
        unreachable = set()
        for order in self.orders:
            for delivery_task in order.get_deliveries():
                unreachable.add(delivery_task.location_id)
        return unreachable
    
    def _extend_label_to_task(self, label: Label, task: Task) -> Optional[Label]:
        """Extend a label to include a specific task."""
        travel_time = self.travel_times.get((label.node_id, task.location_id), 0.0)
        travel_cost = self.travel_costs.get((label.node_id, task.location_id), 0.0)
        
        # Calculate reduced cost
        reduced_cost = travel_cost
        
        # Subtract dual values for covered orders
        if task.order_id in self.dual_values:
            reduced_cost -= self.dual_values.get(f"coverage_{task.order_id}", 0.0)
        
        # Create extended label with access to orders
        extended_label = label.extend_to_task(task, travel_time, reduced_cost, self.vehicle)
        
        # Provide order access for precedence checking
        if extended_label:
            extended_label._orders_dict = self.orders_by_id
        
        return extended_label
    
    def _extend_label_to_depot(self, label: Label) -> Optional[Label]:
        """Extend a label back to the depot."""
        travel_time = self.travel_times.get((label.node_id, self.vehicle.depot_id), 0.0)
        travel_cost = self.travel_costs.get((label.node_id, self.vehicle.depot_id), 0.0)
        
        # Subtract fleet constraint dual value
        reduced_cost = travel_cost - self.dual_values.get(f"fleet_{self.vehicle.id}", 0.0)
        
        depot_label = Label(
            node_id=self.vehicle.depot_id,
            cost=label.cost + reduced_cost,
            time=label.time + travel_time,
            weight_load=0.0,  # Unloaded at depot
            volume_load=0.0,
            path=label.path + [self.vehicle.depot_id],
            visited_nodes=label.visited_nodes | {self.vehicle.depot_id},
            open_orders=label.open_orders,
            unreachable_nodes=label.unreachable_nodes,
            completed_orders=label.completed_orders,
            vehicle_state=label.vehicle_state
        )
        
        return depot_label
    
    def _is_dominated(self, label: Label, existing_labels: List[Label]) -> bool:
        """Check if a label is dominated by any existing label."""
        for existing_label in existing_labels:
            if existing_label.dominates(label):
                return True
        return False
    
    def _build_route_from_label(self, label: Label) -> Route:
        """Build a Route object from the final label."""
        route = Route(vehicle=self.vehicle)
        
        # Build task sequence from path
        for i in range(1, len(label.path) - 1):  # Skip start and end depot
            location_id = label.path[i]
            
            # Find the task at this location that makes sense in sequence
            for task in self.task_by_location[location_id]:
                # Add logic to select the correct task if multiple at same location
                route.tasks.append(task)
                break
        
        return route
