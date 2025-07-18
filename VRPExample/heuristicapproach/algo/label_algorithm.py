"""
Advanced Label Algorithm for EPDT Column Generation

This module implements the specialized label correcting algorithm with custom dominance rules
for the Elementary Shortest Path Problem with Resource Constraints (ESPPRC) in EPDT.

Key Features:
- Advanced dominance rules for multi-pickup, multi-delivery precedence constraints
- Efficient label management and pruning
- Resource-constrained path extension
- Multi-day planning support with temporal constraints
- LIFO loading constraint handling
- Performance optimization for large-scale problems

The algorithm is specifically adapted to handle the unique characteristics of EPDT:
1. Multi-pickup, multi-delivery orders with precedence constraints
2. Vehicle capacity constraints (weight and volume)
3. Time windows and working time regulations
4. LIFO loading requirements for vehicles without side doors
5. Multi-day planning with temporal sequencing
"""

from typing import List, Dict, Set, Tuple, Optional, Any, Iterator
from dataclasses import dataclass, field
from collections import defaultdict, deque
import bisect
import logging

from .epdt_data_structures import Order, Vehicle, Task, TaskType
from .second_level import _check_time_windows, _check_hos

logger = logging.getLogger(__name__)


@dataclass
class LabelState:
    """
    Enhanced state representation for EPDT labels.
    
    This class encapsulates all the state information needed for
    label-based shortest path algorithms in EPDT.
    """
    
    # Basic path information
    location_id: str
    cost: float
    time: float
    
    # Resource consumption
    weight_load: float
    volume_load: float
    
    # Path tracking
    path: List[str] = field(default_factory=list)
    visited_tasks: Set[str] = field(default_factory=set)
    
    # Order state tracking
    open_orders: Set[str] = field(default_factory=set)  # Orders with partial pickups
    completed_orders: Set[str] = field(default_factory=set)  # Fully completed orders
    unreachable_tasks: Set[str] = field(default_factory=set)  # Tasks blocked by precedence
    
    # LIFO loading state
    cargo_stack: List[str] = field(default_factory=list)  # LIFO order of loaded cargo
    
    # Working time state
    driving_time: float = 0.0
    working_time: float = 0.0
    last_break_time: float = 0.0
    extensions_used: int = 0
    
    # Multi-day state
    current_day: int = 0
    day_start_time: float = 0.0
    
    def __post_init__(self):
        """Initialize computed properties."""
        self.hash_value = self._compute_hash()
    
    def _compute_hash(self) -> int:
        """Compute hash for efficient storage and comparison."""
        return hash((
            self.location_id,
            round(self.time, 2),
            round(self.weight_load, 2),
            round(self.volume_load, 2),
            frozenset(self.visited_tasks),
            frozenset(self.open_orders),
            frozenset(self.unreachable_tasks),
            tuple(self.cargo_stack)
        ))
    
    def __hash__(self) -> int:
        return self.hash_value
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, LabelState):
            return False
        return self.hash_value == other.hash_value
    
    def copy(self) -> 'LabelState':
        """Create a deep copy of this label state."""
        return LabelState(
            location_id=self.location_id,
            cost=self.cost,
            time=self.time,
            weight_load=self.weight_load,
            volume_load=self.volume_load,
            path=self.path.copy(),
            visited_tasks=self.visited_tasks.copy(),
            open_orders=self.open_orders.copy(),
            completed_orders=self.completed_orders.copy(),
            unreachable_tasks=self.unreachable_tasks.copy(),
            cargo_stack=self.cargo_stack.copy(),
            driving_time=self.driving_time,
            working_time=self.working_time,
            last_break_time=self.last_break_time,
            extensions_used=self.extensions_used,
            current_day=self.current_day,
            day_start_time=self.day_start_time
        )


class EDDominanceChecker:
    """
    Enhanced Dominance checker for EPDT labels.
    
    Implements the sophisticated dominance rules required for
    multi-pickup, multi-delivery problems with resource constraints.
    """
    
    def __init__(self, orders: List[Order]):
        """
        Initialize dominance checker.
        
        Args:
            orders: List of all orders for precedence analysis
        """
        self.orders = orders
        self.orders_by_id = {order.id: order for order in orders}
        
        # Precompute order relationships
        self.order_tasks = defaultdict(set)
        self.task_to_order = {}
        for order in orders:
            for task in order.get_all_tasks():
                self.order_tasks[order.id].add(task.id)
                self.task_to_order[task.id] = order.id
    
    def dominates(self, label1: LabelState, label2: LabelState) -> bool:
        """
        Check if label1 dominates label2 using EPDT-specific rules.
        
        Dominance Rules for EPDT:
        1. Same location
        2. No worse cost and resources
        3. Same or better feasible completions (unreachable tasks)
        4. Same or better order progress
        5. LIFO stack compatibility
        6. Working time regulations compliance
        
        Args:
            label1: Potentially dominating label
            label2: Label to be dominated
            
        Returns:
            True if label1 dominates label2
        """
        # Rule 1: Must be at same location
        if label1.location_id != label2.location_id:
            return False
        
        # Rule 2: No worse in cost and basic resources
        if (label1.cost > label2.cost or
            label1.time > label2.time or
            label1.weight_load > label2.weight_load or
            label1.volume_load > label2.volume_load):
            return False
        
        # Rule 3: Unreachable tasks must be subset (more possibilities)
        if not label1.unreachable_tasks.issubset(label2.unreachable_tasks):
            return False
        
        # Rule 4: Order progress comparison
        if not self._better_or_equal_order_progress(label1, label2):
            return False
        
        # Rule 5: LIFO stack compatibility
        if not self._compatible_lifo_stacks(label1, label2):
            return False
        
        # Rule 6: Working time regulations
        if not self._better_working_time_state(label1, label2):
            return False
        
        # Rule 7: Multi-day progress
        if label1.current_day < label2.current_day:
            return False
        
        return True
    
    def _better_or_equal_order_progress(self, label1: LabelState, label2: LabelState) -> bool:
        """Check if label1 has better or equal order completion progress."""
        # Completed orders should be superset
        if not label1.completed_orders.issuperset(label2.completed_orders):
            return False
        
        # Open orders analysis - more complex due to partial progress
        for order_id in label2.open_orders:
            if order_id in label1.completed_orders:
                continue  # Better - completed vs open
            
            if order_id not in label1.open_orders:
                return False  # Worse - not started vs open
            
            # Both have order open - check progress within order
            if not self._better_order_progress(label1, label2, order_id):
                return False
        
        return True
    
    def _better_order_progress(self, label1: LabelState, label2: LabelState, order_id: str) -> bool:
        """Check progress within a specific order."""
        order = self.orders_by_id.get(order_id)
        if not order:
            return True
        
        # Count completed pickups and deliveries
        order_tasks = self.order_tasks[order_id]
        
        label1_pickups = sum(1 for task_id in label1.visited_tasks 
                           if task_id in order_tasks and self._is_pickup_task(task_id))
        label2_pickups = sum(1 for task_id in label2.visited_tasks 
                           if task_id in order_tasks and self._is_pickup_task(task_id))
        
        label1_deliveries = sum(1 for task_id in label1.visited_tasks 
                              if task_id in order_tasks and not self._is_pickup_task(task_id))
        label2_deliveries = sum(1 for task_id in label2.visited_tasks 
                              if task_id in order_tasks and not self._is_pickup_task(task_id))
        
        # Label1 should have at least as much progress
        return (label1_pickups >= label2_pickups and 
                label1_deliveries >= label2_deliveries)
    
    def _compatible_lifo_stacks(self, label1: LabelState, label2: LabelState) -> bool:
        """Check LIFO stack compatibility for dominance."""
        # If one stack is empty and other isn't, the empty one dominates
        if not label1.cargo_stack and label2.cargo_stack:
            return True
        
        if label1.cargo_stack and not label2.cargo_stack:
            return False
        
        # Both stacks present - label1 stack should be subset/prefix
        if len(label1.cargo_stack) > len(label2.cargo_stack):
            return False
        
        # Check if label1 stack is a prefix of label2 stack
        for i, order_id in enumerate(label1.cargo_stack):
            if i >= len(label2.cargo_stack) or order_id != label2.cargo_stack[i]:
                return False
        
        return True
    
    def _better_working_time_state(self, label1: LabelState, label2: LabelState) -> bool:
        """Check if label1 has better working time state."""
        # Less driving and working time is better
        if (label1.driving_time > label2.driving_time or
            label1.working_time > label2.working_time):
            return False
        
        # Fewer extensions used is better
        if label1.extensions_used > label2.extensions_used:
            return False
        
        return True
    
    def _is_pickup_task(self, task_id: str) -> bool:
        """Check if a task is a pickup task."""
        # This would need access to task objects - simplified for now
        return "pickup" in task_id.lower() or task_id.endswith("P")


class LabelManager:
    """
    Efficient label storage and management for the ESPPRC algorithm.
    
    Handles label storage, dominance checking, and pruning for optimal performance.
    """
    
    def __init__(self, dominance_checker: EDDominanceChecker):
        """
        Initialize label manager.
        
        Args:
            dominance_checker: Dominance checker for label comparison
        """
        self.dominance_checker = dominance_checker
        
        # Label storage by location
        self.labels_by_location: Dict[str, List[LabelState]] = defaultdict(list)
        
        # Performance tracking
        self.total_labels_generated = 0
        self.total_labels_pruned = 0
        
    def add_label(self, label: LabelState) -> bool:
        """
        Add a new label with dominance checking.
        
        Args:
            label: Label to add
            
        Returns:
            True if label was added (not dominated), False otherwise
        """
        self.total_labels_generated += 1
        location_labels = self.labels_by_location[label.location_id]
        
        # Check if new label is dominated
        for existing_label in location_labels:
            if existing_label.dominates(label):
                self.total_labels_pruned += 1
                return False
        
        # Remove labels dominated by new label
        non_dominated = []
        for existing_label in location_labels:
            if not label.dominates(existing_label):
                non_dominated.append(existing_label)
            else:
                self.total_labels_pruned += 1
        
        # Add new label
        non_dominated.append(label)
        self.labels_by_location[label.location_id] = non_dominated
        
        return True
    
    def get_labels_at_location(self, location_id: str) -> List[LabelState]:
        """Get all non-dominated labels at a location."""
        return self.labels_by_location[location_id].copy()
    
    def get_statistics(self) -> Dict[str, int]:
        """Get performance statistics."""
        total_labels_stored = sum(len(labels) for labels in self.labels_by_location.values())
        return {
            'labels_generated': self.total_labels_generated,
            'labels_pruned': self.total_labels_pruned,
            'labels_stored': total_labels_stored,
            'locations_with_labels': len(self.labels_by_location),
            'pruning_ratio': self.total_labels_pruned / max(self.total_labels_generated, 1)
        }


class LabelExtender:
    """
    Handles label extension for EPDT with all constraint checking.
    
    This class manages the complex logic of extending labels to new tasks
    while respecting all EPDT constraints and updating state correctly.
    """
    
    def __init__(self, vehicle: Vehicle, orders: List[Order], travel_matrix: Dict[Tuple[str, str], Tuple[float, float]]):
        """
        Initialize label extender.
        
        Args:
            vehicle: Vehicle for route generation
            orders: All available orders
            travel_matrix: Travel times and costs between locations
        """
        self.vehicle = vehicle
        self.orders = orders
        self.travel_matrix = travel_matrix
        
        # Order and task lookups
        self.orders_by_id = {order.id: order for order in orders}
        self.tasks_by_id = {}
        self.tasks_by_location = defaultdict(list)
        
        for order in orders:
            for task in order.get_all_tasks():
                self.tasks_by_id[task.id] = task
                self.tasks_by_location[task.location_id].append(task)
    
    def extend_label_to_task(self, label: LabelState, task: Task) -> Optional[LabelState]:
        """
        Extend a label to include a specific task.
        
        Args:
            label: Current label state
            task: Task to extend to
            
        Returns:
            Extended label if feasible, None otherwise
        """
        # Check basic feasibility
        if not self._is_task_reachable(label, task):
            return None
        
        # Calculate travel
        travel_time, travel_cost = self.travel_matrix.get(
            (label.location_id, task.location_id), (0.0, 0.0))
        
        # Create extended label
        extended_label = label.copy()
        extended_label.location_id = task.location_id
        extended_label.cost += travel_cost
        extended_label.time += travel_time + task.service_time
        extended_label.driving_time += travel_time
        extended_label.working_time += travel_time + task.service_time
        extended_label.path.append(task.location_id)
        extended_label.visited_tasks.add(task.id)
        
        # Update load
        extended_label.weight_load += task.demand
        extended_label.volume_load += task.volume
        
        # Check capacity constraints
        if (extended_label.weight_load > self.vehicle.weight_capacity or
            extended_label.volume_load > self.vehicle.volume_capacity):
            return None
        
        # Check time windows
        if not self._check_time_feasibility(extended_label, task):
            return None
        
        # Update order state
        if not self._update_order_state(extended_label, task):
            return None
        
        # Update LIFO state
        if not self._update_lifo_state(extended_label, task):
            return None
        
        # Check working time regulations
        if not self._check_working_time_feasibility(extended_label):
            return None
        
        # Update unreachable tasks
        self._update_unreachable_tasks(extended_label, task)
        
        return extended_label
    
    def _is_task_reachable(self, label: LabelState, task: Task) -> bool:
        """Check if a task is reachable from current label state."""
        # Already visited
        if task.id in label.visited_tasks:
            return False
        
        # Explicitly unreachable
        if task.id in label.unreachable_tasks:
            return False
        
        # Check precedence constraints
        if task.is_delivery():
            order_id = task.order_id
            # All pickups for this order must be completed
            if order_id not in label.open_orders and order_id not in label.completed_orders:
                return False
        
        return True
    
    def _check_time_feasibility(self, label: LabelState, task: Task) -> bool:
        """Check time window and temporal constraints."""
        arrival_time = label.time - task.service_time  # Time when we arrive
        
        # Hard time window check
        if hasattr(task, 'latest_time') and task.latest_time is not None:
            if arrival_time > task.latest_time:
                return False
        
        # Earliest time check with waiting
        if hasattr(task, 'earliest_time') and task.earliest_time is not None:
            if arrival_time < task.earliest_time:
                # Can wait - update time
                wait_time = task.earliest_time - arrival_time
                label.time += wait_time
                label.working_time += wait_time
        
        return True
    
    def _update_order_state(self, label: LabelState, task: Task) -> bool:
        """Update order completion state."""
        order_id = task.order_id
        order = self.orders_by_id.get(order_id)
        if not order:
            return False
        
        if task.is_pickup():
            # Add to open orders
            label.open_orders.add(order_id)
            
            # Check if all pickups for this order are now complete
            all_pickups_done = True
            for pickup_task in order.get_pickups():
                if pickup_task.id not in label.visited_tasks:
                    all_pickups_done = False
                    break
            
            # If all pickups done, deliveries become reachable
            if all_pickups_done:
                for delivery_task in order.get_deliveries():
                    label.unreachable_tasks.discard(delivery_task.id)
        
        else:  # delivery
            # Check if all deliveries for this order are now complete
            all_deliveries_done = True
            for delivery_task in order.get_deliveries():
                if (delivery_task.id != task.id and 
                    delivery_task.id not in label.visited_tasks):
                    all_deliveries_done = False
                    break
            
            if all_deliveries_done:
                # Order is complete
                label.open_orders.discard(order_id)
                label.completed_orders.add(order_id)
        
        return True
    
    def _update_lifo_state(self, label: LabelState, task: Task) -> bool:
        """Update LIFO cargo stack state."""
        if not self.vehicle.lifo_required:
            return True
        
        order_id = task.order_id
        
        if task.is_pickup():
            # Add to stack
            label.cargo_stack.append(order_id)
        else:  # delivery
            # Must unload from top of stack
            if not label.cargo_stack or label.cargo_stack[-1] != order_id:
                return False  # LIFO violation
            label.cargo_stack.pop()
        
        return True
    
    def _check_working_time_feasibility(self, label: LabelState) -> bool:
        """Check European Hours of Service regulations."""
        # Simplified check - could be more sophisticated
        max_driving = 540.0  # 9 hours
        max_working = 780.0  # 13 hours
        
        # Check if break is needed
        if label.driving_time - label.last_break_time > 270.0:  # 4.5 hours
            # Need a break
            label.time += 45.0  # 45 minute break
            label.working_time += 45.0
            label.last_break_time = label.driving_time
        
        # Check daily limits
        if label.driving_time > max_driving or label.working_time > max_working:
            # Check if extensions are available
            if label.extensions_used < 2:  # Can extend
                label.extensions_used += 1
                return True
            else:
                return False  # Exceeded limits
        
        return True
    
    def _update_unreachable_tasks(self, label: LabelState, task: Task):
        """Update the set of unreachable tasks based on precedence."""
        # This is handled in _update_order_state for deliveries
        pass


class AdvancedLabelAlgorithm:
    """
    Main label correcting algorithm for EPDT with advanced features.
    
    This is the core algorithm that coordinates all components to solve
    the Elementary Shortest Path Problem with Resource Constraints.
    """
    
    def __init__(self, vehicle: Vehicle, orders: List[Order], travel_matrix: Dict[Tuple[str, str], Tuple[float, float]]):
        """
        Initialize the advanced label algorithm.
        
        Args:
            vehicle: Vehicle for route generation
            orders: Available orders
            travel_matrix: Travel times and costs between locations
        """
        self.vehicle = vehicle
        self.orders = orders
        self.travel_matrix = travel_matrix
        
        # Initialize components
        self.dominance_checker = EDDominanceChecker(orders)
        self.label_manager = LabelManager(self.dominance_checker)
        self.label_extender = LabelExtender(vehicle, orders, travel_matrix)
        
        # Algorithm parameters
        self.max_labels_per_location = 1000
        self.max_total_labels = 50000
        
    def solve(self, dual_values: Dict[str, float]) -> Optional[LabelState]:
        """
        Solve the ESPPRC using the label correcting algorithm.
        
        Args:
            dual_values: Dual values from Master Problem
            
        Returns:
            Best terminal label with negative reduced cost, or None
        """
        logger.info(f"🔍 Starting advanced label algorithm for vehicle {self.vehicle.id}")
        
        # Initialize with depot label
        initial_label = self._create_initial_label()
        self.label_manager.add_label(initial_label)
        
        # Label queue for processing
        label_queue = deque([initial_label])
        best_terminal_label = None
        
        processed_labels = 0
        while label_queue and processed_labels < self.max_total_labels:
            current_label = label_queue.popleft()
            processed_labels += 1
            
            # Try extending to all possible tasks
            for order in self.orders:
                for task in order.get_all_tasks():
                    extended_label = self.label_extender.extend_label_to_task(current_label, task)
                    
                    if extended_label is None:
                        continue
                    
                    # Calculate reduced cost
                    self._update_reduced_cost(extended_label, task, dual_values)
                    
                    # Add if not dominated
                    if self.label_manager.add_label(extended_label):
                        label_queue.append(extended_label)
            
            # Try returning to depot
            terminal_label = self._extend_to_depot(current_label, dual_values)
            if terminal_label and self._is_feasible_termination(terminal_label):
                if (best_terminal_label is None or 
                    terminal_label.cost < best_terminal_label.cost):
                    best_terminal_label = terminal_label
        
        # Log statistics
        stats = self.label_manager.get_statistics()
        logger.info(f"   📊 Processed {processed_labels} labels")
        logger.info(f"   📊 Generated {stats['labels_generated']}, pruned {stats['labels_pruned']}")
        logger.info(f"   📊 Pruning ratio: {stats['pruning_ratio']:.3f}")
        
        return best_terminal_label
    
    def _create_initial_label(self) -> LabelState:
        """Create initial label at depot."""
        # Get initial unreachable tasks (all deliveries)
        unreachable_tasks = set()
        for order in self.orders:
            for delivery_task in order.get_deliveries():
                unreachable_tasks.add(delivery_task.id)
        
        return LabelState(
            location_id=self.vehicle.depot_id,
            cost=0.0,
            time=0.0,
            weight_load=0.0,
            volume_load=0.0,
            path=[self.vehicle.depot_id],
            visited_tasks=set(),
            open_orders=set(),
            completed_orders=set(),
            unreachable_tasks=unreachable_tasks,
            cargo_stack=[],
            driving_time=0.0,
            working_time=0.0,
            last_break_time=0.0,
            extensions_used=0,
            current_day=0,
            day_start_time=0.0
        )
    
    def _update_reduced_cost(self, label: LabelState, task: Task, dual_values: Dict[str, float]):
        """Update label cost with reduced cost calculation."""
        # Subtract dual value for covering this order
        dual_value = dual_values.get(f"coverage_{task.order_id}", 0.0)
        label.cost -= dual_value
    
    def _extend_to_depot(self, label: LabelState, dual_values: Dict[str, float]) -> Optional[LabelState]:
        """Extend label back to depot."""
        if not self._can_return_to_depot(label):
            return None
        
        travel_time, travel_cost = self.travel_matrix.get(
            (label.location_id, self.vehicle.depot_id), (0.0, 0.0))
        
        # Calculate reduced cost including fleet constraint dual
        fleet_dual = dual_values.get(f"fleet_{self.vehicle.id}", 0.0)
        reduced_cost = travel_cost - fleet_dual
        
        terminal_label = label.copy()
        terminal_label.location_id = self.vehicle.depot_id
        terminal_label.cost += reduced_cost
        terminal_label.time += travel_time
        terminal_label.driving_time += travel_time
        terminal_label.working_time += travel_time
        terminal_label.path.append(self.vehicle.depot_id)
        terminal_label.weight_load = 0.0  # Unload at depot
        terminal_label.volume_load = 0.0
        terminal_label.cargo_stack = []
        
        return terminal_label
    
    def _can_return_to_depot(self, label: LabelState) -> bool:
        """Check if vehicle can return to depot."""
        # No open orders (all picked up goods have been delivered)
        return len(label.open_orders) == 0
    
    def _is_feasible_termination(self, label: LabelState) -> bool:
        """Check if label represents a feasible route termination."""
        return (label.location_id == self.vehicle.depot_id and
                len(label.open_orders) == 0 and
                len(label.cargo_stack) == 0)
