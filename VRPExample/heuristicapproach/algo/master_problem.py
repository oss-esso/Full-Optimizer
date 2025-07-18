"""
Master Problem Implementation for EPDT Column Generation

This module implements the Set Covering formulation for the Master Problem (MP)
in the Column Generation approach for EPDT optimality bounds.

The Master Problem:
- Variables: yr (route selection), xo (order unassignment)
- Objective: Minimize total cost (route costs + unassignment penalties)
- Constraints: Set covering (order coverage) + Fleet size limits

Mathematical Formulation:
    minimize: Σ(c_r * y_r) + Σ(p_o * x_o)
    subject to:
        Σ(a_or * y_r) + x_o >= 1    ∀ mandatory orders o
        Σ(a_or * y_r) + x_o >= 0    ∀ non-mandatory orders o  
        Σ(b_vr * y_r) <= 1          ∀ vehicles v
        y_r ∈ {0,1}                 ∀ routes r
        x_o ∈ {0,1}                 ∀ orders o

Where:
    c_r = cost of route r
    p_o = penalty for not serving order o
    a_or = 1 if route r serves order o, 0 otherwise
    b_vr = 1 if route r uses vehicle v, 0 otherwise
"""

from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

# Linear programming solver imports
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

from .epdt_data_structures import Order, Vehicle, Route

logger = logging.getLogger(__name__)


@dataclass
class MasterProblemResult:
    """Result from solving the Master Problem."""
    
    feasible: bool
    objective_value: float
    route_selection: Dict[str, float]  # route_id -> selection value (0-1)
    order_unassignment: Dict[str, float]  # order_id -> unassignment value (0-1)
    dual_values: Dict[str, float]  # constraint_name -> dual value
    solve_time: float
    solver_status: str


@dataclass 
class ColumnGenerationResult:
    """Final result from Column Generation algorithm."""
    
    feasible: bool
    lower_bound: float
    upper_bound: float
    optimality_gap: float
    solution: Optional[Any]  # Solution object
    statistics: Optional[Any]  # ColumnGenerationStats
    selected_routes: List[Route]


class MasterProblem:
    """
    Set Covering formulation for the Master Problem in Column Generation.
    
    This class manages the linear program that selects routes and handles
    order assignments with the objective of minimizing total cost.
    """
    
    def __init__(self, orders: List[Order], vehicles: List[Vehicle], config):
        """
        Initialize the Master Problem.
        
        Args:
            orders: List of orders to be scheduled
            vehicles: List of available vehicles  
            config: ColumnGenerationConfig with algorithm parameters
        """
        self.orders = orders
        self.vehicles = vehicles
        self.config = config
        
        # Problem structure
        self.routes: List[Route] = []
        self.route_costs: List[float] = []
        self.route_order_coverage: List[Set[str]] = []  # Which orders each route covers
        self.route_vehicle_usage: List[str] = []  # Which vehicle each route uses
        
        # Order properties
        self.mandatory_orders: Set[str] = set()
        self.order_penalties: Dict[str, float] = {}
        self._analyze_orders()
        
        # Solver objects (will be created when needed)
        self.lp_model = None
        self.route_vars: Dict[str, Any] = {}  # route_id -> variable
        self.unassignment_vars: Dict[str, Any] = {}  # order_id -> variable
        
        # Constraint tracking for dual values
        self.coverage_constraints: Dict[str, Any] = {}  # order_id -> constraint
        self.fleet_constraints: Dict[str, Any] = {}  # vehicle_id -> constraint
        
        logger.info(f"🏗️  Master Problem initialized")
        logger.info(f"   📦 Orders: {len(orders)} ({len(self.mandatory_orders)} mandatory)")
        logger.info(f"   🚛 Vehicles: {len(vehicles)}")
    
    def _analyze_orders(self):
        """Analyze orders to identify mandatory orders and penalties."""
        for order in self.orders:
            # Check if order is mandatory
            is_mandatory = getattr(order, 'is_mandatory', True)
            if is_mandatory:
                self.mandatory_orders.add(order.id)
            
            # Calculate unassignment penalty
            priority = getattr(order, 'priority', 1)
            base_penalty = 1000.0  # Default penalty
            
            if hasattr(order, 'is_mandatory') and order.is_mandatory:
                penalty = base_penalty * 10.0  # Very high penalty for mandatory
            elif hasattr(order, 'is_urgent') and order.is_urgent:
                penalty = base_penalty * 2.0   # Higher penalty for urgent  
            elif priority == 'mandatory':
                penalty = base_penalty * 10.0
            elif priority == 'urgent':
                penalty = base_penalty * 2.0
            elif priority == 'normal':
                penalty = 0.0  # No penalty for normal orders
            else:
                penalty = base_penalty
            
            self.order_penalties[order.id] = penalty
    
    def add_column(self, route: Route, cost: float):
        """
        Add a new route (column) to the Master Problem.
        
        Args:
            route: Route object representing the column
            cost: Cost coefficient for this route
        """
        route_id = f"route_{len(self.routes)}"
        
        # Store route information
        self.routes.append(route)
        self.route_costs.append(cost)
        self.route_vehicle_usage.append(route.vehicle.id)
        
        # Determine which orders this route covers
        covered_orders = set()
        for task in route.tasks:
            if hasattr(task, 'order_id'):
                covered_orders.add(task.order_id)
        self.route_order_coverage.append(covered_orders)
        
        # Add variable to existing model if it exists
        if self.lp_model is not None:
            self._add_route_variable(route_id, cost, covered_orders, route.vehicle.id)
        
        logger.debug(f"   ➕ Added route {route_id}: cost={cost:.2f}, orders={covered_orders}")
    
    def solve(self, integer: bool = False) -> MasterProblemResult:
        """
        Solve the Master Problem.
        
        Args:
            integer: Whether to solve as integer program (True) or LP relaxation (False)
            
        Returns:
            MasterProblemResult with solution details
        """
        import time
        start_time = time.time()
        
        try:
            # Build or rebuild the model
            self._build_model(integer)
            
            # Solve the model
            if GUROBI_AVAILABLE and self.config.preferred_solver == "gurobi":
                result = self._solve_with_gurobi()
            elif PULP_AVAILABLE:
                result = self._solve_with_pulp()
            else:
                raise RuntimeError("No LP solver available. Install PuLP or Gurobi.")
            
            result.solve_time = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"❌ Master Problem solving failed: {str(e)}")
            raise
    
    def _build_model(self, integer: bool = False):
        """Build the linear programming model."""
        if GUROBI_AVAILABLE and self.config.preferred_solver == "gurobi":
            self._build_gurobi_model(integer)
        elif PULP_AVAILABLE:
            self._build_pulp_model(integer)
        else:
            raise RuntimeError("No LP solver available")
    
    def _build_gurobi_model(self, integer: bool = False):
        """Build model using Gurobi."""
        self.lp_model = gp.Model("EPDT_Master_Problem")
        self.lp_model.setParam('OutputFlag', 0)  # Suppress output
        
        # Decision variables
        var_type = GRB.BINARY if integer else GRB.CONTINUOUS
        
        # Route selection variables y_r
        self.route_vars = {}
        for i, route in enumerate(self.routes):
            route_id = f"route_{i}"
            var = self.lp_model.addVar(
                vtype=var_type,
                lb=0.0,
                ub=1.0,
                obj=self.route_costs[i],
                name=f"y_{route_id}"
            )
            self.route_vars[route_id] = var
        
        # Order unassignment variables x_o
        self.unassignment_vars = {}
        for order in self.orders:
            var = self.lp_model.addVar(
                vtype=var_type,
                lb=0.0,
                ub=1.0,
                obj=self.order_penalties[order.id],
                name=f"x_{order.id}"
            )
            self.unassignment_vars[order.id] = var
        
        # Set covering constraints: Σ(a_or * y_r) + x_o >= b_o
        self.coverage_constraints = {}
        for order in self.orders:
            # Find routes that cover this order
            covering_routes = []
            for i, covered_orders in enumerate(self.route_order_coverage):
                if order.id in covered_orders:
                    route_id = f"route_{i}"
                    covering_routes.append(self.route_vars[route_id])
            
            # Right-hand side: 1 for mandatory orders, 0 for optional
            rhs = 1.0 if order.id in self.mandatory_orders else 0.0
            
            constraint = self.lp_model.addConstr(
                gp.quicksum(covering_routes) + self.unassignment_vars[order.id] >= rhs,
                name=f"coverage_{order.id}"
            )
            self.coverage_constraints[order.id] = constraint
        
        # Fleet size constraints: Σ(b_vr * y_r) <= 1
        self.fleet_constraints = {}
        for vehicle in self.vehicles:
            # Find routes that use this vehicle
            vehicle_routes = []
            for i, vehicle_id in enumerate(self.route_vehicle_usage):
                if vehicle_id == vehicle.id:
                    route_id = f"route_{i}"
                    vehicle_routes.append(self.route_vars[route_id])
            
            if vehicle_routes:  # Only add constraint if vehicle has routes
                constraint = self.lp_model.addConstr(
                    gp.quicksum(vehicle_routes) <= 1,
                    name=f"fleet_{vehicle.id}"
                )
                self.fleet_constraints[vehicle.id] = constraint
        
        self.lp_model.update()
    
    def _build_pulp_model(self, integer: bool = False):
        """Build model using PuLP."""
        self.lp_model = pulp.LpProblem("EPDT_Master_Problem", pulp.LpMinimize)
        
        # Decision variables
        var_type = pulp.LpBinary if integer else pulp.LpContinuous
        
        # Route selection variables
        self.route_vars = {}
        for i, route in enumerate(self.routes):
            route_id = f"route_{i}"
            var = pulp.LpVariable(
                f"y_{route_id}",
                lowBound=0.0,
                upBound=1.0,
                cat=var_type
            )
            self.route_vars[route_id] = var
        
        # Order unassignment variables
        self.unassignment_vars = {}
        for order in self.orders:
            var = pulp.LpVariable(
                f"x_{order.id}",
                lowBound=0.0,
                upBound=1.0,
                cat=var_type
            )
            self.unassignment_vars[order.id] = var
        
        # Objective function
        objective = 0
        for i, cost in enumerate(self.route_costs):
            route_id = f"route_{i}"
            objective += cost * self.route_vars[route_id]
        
        for order in self.orders:
            objective += self.order_penalties[order.id] * self.unassignment_vars[order.id]
        
        self.lp_model += objective
        
        # Set covering constraints
        self.coverage_constraints = {}
        for order in self.orders:
            # Find routes that cover this order
            covering_expr = 0
            for i, covered_orders in enumerate(self.route_order_coverage):
                if order.id in covered_orders:
                    route_id = f"route_{i}"
                    covering_expr += self.route_vars[route_id]
            
            covering_expr += self.unassignment_vars[order.id]
            
            # Right-hand side
            rhs = 1 if order.id in self.mandatory_orders else 0
            
            constraint = covering_expr >= rhs
            self.lp_model += constraint
            self.coverage_constraints[order.id] = constraint
        
        # Fleet size constraints
        self.fleet_constraints = {}
        for vehicle in self.vehicles:
            vehicle_expr = 0
            for i, vehicle_id in enumerate(self.route_vehicle_usage):
                if vehicle_id == vehicle.id:
                    route_id = f"route_{i}"
                    vehicle_expr += self.route_vars[route_id]
            
            if vehicle_expr != 0:  # Only add if vehicle has routes
                constraint = vehicle_expr <= 1
                self.lp_model += constraint
                self.fleet_constraints[vehicle.id] = constraint
    
    def _solve_with_gurobi(self) -> MasterProblemResult:
        """Solve using Gurobi and extract results."""
        self.lp_model.optimize()
        
        # Extract solution
        feasible = self.lp_model.status == GRB.OPTIMAL
        objective_value = self.lp_model.objVal if feasible else float('inf')
        
        # Extract variable values
        route_selection = {}
        for route_id, var in self.route_vars.items():
            route_selection[route_id] = var.x if feasible else 0.0
        
        order_unassignment = {}
        for order_id, var in self.unassignment_vars.items():
            order_unassignment[order_id] = var.x if feasible else 0.0
        
        # Extract dual values (only for continuous relaxation)
        dual_values = {}
        if feasible and self.lp_model.isMIP == 0:  # LP relaxation
            for order_id, constraint in self.coverage_constraints.items():
                dual_values[f"coverage_{order_id}"] = constraint.pi
            for vehicle_id, constraint in self.fleet_constraints.items():
                dual_values[f"fleet_{vehicle_id}"] = constraint.pi
        
        return MasterProblemResult(
            feasible=feasible,
            objective_value=objective_value,
            route_selection=route_selection,
            order_unassignment=order_unassignment,
            dual_values=dual_values,
            solve_time=0.0,  # Will be set by caller
            solver_status=str(self.lp_model.status)
        )
    
    def _solve_with_pulp(self) -> MasterProblemResult:
        """Solve using PuLP and extract results."""
        self.lp_model.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract solution
        feasible = self.lp_model.status == pulp.LpStatusOptimal
        objective_value = pulp.value(self.lp_model.objective) if feasible else float('inf')
        
        # Extract variable values
        route_selection = {}
        for route_id, var in self.route_vars.items():
            route_selection[route_id] = pulp.value(var) if feasible else 0.0
        
        order_unassignment = {}
        for order_id, var in self.unassignment_vars.items():
            order_unassignment[order_id] = pulp.value(var) if feasible else 0.0
        
        # PuLP doesn't provide easy access to dual values
        dual_values = {}
        
        return MasterProblemResult(
            feasible=feasible,
            objective_value=objective_value,
            route_selection=route_selection,
            order_unassignment=order_unassignment,
            dual_values=dual_values,
            solve_time=0.0,  # Will be set by caller
            solver_status=pulp.LpStatus[self.lp_model.status]
        )
    
    def _add_route_variable(self, route_id: str, cost: float, 
                          covered_orders: Set[str], vehicle_id: str):
        """Add a new route variable to existing model (for incremental column generation)."""
        # This would be used for dynamic column addition
        # Implementation depends on the specific solver and is more complex
        pass
    
    def get_problem_size(self) -> Dict[str, int]:
        """Get statistics about the problem size."""
        return {
            'routes': len(self.routes),
            'orders': len(self.orders),
            'vehicles': len(self.vehicles),
            'mandatory_orders': len(self.mandatory_orders),
            'coverage_constraints': len(self.orders),
            'fleet_constraints': len(self.vehicles)
        }
