"""
QUBO Formulation for EPDT Quantum Annealing

This module implements the conversion of the EPDT MILP formulation into a 
Quadratic Unconstrained Binary Optimization (QUBO) model for quantum annealing hardware.

The QUBO formulation converts all constraints into quadratic penalty terms in the objective function,
creating a single polynomial to be minimized. This enables solving EPDT problems on:
- Quantum annealers (D-Wave, etc.)
- Classical QUBO solvers
- Simulated annealing implementations

Mathematical Foundation:
The total QUBO Hamiltonian is:
H = H_objective + H_penalty_tasks + H_penalty_fleet

Where:
- H_objective: Direct translation of MILP objective
- H_penalty_tasks: Quadratic penalties for task servicing constraints  
- H_penalty_fleet: Quadratic penalties for fleet size constraints

Key Features:
- Compatible with existing Column Generation MILP formulation
- Automatic penalty coefficient determination
- Efficient QUBO matrix construction
- Support for quantum annealer submission
- Classical QUBO solver integration
"""

from typing import List, Dict, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import numpy as np
import logging
from collections import defaultdict
import json

# Import EPDT components
from .epdt_data_structures import Order, Vehicle, Route
from .master_problem import MasterProblem
from .column_generation import ColumnGenerationResult

# Quantum/QUBO solver imports
try:
    import dimod
    DIMOD_AVAILABLE = True
except ImportError:
    DIMOD_AVAILABLE = False

try:
    import dwave.system
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False

try:
    import neal  # Simulated annealing solver
    NEAL_AVAILABLE = True
except ImportError:
    NEAL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class QUBOConfig:
    """Configuration parameters for QUBO formulation and solving."""
    
    # Penalty coefficients
    task_penalty_multiplier: float = 10.0  # Multiplier for task servicing penalty
    fleet_penalty_multiplier: float = 5.0  # Multiplier for fleet size penalty
    auto_penalty_calculation: bool = True  # Automatically calculate penalty coefficients
    
    # QUBO construction
    use_set_partitioning: bool = True  # Use set partitioning (exactly one) vs set covering (at least one)
    include_slack_variables: bool = True  # Include slack variables for fleet constraints
    
    # Solver preferences
    preferred_solver: str = "neal"  # "neal", "dwave", "tabu", "exact"
    num_reads: int = 1000  # Number of annealing reads
    chain_strength: float = 1.0  # Chain strength for quantum annealer
    
    # Performance tuning
    max_variables: int = 1000  # Maximum QUBO variables
    sparse_matrix: bool = True  # Use sparse matrix representation
    
    # Output options
    save_qubo_matrix: bool = False  # Save QUBO matrix to file
    verbose: bool = True


@dataclass
class QUBOVariable:
    """Represents a QUBO binary variable."""
    
    id: str  # Unique identifier
    type: str  # "route", "unassignment", "slack"
    route_id: Optional[str] = None  # Associated route (for route variables)
    order_id: Optional[str] = None  # Associated order (for unassignment variables)
    slack_index: Optional[int] = None  # Slack variable index
    coefficient: float = 0.0  # Linear coefficient in QUBO


@dataclass
class QUBOResult:
    """Result from solving QUBO formulation."""
    
    feasible: bool
    objective_value: float
    variable_assignment: Dict[str, int]  # variable_id -> {0, 1}
    selected_routes: List[str]  # Selected route IDs
    unassigned_orders: List[str]  # Unassigned order IDs
    energy: float  # QUBO energy (raw objective)
    solve_time: float
    solver_info: Dict[str, Any]
    
    # Quality metrics
    constraint_violations: Dict[str, int] = field(default_factory=dict)
    penalty_cost: float = 0.0
    original_objective: float = 0.0


class QUBOFormulator:
    """
    Converts EPDT MILP formulation to QUBO for quantum annealing.
    
    This class takes the Master Problem from Column Generation and converts
    all constraints into quadratic penalty terms in the objective function.
    """
    
    def __init__(self, orders: List[Order], vehicles: List[Vehicle], 
                 routes: List[Route], route_costs: List[float],
                 config: QUBOConfig = None):
        """
        Initialize QUBO formulator.
        
        Args:
            orders: List of orders to schedule
            vehicles: List of available vehicles
            routes: List of generated routes (columns from Column Generation)
            route_costs: Costs of corresponding routes
            config: QUBO configuration parameters
        """
        self.orders = orders
        self.vehicles = vehicles
        self.routes = routes
        self.route_costs = route_costs
        self.config = config or QUBOConfig()
        
        # Problem structure
        self.variables: List[QUBOVariable] = []
        self.variable_map: Dict[str, int] = {}  # variable_id -> index
        self.qubo_matrix: Optional[np.ndarray] = None
        
        # Route-order incidence matrix
        self.route_order_matrix: Dict[Tuple[int, str], bool] = {}  # (route_idx, order_id) -> covers
        
        # QUBO components
        self.h_objective: Dict[str, float] = {}  # Linear terms
        self.j_objective: Dict[Tuple[str, str], float] = {}  # Quadratic terms
        
        # Penalty coefficients
        self.task_penalty: float = 0.0
        self.fleet_penalty: float = 0.0
        
        self._build_problem_structure()
        
        logger.info(f"🏗️  QUBO Formulator initialized")
        logger.info(f"   📦 Orders: {len(orders)}")
        logger.info(f"   🚛 Vehicles: {len(vehicles)}")
        logger.info(f"   🛣️  Routes: {len(routes)}")
    
    def _build_problem_structure(self):
        """Build the problem structure and variable mapping."""
        # Build route-order incidence matrix
        for route_idx, route in enumerate(self.routes):
            covered_orders = set()
            for task in route.tasks:
                if hasattr(task, 'order_id'):
                    covered_orders.add(task.order_id)
            
            for order in self.orders:
                self.route_order_matrix[(route_idx, order.id)] = order.id in covered_orders
        
        # Create QUBO variables
        self._create_variables()
        
        logger.info(f"   🔢 QUBO variables: {len(self.variables)}")
    
    def _create_variables(self):
        """Create all QUBO binary variables."""
        variable_index = 0
        
        # Route selection variables (y_r)
        for route_idx, route in enumerate(self.routes):
            var_id = f"y_r{route_idx}"
            variable = QUBOVariable(
                id=var_id,
                type="route",
                route_id=f"route_{route_idx}",
                coefficient=self.route_costs[route_idx]
            )
            self.variables.append(variable)
            self.variable_map[var_id] = variable_index
            variable_index += 1
        
        # Order unassignment variables (x_o)
        for order in self.orders:
            var_id = f"x_o{order.id}"
            
            # Calculate unassignment penalty
            penalty = self._get_order_penalty(order)
            
            variable = QUBOVariable(
                id=var_id,
                type="unassignment",
                order_id=order.id,
                coefficient=penalty
            )
            self.variables.append(variable)
            self.variable_map[var_id] = variable_index
            variable_index += 1
        
        # Slack variables for fleet constraints (s_k)
        if self.config.include_slack_variables:
            max_fleet_size = len(self.vehicles)
            num_slack_bits = int(np.ceil(np.log2(max_fleet_size + 1))) if max_fleet_size > 0 else 1
            
            for k in range(num_slack_bits):
                var_id = f"s_k{k}"
                variable = QUBOVariable(
                    id=var_id,
                    type="slack",
                    slack_index=k,
                    coefficient=0.0  # No direct cost for slack variables
                )
                self.variables.append(variable)
                self.variable_map[var_id] = variable_index
                variable_index += 1
    
    def _get_order_penalty(self, order: Order) -> float:
        """Calculate penalty for not serving an order."""
        base_penalty = 1000.0
        
        # Check order properties
        if hasattr(order, 'is_mandatory') and order.is_mandatory:
            return base_penalty * 10.0  # Very high penalty for mandatory
        elif hasattr(order, 'is_urgent') and order.is_urgent:
            return base_penalty * 2.0   # Higher penalty for urgent
        elif hasattr(order, 'priority'):
            if order.priority == 'mandatory':
                return base_penalty * 10.0
            elif order.priority == 'urgent':
                return base_penalty * 2.0
            elif order.priority == 'normal':
                return 0.0  # No penalty for normal orders
        
        return base_penalty
    
    def formulate_qubo(self) -> np.ndarray:
        """
        Convert MILP to QUBO formulation.
        
        Returns:
            QUBO matrix Q where objective is x^T Q x
        """
        logger.info(f"🔬 Formulating QUBO...")
        
        # Step 1: Calculate penalty coefficients
        self._calculate_penalty_coefficients()
        
        # Step 2: Formulate objective term
        self._formulate_objective_term()
        
        # Step 3: Formulate task servicing penalties
        self._formulate_task_penalties()
        
        # Step 4: Formulate fleet size penalties
        self._formulate_fleet_penalties()
        
        # Step 5: Construct QUBO matrix
        self._construct_qubo_matrix()
        
        logger.info(f"   ✅ QUBO formulated with {len(self.variables)} variables")
        logger.info(f"   📊 Task penalty: {self.task_penalty:.1f}")
        logger.info(f"   📊 Fleet penalty: {self.fleet_penalty:.1f}")
        
        return self.qubo_matrix
    
    def _calculate_penalty_coefficients(self):
        """Calculate appropriate penalty coefficients."""
        if self.config.auto_penalty_calculation:
            # Calculate maximum possible objective value
            max_route_cost = max(self.route_costs) if self.route_costs else 1000.0
            max_objective = max_route_cost * len(self.routes)
            
            # Set penalties to be larger than maximum objective change
            self.task_penalty = max_objective * self.config.task_penalty_multiplier
            self.fleet_penalty = max_objective * self.config.fleet_penalty_multiplier
        else:
            # Use fixed penalties
            self.task_penalty = 10000.0 * self.config.task_penalty_multiplier
            self.fleet_penalty = 5000.0 * self.config.fleet_penalty_multiplier
    
    def _formulate_objective_term(self):
        """Formulate H_objective term."""
        # Add route costs (linear terms)
        for route_idx, cost in enumerate(self.route_costs):
            var_id = f"y_r{route_idx}"
            self.h_objective[var_id] = cost
        
        # Add order unassignment penalties (linear terms)
        for order in self.orders:
            var_id = f"x_o{order.id}"
            penalty = self._get_order_penalty(order)
            self.h_objective[var_id] = penalty
    
    def _formulate_task_penalties(self):
        """Formulate H_penalty_tasks term."""
        # For each order, add penalty: P_task * ((Σ_r a_or * y_r + x_o) - 1)^2
        
        for order in self.orders:
            # Find routes that cover this order
            covering_route_vars = []
            for route_idx, route in enumerate(self.routes):
                if self.route_order_matrix.get((route_idx, order.id), False):
                    covering_route_vars.append(f"y_r{route_idx}")
            
            unassignment_var = f"x_o{order.id}"
            
            # Expand (Σ_r a_or * y_r + x_o - 1)^2
            # = (Σ_r y_r + x_o)^2 - 2(Σ_r y_r + x_o) + 1
            # = Σ_r y_r^2 + x_o^2 + 2*Σ_r,s y_r*y_s + 2*Σ_r y_r*x_o - 2*Σ_r y_r - 2*x_o + 1
            # Since y_r^2 = y_r and x_o^2 = x_o for binary variables:
            # = Σ_r y_r + x_o + 2*Σ_r,s y_r*y_s + 2*Σ_r y_r*x_o - 2*Σ_r y_r - 2*x_o + 1
            # = 2*Σ_r,s y_r*y_s + 2*Σ_r y_r*x_o - Σ_r y_r - x_o + 1
            
            # Linear terms: -Σ_r y_r - x_o
            for route_var in covering_route_vars:
                if route_var not in self.h_objective:
                    self.h_objective[route_var] = 0.0
                self.h_objective[route_var] -= self.task_penalty
            
            if unassignment_var not in self.h_objective:
                self.h_objective[unassignment_var] = 0.0
            self.h_objective[unassignment_var] -= self.task_penalty
            
            # Quadratic terms: 2*Σ_r,s y_r*y_s (for r != s)
            for i, route_var1 in enumerate(covering_route_vars):
                for j, route_var2 in enumerate(covering_route_vars):
                    if i < j:  # Avoid double counting
                        key = (route_var1, route_var2)
                        if key not in self.j_objective:
                            self.j_objective[key] = 0.0
                        self.j_objective[key] += 2.0 * self.task_penalty
            
            # Quadratic terms: 2*Σ_r y_r*x_o
            for route_var in covering_route_vars:
                key = (route_var, unassignment_var) if route_var < unassignment_var else (unassignment_var, route_var)
                if key not in self.j_objective:
                    self.j_objective[key] = 0.0
                self.j_objective[key] += 2.0 * self.task_penalty
            
            # Constant term: +1 (handled separately)
    
    def _formulate_fleet_penalties(self):
        """Formulate H_penalty_fleet term."""
        if not self.config.include_slack_variables:
            return
        
        # Fleet constraint: Σ_r y_r + Σ_k (2^k * s_k) = |V|
        # Penalty: P_fleet * (Σ_r y_r + Σ_k (2^k * s_k) - |V|)^2
        
        fleet_size = len(self.vehicles)
        
        # Collect all variables in the constraint
        route_vars = [f"y_r{i}" for i in range(len(self.routes))]
        slack_vars = []
        slack_coeffs = []
        
        if self.config.include_slack_variables:
            max_fleet_size = len(self.vehicles)
            num_slack_bits = int(np.ceil(np.log2(max_fleet_size + 1))) if max_fleet_size > 0 else 1
            
            for k in range(num_slack_bits):
                slack_var = f"s_k{k}"
                slack_vars.append(slack_var)
                slack_coeffs.append(2**k)
        
        # Expand (Σ_r y_r + Σ_k (2^k * s_k) - |V|)^2
        
        # Linear terms: -2*|V|*(Σ_r y_r + Σ_k (2^k * s_k))
        for route_var in route_vars:
            if route_var not in self.h_objective:
                self.h_objective[route_var] = 0.0
            self.h_objective[route_var] -= 2.0 * fleet_size * self.fleet_penalty
        
        for slack_var, coeff in zip(slack_vars, slack_coeffs):
            if slack_var not in self.h_objective:
                self.h_objective[slack_var] = 0.0
            self.h_objective[slack_var] -= 2.0 * fleet_size * coeff * self.fleet_penalty
        
        # Quadratic terms: (Σ_r y_r + Σ_k (2^k * s_k))^2
        
        # Route-route interactions
        for i, route_var1 in enumerate(route_vars):
            for j, route_var2 in enumerate(route_vars):
                if i < j:
                    key = (route_var1, route_var2)
                    if key not in self.j_objective:
                        self.j_objective[key] = 0.0
                    self.j_objective[key] += 2.0 * self.fleet_penalty
        
        # Slack-slack interactions
        for i, (slack_var1, coeff1) in enumerate(zip(slack_vars, slack_coeffs)):
            for j, (slack_var2, coeff2) in enumerate(zip(slack_vars, slack_coeffs)):
                if i < j:
                    key = (slack_var1, slack_var2) if slack_var1 < slack_var2 else (slack_var2, slack_var1)
                    if key not in self.j_objective:
                        self.j_objective[key] = 0.0
                    self.j_objective[key] += 2.0 * coeff1 * coeff2 * self.fleet_penalty
        
        # Route-slack interactions
        for route_var in route_vars:
            for slack_var, coeff in zip(slack_vars, slack_coeffs):
                key = (route_var, slack_var) if route_var < slack_var else (slack_var, route_var)
                if key not in self.j_objective:
                    self.j_objective[key] = 0.0
                self.j_objective[key] += 2.0 * coeff * self.fleet_penalty
    
    def _construct_qubo_matrix(self):
        """Construct the final QUBO matrix."""
        n_vars = len(self.variables)
        
        if self.config.sparse_matrix:
            self.qubo_matrix = np.zeros((n_vars, n_vars))
        else:
            self.qubo_matrix = np.zeros((n_vars, n_vars))
        
        # Add linear terms (diagonal)
        for var_id, coeff in self.h_objective.items():
            if var_id in self.variable_map:
                i = self.variable_map[var_id]
                self.qubo_matrix[i, i] += coeff
        
        # Add quadratic terms (off-diagonal)
        for (var1_id, var2_id), coeff in self.j_objective.items():
            if var1_id in self.variable_map and var2_id in self.variable_map:
                i = self.variable_map[var1_id]
                j = self.variable_map[var2_id]
                # QUBO convention: upper triangular matrix
                if i <= j:
                    self.qubo_matrix[i, j] += coeff
                else:
                    self.qubo_matrix[j, i] += coeff
    
    def get_variable_mapping(self) -> Dict[int, str]:
        """Get mapping from matrix index to variable ID."""
        return {idx: var_id for var_id, idx in self.variable_map.items()}
    
    def get_qubo_info(self) -> Dict[str, Any]:
        """Get information about the QUBO formulation."""
        return {
            'num_variables': len(self.variables),
            'num_routes': len([v for v in self.variables if v.type == "route"]),
            'num_orders': len([v for v in self.variables if v.type == "unassignment"]),
            'num_slack': len([v for v in self.variables if v.type == "slack"]),
            'task_penalty': self.task_penalty,
            'fleet_penalty': self.fleet_penalty,
            'matrix_density': np.count_nonzero(self.qubo_matrix) / (self.qubo_matrix.size) if self.qubo_matrix is not None else 0.0
        }


class QUBOSolver:
    """
    Solver for QUBO formulations using various backends.
    
    Supports quantum annealers, classical QUBO solvers, and simulated annealing.
    """
    
    def __init__(self, config: QUBOConfig = None):
        """Initialize QUBO solver."""
        self.config = config or QUBOConfig()
        
    def solve(self, qubo_matrix: np.ndarray, formulator: QUBOFormulator) -> QUBOResult:
        """
        Solve QUBO formulation.
        
        Args:
            qubo_matrix: QUBO matrix Q
            formulator: QUBOFormulator for variable mapping
            
        Returns:
            QUBOResult with solution and metrics
        """
        import time
        start_time = time.time()
        
        logger.info(f"🔍 Solving QUBO with {self.config.preferred_solver}")
        
        try:
            if self.config.preferred_solver == "neal" and NEAL_AVAILABLE:
                result = self._solve_with_neal(qubo_matrix, formulator)
            elif self.config.preferred_solver == "dwave" and DWAVE_AVAILABLE:
                result = self._solve_with_dwave(qubo_matrix, formulator)
            elif self.config.preferred_solver == "exact":
                result = self._solve_exact(qubo_matrix, formulator)
            else:
                # Fallback to simulated annealing
                result = self._solve_with_simulated_annealing(qubo_matrix, formulator)
            
            result.solve_time = time.time() - start_time
            
            logger.info(f"   ✅ QUBO solved: feasible={result.feasible}")
            logger.info(f"   📊 Energy: {result.energy:.2f}")
            logger.info(f"   ⏱️  Solve time: {result.solve_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ QUBO solving failed: {str(e)}")
            raise
    
    def _solve_with_neal(self, qubo_matrix: np.ndarray, formulator: QUBOFormulator) -> QUBOResult:
        """Solve using Neal simulated annealing."""
        import neal
        
        # Convert to dimod BQM
        bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(qubo_matrix)
        
        # Solve with simulated annealing
        sampler = neal.SimulatedAnnealingSampler()
        response = sampler.sample(bqm, num_reads=self.config.num_reads)
        
        # Extract best solution
        best_sample = response.lowest().first
        
        return self._create_result_from_sample(best_sample, formulator, "neal")
    
    def _solve_with_dwave(self, qubo_matrix: np.ndarray, formulator: QUBOFormulator) -> QUBOResult:
        """Solve using D-Wave quantum annealer."""
        import dwave.system
        
        # Convert to dimod BQM
        bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(qubo_matrix)
        
        # Use D-Wave sampler
        sampler = dwave.system.DWaveSampler()
        composite_sampler = dwave.system.EmbeddingComposite(sampler)
        
        response = composite_sampler.sample(
            bqm, 
            num_reads=self.config.num_reads,
            chain_strength=self.config.chain_strength
        )
        
        # Extract best solution
        best_sample = response.lowest().first
        
        return self._create_result_from_sample(best_sample, formulator, "dwave")
    
    def _solve_exact(self, qubo_matrix: np.ndarray, formulator: QUBOFormulator) -> QUBOResult:
        """Solve exactly (for small problems only)."""
        if qubo_matrix.shape[0] > 20:
            raise ValueError("Exact solver only supports problems with ≤20 variables")
        
        import dimod
        
        # Convert to dimod BQM
        bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(qubo_matrix)
        
        # Solve exactly
        sampler = dimod.ExactSolver()
        response = sampler.sample(bqm)
        
        # Extract best solution
        best_sample = response.lowest().first
        
        return self._create_result_from_sample(best_sample, formulator, "exact")
    
    def _solve_with_simulated_annealing(self, qubo_matrix: np.ndarray, formulator: QUBOFormulator) -> QUBOResult:
        """Fallback simulated annealing implementation."""
        # Simple random search as fallback
        n_vars = qubo_matrix.shape[0]
        best_energy = float('inf')
        best_solution = None
        
        for _ in range(self.config.num_reads):
            # Random binary solution
            x = np.random.randint(0, 2, n_vars)
            
            # Calculate energy
            energy = x.T @ qubo_matrix @ x
            
            if energy < best_energy:
                best_energy = energy
                best_solution = x
        
        # Convert to result format
        variable_assignment = {}
        var_mapping = formulator.get_variable_mapping()
        
        for i, value in enumerate(best_solution):
            var_id = var_mapping[i]
            variable_assignment[var_id] = int(value)
        
        return self._create_result_from_assignment(variable_assignment, best_energy, formulator, "fallback")
    
    def _create_result_from_sample(self, sample, formulator: QUBOFormulator, solver_name: str) -> QUBOResult:
        """Create QUBOResult from dimod sample."""
        # Convert sample to variable assignment with proper variable IDs
        variable_assignment = {}
        var_mapping = formulator.get_variable_mapping()
        
        # Handle both integer keys (variable indices) and string keys (variable IDs)
        sample_dict = dict(sample.sample)
        
        for key, value in sample_dict.items():
            if isinstance(key, int):
                # Key is variable index, convert to variable ID
                if key in var_mapping:
                    var_id = var_mapping[key]
                    variable_assignment[var_id] = int(value)
            elif isinstance(key, str):
                # Key is already variable ID
                variable_assignment[key] = int(value)
        
        energy = sample.energy
        
        return self._create_result_from_assignment(variable_assignment, energy, formulator, solver_name)
    
    def _create_result_from_assignment(self, variable_assignment: Dict[str, int], 
                                     energy: float, formulator: QUBOFormulator,
                                     solver_name: str) -> QUBOResult:
        """Create QUBOResult from variable assignment."""
        # Extract selected routes and unassigned orders
        selected_routes = []
        unassigned_orders = []
        
        for var_id, value in variable_assignment.items():
            if value == 1:
                if var_id.startswith("y_r"):
                    route_idx = int(var_id[3:])
                    selected_routes.append(f"route_{route_idx}")
                elif var_id.startswith("x_o"):
                    order_id = var_id[3:]
                    unassigned_orders.append(order_id)
        
        # Calculate constraint violations and penalties
        violations = self._calculate_violations(variable_assignment, formulator)
        
        # Calculate original objective (without penalties)
        original_obj = 0.0
        for var_id, value in variable_assignment.items():
            if var_id.startswith("y_r") and value == 1:
                route_idx = int(var_id[3:])
                if route_idx < len(formulator.route_costs):
                    original_obj += formulator.route_costs[route_idx]
            elif var_id.startswith("x_o") and value == 1:
                for var in formulator.variables:
                    if var.id == var_id:
                        original_obj += var.coefficient
                        break
        
        return QUBOResult(
            feasible=sum(violations.values()) == 0,
            objective_value=original_obj,
            variable_assignment=variable_assignment,
            selected_routes=selected_routes,
            unassigned_orders=unassigned_orders,
            energy=energy,
            solve_time=0.0,  # Set by caller
            solver_info={'solver': solver_name},
            constraint_violations=violations,
            penalty_cost=energy - original_obj,
            original_objective=original_obj
        )
    
    def _calculate_violations(self, assignment: Dict[str, int], 
                            formulator: QUBOFormulator) -> Dict[str, int]:
        """Calculate constraint violations."""
        violations = {}
        
        # Check task servicing constraints
        for order in formulator.orders:
            # Count how many routes serve this order
            routes_serving = 0
            for route_idx, route in enumerate(formulator.routes):
                var_id = f"y_r{route_idx}"
                if assignment.get(var_id, 0) == 1:
                    if formulator.route_order_matrix.get((route_idx, order.id), False):
                        routes_serving += 1
            
            # Check if order is unassigned
            unassignment_var = f"x_o{order.id}"
            is_unassigned = assignment.get(unassignment_var, 0)
            
            # Violation if not exactly one of: served by route OR unassigned
            total_coverage = routes_serving + is_unassigned
            if formulator.config.use_set_partitioning:
                if total_coverage != 1:
                    violations[f"task_{order.id}"] = abs(total_coverage - 1)
            else:  # set covering
                if total_coverage < 1:
                    violations[f"task_{order.id}"] = 1 - total_coverage
        
        # Check fleet size constraints
        total_routes_used = sum(assignment.get(f"y_r{i}", 0) for i in range(len(formulator.routes)))
        if total_routes_used > len(formulator.vehicles):
            violations["fleet_size"] = total_routes_used - len(formulator.vehicles)
        
        return violations


def solve_epdt_with_qubo(orders: List[Order], vehicles: List[Vehicle], 
                        routes: List[Route], route_costs: List[float],
                        config: QUBOConfig = None) -> QUBOResult:
    """
    Convenience function to solve EPDT problem using QUBO formulation.
    
    Args:
        orders: List of orders to schedule
        vehicles: List of available vehicles
        routes: List of generated routes (from Column Generation)
        route_costs: Costs of corresponding routes
        config: QUBO configuration
        
    Returns:
        QUBOResult with solution and metrics
    """
    # Initialize formulator
    formulator = QUBOFormulator(orders, vehicles, routes, route_costs, config)
    
    # Formulate QUBO
    qubo_matrix = formulator.formulate_qubo()
    
    # Solve QUBO
    solver = QUBOSolver(config)
    result = solver.solve(qubo_matrix, formulator)
    
    return result


# Example usage and testing
if __name__ == "__main__":
    # This will be used for testing the implementation
    from .epdt_data_structures import TaskType, Task
    
    print("🧪 Testing QUBO Formulation implementation...")
    
    # Check available solvers
    print(f"📦 Available QUBO solvers:")
    print(f"   DIMOD: {DIMOD_AVAILABLE}")
    print(f"   Neal (simulated annealing): {NEAL_AVAILABLE}")
    print(f"   D-Wave: {DWAVE_AVAILABLE}")
    
    if not DIMOD_AVAILABLE:
        print("⚠️  Install dimod for QUBO support: pip install dimod")
    
    if NEAL_AVAILABLE:
        print("✅ Neal available for testing")
        
        # Create simple test problem
        test_orders = []
        test_vehicles = []
        test_routes = []
        test_costs = []
        
        print("🎯 QUBO formulation ready for integration with Column Generation")
    else:
        print("⚠️  Install neal for simulated annealing: pip install dwave-neal")
