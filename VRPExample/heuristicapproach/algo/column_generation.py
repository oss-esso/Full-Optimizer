"""
Column Generation Algorithm for EPDT Optimality Bounds

This module implements the Column Generation approach described in Chapter 6 of the thesis
to find optimality bounds for the EPDT (Enhanced Pickup and Delivery with Time Windows) problem.

The algorithm decomposes the problem into:
1. Master Problem (MP): Set covering formulation with route selection
2. Pricing Problem (PP): Elementary Shortest Path Problem with Resource Constraints (ESPPRC)

Key Features:
- Set covering formulation for route selection
- Custom label correcting algorithm for ESPPRC
- Advanced dominance rules for multi-pickup, multi-delivery constraints
- Integration with enhanced EPDT data structures
- Support for multi-day planning and advanced constraints

References:
- Chapter 6: MILP Formulation for Optimality Bounds
- Desaulniers, G., & Villeneuve, D. (2000). The shortest path problem with time windows and linear waiting costs
"""

from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
import logging
from collections import defaultdict

# Linear programming solver imports
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("⚠️  Warning: PuLP not available. Install with: pip install pulp")

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

# Import EPDT data structures
from .epdt_data_structures import Order, Vehicle, Task, Route, Solution
from .master_problem import MasterProblem, ColumnGenerationResult
from .pricing_problem import PricingProblem, ESPPRCResult
from .second_level import calculate_z2_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ColumnGenerationConfig:
    """Configuration parameters for the Column Generation algorithm."""
    
    # Termination criteria
    max_iterations: int = 100
    optimality_gap_tolerance: float = 1e-6
    time_limit_seconds: float = 3600.0  # 1 hour
    
    # Solver preferences
    preferred_solver: str = "gurobi"  # "gurobi", "pulp", "cplex"
    solver_time_limit: float = 300.0  # 5 minutes per subproblem
    
    # Algorithm parameters
    initial_routes_strategy: str = "single_order"  # "single_order", "empty", "heuristic"
    pricing_heuristic: bool = True  # Use heuristic pricing before exact pricing
    
    # Debugging and output
    verbose: bool = True
    save_intermediate_solutions: bool = False
    log_dual_values: bool = False


@dataclass
class ColumnGenerationStats:
    """Statistics tracking for the Column Generation algorithm."""
    
    iterations: int = 0
    total_time: float = 0.0
    master_problem_time: float = 0.0
    pricing_problem_time: float = 0.0
    
    # Solution quality metrics
    best_lower_bound: float = float('-inf')
    best_upper_bound: float = float('inf')
    optimality_gap: float = float('inf')
    
    # Column generation metrics
    total_columns_generated: int = 0
    columns_per_iteration: List[int] = field(default_factory=list)
    
    # Solver statistics
    master_problem_status: str = "Not solved"
    pricing_problem_calls: int = 0
    
    def update_bounds(self, lower_bound: float, upper_bound: float = None):
        """Update bounds and calculate optimality gap."""
        self.best_lower_bound = max(self.best_lower_bound, lower_bound)
        if upper_bound is not None:
            self.best_upper_bound = min(self.best_upper_bound, upper_bound)
        
        if self.best_upper_bound != float('inf') and self.best_lower_bound != float('-inf'):
            self.optimality_gap = abs(self.best_upper_bound - self.best_lower_bound) / max(abs(self.best_upper_bound), 1e-10)


class ColumnGenerationSolver:
    """
    Main Column Generation solver for EPDT optimality bounds.
    
    This class coordinates the Master Problem and Pricing Problem to iteratively
    generate columns and improve the lower bound until optimality is reached.
    """
    
    def __init__(self, orders: List[Order], vehicles: List[Vehicle], 
                 config: ColumnGenerationConfig = None):
        """
        Initialize the Column Generation solver.
        
        Args:
            orders: List of orders to be scheduled
            vehicles: List of available vehicles
            config: Algorithm configuration parameters
        """
        self.orders = orders
        self.vehicles = vehicles
        self.config = config or ColumnGenerationConfig()
        
        # Initialize subproblems
        self.master_problem = MasterProblem(orders, vehicles, self.config)
        self.pricing_problems = {
            vehicle.id: PricingProblem(vehicle, orders, self.config)
            for vehicle in vehicles
        }
        
        # Initialize statistics
        self.stats = ColumnGenerationStats()
        
        # Current solution state
        self.current_routes: List[Route] = []
        self.dual_values: Dict[str, float] = {}
        
        logger.info(f"🏗️  Initialized Column Generation solver")
        logger.info(f"   📦 Orders: {len(orders)}")
        logger.info(f"   🚛 Vehicles: {len(vehicles)}")
        
    def solve(self) -> ColumnGenerationResult:
        """
        Main solving method using Column Generation.
        
        Returns:
            ColumnGenerationResult with optimality bounds and solution
        """
        logger.info(f"🚀 Starting Column Generation algorithm")
        start_time = time.time()
        
        try:
            # Phase 1: Initialize with basic feasible solution
            self._initialize_columns()
            
            # Phase 2: Column Generation main loop
            iteration = 0
            while iteration < self.config.max_iterations:
                iteration += 1
                self.stats.iterations = iteration
                
                if self.config.verbose:
                    logger.info(f"\n🔄 Iteration {iteration}")
                
                # Solve Master Problem
                mp_result = self._solve_master_problem()
                if not mp_result.feasible:
                    logger.error("❌ Master Problem infeasible")
                    break
                
                # Update bounds
                self.stats.update_bounds(mp_result.objective_value)
                self.dual_values = mp_result.dual_values
                
                # Check termination criteria
                if self._check_termination():
                    logger.info(f"✅ Optimality reached at iteration {iteration}")
                    break
                
                # Solve Pricing Problems
                new_columns = self._solve_pricing_problems()
                
                if not new_columns:
                    logger.info(f"✅ No improving columns found - optimal solution")
                    break
                
                # Add new columns to Master Problem
                self._add_columns_to_master(new_columns)
                self.stats.total_columns_generated += len(new_columns)
                self.stats.columns_per_iteration.append(len(new_columns))
                
                # Check time limit
                elapsed_time = time.time() - start_time
                if elapsed_time > self.config.time_limit_seconds:
                    logger.warning(f"⏰ Time limit reached after {elapsed_time:.1f} seconds")
                    break
            
            # Phase 3: Finalize solution
            final_result = self._finalize_solution()
            
        except Exception as e:
            logger.error(f"❌ Column Generation failed: {str(e)}")
            raise
        
        finally:
            self.stats.total_time = time.time() - start_time
            self._log_final_statistics()
        
        return final_result
    
    def _initialize_columns(self):
        """Initialize Master Problem with basic feasible columns."""
        logger.info(f"🏗️  Initializing columns...")
        
        if self.config.initial_routes_strategy == "single_order":
            # Create one route per order using shortest vehicle
            initial_routes = []
            for order in self.orders:
                # Find vehicle with minimum cost for this order
                best_vehicle = min(self.vehicles, key=lambda v: v.cost_per_km)
                route = Route(vehicle=best_vehicle)
                
                # Add all pickup tasks first, then delivery tasks
                for task in order.get_pickups():
                    route.tasks.append(task)
                for task in order.get_deliveries():
                    route.tasks.append(task)
                
                initial_routes.append(route)
                
        elif self.config.initial_routes_strategy == "empty":
            # Start with empty routes for each vehicle
            initial_routes = [Route(vehicle=vehicle) for vehicle in self.vehicles]
            
        else:  # "heuristic"
            # Use EPDT heuristic to generate initial routes
            try:
                from .first_level import l1_heuristic
                params = {'tabu_tenure': 5, 'M1': 10, 'M2': 50}
                solution = l1_heuristic(self.orders, self.vehicles, params)
                initial_routes = list(solution.routes.values())
            except ImportError:
                logger.warning("⚠️  EPDT heuristic not available, using single_order strategy")
                initial_routes = []
                # Fall back to single_order strategy
                for order in self.orders:
                    best_vehicle = min(self.vehicles, key=lambda v: v.cost_per_km)
                    route = Route(vehicle=best_vehicle)
                    for task in order.get_pickups():
                        route.tasks.append(task)
                    for task in order.get_deliveries():
                        route.tasks.append(task)
                    initial_routes.append(route)
        
        # Add initial routes to Master Problem
        for route in initial_routes:
            route_cost = calculate_z2_score(route)
            self.master_problem.add_column(route, route_cost)
        
        self.current_routes = initial_routes
        logger.info(f"   ✅ Added {len(initial_routes)} initial routes")
    
    def _solve_master_problem(self) -> Any:
        """Solve the Master Problem and return results."""
        start_time = time.time()
        
        try:
            result = self.master_problem.solve()
            self.stats.master_problem_time += time.time() - start_time
            self.stats.master_problem_status = "Optimal" if result.feasible else "Infeasible"
            
            if self.config.verbose and result.feasible:
                logger.info(f"   📊 MP Objective: {result.objective_value:.2f}")
                logger.info(f"   🚛 Routes selected: {len([r for r in result.route_selection.values() if r > 0.5])}")
                
            return result
            
        except Exception as e:
            logger.error(f"❌ Master Problem solving failed: {str(e)}")
            raise
    
    def _solve_pricing_problems(self) -> List[Route]:
        """Solve Pricing Problems for all vehicles and return improving routes."""
        start_time = time.time()
        new_routes = []
        
        for vehicle_id, pricing_problem in self.pricing_problems.items():
            self.stats.pricing_problem_calls += 1
            
            # Set dual values from Master Problem
            pricing_problem.set_dual_values(self.dual_values)
            
            # Solve ESPPRC
            result = pricing_problem.solve()
            
            if result.feasible and result.reduced_cost < -self.config.optimality_gap_tolerance:
                new_routes.append(result.route)
                if self.config.verbose:
                    logger.info(f"   🔍 Vehicle {vehicle_id}: New route with reduced cost {result.reduced_cost:.4f}")
        
        self.stats.pricing_problem_time += time.time() - start_time
        
        if self.config.verbose:
            logger.info(f"   ➕ Generated {len(new_routes)} new columns")
        
        return new_routes
    
    def _add_columns_to_master(self, new_routes: List[Route]):
        """Add new routes as columns to the Master Problem."""
        for route in new_routes:
            route_cost = calculate_z2_score(route)
            self.master_problem.add_column(route, route_cost)
            self.current_routes.append(route)
    
    def _check_termination(self) -> bool:
        """Check if termination criteria are met."""
        if self.stats.optimality_gap < self.config.optimality_gap_tolerance:
            return True
        
        # Add other termination criteria as needed
        return False
    
    def _finalize_solution(self) -> ColumnGenerationResult:
        """Create final solution result."""
        # Solve Master Problem one final time to get integer solution
        final_mp_result = self.master_problem.solve(integer=True)
        
        # Extract selected routes
        selected_routes = []
        if final_mp_result.feasible:
            for route_id, selection_value in final_mp_result.route_selection.items():
                if selection_value > 0.5:  # Selected route
                    route = self.current_routes[int(route_id)]
                    selected_routes.append(route)
        
        # Create solution object
        solution = Solution()
        for route in selected_routes:
            solution.add_route(route.vehicle.id, route)
        
        # Create result
        result = ColumnGenerationResult(
            feasible=final_mp_result.feasible,
            lower_bound=self.stats.best_lower_bound,
            upper_bound=final_mp_result.objective_value if final_mp_result.feasible else float('inf'),
            optimality_gap=self.stats.optimality_gap,
            solution=solution,
            statistics=self.stats,
            selected_routes=selected_routes
        )
        
        return result
    
    def _log_final_statistics(self):
        """Log comprehensive algorithm statistics."""
        logger.info(f"\n📊 Column Generation Statistics")
        logger.info(f"{'='*50}")
        logger.info(f"🔄 Total iterations: {self.stats.iterations}")
        logger.info(f"⏱️  Total time: {self.stats.total_time:.2f} seconds")
        logger.info(f"   Master Problem time: {self.stats.master_problem_time:.2f} seconds")
        logger.info(f"   Pricing Problem time: {self.stats.pricing_problem_time:.2f} seconds")
        logger.info(f"📈 Best lower bound: {self.stats.best_lower_bound:.2f}")
        logger.info(f"📉 Best upper bound: {self.stats.best_upper_bound:.2f}")
        logger.info(f"🎯 Optimality gap: {self.stats.optimality_gap:.6f}")
        logger.info(f"➕ Total columns generated: {self.stats.total_columns_generated}")
        logger.info(f"🔍 Pricing problem calls: {self.stats.pricing_problem_calls}")


def solve_epdt_with_column_generation(orders: List[Order], vehicles: List[Vehicle], 
                                    config: ColumnGenerationConfig = None) -> ColumnGenerationResult:
    """
    Convenience function to solve EPDT problem using Column Generation.
    
    Args:
        orders: List of orders to schedule
        vehicles: List of available vehicles  
        config: Algorithm configuration
        
    Returns:
        ColumnGenerationResult with optimality bounds and solution
    """
    solver = ColumnGenerationSolver(orders, vehicles, config)
    return solver.solve()


# Example usage and testing
if __name__ == "__main__":
    # This will be used for testing the implementation
    from .epdt_data_structures import TaskType
    
    # Create simple test problem
    test_orders = [
        Order(
            id="O1",
            pickup_tasks=[Task(
                id="P1", location_id="L1", task_type=TaskType.PICKUP,
                order_id="O1", lat=0.0, lon=0.0, service_time=10.0
            )],
            delivery_tasks=[Task(
                id="D1", location_id="L2", task_type=TaskType.DELIVERY,
                order_id="O1", lat=1.0, lon=1.0, service_time=10.0
            )]
        )
    ]
    
    test_vehicles = [
        Vehicle(
            id="V1", depot_id="DEPOT", weight_capacity=1000.0,
            volume_capacity=10.0, cost_per_km=1.0
        )
    ]
    
    # Test configuration
    test_config = ColumnGenerationConfig(
        max_iterations=10,
        verbose=True
    )
    
    print("🧪 Testing Column Generation implementation...")
    try:
        result = solve_epdt_with_column_generation(test_orders, test_vehicles, test_config)
        print(f"✅ Test completed successfully")
        print(f"   Lower bound: {result.lower_bound:.2f}")
        print(f"   Upper bound: {result.upper_bound:.2f}")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
