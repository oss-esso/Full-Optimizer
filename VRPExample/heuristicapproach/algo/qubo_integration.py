"""
QUBO Integration Utilities

This module provides integration between the Column Generation MILP formulation
and the QUBO formulation for quantum annealing. It enables a hybrid approach:

1. Use Column Generation to generate high-quality routes
2. Convert the problem to QUBO using the generated routes
3. Solve on quantum annealing hardware or classical QUBO solvers

This hybrid approach combines the strengths of both methods:
- Column Generation: Efficient route generation with complex constraints
- QUBO: Quantum advantage for combinatorial optimization

Key Features:
- Seamless integration between MILP and QUBO
- Route quality filtering and selection
- Performance comparison framework
- Quantum vs classical solver benchmarking
"""

from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
import logging
import json

# Import EPDT and optimization components
from .epdt_data_structures import Order, Vehicle, Route, Solution
from .column_generation import (
    ColumnGenerationSolver, ColumnGenerationConfig, 
    solve_epdt_with_column_generation
)
from .qubo_formulation import (
    QUBOFormulator, QUBOSolver, QUBOConfig, 
    solve_epdt_with_qubo, QUBOResult
)

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    """Configuration for hybrid Column Generation + QUBO approach."""
    
    # Column Generation phase
    cg_config: ColumnGenerationConfig = field(default_factory=ColumnGenerationConfig)
    use_column_generation: bool = True
    max_routes_for_qubo: int = 100  # Limit routes for QUBO complexity
    
    # QUBO phase
    qubo_config: QUBOConfig = field(default_factory=QUBOConfig)
    use_qubo_solver: bool = True
    
    # Route filtering
    route_quality_threshold: float = 0.1  # Filter routes within X% of best
    include_all_single_order_routes: bool = True  # Always include trivial routes
    
    # Performance comparison
    compare_with_milp: bool = True  # Compare QUBO result with MILP
    benchmark_solvers: bool = False  # Compare multiple QUBO solvers


@dataclass 
class HybridResult:
    """Result from hybrid Column Generation + QUBO approach."""
    
    # Column Generation results
    cg_feasible: bool
    cg_lower_bound: float
    cg_upper_bound: float
    cg_routes_generated: int
    cg_solve_time: float
    
    # QUBO results
    qubo_feasible: bool
    qubo_objective: float
    qubo_energy: float
    qubo_routes_used: int
    qubo_solve_time: float
    qubo_solver_info: Dict[str, Any]
    
    # Combined solution
    final_solution: Optional[Solution]
    selected_routes: List[Route]
    unassigned_orders: List[str]
    
    # Quality metrics
    optimality_gap: float = float('inf')  # vs Column Generation lower bound
    constraint_violations: Dict[str, int] = field(default_factory=dict)
    
    # Performance comparison
    total_solve_time: float = 0.0
    hybrid_vs_cg_ratio: float = 1.0  # QUBO objective / CG upper bound


class HybridSolver:
    """
    Hybrid solver combining Column Generation with QUBO for quantum annealing.
    
    This solver implements a two-phase approach:
    1. Column Generation to generate high-quality routes
    2. QUBO formulation and solving for final route selection
    """
    
    def __init__(self, orders: List[Order], vehicles: List[Vehicle], 
                 config: HybridConfig = None):
        """
        Initialize hybrid solver.
        
        Args:
            orders: List of orders to schedule
            vehicles: List of available vehicles
            config: Hybrid algorithm configuration
        """
        self.orders = orders
        self.vehicles = vehicles
        self.config = config or HybridConfig()
        
        # Generated routes and costs
        self.all_routes: List[Route] = []
        self.all_route_costs: List[float] = []
        self.filtered_routes: List[Route] = []
        self.filtered_costs: List[float] = []
        
        logger.info(f"🔀 Hybrid CG+QUBO solver initialized")
        logger.info(f"   📦 Orders: {len(orders)}")
        logger.info(f"   🚛 Vehicles: {len(vehicles)}")
    
    def solve(self) -> HybridResult:
        """
        Solve EPDT using hybrid Column Generation + QUBO approach.
        
        Returns:
            HybridResult with comprehensive solution information
        """
        logger.info(f"🚀 Starting Hybrid CG+QUBO solving")
        total_start_time = time.time()
        
        # Phase 1: Column Generation
        cg_result = self._solve_column_generation()
        
        # Phase 2: Route filtering and selection
        self._filter_routes(cg_result)
        
        # Phase 3: QUBO formulation and solving
        qubo_result = self._solve_qubo()
        
        # Phase 4: Solution construction and analysis
        final_result = self._construct_final_result(cg_result, qubo_result)
        final_result.total_solve_time = time.time() - total_start_time
        
        self._log_results(final_result)
        return final_result
    
    def _solve_column_generation(self) -> Any:
        """Solve using Column Generation to generate routes."""
        logger.info(f"📊 Phase 1: Column Generation route generation")
        
        if not self.config.use_column_generation:
            # Generate simple routes manually
            return self._generate_simple_routes()
        
        cg_start_time = time.time()
        
        try:
            cg_result = solve_epdt_with_column_generation(
                self.orders, self.vehicles, self.config.cg_config)
            
            cg_solve_time = time.time() - cg_start_time
            
            logger.info(f"   ✅ Column Generation completed")
            logger.info(f"   📊 Routes generated: {cg_result.statistics.total_columns_generated if cg_result.statistics else 0}")
            logger.info(f"   📊 Lower bound: {cg_result.lower_bound:.2f}")
            logger.info(f"   ⏱️  CG solve time: {cg_solve_time:.2f} seconds")
            
            # Extract routes from Column Generation
            if hasattr(cg_result, 'selected_routes') and cg_result.selected_routes:
                self.all_routes = cg_result.selected_routes
            else:
                # Generate routes from solution if available
                self.all_routes = self._extract_routes_from_cg_solution(cg_result)
            
            # Calculate route costs
            from .second_level import calculate_z2_score
            self.all_route_costs = [calculate_z2_score(route) for route in self.all_routes]
            
            return cg_result
            
        except Exception as e:
            logger.error(f"❌ Column Generation failed: {str(e)}")
            # Fallback to simple route generation
            return self._generate_simple_routes()
    
    def _generate_simple_routes(self) -> Any:
        """Generate simple routes as fallback."""
        logger.info(f"   🔄 Generating simple fallback routes")
        
        from .epdt_data_structures import Route
        from .second_level import calculate_z2_score
        
        # Create simple routes (one per order)
        for order in self.orders:
            for vehicle in self.vehicles:
                route = Route(vehicle=vehicle)
                
                # Add all pickup tasks first
                for task in order.get_pickups():
                    route.tasks.append(task)
                
                # Then all delivery tasks
                for task in order.get_deliveries():
                    route.tasks.append(task)
                
                self.all_routes.append(route)
                cost = calculate_z2_score(route)
                self.all_route_costs.append(cost)
        
        # Create mock CG result
        @dataclass
        class MockCGResult:
            feasible: bool = True
            lower_bound: float = 0.0
            upper_bound: float = sum(self.all_route_costs)
            optimality_gap: float = 0.0
            statistics: Any = None
        
        return MockCGResult()
    
    def _extract_routes_from_cg_solution(self, cg_result) -> List[Route]:
        """Extract routes from Column Generation solution."""
        routes = []
        
        if hasattr(cg_result, 'solution') and cg_result.solution:
            # Extract from solution object
            for vehicle_id, route in cg_result.solution.routes.items():
                if route.tasks:  # Only non-empty routes
                    routes.append(route)
        
        # If no routes extracted, generate simple ones
        if not routes:
            logger.warning("⚠️  No routes extracted from CG solution, generating simple routes")
            routes = self._generate_simple_routes_list()
        
        return routes
    
    def _generate_simple_routes_list(self) -> List[Route]:
        """Generate simple routes for each order."""
        routes = []
        
        for order in self.orders:
            # Use first available vehicle
            vehicle = self.vehicles[0] if self.vehicles else None
            if not vehicle:
                continue
            
            route = Route(vehicle=vehicle)
            
            # Add pickup tasks
            for task in order.get_pickups():
                route.tasks.append(task)
            
            # Add delivery tasks
            for task in order.get_deliveries():
                route.tasks.append(task)
            
            routes.append(route)
        
        return routes
    
    def _filter_routes(self, cg_result):
        """Filter and select routes for QUBO formulation."""
        logger.info(f"🔍 Phase 2: Route filtering for QUBO")
        
        if not self.all_routes:
            logger.error("❌ No routes available for filtering")
            return
        
        # Sort routes by cost
        route_cost_pairs = list(zip(self.all_routes, self.all_route_costs))
        route_cost_pairs.sort(key=lambda x: x[1])
        
        # Filter by quality threshold
        if route_cost_pairs:
            best_cost = route_cost_pairs[0][1]
            threshold = best_cost * (1 + self.config.route_quality_threshold)
            
            filtered_pairs = [
                (route, cost) for route, cost in route_cost_pairs 
                if cost <= threshold
            ]
        else:
            filtered_pairs = route_cost_pairs
        
        # Limit number of routes for QUBO complexity
        if len(filtered_pairs) > self.config.max_routes_for_qubo:
            filtered_pairs = filtered_pairs[:self.config.max_routes_for_qubo]
        
        # Ensure we have at least one route per order if possible
        if self.config.include_all_single_order_routes:
            self._ensure_order_coverage(filtered_pairs)
        
        # Extract filtered routes and costs
        self.filtered_routes = [route for route, cost in filtered_pairs]
        self.filtered_costs = [cost for route, cost in filtered_pairs]
        
        logger.info(f"   📊 Routes for QUBO: {len(self.filtered_routes)} (from {len(self.all_routes)} total)")
        
        if self.filtered_costs:
            avg_cost = sum(self.filtered_costs) / len(self.filtered_costs)
            logger.info(f"   📊 Average route cost: {avg_cost:.2f}")
    
    def _ensure_order_coverage(self, filtered_pairs: List[Tuple[Route, float]]):
        """Ensure each order has at least one covering route."""
        covered_orders = set()
        
        # Check which orders are covered by filtered routes
        for route, cost in filtered_pairs:
            for task in route.tasks:
                if hasattr(task, 'order_id'):
                    covered_orders.add(task.order_id)
        
        # Add simple routes for uncovered orders
        for order in self.orders:
            if order.id not in covered_orders:
                # Create simple route for this order
                vehicle = self.vehicles[0] if self.vehicles else None
                if vehicle:
                    route = Route(vehicle=vehicle)
                    for task in order.get_all_tasks():
                        route.tasks.append(task)
                    
                    from .second_level import calculate_z2_score
                    cost = calculate_z2_score(route)
                    filtered_pairs.append((route, cost))
    
    def _solve_qubo(self) -> QUBOResult:
        """Solve using QUBO formulation."""
        logger.info(f"⚛️  Phase 3: QUBO formulation and solving")
        
        if not self.config.use_qubo_solver:
            return self._create_mock_qubo_result()
        
        if not self.filtered_routes:
            logger.error("❌ No routes available for QUBO formulation")
            return self._create_mock_qubo_result()
        
        qubo_start_time = time.time()
        
        try:
            qubo_result = solve_epdt_with_qubo(
                self.orders, self.vehicles, 
                self.filtered_routes, self.filtered_costs,
                self.config.qubo_config
            )
            
            qubo_solve_time = time.time() - qubo_start_time
            qubo_result.solve_time = qubo_solve_time
            
            logger.info(f"   ✅ QUBO solved")
            logger.info(f"   📊 QUBO energy: {qubo_result.energy:.2f}")
            logger.info(f"   📊 Routes selected: {len(qubo_result.selected_routes)}")
            logger.info(f"   ⏱️  QUBO solve time: {qubo_solve_time:.2f} seconds")
            
            return qubo_result
            
        except Exception as e:
            logger.error(f"❌ QUBO solving failed: {str(e)}")
            return self._create_mock_qubo_result()
    
    def _create_mock_qubo_result(self) -> QUBOResult:
        """Create mock QUBO result as fallback."""
        return QUBOResult(
            feasible=False,
            objective_value=float('inf'),
            variable_assignment={},
            selected_routes=[],
            unassigned_orders=[],
            energy=float('inf'),
            solve_time=0.0,
            solver_info={'solver': 'mock'}
        )
    
    def _construct_final_result(self, cg_result, qubo_result: QUBOResult) -> HybridResult:
        """Construct final hybrid result."""
        logger.info(f"🔧 Phase 4: Solution construction and analysis")
        
        # Extract CG metrics
        cg_feasible = getattr(cg_result, 'feasible', False)
        cg_lower_bound = getattr(cg_result, 'lower_bound', 0.0)
        cg_upper_bound = getattr(cg_result, 'upper_bound', float('inf'))
        cg_routes_generated = len(self.all_routes)
        cg_solve_time = 0.0  # Would need to track separately
        
        # Extract QUBO metrics
        qubo_feasible = qubo_result.feasible
        qubo_objective = qubo_result.objective_value
        qubo_energy = qubo_result.energy
        
        # Build final solution
        final_solution = None
        selected_routes = []
        
        if qubo_feasible and qubo_result.selected_routes:
            # Map selected route IDs back to Route objects
            route_id_map = {f"route_{i}": route for i, route in enumerate(self.filtered_routes)}
            
            for route_id in qubo_result.selected_routes:
                if route_id in route_id_map:
                    selected_routes.append(route_id_map[route_id])
            
            # Create solution object
            if selected_routes:
                final_solution = Solution()
                for route in selected_routes:
                    final_solution.add_route(route.vehicle.id, route)
        
        # Calculate quality metrics
        optimality_gap = float('inf')
        if cg_lower_bound > 0 and qubo_objective < float('inf'):
            optimality_gap = abs(qubo_objective - cg_lower_bound) / cg_lower_bound
        
        hybrid_vs_cg_ratio = 1.0
        if cg_upper_bound < float('inf') and qubo_objective < float('inf'):
            hybrid_vs_cg_ratio = qubo_objective / cg_upper_bound
        
        return HybridResult(
            cg_feasible=cg_feasible,
            cg_lower_bound=cg_lower_bound,
            cg_upper_bound=cg_upper_bound,
            cg_routes_generated=cg_routes_generated,
            cg_solve_time=cg_solve_time,
            
            qubo_feasible=qubo_feasible,
            qubo_objective=qubo_objective,
            qubo_energy=qubo_energy,
            qubo_routes_used=len(selected_routes),
            qubo_solve_time=qubo_result.solve_time,
            qubo_solver_info=qubo_result.solver_info,
            
            final_solution=final_solution,
            selected_routes=selected_routes,
            unassigned_orders=qubo_result.unassigned_orders,
            
            optimality_gap=optimality_gap,
            constraint_violations=qubo_result.constraint_violations,
            hybrid_vs_cg_ratio=hybrid_vs_cg_ratio
        )
    
    def _log_results(self, result: HybridResult):
        """Log comprehensive results."""
        logger.info(f"\n📊 Hybrid CG+QUBO Results")
        logger.info(f"="*50)
        
        # Column Generation phase
        logger.info(f"🔢 Column Generation:")
        logger.info(f"   Feasible: {result.cg_feasible}")
        logger.info(f"   Lower bound: {result.cg_lower_bound:.2f}")
        logger.info(f"   Upper bound: {result.cg_upper_bound:.2f}")
        logger.info(f"   Routes generated: {result.cg_routes_generated}")
        
        # QUBO phase
        logger.info(f"⚛️  QUBO Phase:")
        logger.info(f"   Feasible: {result.qubo_feasible}")
        logger.info(f"   Objective: {result.qubo_objective:.2f}")
        logger.info(f"   Energy: {result.qubo_energy:.2f}")
        logger.info(f"   Routes used: {result.qubo_routes_used}")
        logger.info(f"   Solver: {result.qubo_solver_info.get('solver', 'unknown')}")
        
        # Quality assessment
        logger.info(f"🎯 Quality Assessment:")
        logger.info(f"   Optimality gap: {result.optimality_gap:.6f}")
        logger.info(f"   vs CG ratio: {result.hybrid_vs_cg_ratio:.3f}")
        logger.info(f"   Constraint violations: {sum(result.constraint_violations.values())}")
        
        # Performance
        logger.info(f"⏱️  Performance:")
        logger.info(f"   Total solve time: {result.total_solve_time:.2f} seconds")
        logger.info(f"   CG time: {result.cg_solve_time:.2f} seconds")
        logger.info(f"   QUBO time: {result.qubo_solve_time:.2f} seconds")


def solve_epdt_hybrid(orders: List[Order], vehicles: List[Vehicle], 
                     config: HybridConfig = None) -> HybridResult:
    """
    Convenience function to solve EPDT using hybrid CG+QUBO approach.
    
    Args:
        orders: List of orders to schedule
        vehicles: List of available vehicles
        config: Hybrid algorithm configuration
        
    Returns:
        HybridResult with comprehensive solution information
    """
    solver = HybridSolver(orders, vehicles, config)
    return solver.solve()


def benchmark_quantum_vs_classical(orders: List[Order], vehicles: List[Vehicle],
                                 configs: Dict[str, HybridConfig] = None) -> Dict[str, HybridResult]:
    """
    Benchmark quantum vs classical QUBO solvers.
    
    Args:
        orders: List of orders to schedule
        vehicles: List of available vehicles
        configs: Dictionary of solver_name -> HybridConfig
        
    Returns:
        Dictionary of solver_name -> HybridResult
    """
    if configs is None:
        # Default configurations for different solvers
        configs = {
            'neal_sa': HybridConfig(
                qubo_config=QUBOConfig(preferred_solver="neal", num_reads=1000)
            ),
            'exact': HybridConfig(
                qubo_config=QUBOConfig(preferred_solver="exact")
            )
        }
        
        # Add D-Wave if available
        try:
            import dwave.system
            configs['dwave'] = HybridConfig(
                qubo_config=QUBOConfig(preferred_solver="dwave", num_reads=1000)
            )
        except ImportError:
            pass
    
    logger.info(f"🏁 Benchmarking quantum vs classical QUBO solvers")
    logger.info(f"   Solvers: {list(configs.keys())}")
    
    results = {}
    
    for solver_name, config in configs.items():
        logger.info(f"\n🔍 Testing solver: {solver_name}")
        
        try:
            result = solve_epdt_hybrid(orders, vehicles, config)
            results[solver_name] = result
            
            logger.info(f"   ✅ {solver_name}: Objective = {result.qubo_objective:.2f}")
            
        except Exception as e:
            logger.error(f"   ❌ {solver_name} failed: {str(e)}")
    
    return results


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Hybrid CG+QUBO implementation...")
    
    # Check QUBO solver availability
    try:
        import dimod
        print("✅ DIMOD available")
    except ImportError:
        print("❌ DIMOD not available - install with: pip install dimod")
    
    try:
        import neal
        print("✅ Neal simulated annealing available")
    except ImportError:
        print("❌ Neal not available - install with: pip install dwave-neal")
    
    print("🎯 Hybrid CG+QUBO solver ready for integration")
