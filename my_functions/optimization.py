"""
Optimization Module

This module provides various optimization methods including:
1. Classical optimization (Gurobi, scipy)
2. Simulated annealing
3. QUBO solvers
4. Food production optimization
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Callable, Tuple, Optional, Dict, Any, Union, List
from scipy.optimize import minimize
import random
import math
import logging
from dataclasses import dataclass
import time
import pulp
from enum import Enum

@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    solution: np.ndarray
    objective_value: float
    runtime: float
    status: str
    gap: Optional[float] = None
    optimal_params: Optional[np.ndarray] = None
    optimal_value: Optional[float] = None

class OptimizationObjective(Enum):
    """Types of optimization objectives."""
    NUTRITIONAL_VALUE = "nutritional_value"
    NUTRIENT_DENSITY = "nutrient_density"
    ENVIRONMENTAL_IMPACT = "environmental_impact"
    AFFORDABILITY = "affordability"
    PROFITABILITY = "profitability"
    SUSTAINABILITY = "sustainability"

class ClassicalOptimizer:
    """
    Base class for classical optimization methods.
    """
    
    def __init__(self, objective_function: Callable, 
                 initial_params: np.ndarray,
                 bounds: Optional[Tuple[float, float]] = None,
                 log_file: Optional[str] = None):
        """
        Initialize the optimizer.
        
        Args:
            objective_function: Function to minimize
            initial_params: Initial parameter values
            bounds: Optional parameter bounds
            log_file: Optional file to write logs to
        """
        self.objective_function = objective_function
        self.initial_params = initial_params
        self.bounds = bounds
        self.optimal_params = None
        self.optimal_value = None
        self.log_file = log_file
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for the optimizer."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file) if self.log_file else logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def optimize(self, method: str = 'L-BFGS-B', 
                options: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """
        Optimize using scipy's minimize function.
        
        Args:
            method: Optimization method to use
            options: Additional options for the optimizer
            
        Returns:
            OptimizationResult containing the solution and metrics
        """
        start_time = time.time()
        
        try:
            result = minimize(self.objective_function, 
                            self.initial_params,
                            method=method,
                            bounds=self.bounds,
                            options=options)
            
            self.optimal_params = result.x
            self.optimal_value = result.fun
            
            runtime = time.time() - start_time
            
            return OptimizationResult(
                solution=result.x,
                objective_value=result.fun,
                runtime=runtime,
                status=result.status,
                optimal_params=result.x,
                optimal_value=result.fun
            )
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise

class GurobiOptimizer:
    """
    Gurobi optimizer for solving QUBO problems.
    
    This class provides an interface to solve QUBO problems using Gurobi's
    mixed-integer quadratic programming solver.
    """
    
    def __init__(
        self,
        time_limit: Optional[float] = None,
        mip_gap: Optional[float] = None,
        threads: Optional[int] = None,
        log_file: Optional[str] = None
    ):
        """
        Initialize the Gurobi optimizer.
        
        Args:
            time_limit: Maximum time in seconds for the optimization
            mip_gap: Relative MIP gap tolerance
            threads: Number of threads to use
            log_file: File to write Gurobi log to
        """
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.threads = threads
        self.log_file = log_file
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for the optimizer."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file) if self.log_file else logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def solve_qubo(self, Q: np.ndarray) -> Tuple[np.ndarray, float]:
        """Solve QUBO problem and return solution/objective value."""
        result = self.optimize(Q=Q, c=None)
        return result.solution, result.objective_value
    
    def optimize(self, Q: np.ndarray, c: Optional[np.ndarray] = None) -> OptimizationResult:
        """
        Solve a QUBO problem using Gurobi.
        
        The QUBO problem is defined as:
            min x^T Q x + c^T x
            s.t. x ∈ {0,1}^n
        
        Args:
            Q: Quadratic coefficient matrix (n x n)
            c: Linear coefficient vector (n,)
            
        Returns:
            OptimizationResult containing the solution and metrics
        """
        start_time = time.time()
        
        try:
            # Create model
            model = gp.Model("qubo")
            
            # Set parameters
            if self.time_limit is not None:
                model.setParam('TimeLimit', self.time_limit)
            if self.mip_gap is not None:
                model.setParam('MIPGap', self.mip_gap)
            if self.threads is not None:
                model.setParam('Threads', self.threads)
            if self.log_file is not None:
                model.setParam('LogFile', self.log_file)
            
            # Create variables
            n = Q.shape[0]
            x = model.addVars(n, vtype=GRB.BINARY, name="x")
            
            # Set objective
            if c is None:
                c = np.zeros(n)
            
            # Quadratic terms
            quad_expr = gp.QuadExpr()
            for i in range(n):
                for j in range(n):
                    if Q[i,j] != 0:
                        quad_expr.add(x[i] * x[j] * Q[i,j])
            
            # Linear terms
            lin_expr = gp.LinExpr()
            for i in range(n):
                if c[i] != 0:
                    lin_expr.add(x[i] * c[i])
            
            # Set objective
            model.setObjective(quad_expr + lin_expr, GRB.MINIMIZE)
            
            # Optimize
            model.optimize()
            
            # Get solution
            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                solution = np.array([x[i].X for i in range(n)])
                objective_value = model.objVal
                gap = model.MIPGap if hasattr(model, 'MIPGap') else None
            else:
                raise RuntimeError(f"Optimization failed with status {model.status}")
            
            # Calculate runtime
            runtime = time.time() - start_time
            
            # Create result
            result = OptimizationResult(
                solution=solution,
                objective_value=objective_value,
                runtime=runtime,
                status=model.status,
                gap=gap
            )
            
            self.logger.info(f"Optimization completed in {runtime:.2f} seconds")
            self.logger.info(f"Objective value: {objective_value:.4f}")
            if gap is not None:
                self.logger.info(f"MIP gap: {gap:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise

class SimulatedAnnealing:
    """
    Simulated annealing optimizer for solving optimization problems.
    
    This class implements a simulated annealing algorithm that can be used
    for both general optimization problems and QUBO problems.
    """
    
    def __init__(
        self,
        objective_function: Callable,
        initial_params: np.ndarray,
        temperature: float = 1.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 1e-3,
        max_iterations: int = 1000,
        neighbor_function: Optional[Callable] = None,
        log_file: Optional[str] = None
    ):
        """
        Initialize simulated annealing.
        
        Args:
            objective_function: Function to minimize
            initial_params: Initial parameter values
            temperature: Initial temperature
            cooling_rate: Rate at which temperature decreases
            min_temperature: Minimum temperature to reach
            max_iterations: Maximum number of iterations
            neighbor_function: Optional function to generate neighbors
            log_file: Optional file to write logs to
        """
        self.objective_function = objective_function
        self.current_params = initial_params.copy()
        self.current_value = objective_function(initial_params)
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.neighbor_function = neighbor_function or self._default_neighbor_function
        self.log_file = log_file
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for the optimizer."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file) if self.log_file else logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _default_neighbor_function(self, params: np.ndarray) -> np.ndarray:
        """
        Default neighbor generation function.
        
        Args:
            params: Current parameters
            
        Returns:
            New parameter values
        """
        return params + np.random.normal(0, 0.1, size=params.shape)
    
    def optimize(self) -> OptimizationResult:
        """
        Run simulated annealing optimization.
        
        Returns:
            OptimizationResult containing the best solution found
        """
        start_time = time.time()
        
        try:
            self.optimal_params = self.current_params.copy()
            self.optimal_value = self.current_value
            
            iteration = 0
            while self.temperature > self.min_temperature and iteration < self.max_iterations:
                # Generate neighbor
                neighbor_params = self.neighbor_function(self.current_params)
                neighbor_value = self.objective_function(neighbor_params)
                
                # Accept or reject
                if neighbor_value < self.current_value:
                    self.current_params = neighbor_params
                    self.current_value = neighbor_value
                    
                    if neighbor_value < self.optimal_value:
                        self.optimal_params = neighbor_params
                        self.optimal_value = neighbor_value
                else:
                    # Metropolis acceptance criterion
                    acceptance_prob = math.exp((self.current_value - neighbor_value) / self.temperature)
                    if random.random() < acceptance_prob:
                        self.current_params = neighbor_params
                        self.current_value = neighbor_value
                
                # Cool down
                self.temperature *= self.cooling_rate
                iteration += 1
            
            # Calculate runtime
            runtime = time.time() - start_time
            
            # Create result
            result = OptimizationResult(
                solution=self.optimal_params,
                objective_value=self.optimal_value,
                runtime=runtime,
                status="COMPLETED",
                optimal_params=self.optimal_params,
                optimal_value=self.optimal_value
            )
            
            self.logger.info(f"Optimization completed in {runtime:.2f} seconds")
            self.logger.info(f"Objective value: {self.optimal_value:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise

class FoodProductionOptimizer:
    """Class for optimizing food production using multiple objectives."""
    
    def __init__(self, 
                 farms: List[str], 
                 foods: Dict[str, Dict[str, float]], 
                 food_groups: Dict[str, List[str]],
                 config: Optional[Dict] = None):
        """
        Initialize the multi-objective food production optimization model
        
        Args:
            farms: List of farm identifiers
            foods: Dictionary of foods with their scores
            food_groups: Mapping of food groups and their constituents
            config: Optional configuration dictionary
        """
        # Logging configuration
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Core configuration
        self.farms = farms
        self.foods = foods
        self.food_groups = food_groups
        self.config = config or {}
        
        # Model parameters
        self.parameters = self._generate_model_parameters()
        
        # Validation
        self._validate_inputs()
        
        # Initialize results storage
        self.results = []
    
    def _validate_inputs(self):
        """Validate input configurations and raise informative errors."""
        # Validate farms
        if not self.farms:
            raise ValueError("At least one farm must be provided")
        
        # Validate foods
        required_keys = [obj.value for obj in OptimizationObjective]
        
        for food, scores in self.foods.items():
            missing_keys = [key for key in required_keys if key not in scores]
            if missing_keys:
                raise ValueError(f"Missing keys for food {food}: {missing_keys}")
            
            # Validate score ranges
            for key, value in scores.items():
                if not 0 <= value <= 1:
                    raise ValueError(f"Invalid score range for {food}.{key}: {value}")
        
        # Validate food groups
        for group, foods in self.food_groups.items():
            unknown_foods = set(foods) - set(self.foods.keys())
            if unknown_foods:
                raise ValueError(f"Unknown foods in group {group}: {unknown_foods}")
    
    def _generate_model_parameters(self) -> Dict:
        """Generate model parameters with default values."""
        # Get configuration or use defaults
        config = self.config.get('parameters', {})
        
        return {
            'land_availability': config.get('land_availability', {
                farm: np.random.uniform(10, 100)  # hectares
                for farm in self.farms
            }),
            'minimum_planting_area': config.get('minimum_planting_area', {
                food: np.random.uniform(1, 5)  # minimum hectares
                for food in self.foods
            }),
            'food_group_constraints': config.get('food_group_constraints', {
                group: {
                    'min_foods': max(1, len(foods) // 2),
                    'max_foods': len(foods)
                }
                for group, foods in self.food_groups.items()
            }),
            'objective_weights': config.get('objective_weights', {
                obj.value: 1.0 / len(OptimizationObjective)
                for obj in OptimizationObjective
            }),
            'seasonal_factors': config.get('seasonal_factors', {
                food: np.random.uniform(0.8, 1.2)
                for food in self.foods
            }),
            'market_demand': config.get('market_demand', {
                food: np.random.uniform(0.5, 2.0)
                for food in self.foods
            })
        }
    
    def solve_optimization_problem(self, 
                                 timeout: Optional[float] = None) -> OptimizationResult:
        """Solve the multi-objective food production optimization problem."""
        try:
            start_time = time.time()
            
            # Create the linear programming problem
            prob = pulp.LpProblem("Multi-Objective Food Production", pulp.LpMaximize)
            
            # Decision Variables
            area_vars = pulp.LpVariable.dicts(
                "Area", 
                ((f, c) for f in self.farms for c in self.foods), 
                lowBound=0
            )
            
            plant_vars = pulp.LpVariable.dicts(
                "Plant", 
                ((f, c) for f in self.farms for c in self.foods), 
                cat='Binary'
            )
            
            # Objective Function
            objective_terms = []
            total_area_terms = []
            
            for f in self.farms:
                for c in self.foods:
                    # Weighted objective terms
                    weights = self.parameters['objective_weights']
                    seasonal_factor = self.parameters['seasonal_factors'][c]
                    
                    objective_terms.extend([
                        weights[obj.value] * 
                        self.foods[c][obj.value] * 
                        area_vars[f, c] * 
                        seasonal_factor
                        for obj in OptimizationObjective
                    ])
                    
                    total_area_terms.append(area_vars[f, c])
            
            # Set the objective
            prob += pulp.lpSum(objective_terms) / pulp.lpSum(total_area_terms)
            
            # Constraints
            self._add_land_availability_constraints(prob, area_vars)
            self._add_planting_area_constraints(prob, area_vars, plant_vars)
            self._add_food_group_constraints(prob, plant_vars)
            self._add_market_demand_constraints(prob, area_vars)
            
            # Solve the problem
            if timeout:
                prob.solve(pulp.PULP_CBC_CMD(timeLimit=timeout))
            else:
                prob.solve()
            
            # Calculate runtime
            runtime = time.time() - start_time
            
            # Extract solution
            solution = {
                (f, c): area_vars[f, c].varValue
                for f in self.farms
                for c in self.foods
                if area_vars[f, c].varValue > 0
            }
            
            # Calculate metrics
            metrics = self._calculate_metrics(prob, solution)
            
            # Create result object
            result = OptimizationResult(
                solution=np.array(list(solution.values())),
                objective_value=pulp.value(prob.objective),
                runtime=runtime,
                status=pulp.LpStatus[prob.status]
            )
            
            # Store result
            self.results.append(result)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise
    
    def _add_land_availability_constraints(self, prob, area_vars):
        """Add land availability constraints."""
        for f in self.farms:
            prob += pulp.lpSum(area_vars[f, c] for c in self.foods) <= \
                    self.parameters['land_availability'][f]
    
    def _add_planting_area_constraints(self, prob, area_vars, plant_vars):
        """Add minimum and maximum planting area constraints."""
        for f in self.farms:
            for c in self.foods:
                min_area = self.parameters['minimum_planting_area'][c]
                farm_land = self.parameters['land_availability'][f]
                
                prob += area_vars[f, c] >= min_area * plant_vars[f, c]
                prob += area_vars[f, c] <= farm_land * plant_vars[f, c]
    
    def _add_food_group_constraints(self, prob, plant_vars):
        """Add food group diversity constraints."""
        for g, group_foods in self.food_groups.items():
            constraints = self.parameters['food_group_constraints'][g]
            
            prob += pulp.lpSum(
                plant_vars[f, c] 
                for f in self.farms 
                for c in group_foods
            ) >= constraints['min_foods']
            
            prob += pulp.lpSum(
                plant_vars[f, c] 
                for f in self.farms 
                for c in group_foods
            ) <= constraints['max_foods']
    
    def _add_market_demand_constraints(self, prob, area_vars):
        """Add market demand constraints."""
        for c in self.foods:
            total_demand = self.parameters['market_demand'][c]
            prob += pulp.lpSum(area_vars[f, c] for f in self.farms) >= total_demand
    
    def _calculate_metrics(self, prob, solution) -> Dict[str, float]:
        """Calculate optimization metrics."""
        metrics = {}
        
        # Calculate objective contributions
        for obj in OptimizationObjective:
            metrics[obj.value] = sum(
                self.parameters['objective_weights'][obj.value] *
                self.foods[c][obj.value] *
                area
                for (f, c), area in solution.items()
            )
        
        # Calculate total area
        total_area = sum(solution.values())
        metrics['total_area'] = total_area
        
        # Calculate utilization
        for f in self.farms:
            farm_area = sum(
                area for (farm, _), area in solution.items() 
                if farm == f
            )
            metrics[f'utilization_{f}'] = (
                farm_area / self.parameters['land_availability'][f]
            )
        
        return metrics 