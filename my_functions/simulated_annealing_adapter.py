"""
Simulated Annealing Adapter for Food Production Optimization

This module provides a simulated annealing adapter that converts the food production 
optimization problem for use with classical simulated annealing algorithms. It integrates 
with the existing QUBO converter and Benders decomposition framework to enable 
classical annealing approaches as an alternative to quantum methods.

Key Features:
- Classical simulated annealing for QUBO problems
- Multiple cooling schedules (linear, exponential, logarithmic, adaptive)
- Comprehensive parameter tuning and configuration
- Integration with existing Benders decomposition approach
- Performance monitoring and convergence tracking
- Binary and Ising problem support
- Multiple neighborhood search strategies

Dependencies:
- numpy: Numerical computing
- logging: Error tracking and debugging
- scipy: Additional optimization utilities (optional)
"""

import numpy as np
import logging
import time
import math
import random
import traceback
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

# Optional scipy import for advanced features
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    logging.warning("scipy not available. Some advanced features will be disabled.")
    SCIPY_AVAILABLE = False

# Import existing QUBO infrastructure
try:
    from .qubo_converter import QUBOConverter, convert_benders_master_to_qubo
    from .mean_field_base import qubo_to_ising
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from my_functions.qubo_converter import QUBOConverter, convert_benders_master_to_qubo
        from my_functions.mean_field_base import qubo_to_ising
    except ImportError:
        logging.error("Could not import QUBO converter. Some functionality may be limited.")
        QUBOConverter = None
        convert_benders_master_to_qubo = None
        qubo_to_ising = None


class CoolingSchedule(Enum):
    """Types of cooling schedules for simulated annealing."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    ADAPTIVE = "adaptive"
    BOLTZMANN = "boltzmann"
    CAUCHY = "cauchy"


class NeighborhoodType(Enum):
    """Types of neighborhood search strategies."""
    SINGLE_FLIP = "single_flip"  # Flip one bit
    MULTI_FLIP = "multi_flip"    # Flip multiple bits
    SWAP = "swap"                # Swap two bits
    RANDOM_WALK = "random_walk"  # Random perturbation
    GUIDED = "guided"            # Objective-guided search


@dataclass
class SimulatedAnnealingConfig:
    """Configuration parameters for simulated annealing."""
    
    # Temperature schedule
    initial_temperature: float = 100.0
    min_temperature: float = 1e-6
    cooling_schedule: CoolingSchedule = CoolingSchedule.EXPONENTIAL
    cooling_rate: float = 0.95
    
    # Iteration control
    max_iterations: int = 10000
    max_iterations_per_temperature: int = 100
    max_stagnation_iterations: int = 1000
    
    # Neighborhood search
    neighborhood_type: NeighborhoodType = NeighborhoodType.SINGLE_FLIP
    flip_probability: float = 0.1  # For multi-flip neighborhood
    max_flips: int = 5  # Maximum number of bits to flip
    
    # Convergence criteria
    tolerance: float = 1e-8
    relative_tolerance: float = 1e-6
    target_energy: Optional[float] = None
    
    # Adaptive parameters
    adaptive_cooling: bool = True
    reheat_threshold: float = 0.1  # Reheat if acceptance rate drops below this
    reheat_factor: float = 2.0
    
    # Advanced features
    use_tabu_list: bool = False
    tabu_list_size: int = 100
    use_restart: bool = True
    restart_threshold: int = 1000  # Restart after this many stagnant iterations
    
    # Logging and debugging
    log_interval: int = 100
    track_history: bool = True
    random_seed: Optional[int] = None


class SimulatedAnnealingAdapter:
    """
    Simulated annealing adapter for food production optimization problems.
    
    This class provides an interface between the existing QUBO conversion framework
    and classical simulated annealing algorithms, enabling classical optimization
    approaches for the food production problem.
    """
    
    def __init__(self, config: Optional[SimulatedAnnealingConfig] = None, 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the simulated annealing adapter.
        
        Args:
            config: Simulated annealing configuration parameters
            logger: Logger instance for debugging and monitoring
        """
        self.config = config or SimulatedAnnealingConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Set random seed if specified
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            random.seed(self.config.random_seed)
        
        # Performance metrics
        self.metrics = {
            'total_iterations': 0,
            'accepted_moves': 0,
            'rejected_moves': 0,
            'temperature_reductions': 0,
            'reheats': 0,
            'restarts': 0,
            'best_energies': [],
            'temperatures': [],
            'acceptance_rates': [],
            'convergence_history': []
        }
        
        # State tracking
        self.current_solution = None
        self.current_energy = None
        self.best_solution = None
        self.best_energy = float('inf')
        self.temperature = self.config.initial_temperature
        
        # Tabu list for avoiding cycles
        self.tabu_list = [] if self.config.use_tabu_list else None
        
        # History tracking
        self.energy_history = [] if self.config.track_history else None
        self.solution_history = [] if self.config.track_history else None
    
    def solve_benders_master_with_sa(self,
                                     f_coeffs: np.ndarray,
                                     D_matrix: np.ndarray,
                                     d_vector: np.ndarray,
                                     optimality_cuts: List,
                                     feasibility_cuts: List,
                                     B_matrix: Optional[np.ndarray] = None,
                                     b_vector: Optional[np.ndarray] = None,
                                     Ny: int = None,
                                     config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Solve Benders master problem using simulated annealing.
        
        This method converts the Benders master problem to QUBO form and solves it
        using classical simulated annealing.
        
        Args:
            f_coeffs: Objective coefficients for binary variables
            D_matrix: Constraint matrix for binary variables
            d_vector: RHS vector for binary constraints
            optimality_cuts: List of optimality cuts from previous iterations
            feasibility_cuts: List of feasibility cuts from previous iterations
            B_matrix: Optional constraint matrix linking continuous and binary variables
            b_vector: Optional RHS vector for linking constraints
            Ny: Number of binary variables
            config: QUBO conversion configuration parameters
            
        Returns:
            Dictionary containing solution, objective value, and metrics
        """
        start_time = time.time()
        
        try:
            # Convert Benders master problem to QUBO
            self.logger.info("Converting Benders master problem to QUBO for simulated annealing...")
            
            if config is None:
                config = {
                    "eta_min": -1000.0,
                    "eta_max": 1000.0,
                    "eta_num_bits": 6,
                    "penalty_coefficient": 10000.0,
                    "penalty_slack_num_bits": 4
                }
            
            qubo_model = convert_benders_master_to_qubo(
                f_coeffs, D_matrix, d_vector, optimality_cuts, feasibility_cuts,
                B_matrix, b_vector, Ny, config, self.logger
            )
            
            if qubo_model is None:
                return {"error": "Failed to convert problem to QUBO format"}
            
            # Extract QUBO components
            Q_matrix = qubo_model.Q
            c_vector = qubo_model.c
            offset = qubo_model.offset
            
            # Incorporate linear terms into Q matrix diagonal
            Q_full = Q_matrix.copy()
            np.fill_diagonal(Q_full, np.diag(Q_full) + c_vector)
            
            problem_size = Q_full.shape[0]
            self.logger.info(f"QUBO problem size: {problem_size} variables")
            
            # Create objective function for simulated annealing
            def qubo_objective(x):
                return x.T @ Q_full @ x + offset
            
            # Solve using simulated annealing
            result = self._solve_qubo_sa(Q_full, offset, problem_size)
            
            if result.get('error'):
                return result
            
            # Map back to original problem variables
            sa_solution = result['solution']
            original_solution = np.zeros(Ny)
            
            if hasattr(qubo_model, 'reverse_mapping'):
                for qubo_idx, orig_idx in qubo_model.reverse_mapping.items():
                    if qubo_idx < len(sa_solution) and orig_idx < Ny:
                        original_solution[orig_idx] = sa_solution[qubo_idx]
            else:
                # Direct mapping for first Ny variables
                copy_size = min(Ny, len(sa_solution))
                original_solution[:copy_size] = sa_solution[:copy_size]
            
            # Calculate objective value
            objective_value = float(f_coeffs.T.dot(original_solution.reshape(-1, 1))[0, 0])
            
            # Update metrics
            wall_time = time.time() - start_time
            
            self.logger.info(f"Simulated annealing solution found in {wall_time:.2f}s "
                           f"with energy {result['energy']:.6f}")
            
            return {
                'solution': original_solution,
                'objective': objective_value,
                'energy': result['energy'],
                'iterations': result['iterations'],
                'final_temperature': result['final_temperature'],
                'acceptance_rate': result['acceptance_rate'],
                'wall_time': wall_time,
                'problem_size': problem_size,
                'qubo_matrix': Q_full,
                'sa_info': {
                    'cooling_schedule': self.config.cooling_schedule.value,
                    'neighborhood_type': self.config.neighborhood_type.value,
                    'final_temperature': result['final_temperature'],
                    'total_iterations': result['iterations']
                },
                'metrics': self._get_performance_metrics(),
                'convergence_history': result.get('convergence_history', [])
            }
            
        except Exception as e:
            self.logger.error(f"Error solving with simulated annealing: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def _solve_qubo_sa(self, Q_matrix: np.ndarray, offset: float, 
                       problem_size: int) -> Dict[str, Any]:
        """
        Solve a QUBO problem using simulated annealing.
        
        Args:
            Q_matrix: QUBO coefficient matrix
            offset: Constant offset term
            problem_size: Number of binary variables
            
        Returns:
            Dictionary with solution and performance data
        """
        try:
            # Initialize solution randomly
            current_solution = np.random.randint(0, 2, size=problem_size)
            current_energy = self._evaluate_energy(current_solution, Q_matrix, offset)
            
            # Track best solution
            best_solution = current_solution.copy()
            best_energy = current_energy
            
            # Initialize temperature
            temperature = self.config.initial_temperature
            
            # Tracking variables
            iteration = 0
            stagnation_count = 0
            accepted_moves = 0
            total_moves = 0
            convergence_history = []
            
            self.logger.info(f"Starting simulated annealing with T={temperature:.4f}, "
                           f"initial energy={current_energy:.6f}")
            
            while (temperature > self.config.min_temperature and 
                   iteration < self.config.max_iterations and
                   stagnation_count < self.config.max_stagnation_iterations):
                
                # Perform moves at current temperature
                temp_accepted = 0
                temp_total = 0
                
                for _ in range(self.config.max_iterations_per_temperature):
                    # Generate neighbor
                    neighbor_solution = self._generate_neighbor(current_solution)
                    
                    # Skip if in tabu list
                    if self._is_tabu(neighbor_solution):
                        continue
                    
                    neighbor_energy = self._evaluate_energy(neighbor_solution, Q_matrix, offset)
                    
                    # Accept or reject move
                    if self._accept_move(current_energy, neighbor_energy, temperature):
                        current_solution = neighbor_solution
                        current_energy = neighbor_energy
                        temp_accepted += 1
                        accepted_moves += 1
                        
                        # Update tabu list
                        if self.config.use_tabu_list:
                            self._update_tabu_list(current_solution)
                        
                        # Check if new best
                        if current_energy < best_energy:
                            best_solution = current_solution.copy()
                            best_energy = current_energy
                            stagnation_count = 0
                            
                            self.logger.debug(f"New best energy: {best_energy:.6f} "
                                            f"at iteration {iteration}")
                    
                    temp_total += 1
                    total_moves += 1
                    iteration += 1
                    
                    # Check convergence
                    if self._check_convergence(best_energy):
                        self.logger.info(f"Converged at iteration {iteration}")
                        break
                
                # Calculate acceptance rate for this temperature
                acceptance_rate = temp_accepted / max(temp_total, 1)
                
                # Log progress
                if iteration % self.config.log_interval == 0:
                    self.logger.info(f"Iter {iteration}: T={temperature:.6f}, "
                                   f"E={current_energy:.6f}, Best={best_energy:.6f}, "
                                   f"AccRate={acceptance_rate:.3f}")
                
                # Store convergence data
                if self.config.track_history:
                    convergence_history.append({
                        'iteration': iteration,
                        'temperature': temperature,
                        'current_energy': current_energy,
                        'best_energy': best_energy,
                        'acceptance_rate': acceptance_rate
                    })
                
                # Update temperature
                temperature = self._update_temperature(temperature, iteration, acceptance_rate)
                
                # Check for reheat
                if (self.config.adaptive_cooling and 
                    acceptance_rate < self.config.reheat_threshold):
                    temperature *= self.config.reheat_factor
                    self.metrics['reheats'] += 1
                    self.logger.debug(f"Reheating to T={temperature:.6f}")
                
                stagnation_count += 1
                
                # Restart if needed
                if (self.config.use_restart and 
                    stagnation_count >= self.config.restart_threshold):
                    current_solution = np.random.randint(0, 2, size=problem_size)
                    current_energy = self._evaluate_energy(current_solution, Q_matrix, offset)
                    temperature = self.config.initial_temperature
                    stagnation_count = 0
                    self.metrics['restarts'] += 1
                    self.logger.info(f"Restarting at iteration {iteration}")
            
            # Update final metrics
            final_acceptance_rate = accepted_moves / max(total_moves, 1)
            self.metrics['total_iterations'] += iteration
            self.metrics['accepted_moves'] += accepted_moves
            self.metrics['rejected_moves'] += (total_moves - accepted_moves)
            
            self.logger.info(f"Simulated annealing completed: {iteration} iterations, "
                           f"final energy={best_energy:.6f}, "
                           f"acceptance rate={final_acceptance_rate:.3f}")
            
            return {
                'solution': best_solution,
                'energy': best_energy,
                'iterations': iteration,
                'final_temperature': temperature,
                'acceptance_rate': final_acceptance_rate,
                'convergence_history': convergence_history
            }
            
        except Exception as e:
            self.logger.error(f"Error during simulated annealing: {e}")
            return {'error': str(e)}
    
    def _evaluate_energy(self, solution: np.ndarray, Q_matrix: np.ndarray, 
                        offset: float) -> float:
        """Evaluate the energy of a solution."""
        return float(solution.T @ Q_matrix @ solution + offset)
    
    def _generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        """Generate a neighbor solution based on the neighborhood type."""
        neighbor = solution.copy()
        
        if self.config.neighborhood_type == NeighborhoodType.SINGLE_FLIP:
            # Flip a single random bit
            idx = np.random.randint(0, len(solution))
            neighbor[idx] = 1 - neighbor[idx]
            
        elif self.config.neighborhood_type == NeighborhoodType.MULTI_FLIP:
            # Flip multiple bits with given probability
            for i in range(len(solution)):
                if np.random.random() < self.config.flip_probability:
                    neighbor[i] = 1 - neighbor[i]
                    
        elif self.config.neighborhood_type == NeighborhoodType.SWAP:
            # Swap two random bits
            if len(solution) >= 2:
                idx1, idx2 = np.random.choice(len(solution), 2, replace=False)
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                
        elif self.config.neighborhood_type == NeighborhoodType.RANDOM_WALK:
            # Random number of flips (1 to max_flips)
            num_flips = np.random.randint(1, min(self.config.max_flips + 1, len(solution) + 1))
            indices = np.random.choice(len(solution), num_flips, replace=False)
            for idx in indices:
                neighbor[idx] = 1 - neighbor[idx]
        
        return neighbor
    
    def _accept_move(self, current_energy: float, neighbor_energy: float, 
                     temperature: float) -> bool:
        """Determine whether to accept a move using Metropolis criterion."""
        if neighbor_energy <= current_energy:
            return True
        else:
            delta_energy = neighbor_energy - current_energy
            acceptance_prob = math.exp(-delta_energy / temperature) if temperature > 0 else 0
            return np.random.random() < acceptance_prob
    
    def _update_temperature(self, temperature: float, iteration: int, 
                          acceptance_rate: float) -> float:
        """Update temperature according to cooling schedule."""
        if self.config.cooling_schedule == CoolingSchedule.LINEAR:
            return temperature - (self.config.initial_temperature - self.config.min_temperature) / self.config.max_iterations
            
        elif self.config.cooling_schedule == CoolingSchedule.EXPONENTIAL:
            return temperature * self.config.cooling_rate
            
        elif self.config.cooling_schedule == CoolingSchedule.LOGARITHMIC:
            return self.config.initial_temperature / (1 + self.config.cooling_rate * math.log(1 + iteration))
            
        elif self.config.cooling_schedule == CoolingSchedule.BOLTZMANN:
            return self.config.initial_temperature / (1 + iteration)
            
        elif self.config.cooling_schedule == CoolingSchedule.CAUCHY:
            return self.config.initial_temperature / (1 + iteration) ** 0.5
            
        elif self.config.cooling_schedule == CoolingSchedule.ADAPTIVE:
            # Adjust cooling rate based on acceptance rate
            if acceptance_rate > 0.5:
                return temperature * 0.99  # Cool faster
            elif acceptance_rate < 0.1:
                return temperature * 0.999  # Cool slower
            else:
                return temperature * self.config.cooling_rate
        
        return temperature * self.config.cooling_rate
    
    def _check_convergence(self, best_energy: float) -> bool:
        """Check if convergence criteria are met."""
        if self.config.target_energy is not None and best_energy <= self.config.target_energy:
            return True
        return False
    
    def _is_tabu(self, solution: np.ndarray) -> bool:
        """Check if solution is in tabu list."""
        if not self.config.use_tabu_list or self.tabu_list is None:
            return False
        
        solution_str = ''.join(map(str, solution))
        return solution_str in self.tabu_list
    
    def _update_tabu_list(self, solution: np.ndarray):
        """Update tabu list with new solution."""
        if not self.config.use_tabu_list or self.tabu_list is None:
            return
        
        solution_str = ''.join(map(str, solution))
        self.tabu_list.append(solution_str)
        
        # Maintain tabu list size
        if len(self.tabu_list) > self.config.tabu_list_size:
            self.tabu_list.pop(0)
    
    def solve_food_optimization_with_sa(self, optimizer: Any) -> Dict[str, Any]:
        """
        Solve the complete food production optimization problem using simulated annealing.
        
        This method integrates with the existing food production optimizer framework
        and uses simulated annealing to solve the optimization problem.
        
        Args:
            optimizer: FoodProductionOptimizer instance
            
        Returns:
            Dictionary containing solution, metrics, and performance data
        """
        try:
            self.logger.info("Starting simulated annealing optimization for food production")
            
            # Extract problem dimensions
            F = len(optimizer.farms)
            C = len(optimizer.foods)
            Ny = F * C  # Binary variables
            
            # Build objective coefficients
            f = np.zeros((Ny, 1))
            for fi, farm in enumerate(optimizer.farms):
                for food_idx, food in enumerate(optimizer.foods):
                    pos = fi * C + food_idx
                    food_data = optimizer.foods[food]
                    weights = optimizer.parameters.get('weights', {})
                    
                    # Calculate objective score
                    pos_score = (
                        weights.get('nutritional_value', 0.2) * food_data.get('nutritional_value', 0) +
                        weights.get('nutrient_density', 0.2) * food_data.get('nutrient_density', 0) +
                        weights.get('affordability', 0.2) * food_data.get('affordability', 0) +
                        weights.get('sustainability', 0.2) * food_data.get('sustainability', 0)
                    )
                    neg_score = weights.get('environmental_impact', 0.2) * food_data.get('environmental_impact', 0)
                    
                    f[pos, 0] = -(pos_score - neg_score)  # Negative for minimization
            
            # Build constraints for binary variables only (simplified)
            # Constraint: each farm must select at least 1 food
            D = np.zeros((F, Ny))
            d = np.ones((F, 1))
            
            for fi in range(F):
                for food_idx in range(C):
                    pos = fi * C + food_idx
                    D[fi, pos] = 1
            
            # Convert to QUBO and solve with simulated annealing
            result = self.solve_benders_master_with_sa(
                f_coeffs=f,
                D_matrix=D,
                d_vector=d,
                optimality_cuts=[],
                feasibility_cuts=[],
                Ny=Ny,
                config={
                    "eta_min": -500.0,
                    "eta_max": 500.0,
                    "eta_num_bits": 5,
                    "penalty_coefficient": 5000.0,
                    "penalty_slack_num_bits": 3
                }
            )
            
            if result.get('error'):
                return result
            
            # Convert binary solution to food allocation
            y_solution = result['solution']
            solution = {}
            
            for i, val in enumerate(y_solution):
                if val > 0.5:  # Binary variable is active
                    farm_idx = i // C
                    food_idx = i % C
                    
                    if farm_idx < len(optimizer.farms) and food_idx < len(list(optimizer.foods.keys())):
                        farm = optimizer.farms[farm_idx]
                        food = list(optimizer.foods.keys())[food_idx]
                        
                        # Assign a reasonable land allocation
                        land_available = optimizer.parameters['land_availability'].get(farm, 100)
                        allocation = min(50, land_available * 0.3)  # 30% of available land
                        solution[(farm, food)] = allocation
            
            # Calculate metrics
            total_land = sum(solution.values())
            num_foods = len(solution)
            
            metrics = {
                'total_land_used': total_land,
                'num_food_selections': num_foods,
                'avg_land_per_food': total_land / max(num_foods, 1),
                'sa_energy': result['energy'],
                'sa_iterations': result['iterations'],
                'sa_acceptance_rate': result['acceptance_rate'],
                'final_temperature': result['final_temperature']
            }
            
            return {
                'solution': solution,
                'objective_value': -result['objective'],  # Convert back to maximization
                'metrics': metrics,
                'sa_result': result,
                'performance_metrics': self._get_performance_metrics()
            }
            
        except Exception as e:
            self.logger.error(f"Error in simulated annealing food optimization: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            'total_iterations': self.metrics['total_iterations'],
            'accepted_moves': self.metrics['accepted_moves'],
            'rejected_moves': self.metrics['rejected_moves'],
            'acceptance_rate': (self.metrics['accepted_moves'] / 
                               max(self.metrics['accepted_moves'] + self.metrics['rejected_moves'], 1)),
            'temperature_reductions': self.metrics['temperature_reductions'],
            'reheats': self.metrics['reheats'],
            'restarts': self.metrics['restarts'],
            'best_energy': min(self.metrics['best_energies']) if self.metrics['best_energies'] else float('inf'),
            'avg_temperature': np.mean(self.metrics['temperatures']) if self.metrics['temperatures'] else 0,
            'cooling_schedule': self.config.cooling_schedule.value,
            'neighborhood_type': self.config.neighborhood_type.value
        }
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = {
            'total_iterations': 0,
            'accepted_moves': 0,
            'rejected_moves': 0,
            'temperature_reductions': 0,
            'reheats': 0,
            'restarts': 0,
            'best_energies': [],
            'temperatures': [],
            'acceptance_rates': [],
            'convergence_history': []
        }
        self.logger.info("Performance metrics reset")
    
    def tune_parameters(self, Q_matrix: np.ndarray, offset: float = 0.0,
                       num_trials: int = 10) -> Dict[str, Any]:
        """
        Automatically tune simulated annealing parameters for a given problem.
        
        Args:
            Q_matrix: QUBO coefficient matrix
            offset: Constant offset term
            num_trials: Number of tuning trials
            
        Returns:
            Dictionary with optimal parameters and performance data
        """
        try:
            self.logger.info(f"Starting parameter tuning with {num_trials} trials")
            
            # Parameter ranges to test
            cooling_rates = [0.9, 0.95, 0.99]
            initial_temps = [10.0, 100.0, 1000.0]
            neighborhood_types = [NeighborhoodType.SINGLE_FLIP, NeighborhoodType.MULTI_FLIP]
            
            best_config = None
            best_energy = float('inf')
            tuning_results = []
            
            for cooling_rate in cooling_rates:
                for initial_temp in initial_temps:
                    for neighborhood_type in neighborhood_types:
                        # Create test configuration
                        test_config = SimulatedAnnealingConfig(
                            initial_temperature=initial_temp,
                            cooling_rate=cooling_rate,
                            neighborhood_type=neighborhood_type,
                            max_iterations=1000,  # Reduced for tuning
                            log_interval=1000  # Reduce logging during tuning
                        )
                        
                        # Run trials
                        trial_energies = []
                        for trial in range(num_trials):
                            test_adapter = SimulatedAnnealingAdapter(config=test_config, logger=self.logger)
                            result = test_adapter._solve_qubo_sa(Q_matrix, offset, Q_matrix.shape[0])
                            
                            if not result.get('error'):
                                trial_energies.append(result['energy'])
                        
                        if trial_energies:
                            avg_energy = np.mean(trial_energies)
                            std_energy = np.std(trial_energies)
                            min_energy = min(trial_energies)
                            
                            tuning_results.append({
                                'cooling_rate': cooling_rate,
                                'initial_temperature': initial_temp,
                                'neighborhood_type': neighborhood_type.value,
                                'avg_energy': avg_energy,
                                'std_energy': std_energy,
                                'min_energy': min_energy,
                                'num_successful_trials': len(trial_energies)
                            })
                            
                            # Update best configuration
                            if min_energy < best_energy:
                                best_energy = min_energy
                                best_config = test_config
                            
                            self.logger.debug(f"Config test: cooling={cooling_rate}, "
                                            f"temp={initial_temp}, neighborhood={neighborhood_type.value}, "
                                            f"avg_energy={avg_energy:.6f}")
            
            self.logger.info(f"Parameter tuning completed. Best energy: {best_energy:.6f}")
            
            return {
                'best_config': best_config,
                'best_energy': best_energy,
                'tuning_results': tuning_results,
                'num_configurations_tested': len(tuning_results)
            }
            
        except Exception as e:
            self.logger.error(f"Error during parameter tuning: {e}")
            return {"error": str(e)}


# Convenience functions for integration with existing framework

def solve_benders_master_with_sa(f_coeffs: np.ndarray,
                                 D_matrix: np.ndarray,
                                 d_vector: np.ndarray,
                                 optimality_cuts: List,
                                 feasibility_cuts: List,
                                 B_matrix: Optional[np.ndarray] = None,
                                 b_vector: Optional[np.ndarray] = None,
                                 Ny: int = None,
                                 config: Optional[Dict] = None,
                                 sa_config: Optional[SimulatedAnnealingConfig] = None,
                                 logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Convenience function to solve Benders master problem with simulated annealing.
    
    This function provides a direct interface compatible with existing QAOA solvers
    in the optimization framework.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        adapter = SimulatedAnnealingAdapter(config=sa_config, logger=logger)
        return adapter.solve_benders_master_with_sa(
            f_coeffs, D_matrix, d_vector, optimality_cuts, feasibility_cuts,
            B_matrix, b_vector, Ny, config
        )
    except Exception as e:
        logger.error(f"Error in simulated annealing solver: {e}")
        return {"error": str(e)}


def create_sa_food_optimizer(optimizer: Any, 
                            sa_config: Optional[SimulatedAnnealingConfig] = None) -> SimulatedAnnealingAdapter:
    """
    Create a simulated annealing adapter for an existing food production optimizer.
    
    Args:
        optimizer: FoodProductionOptimizer instance
        sa_config: Simulated annealing configuration parameters
        
    Returns:
        SimulatedAnnealingAdapter instance configured for the optimizer
    """
    if hasattr(optimizer, 'logger'):
        logger = optimizer.logger
    else:
        logger = logging.getLogger(__name__)
    
    return SimulatedAnnealingAdapter(config=sa_config, logger=logger)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test simulated annealing adapter
    config = SimulatedAnnealingConfig(
        initial_temperature=100.0,
        cooling_rate=0.95,
        max_iterations=1000,
        neighborhood_type=NeighborhoodType.SINGLE_FLIP
    )
    
    adapter = SimulatedAnnealingAdapter(config=config, logger=logger)
    
    # Test simple QUBO problem
    print("Testing simple QUBO problem with simulated annealing...")
    
    # Create a small test QUBO (MaxCut problem)
    Q = np.array([[1, -2], [-2, 1]])
    
    result = adapter._solve_qubo_sa(Q, 0.0, 2)
    print(f"Test result: {result}")
    
    print(f"Performance metrics: {adapter._get_performance_metrics()}")
    
    # Test parameter tuning
    print("\nTesting parameter tuning...")
    tuning_result = adapter.tune_parameters(Q, num_trials=3)
    print(f"Tuning result: {tuning_result}")
