"""
Enhanced Simulated Annealing Implementation

This module provides an enhanced simulated annealing optimizer that incorporates
advanced features learned from the simulated_annealing_adapter.py file.

Key enhancements:
- Multiple cooling schedules (linear, exponential, logarithmic, adaptive)
- Advanced neighborhood search strategies  
- Comprehensive performance monitoring
- Automatic parameter tuning
- Robust convergence criteria
- Enhanced logging and metrics
"""

import numpy as np
import random
import math
import logging
import time
from typing import Callable, Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum


class CoolingSchedule(Enum):
    """Types of cooling schedules for simulated annealing."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    ADAPTIVE = "adaptive"
    BOLTZMANN = "boltzmann"


class NeighborhoodType(Enum):
    """Types of neighborhood search strategies."""
    SINGLE_FLIP = "single_flip"     # Flip one bit
    MULTI_FLIP = "multi_flip"       # Flip multiple bits
    SWAP = "swap"                   # Swap two bits
    RANDOM_WALK = "random_walk"     # Random perturbation
    GUIDED = "guided"               # Objective-guided search


@dataclass
class SAConfig:
    """Configuration parameters for enhanced simulated annealing."""
    
    # Temperature schedule
    initial_temperature: float = 100.0
    min_temperature: float = 1e-6
    cooling_schedule: CoolingSchedule = CoolingSchedule.EXPONENTIAL
    cooling_rate: float = 0.95
    
    # Iteration control
    max_iterations: int = 10000
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
    use_restart: bool = True
    restart_threshold: int = 1000  # Restart after this many stagnant iterations
    
    # Logging and debugging
    log_interval: int = 100
    track_history: bool = True
    random_seed: Optional[int] = None


@dataclass
class OptimizationResult:
    """Enhanced result of an optimization run."""
    solution: np.ndarray
    objective_value: float
    runtime: float
    status: str
    iterations: int
    final_temperature: float
    acceptance_rate: float
    stagnation_count: int
    restart_count: int
    convergence_history: Optional[List[float]] = None
    temperature_history: Optional[List[float]] = None


class EnhancedSimulatedAnnealing:
    """
    Enhanced simulated annealing optimizer with advanced features.
    
    This class implements multiple cooling schedules, neighborhood strategies,
    adaptive mechanisms, and comprehensive monitoring capabilities.
    """
    
    def __init__(
        self,
        objective_function: Callable,
        initial_params: np.ndarray,
        config: Optional[SAConfig] = None,
        neighbor_function: Optional[Callable] = None,
        log_file: Optional[str] = None
    ):
        """
        Initialize enhanced simulated annealing.
        
        Args:
            objective_function: Function to minimize
            initial_params: Initial parameter values
            config: SA configuration parameters
            neighbor_function: Optional custom neighbor function
            log_file: Optional file to write logs to
        """
        self.objective_function = objective_function
        self.initial_params = initial_params.copy()
        self.config = config or SAConfig()
        self.neighbor_function = neighbor_function
        self.log_file = log_file
        
        # Set random seed if specified
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
        
        # Setup logging
        self._setup_logging()
        
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
        
        # History tracking
        self.energy_history = [] if self.config.track_history else None
        self.temperature_history = [] if self.config.track_history else None
    
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
    
    def optimize(self) -> OptimizationResult:
        """
        Run enhanced simulated annealing optimization.
        
        Returns:
            OptimizationResult containing comprehensive solution data
        """
        start_time = time.time()
        
        try:
            # Initialize
            self.current_solution = self.initial_params.copy()
            self.current_energy = self.objective_function(self.current_solution)
            self.best_solution = self.current_solution.copy()
            self.best_energy = self.current_energy
            
            iteration = 0
            stagnation_count = 0
            restart_count = 0
            recent_acceptances = []
            
            self.logger.info(f"Starting SA optimization with {self.config.cooling_schedule.value} cooling")
            self.logger.info(f"Initial energy: {self.current_energy:.6f}")
            
            while (self.temperature > self.config.min_temperature and 
                   iteration < self.config.max_iterations):
                
                # Generate neighbor
                if self.neighbor_function:
                    neighbor_solution = self.neighbor_function(self.current_solution)
                else:
                    neighbor_solution = self._generate_neighbor(self.current_solution)
                
                neighbor_energy = self.objective_function(neighbor_solution)
                
                # Accept or reject move
                accepted = self._accept_move(self.current_energy, neighbor_energy, self.temperature)
                
                if accepted:
                    self.current_solution = neighbor_solution
                    self.current_energy = neighbor_energy
                    self.metrics['accepted_moves'] += 1
                    
                    # Update best solution
                    if neighbor_energy < self.best_energy:
                        self.best_solution = neighbor_solution.copy()
                        self.best_energy = neighbor_energy
                        stagnation_count = 0
                        self.logger.debug(f"New best energy: {self.best_energy:.6f}")
                    else:
                        stagnation_count += 1
                else:
                    self.metrics['rejected_moves'] += 1
                    stagnation_count += 1
                
                # Track acceptance for adaptive cooling
                recent_acceptances.append(accepted)
                if len(recent_acceptances) > 100:
                    recent_acceptances.pop(0)
                
                # Update temperature
                acceptance_rate = sum(recent_acceptances) / len(recent_acceptances) if recent_acceptances else 0
                self.temperature = self._update_temperature(
                    self.temperature, iteration, acceptance_rate
                )
                
                # Track history
                if self.config.track_history:
                    self.energy_history.append(self.best_energy)
                    self.temperature_history.append(self.temperature)
                
                # Check for restart
                if (self.config.use_restart and 
                    stagnation_count >= self.config.restart_threshold):
                    self._restart()
                    restart_count += 1
                    stagnation_count = 0
                    self.logger.info(f"Restarting after {self.config.restart_threshold} stagnant iterations")
                
                # Log progress
                if iteration % self.config.log_interval == 0:
                    self.logger.info(
                        f"Iter {iteration:5d}: T={self.temperature:.2e}, "
                        f"Energy={self.current_energy:.6f}, Best={self.best_energy:.6f}, "
                        f"AccRate={acceptance_rate:.2%}"
                    )
                
                # Check convergence
                if self._check_convergence():
                    self.logger.info("Convergence criteria met")
                    break
                
                iteration += 1
                self.metrics['total_iterations'] = iteration
            
            # Calculate final metrics
            runtime = time.time() - start_time
            final_acceptance_rate = sum(recent_acceptances) / len(recent_acceptances) if recent_acceptances else 0
            
            # Create result
            result = OptimizationResult(
                solution=self.best_solution,
                objective_value=self.best_energy,
                runtime=runtime,
                status="COMPLETED",
                iterations=iteration,
                final_temperature=self.temperature,
                acceptance_rate=final_acceptance_rate,
                stagnation_count=stagnation_count,
                restart_count=restart_count,
                convergence_history=self.energy_history,
                temperature_history=self.temperature_history
            )
            
            self.logger.info(f"Optimization completed in {runtime:.2f} seconds")
            self.logger.info(f"Final objective value: {self.best_energy:.6f}")
            self.logger.info(f"Total iterations: {iteration}")
            self.logger.info(f"Final acceptance rate: {final_acceptance_rate:.2%}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise
    
    def _generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        """Generate a neighbor solution based on the neighborhood type."""
        neighbor = solution.copy()
        
        if self.config.neighborhood_type == NeighborhoodType.SINGLE_FLIP:
            # Flip one random bit
            flip_idx = np.random.randint(0, len(solution))
            neighbor[flip_idx] = 1.0 - neighbor[flip_idx]
            
        elif self.config.neighborhood_type == NeighborhoodType.MULTI_FLIP:
            # Flip multiple bits with given probability
            flip_mask = np.random.random(len(solution)) < self.config.flip_probability
            neighbor[flip_mask] = 1.0 - neighbor[flip_mask]
                    
        elif self.config.neighborhood_type == NeighborhoodType.SWAP:
            # Swap two random bits
            if len(solution) >= 2:
                idx1, idx2 = np.random.choice(len(solution), 2, replace=False)
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                
        elif self.config.neighborhood_type == NeighborhoodType.RANDOM_WALK:
            # Random perturbation for continuous variables
            neighbor += np.random.normal(0, 0.1 * self.temperature, size=neighbor.shape)
            neighbor = np.clip(neighbor, 0, 1)  # Keep in bounds
        
        return neighbor
    
    def _accept_move(self, current_energy: float, neighbor_energy: float, 
                     temperature: float) -> bool:
        """Determine whether to accept a move using Metropolis criterion."""
        if neighbor_energy <= current_energy:
            return True
        else:
            delta_energy = neighbor_energy - current_energy
            acceptance_prob = math.exp(-delta_energy / temperature)
            return random.random() < acceptance_prob
    
    def _update_temperature(self, temperature: float, iteration: int, 
                          acceptance_rate: float) -> float:
        """Update temperature according to cooling schedule."""
        
        if self.config.cooling_schedule == CoolingSchedule.LINEAR:
            # Linear cooling
            return self.config.initial_temperature * (1 - iteration / self.config.max_iterations)
            
        elif self.config.cooling_schedule == CoolingSchedule.EXPONENTIAL:
            # Exponential cooling
            new_temp = temperature * self.config.cooling_rate
            
        elif self.config.cooling_schedule == CoolingSchedule.LOGARITHMIC:
            # Logarithmic cooling
            new_temp = self.config.initial_temperature / math.log(iteration + 2)
            
        elif self.config.cooling_schedule == CoolingSchedule.BOLTZMANN:
            # Boltzmann cooling
            new_temp = self.config.initial_temperature / (iteration + 1)
            
        elif self.config.cooling_schedule == CoolingSchedule.ADAPTIVE:
            # Adaptive cooling based on acceptance rate
            if acceptance_rate > 0.9:
                new_temp = temperature * 0.9  # Cool faster if accepting too much
            elif acceptance_rate < 0.1:
                new_temp = temperature * 1.1  # Cool slower if accepting too little
            else:
                new_temp = temperature * self.config.cooling_rate
        else:
            new_temp = temperature * self.config.cooling_rate
        
        # Apply reheating if acceptance rate is too low
        if (self.config.adaptive_cooling and 
            acceptance_rate < self.config.reheat_threshold and
            iteration > 100):  # Don't reheat too early
            new_temp *= self.config.reheat_factor
            self.metrics['reheats'] += 1
            self.logger.debug(f"Reheating: T={new_temp:.2e}")
        
        return max(new_temp, self.config.min_temperature)
    
    def _check_convergence(self) -> bool:
        """Check if convergence criteria are met."""
        # Check target energy
        if (self.config.target_energy is not None and 
            self.best_energy <= self.config.target_energy):
            return True
        
        # Check relative tolerance (improvement over recent history)
        if (self.energy_history and len(self.energy_history) > 100):
            recent_improvement = abs(self.energy_history[-1] - self.energy_history[-100])
            relative_improvement = recent_improvement / abs(self.energy_history[-100])
            if relative_improvement < self.config.relative_tolerance:
                return True
        
        return False
    
    def _restart(self):
        """Restart with a new random solution."""
        self.current_solution = np.random.choice([0.0, 1.0], size=len(self.initial_params))
        self.current_energy = self.objective_function(self.current_solution)
        self.temperature = self.config.initial_temperature * 0.5  # Start with lower temperature
        self.metrics['restarts'] += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        total_moves = self.metrics['accepted_moves'] + self.metrics['rejected_moves']
        acceptance_rate = self.metrics['accepted_moves'] / total_moves if total_moves > 0 else 0
        
        return {
            'total_iterations': self.metrics['total_iterations'],
            'accepted_moves': self.metrics['accepted_moves'],
            'rejected_moves': self.metrics['rejected_moves'],
            'acceptance_rate': acceptance_rate,
            'reheats': self.metrics['reheats'],
            'restarts': self.metrics['restarts'],
            'best_energy': self.best_energy,
            'final_temperature': self.temperature
        }


# Backward compatibility - simple wrapper that matches the original interface
class SimulatedAnnealing:
    """
    Backward-compatible wrapper for enhanced simulated annealing.
    
    This maintains the same interface as the original SimulatedAnnealing class
    while providing access to enhanced features.
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
        """Initialize with original interface."""
        
        # Create config from original parameters
        config = SAConfig(
            initial_temperature=temperature,
            cooling_rate=cooling_rate,
            min_temperature=min_temperature,
            max_iterations=max_iterations,
            cooling_schedule=CoolingSchedule.EXPONENTIAL,
            neighborhood_type=NeighborhoodType.SINGLE_FLIP,
            track_history=False  # Keep it simple for backward compatibility
        )
        
        # Create enhanced SA instance
        self.enhanced_sa = EnhancedSimulatedAnnealing(
            objective_function=objective_function,
            initial_params=initial_params,
            config=config,
            neighbor_function=neighbor_function,
            log_file=log_file
        )
        
        # Expose some attributes for compatibility
        self.logger = self.enhanced_sa.logger
    
    def optimize(self) -> OptimizationResult:
        """Run optimization with original interface."""
        enhanced_result = self.enhanced_sa.optimize()
        
        # Convert to original OptimizationResult format
        from my_functions.optimization import OptimizationResult as OriginalResult
        
        return OriginalResult(
            solution=enhanced_result.solution,
            objective_value=enhanced_result.objective_value,
            runtime=enhanced_result.runtime,
            status=enhanced_result.status,
            optimal_params=enhanced_result.solution,
            optimal_value=enhanced_result.objective_value
        )


# Example usage
if __name__ == "__main__":
    # Test the enhanced SA on a simple problem
    def test_objective(x):
        return sum((x - 0.5) ** 2)  # Minimize distance from 0.5
    
    initial = np.random.random(10)
    
    # Test enhanced SA
    config = SAConfig(
        initial_temperature=10.0,
        cooling_schedule=CoolingSchedule.ADAPTIVE,
        neighborhood_type=NeighborhoodType.MULTI_FLIP,
        max_iterations=1000,
        track_history=True
    )
    
    enhanced_sa = EnhancedSimulatedAnnealing(test_objective, initial, config)
    result = enhanced_sa.optimize()
    
    print(f"Enhanced SA result: {result.objective_value:.6f}")
    print(f"Performance metrics: {enhanced_sa.get_performance_metrics()}")
    
    # Test backward-compatible wrapper
    original_sa = SimulatedAnnealing(test_objective, initial)
    original_result = original_sa.optimize()
    
    print(f"Original SA result: {original_result.objective_value:.6f}")
