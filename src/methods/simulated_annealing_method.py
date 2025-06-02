"""
Simulated Annealing Method for Food Production Optimization

This module implements a simulated annealing approach using the SimulatedAnnealing
class from my_functions.optimization. It provides a clean alternative to
the complex Benders decomposition approach while maintaining compatibility
with the existing optimization pipeline.
"""

import os
import sys
import time
import logging
import numpy as np
import math
import random
from typing import Dict, List, Optional, Any, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from my_functions.optimization import SimulatedAnnealing, OptimizationResult as OptResult
    from src.data_models import OptimizationResult
    # Enhanced SA implementation with advanced features
    from my_functions.enhanced_simulated_annealing import (
        EnhancedSimulatedAnnealing, SAConfig, CoolingSchedule, 
        NeighborhoodType, OptimizationResult as EnhancedResult
    )
    ENHANCED_SA_AVAILABLE = True
except ImportError as e:
    logging.error(f"Error importing required modules: {e}")
    SimulatedAnnealing = None
    ENHANCED_SA_AVAILABLE = False


def _create_error_result(error_message: str) -> 'OptimizationResult':
    """Create a standardized error result."""
    return OptimizationResult(
        status='ERROR',
        objective_value=float('inf'),
        solution={},
        metrics={'error': error_message},
        runtime=0,
        benders_data={'error': error_message},
        quantum_metrics={}
    )


def optimize_with_simulated_annealing_benders(
    self,
    cooling_schedule: str = "exponential",
    initial_temperature: float = 200.0,
    cooling_rate: float = 0.98,
    min_temperature: float = 1e-6,
    max_iterations: int = 10000,
    max_stagnation: int = 1000,
    neighborhood_type: str = "single_flip",
    adaptive_cooling: bool = True,
    use_restart: bool = True,
    enhanced_sa: bool = False,  # New parameter to select enhanced SA
    **kwargs
) -> 'OptimizationResult':
    """
    Optimize food production using simulated annealing.
    
    This method uses the SimulatedAnnealing class from optimization.py to solve 
    the food production optimization problem. The function maintains compatibility
    with the existing pipeline by keeping the original function signature.
    
    Args:
        cooling_schedule: Type of cooling schedule (for compatibility, currently unused)
        initial_temperature: Starting temperature for annealing
        cooling_rate: Rate of temperature reduction 
        min_temperature: Minimum temperature before termination
        max_iterations: Maximum number of SA iterations
        max_stagnation: Maximum iterations without improvement (for compatibility, currently unused)
        neighborhood_type: Type of neighborhood search (for compatibility, currently unused)
        adaptive_cooling: Whether to use adaptive cooling (for compatibility, currently unused)
        use_restart: Whether to restart from different initial solutions (for compatibility, currently unused)
        enhanced_sa: Whether to use the enhanced simulated annealing implementation
        **kwargs: Additional parameters
      Returns:
        OptimizationResult: Complete optimization results with SA-specific metrics
    """
    if SimulatedAnnealing is None:
        self.logger.error("SimulatedAnnealing class not available")
        return _create_error_result("Import error: SimulatedAnnealing class not available")
    
    # Check for enhanced SA usage
    if enhanced_sa and not ENHANCED_SA_AVAILABLE:
        self.logger.warning("Enhanced SA requested but not available, falling back to basic SA")
        enhanced_sa = False
    
    if enhanced_sa:
        self.logger.info("Starting Enhanced Simulated Annealing optimization")
    else:
        self.logger.info("Starting Simulated Annealing optimization")
    self.logger.info(f"Configuration: T₀={initial_temperature}, rate={cooling_rate}, max_iter={max_iterations}")
    
    start_time = time.time()
    
    try:
        # Problem dimensions
        F = len(self.farms)
        C = len(self.foods)
        
        self.logger.info(f"Problem size: {F} farms, {C} foods")
        
        # Create objective function for SA
        def objective_function(params):
            """
            Objective function for the food production optimization.
            
            Args:
                params: Binary array representing food selection (F*C elements)
                
            Returns:
                float: Objective value to minimize
            """
            try:
                # Ensure params is the right shape and type
                if len(params) != F * C:
                    return float('inf')
                
                # Reshape params to (F, C) matrix for easier processing
                selection_matrix = params.reshape((F, C))
                
                # Calculate total objective value
                total_objective = 0.0                # Use scenario weights loaded from food data
                scenario_weights = getattr(self, 'parameters', {}).get('weights', {})
                
                # Fallback if no weights in parameters
                if not scenario_weights:
                    scenario_weights = getattr(self, 'weights', {})                # Calculate objective contributions using area-based approach (same as PuLP)
                # For each farm, distribute land equally among selected foods
                
                for fi, farm in enumerate(self.farms):
                    # Count selected foods for this farm
                    selected_foods = []
                    for ci, food in enumerate(self.foods.keys()):
                        if selection_matrix[fi, ci] > 0.5:
                            selected_foods.append((ci, food))
                    
                    if selected_foods:
                        # Get available land for this farm
                        farm_land = self.parameters['land_availability'][farm]
                        
                        # Distribute land equally among selected foods (like PuLP does)
                        area_per_food = farm_land / len(selected_foods)
                        
                        # Calculate objective contribution for each selected food
                        for ci, food in selected_foods:
                            food_data = self.foods[food]
                            
                            # Calculate weighted score per unit area (same as PuLP)
                            weighted_score = 0.0
                            
                            # Nutritional objectives (positive contribution)
                            if 'nutritional_value' in scenario_weights and 'nutritional_value' in food_data:
                                weighted_score += scenario_weights['nutritional_value'] * food_data['nutritional_value']
                            
                            if 'nutrient_density' in scenario_weights and 'nutrient_density' in food_data:
                                weighted_score += scenario_weights['nutrient_density'] * food_data['nutrient_density']
                            
                            # Economic objectives (positive contribution)
                            if 'affordability' in scenario_weights and 'affordability' in food_data:
                                weighted_score += scenario_weights['affordability'] * food_data['affordability']
                            
                            if 'sustainability' in scenario_weights and 'sustainability' in food_data:
                                weighted_score += scenario_weights['sustainability'] * food_data['sustainability']
                            
                            # Environmental impact (negative contribution - lower is better)
                            if 'environmental_impact' in scenario_weights and 'environmental_impact' in food_data:
                                weighted_score -= scenario_weights['environmental_impact'] * food_data['environmental_impact']
                            
                            # Multiply by allocated area (same as PuLP: weighted_score * area)
                            total_objective += weighted_score * area_per_food
                
                # Add penalty for constraint violations
                penalty = 0.0
                
                # Minimum and maximum foods per farm penalties
                for fi in range(F):
                    foods_selected = sum(1 for ci in range(C) if selection_matrix[fi, ci] > 0.5)
                    
                    # Minimum foods per farm (at least 1)
                    if foods_selected < 1:
                        penalty += 500.0
                    
                    # Maximum foods per farm (at most 3)
                    elif foods_selected > 3:
                        penalty += 200.0 * (foods_selected - 3)
                
                # Return negative for minimization (SA minimizes, but we want to maximize objective)
                return -(total_objective - penalty)
                
            except Exception as e:
                self.logger.error(f"Error in objective function: {e}")
                return float('inf')
        
        # Create binary neighbor function for food selection
        def binary_neighbor_function(params):
            """Generate a neighbor by flipping a random bit."""
            neighbor = params.copy()
            
            # Flip a random bit
            flip_idx = np.random.randint(0, len(params))
            neighbor[flip_idx] = 1.0 - neighbor[flip_idx]
            
            return neighbor
        
        # Initialize with a smart binary solution
        initial_params = np.zeros(F * C, dtype=float)
        
        # Ensure at least one food per farm
        for fi in range(F):
            farm_start = fi * C
            # Select 1-2 foods per farm randomly
            num_foods_to_select = min(np.random.choice([1, 2], p=[0.7, 0.3]), C)
            selected_foods = np.random.choice(C, size=num_foods_to_select, replace=False)
            
            for food_idx in selected_foods:
                initial_params[farm_start + food_idx] = 1.0
        
        self.logger.info(f"Initial solution has {np.sum(initial_params)} foods selected")
          # Create and run simulated annealing
        if enhanced_sa and ENHANCED_SA_AVAILABLE:
            # Create enhanced SA configuration
            config = SAConfig(
                initial_temperature=initial_temperature,
                cooling_rate=cooling_rate,
                min_temperature=min_temperature,
                max_iterations=max_iterations,
                max_stagnation_iterations=max_stagnation,
                cooling_schedule=CoolingSchedule.ADAPTIVE if adaptive_cooling else CoolingSchedule.EXPONENTIAL,
                neighborhood_type=NeighborhoodType.SINGLE_FLIP if neighborhood_type == "single_flip" else NeighborhoodType.MULTI_FLIP,
                adaptive_cooling=adaptive_cooling,
                use_restart=use_restart,
                track_history=True,  # Enable history tracking for enhanced features
                log_interval=max(1, max_iterations // 20)  # Log every 5% of iterations
            )
            
            # Create enhanced SA optimizer
            sa_optimizer = EnhancedSimulatedAnnealing(
                objective_function=objective_function,
                initial_params=initial_params,
                config=config,
                neighbor_function=binary_neighbor_function,
                log_file=None
            )
            
            self.logger.info("Running enhanced simulated annealing optimization...")
            sa_result = sa_optimizer.optimize()
            
            # Get enhanced metrics
            metrics = sa_optimizer.get_performance_metrics()
            self.logger.info(f"Enhanced SA metrics: {metrics}")
            
        else:
            # Use basic SA
            sa_optimizer = SimulatedAnnealing(
                objective_function=objective_function,
                initial_params=initial_params,
                temperature=initial_temperature,
                cooling_rate=cooling_rate,
                min_temperature=min_temperature,
                max_iterations=max_iterations,
                neighbor_function=binary_neighbor_function,
                log_file=None
            )
            
            self.logger.info("Running simulated annealing optimization...")
            sa_result = sa_optimizer.optimize()
          # Extract results
        solution_binary = sa_result.solution
        objective_value = -sa_result.objective_value  # Convert back from minimization to maximization
        runtime = sa_result.runtime
        
        self.logger.info(f"SA optimization completed in {runtime:.2f}s")
        self.logger.info(f"Final objective value (converted to maximization): {objective_value:.6f}")
        
        # Convert binary solution to production amounts
        solution_dict = {}
        total_foods_selected = 0
        
        for fi, farm in enumerate(self.farms):
            farm_foods_selected = []
            for ci, food in enumerate(self.foods.keys()):
                idx = fi * C + ci
                if idx < len(solution_binary) and solution_binary[idx] > 0.5:  # Selected
                    farm_foods_selected.append(food)
                    total_foods_selected += 1
            
            # Assign equal production for selected foods
            if farm_foods_selected:
                farm_land = self.parameters.get('land_availability', {}).get(farm, 100.0)
                production_per_food = farm_land / len(farm_foods_selected)
                
                for food in farm_foods_selected:
                    solution_dict[(farm, food)] = production_per_food
            
            # Set non-selected foods to zero
            for food in self.foods.keys():
                if food not in farm_foods_selected:
                    solution_dict[(farm, food)] = 0.0
        self.logger.info(f"Total foods selected: {total_foods_selected}")
        
        # Calculate detailed metrics (same as PuLP and Benders methods)
        detailed_metrics = self._calculate_metrics(solution_dict)
        self.logger.info("Calculated detailed metrics:")
        for metric, value in detailed_metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        # Combine SA-specific metrics with detailed metrics
        combined_metrics = {
            # Detailed nutritional and operational metrics (same as other methods)
            **detailed_metrics,
            # SA-specific metrics
            'total_foods_selected': total_foods_selected,
            'sa_iterations': max_iterations,
            'wall_time': runtime,
            'farms_count': F,
            'foods_count': C,
            'initial_temperature': initial_temperature,
            'final_temperature': min_temperature,
            'cooling_rate': cooling_rate,
            'method': 'simulated_annealing'
        }
        
        # Create optimization result
        optimization_result = OptimizationResult(
            status='OPTIMAL',
            objective_value=objective_value,
            solution=solution_dict,
            metrics=combined_metrics,
            runtime=runtime,
            benders_data={
                'method': 'simulated_annealing_replacement',
                'original_benders': False,
                'sa_based': True
            },
            quantum_metrics={
                'method': 'simulated_annealing',
                'temperature_initial': initial_temperature,
                'temperature_final': min_temperature,
                'cooling_rate': cooling_rate,
                'max_iterations': max_iterations,
                'total_foods_selected': total_foods_selected
            }
        )
        
        self.logger.info("Simulated Annealing optimization completed successfully")
        return optimization_result
        
    except Exception as e:
        self.logger.error(f"Error during simulated annealing optimization: {e}")
        import traceback
        self.logger.error(traceback.format_exc())
        return _create_error_result(f"Optimization error: {e}")
