"""
D-Wave QPU Adapter for Food Production Optimization

This module provides a D-Wave quantum processing unit (QPU) adapter that converts
the food production optimization problem for use with D-Wave's quantum annealing
hardware using the dimod library. It integrates with the existing QUBO converter
and Benders decomposition framework to enable quantum annealing approaches.

Key Features:
- D-Wave QPU integration using dimod BinaryQuadraticModel
- Quantum annealing solver interface compatible with existing optimization framework
- Support for both simulated annealing and real QPU execution
- Integration with existing Benders decomposition approach
- Comprehensive error handling and fallback mechanisms
- Performance monitoring and metrics collection

Dependencies:
- dimod: D-Wave binary quadratic model library
- dwave-system: D-Wave system tools and samplers
- numpy: Numerical computing
- logging: Error tracking and debugging
"""

import numpy as np
import logging
import time
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass

try:
    import dimod
    from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler
    DIMOD_AVAILABLE = True
except ImportError:
    logging.warning("dimod not available. D-Wave functionality will be limited.")
    dimod = None
    BinaryQuadraticModel = None
    SimulatedAnnealingSampler = None
    DIMOD_AVAILABLE = False

try:
    from dwave.system import DWaveSampler, EmbeddingComposite, LazyFixedEmbeddingComposite
    from dwave.cloud.exceptions import SolverNotFoundError
    DWAVE_SYSTEM_AVAILABLE = True
except ImportError:
    logging.warning("dwave-system not available. Real QPU access will be disabled.")
    DWaveSampler = None
    EmbeddingComposite = None
    LazyFixedEmbeddingComposite = None
    SolverNotFoundError = None
    DWAVE_SYSTEM_AVAILABLE = False

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


@dataclass
class DWaveConfig:
    """Configuration parameters for D-Wave quantum annealing."""
    
    # Sampler configuration
    use_real_qpu: bool = False  # Whether to use real QPU or simulator
    solver_name: Optional[str] = None  # Specific solver name (e.g., 'Advantage_system4.1')
    num_reads: int = 1000  # Number of annealing cycles
    
    # Annealing parameters
    annealing_time: float = 20.0  # Microseconds for quantum annealing
    programming_thermalization: float = 1000.0  # Microseconds
    readout_thermalization: float = 1000.0  # Microseconds
    
    # Problem embedding
    chain_strength: Optional[float] = None  # Auto-calculated if None
    auto_scale: bool = True  # Auto-scale problem for QPU
    
    # Advanced parameters
    anneal_schedule: Optional[List[Tuple[float, float]]] = None  # Custom annealing schedule
    initial_state: Optional[Dict[int, int]] = None  # Warm start state
    h_gain_schedule: Optional[List[Tuple[float, float]]] = None  # h-gain schedule
    
    # Timeout and retry
    timeout: float = 300.0  # Maximum time to wait for results (seconds)
    max_retries: int = 3  # Number of retry attempts
    
    # Preprocessing
    reduce_intersample_correlation: bool = True
    reinitialize_state: bool = True
    postprocess: str = 'optimization'  # 'sampling' or 'optimization'
    
    # Cost estimation settings
    estimate_cost_only: bool = False  # If True, only estimate cost without solving
    max_budget_usd: float = 100.0  # Maximum budget for QPU usage
    warn_cost_threshold: float = 10.0  # Warn if estimated cost exceeds this


@dataclass
class ComplexityLevel:
    """Defines different complexity levels for problem estimation."""
    name: str
    num_farms: int
    num_foods: int
    num_nutrients: int
    num_constraints: int
    description: str
    
    @property
    def total_variables(self) -> int:
        """Calculate total number of variables for this complexity level."""
        return self.num_farms * self.num_foods
    
    @property
    def estimated_qubits(self) -> int:
        """Estimate number of qubits needed (including auxiliary variables)."""
        base_vars = self.total_variables
        # Add auxiliary variables for constraints and penalties
        aux_vars = min(self.num_constraints * 3, base_vars)  # Conservative estimate
        return base_vars + aux_vars


@dataclass
class CostEstimation:
    """Cost estimation results for D-Wave QPU usage."""
    complexity_level: str
    num_variables: int
    estimated_qubits: int
    estimated_qpu_time_us: float
    estimated_cost_usd: float
    num_reads: int
    is_feasible: bool
    warnings: List[str]
    recommendations: List[str]


class DWaveQPUAdapter:
    """
    D-Wave QPU adapter for quantum annealing optimization of food production problems.
    
    This class provides an interface between the existing QUBO conversion framework
    and D-Wave's quantum processing units, enabling quantum annealing approaches
    for the food production optimization problem.
    """
    
    # Predefined complexity levels for testing and estimation
    COMPLEXITY_LEVELS = {
        'tiny': ComplexityLevel(
            name='tiny',
            num_farms=2,
            num_foods=3,
            num_nutrients=5,
            num_constraints=8,
            description='Minimal test case for algorithm validation'
        ),
        'small': ComplexityLevel(
            name='small',
            num_farms=5,
            num_foods=8,
            num_nutrients=12,
            num_constraints=25,
            description='Small-scale optimization suitable for prototyping'
        ),
        'medium': ComplexityLevel(
            name='medium',
            num_farms=10,
            num_foods=15,
            num_nutrients=20,
            num_constraints=50,
            description='Medium-scale optimization for regional planning'
        ),
        'large': ComplexityLevel(
            name='large',
            num_farms=25,
            num_foods=30,
            num_nutrients=30,
            num_constraints=100,
            description='Large-scale optimization for national planning'
        ),
        'enterprise': ComplexityLevel(
            name='enterprise',
            num_farms=50,
            num_foods=50,
            num_nutrients=40,
            num_constraints=200,
            description='Enterprise-scale optimization for global planning'
        )
    }
    
    # D-Wave pricing estimates (approximate, as of 2024)
    DWAVE_PRICING = {
        'cost_per_second': 0.00015,  # USD per microsecond of QPU time
        'minimum_charge': 0.001,     # Minimum charge per problem
        'overhead_factor': 1.2,      # Factor for programming and readout overhead
    }
    
    def __init__(self, config: Optional[DWaveConfig] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the D-Wave QPU adapter.
        
        Args:
            config: D-Wave configuration parameters
            logger: Logger instance for debugging and monitoring
        """
        self.config = config or DWaveConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize samplers
        self.qpu_sampler = None
        self.sim_sampler = None
        self.active_sampler = None
        
        # Performance metrics
        self.metrics = {
            'qpu_calls': 0,
            'successful_calls': 0,
            'total_qpu_time': 0.0,
            'total_wall_time': 0.0,
            'chain_breaks': 0,
            'embedding_retries': 0,
            'problem_sizes': [],
            'energies': [],
            'num_occurrences': [],
            'cost_estimates': [],
            'actual_costs': []
        }
        
        # Problem cache for efficiency
        self.problem_cache = {}
        self.embedding_cache = {}
        
        if not DIMOD_AVAILABLE:
            raise ImportError("dimod library is required for D-Wave functionality")
        
        self._initialize_samplers()
    
    def _initialize_samplers(self):
        """Initialize D-Wave samplers (real QPU and simulator)."""
        try:
            # Initialize simulated annealing sampler (always available)
            self.sim_sampler = SimulatedAnnealingSampler()
            self.logger.info("Initialized simulated annealing sampler")
            
            # Try to initialize real QPU sampler
            if self.config.use_real_qpu and DWAVE_SYSTEM_AVAILABLE:
                try:
                    if self.config.solver_name:
                        self.qpu_sampler = DWaveSampler(solver=self.config.solver_name)
                    else:
                        self.qpu_sampler = DWaveSampler()
                    
                    # Wrap with embedding composite
                    self.qpu_sampler = EmbeddingComposite(self.qpu_sampler)
                    
                    self.logger.info(f"Initialized D-Wave QPU sampler: {self.qpu_sampler.solver.name}")
                    self.active_sampler = self.qpu_sampler
                    
                except (SolverNotFoundError, Exception) as e:
                    self.logger.warning(f"Failed to initialize D-Wave QPU: {e}")
                    self.logger.info("Falling back to simulated annealing")
                    self.active_sampler = self.sim_sampler
            else:
                self.active_sampler = self.sim_sampler
                
        except Exception as e:
            self.logger.error(f"Error initializing samplers: {e}")
            raise
    
    def create_bqm_from_qubo(self, Q_matrix: np.ndarray, 
                            offset: float = 0.0,
                            variable_labels: Optional[List[str]] = None) -> BinaryQuadraticModel:
        """
        Create a D-Wave BinaryQuadraticModel from a QUBO matrix.
        
        Args:
            Q_matrix: QUBO coefficient matrix
            offset: Constant offset term
            variable_labels: Labels for variables (defaults to integers)
            
        Returns:
            BinaryQuadraticModel instance
        """
        n_vars = Q_matrix.shape[0]
        
        if variable_labels is None:
            variable_labels = list(range(n_vars))
        elif len(variable_labels) != n_vars:
            raise ValueError(f"Number of labels ({len(variable_labels)}) must match matrix size ({n_vars})")
        
        # Create BQM
        bqm = BinaryQuadraticModel('BINARY')
        
        # Add linear terms (diagonal elements)
        for i in range(n_vars):
            if Q_matrix[i, i] != 0:
                bqm.add_variable(variable_labels[i], Q_matrix[i, i])
        
        # Add quadratic terms (off-diagonal elements)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if Q_matrix[i, j] != 0:
                    bqm.add_interaction(variable_labels[i], variable_labels[j], Q_matrix[i, j])
        
        # Add offset
        bqm.offset = offset
        
        self.logger.debug(f"Created BQM with {len(bqm.variables)} variables, "
                         f"{len(bqm.quadratic)} interactions, offset={offset}")
        
        return bqm
    
    def solve_benders_master_with_dwave(self,
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
        Solve Benders master problem using D-Wave quantum annealing.
        
        This method converts the Benders master problem to QUBO form and solves it
        using D-Wave's quantum annealing approach.
        
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
            # Check if we should only estimate cost
            if self.config.estimate_cost_only:
                problem_size = f_coeffs.shape[0]
                estimation = self.estimate_qpu_cost(problem_size)
                
                self.logger.info(f"Cost estimation only: {problem_size} variables, "
                               f"~${estimation.estimated_cost_usd:.4f}")
                
                return {
                    'cost_estimation': estimation,
                    'estimation_only': True,
                    'recommendations': estimation.recommendations,
                    'warnings': estimation.warnings
                }
            
            # Proceed with normal solving
            self.logger.info("Converting Benders master problem to QUBO for D-Wave...")
            
            if config is None:
                config = {
                    "eta_min": -1000.0,
                    "eta_max": 1000.0,
                    "eta_num_bits": 6,
                    "penalty_coefficient": 10000.0,
                    "penalty_slack_num_bits": 4
                }
            
            # Get cost estimation before solving
            problem_size = f_coeffs.shape[0]
            cost_estimation = self.estimate_qpu_cost(problem_size)
            
            # Check if cost exceeds budget
            if cost_estimation.estimated_cost_usd > self.config.max_budget_usd:
                self.logger.warning(f"Estimated cost (${cost_estimation.estimated_cost_usd:.4f}) "
                                  f"exceeds budget (${self.config.max_budget_usd})")
                return {
                    "error": "Cost exceeds budget",
                    "cost_estimation": cost_estimation,
                    "suggested_action": "Use simulated annealing or reduce problem size"
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
            variable_mapping = getattr(qubo_model, 'variable_map', getattr(qubo_model, 'variable_mapping', None))
            
            # Incorporate linear terms into Q matrix diagonal
            Q_full = Q_matrix.copy()
            np.fill_diagonal(Q_full, np.diag(Q_full) + c_vector)
            
            problem_size = Q_full.shape[0]
            self.logger.info(f"QUBO problem size: {problem_size} variables")
            
            # Check problem size limits
            if self.active_sampler == self.qpu_sampler and problem_size > 5000:
                self.logger.warning(f"Problem size ({problem_size}) may be too large for QPU. "
                                  "Consider using problem decomposition.")
            
            # Create BinaryQuadraticModel
            variable_labels = [f"x_{i}" for i in range(problem_size)]
            bqm = self.create_bqm_from_qubo(Q_full, offset, variable_labels)
            
            # Solve using D-Wave
            result = self._solve_bqm(bqm)
            
            if result.get('error'):
                return result
            
            # Convert solution back to original variable space
            dwave_solution = result['sample']
            binary_solution = np.zeros(problem_size)
            
            for i, var_label in enumerate(variable_labels):
                if var_label in dwave_solution:
                    binary_solution[i] = dwave_solution[var_label]
            
            # Map back to original problem variables
            original_solution = np.zeros(Ny)
            if hasattr(qubo_model, 'reverse_mapping'):
                for qubo_idx, orig_idx in qubo_model.reverse_mapping.items():
                    if qubo_idx < len(binary_solution) and orig_idx < Ny:
                        original_solution[orig_idx] = binary_solution[qubo_idx]
            else:
                # Direct mapping for first Ny variables
                copy_size = min(Ny, len(binary_solution))
                original_solution[:copy_size] = binary_solution[:copy_size]
            
            # Calculate objective value
            objective_value = float(f_coeffs.T.dot(original_solution.reshape(-1, 1))[0, 0])
            
            # Update metrics with cost information
            wall_time = time.time() - start_time
            self.metrics['successful_calls'] += 1
            self.metrics['total_wall_time'] += wall_time
            self.metrics['problem_sizes'].append(problem_size)
            self.metrics['energies'].append(result['energy'])
            self.metrics['cost_estimates'].append(cost_estimation.estimated_cost_usd)
            
            self.logger.info(f"D-Wave solution found in {wall_time:.2f}s with energy {result['energy']:.6f}, "
                           f"estimated cost: ${cost_estimation.estimated_cost_usd:.4f}")
            
            return {
                'solution': original_solution,
                'objective': objective_value,
                'energy': result['energy'],
                'num_occurrences': result['num_occurrences'],
                'chain_break_fraction': result.get('chain_break_fraction', 0.0),
                'timing': result.get('timing', {}),
                'wall_time': wall_time,
                'problem_size': problem_size,
                'cost_estimation': cost_estimation,
                'qubo_matrix': Q_full,
                'bqm_info': {
                    'num_variables': len(bqm.variables),
                    'num_interactions': len(bqm.quadratic),
                    'offset': bqm.offset
                },
                'metrics': self._get_performance_metrics()
            }
            
        except Exception as e:
            self.logger.error(f"Error solving with D-Wave: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def _solve_bqm(self, bqm: BinaryQuadraticModel) -> Dict[str, Any]:
        """
        Solve a BinaryQuadraticModel using the active D-Wave sampler.
        
        Args:
            bqm: BinaryQuadraticModel to solve
            
        Returns:
            Dictionary with solution and performance data
        """
        self.metrics['qpu_calls'] += 1
        
        try:
            # Prepare sampler parameters
            sampler_params = {
                'num_reads': self.config.num_reads,
            }
            
            # Add QPU-specific parameters
            if self.active_sampler == self.qpu_sampler:
                sampler_params.update({
                    'annealing_time': self.config.annealing_time,
                    'programming_thermalization': self.config.programming_thermalization,
                    'readout_thermalization': self.config.readout_thermalization,
                    'reduce_intersample_correlation': self.config.reduce_intersample_correlation,
                    'reinitialize_state': self.config.reinitialize_state,
                    'postprocess': self.config.postprocess
                })
                
                if self.config.chain_strength is not None:
                    sampler_params['chain_strength'] = self.config.chain_strength
                if self.config.auto_scale:
                    sampler_params['auto_scale'] = True
                if self.config.anneal_schedule:
                    sampler_params['anneal_schedule'] = self.config.anneal_schedule
                if self.config.initial_state:
                    sampler_params['initial_state'] = self.config.initial_state
                if self.config.h_gain_schedule:
                    sampler_params['h_gain_schedule'] = self.config.h_gain_schedule
            
            # Add simulated annealing specific parameters
            else:
                sampler_params.update({
                    'beta_range': [0.1, 10.0],
                    'num_sweeps': 1000,
                    'seed': None
                })
            
            # Solve the problem
            self.logger.debug(f"Submitting BQM to {type(self.active_sampler).__name__}")
            qpu_start = time.time()
            
            sampleset = self.active_sampler.sample(bqm, **sampler_params)
            
            qpu_time = time.time() - qpu_start
            self.metrics['total_qpu_time'] += qpu_time
            
            # Extract best solution
            best_sample = sampleset.first
            sample_dict = dict(best_sample.sample)
            energy = best_sample.energy
            num_occurrences = best_sample.num_occurrences
            
            # Extract chain break information (QPU only)
            chain_break_fraction = 0.0
            if hasattr(sampleset, 'data_vectors') and 'chain_break_fraction' in sampleset.data_vectors:
                chain_break_fractions = sampleset.data_vectors['chain_break_fraction']
                if len(chain_break_fractions) > 0:
                    chain_break_fraction = chain_break_fractions[0]
                    self.metrics['chain_breaks'] += int(chain_break_fraction > 0)
            
            # Extract timing information (QPU only)
            timing_info = {}
            if hasattr(sampleset, 'info') and 'timing' in sampleset.info:
                timing_info = sampleset.info['timing']
            
            self.logger.debug(f"D-Wave sampling completed in {qpu_time:.2f}s, "
                            f"energy={energy:.6f}, chain_breaks={chain_break_fraction:.3f}")
            
            return {
                'sample': sample_dict,
                'energy': energy,
                'num_occurrences': num_occurrences,
                'chain_break_fraction': chain_break_fraction,
                'timing': timing_info,
                'qpu_time': qpu_time,
                'sampleset_info': sampleset.info if hasattr(sampleset, 'info') else {}
            }
            
        except Exception as e:
            self.logger.error(f"Error during D-Wave sampling: {e}")
            return {'error': str(e)}
    
    def solve_food_optimization_with_dwave(self, optimizer: Any) -> Dict[str, Any]:
        """
        Solve the complete food production optimization problem using D-Wave quantum annealing.
        
        This method integrates with the existing food production optimizer framework
        and uses D-Wave's quantum annealing to solve the optimization problem.
        
        Args:
            optimizer: FoodProductionOptimizer instance
            
        Returns:
            Dictionary containing solution, metrics, and performance data
        """
        try:
            self.logger.info("Starting D-Wave quantum annealing optimization for food production")
            
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
            
            # Convert to QUBO and solve with D-Wave
            result = self.solve_benders_master_with_dwave(
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
                'dwave_energy': result['energy'],
                'dwave_timing': result.get('timing', {}),
                'chain_break_fraction': result.get('chain_break_fraction', 0.0)
            }
            
            return {
                'solution': solution,
                'objective_value': -result['objective'],  # Convert back to maximization
                'metrics': metrics,
                'dwave_result': result,
                'performance_metrics': self._get_performance_metrics()
            }
            
        except Exception as e:
            self.logger.error(f"Error in D-Wave food optimization: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def estimate_qpu_cost(self, 
                         problem_size: int, 
                         num_reads: Optional[int] = None,
                         complexity_level: Optional[str] = None) -> CostEstimation:
        """
        Estimate the cost of running a problem on D-Wave QPU.
        
        Args:
            problem_size: Number of variables in the problem
            num_reads: Number of annealing cycles (uses config default if None)
            complexity_level: Name of complexity level (for reference)
            
        Returns:
            CostEstimation object with detailed cost analysis
        """
        if num_reads is None:
            num_reads = self.config.num_reads
        
        warnings = []
        recommendations = []
        
        # Estimate qubits needed (including overhead for embedding)
        estimated_qubits = problem_size
        if problem_size > 100:
            # Add embedding overhead for larger problems
            embedding_overhead = min(problem_size * 0.3, 1000)
            estimated_qubits += int(embedding_overhead)
        
        # Check feasibility based on QPU limits - FIXED: More realistic limits
        max_qubits = 5000  # Current D-Wave systems can handle this
        is_feasible = estimated_qubits <= max_qubits
        
        if not is_feasible:
            warnings.append(f"Problem may exceed QPU capacity ({estimated_qubits} > {max_qubits} qubits)")
            recommendations.append("Consider problem decomposition or use classical algorithms")
        
        # Estimate QPU time - FIXED: More realistic pricing
        base_annealing_time = self.config.annealing_time  # microseconds
        programming_time = self.config.programming_thermalization
        readout_time = self.config.readout_thermalization
        
        total_time_per_read = base_annealing_time + programming_time + readout_time
        total_qpu_time = total_time_per_read * num_reads
        
        # Apply overhead factor
        total_qpu_time *= self.DWAVE_PRICING['overhead_factor']
        
        # FIXED: Correct cost calculation (convert microseconds to seconds)
        total_qpu_time_seconds = total_qpu_time / 1_000_000  # Convert microseconds to seconds
        cost_usd = max(
            total_qpu_time_seconds * 2.0,  # More realistic $2 per second of QPU time
            self.DWAVE_PRICING['minimum_charge']
        )
        
        # Add cost warnings and recommendations
        if cost_usd > self.config.warn_cost_threshold:
            warnings.append(f"Estimated cost (${cost_usd:.4f}) exceeds warning threshold")
            
        if cost_usd > self.config.max_budget_usd:
            warnings.append(f"Estimated cost (${cost_usd:.4f}) exceeds budget (${self.config.max_budget_usd})")
            recommendations.append("Reduce num_reads or use simulated annealing")
            # Don't mark as infeasible just because of budget
        
        # Size-based recommendations
        if problem_size < 50:
            recommendations.append("Problem is small enough for classical solvers - consider alternatives")
        elif problem_size < 200:
            recommendations.append("Good candidate for quantum annealing")
        else:
            recommendations.append("Large problem - monitor embedding quality and chain breaks")
        
        return CostEstimation(
            complexity_level=complexity_level or 'custom',
            num_variables=problem_size,
            estimated_qubits=estimated_qubits,
            estimated_qpu_time_us=total_qpu_time,
            estimated_cost_usd=cost_usd,
            num_reads=num_reads,
            is_feasible=is_feasible,  # Based on QPU capacity, not budget
            warnings=warnings,
            recommendations=recommendations
        )
    
    def estimate_all_complexity_levels(self) -> Dict[str, CostEstimation]:
        """
        Estimate costs for all predefined complexity levels.
        
        Returns:
            Dictionary mapping complexity level names to cost estimations
        """
        estimations = {}
        
        for level_name, level_config in self.COMPLEXITY_LEVELS.items():
            estimation = self.estimate_qpu_cost(
                problem_size=level_config.total_variables,
                complexity_level=level_name
            )
            estimations[level_name] = estimation
            
            self.logger.info(f"Complexity '{level_name}': {level_config.total_variables} vars, "
                           f"~${estimation.estimated_cost_usd:.4f}, feasible={estimation.is_feasible}")
        
        return estimations
    
    def get_complexity_recommendation(self, 
                                    num_farms: int, 
                                    num_foods: int,
                                    budget_usd: Optional[float] = None) -> Dict[str, Any]:
        """
        Get complexity level recommendation based on problem size and budget.
        
        Args:
            num_farms: Number of farms in the problem
            num_foods: Number of food types
            budget_usd: Available budget (uses config default if None)
            
        Returns:
            Dictionary with recommendation and analysis
        """
        if budget_usd is None:
            budget_usd = self.config.max_budget_usd
        
        problem_size = num_farms * num_foods
        estimation = self.estimate_qpu_cost(problem_size)
        
        # Find closest predefined complexity level
        closest_level = None
        min_diff = float('inf')
        
        for level_name, level_config in self.COMPLEXITY_LEVELS.items():
            diff = abs(level_config.total_variables - problem_size)
            if diff < min_diff:
                min_diff = diff
                closest_level = level_name
        
        recommendation = {
            'problem_size': problem_size,
            'estimated_cost': estimation.estimated_cost_usd,
            'is_affordable': estimation.estimated_cost_usd <= budget_usd,
            'closest_complexity_level': closest_level,
            'complexity_description': self.COMPLEXITY_LEVELS[closest_level].description if closest_level else None,
            'estimation': estimation,
            'alternatives': []
        }
        
        # Suggest alternatives if not affordable
        if not recommendation['is_affordable']:
            recommendation['alternatives'].extend([
                'Use simulated annealing (free)',
                'Reduce problem size',
                'Use classical optimization methods'
            ])
        
        # Suggest smaller complexity levels that fit budget
        affordable_levels = []
        for level_name, level_config in self.COMPLEXITY_LEVELS.items():
            level_est = self.estimate_qpu_cost(level_config.total_variables)
            if level_est.estimated_cost_usd <= budget_usd:
                affordable_levels.append({
                    'level': level_name,
                    'size': level_config.total_variables,
                    'cost': level_est.estimated_cost_usd,
                    'description': level_config.description
                })
        
        recommendation['affordable_levels'] = sorted(affordable_levels, key=lambda x: x['cost'], reverse=True)
        
        return recommendation
    
    def run_free_complexity_analysis(self) -> Dict[str, Any]:
        """
        Run a complete free analysis of all complexity levels without using QPU resources.
        
        Returns:
            Comprehensive analysis report
        """
        self.logger.info("Running free complexity analysis for D-Wave quantum annealing...")
        
        analysis_start = time.time()
        
        # Estimate all complexity levels
        estimations = self.estimate_all_complexity_levels()
        
        # Calculate summary statistics
        total_costs = [est.estimated_cost_usd for est in estimations.values()]
        feasible_levels = [name for name, est in estimations.items() if est.is_feasible]
        affordable_levels = [name for name, est in estimations.items() 
                           if est.estimated_cost_usd <= self.config.max_budget_usd]
        
        # Generate recommendations
        recommendations = []
        if not affordable_levels:
            recommendations.append("All complexity levels exceed budget - use simulated annealing")
        else:
            max_affordable = max(affordable_levels, key=lambda x: estimations[x].num_variables)
            recommendations.append(f"Maximum affordable complexity: '{max_affordable}'")
        
        if len(feasible_levels) < len(estimations):
            recommendations.append("Some complexity levels may not fit on current QPU hardware")
        
        # Create detailed report
        report = {
            'analysis_timestamp': time.time(),
            'analysis_duration': time.time() - analysis_start,
            'budget_usd': self.config.max_budget_usd,
            'estimations': estimations,
            'summary': {
                'total_levels_analyzed': len(estimations),
                'feasible_levels': feasible_levels,
                'affordable_levels': affordable_levels,
                'cost_range': {
                    'min': min(total_costs),
                    'max': max(total_costs),
                    'mean': np.mean(total_costs)
                }
            },
            'recommendations': recommendations,
            'sampler_info': {
                'simulator_available': self.sim_sampler is not None,
                'qpu_configured': self.config.use_real_qpu,
                'qpu_available': self.qpu_sampler is not None
            }
        }
        
        # Log summary
        self.logger.info(f"Analysis complete: {len(feasible_levels)}/{len(estimations)} levels feasible, "
                        f"{len(affordable_levels)} affordable within ${self.config.max_budget_usd}")
        
        return report

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            'total_qpu_calls': self.metrics['qpu_calls'],
            'successful_calls': self.metrics['successful_calls'],
            'success_rate': self.metrics['successful_calls'] / max(self.metrics['qpu_calls'], 1),
            'avg_qpu_time': self.metrics['total_qpu_time'] / max(self.metrics['successful_calls'], 1),
            'avg_wall_time': self.metrics['total_wall_time'] / max(self.metrics['successful_calls'], 1),
            'total_chain_breaks': self.metrics['chain_breaks'],
            'avg_problem_size': np.mean(self.metrics['problem_sizes']) if self.metrics['problem_sizes'] else 0,
            'avg_energy': np.mean(self.metrics['energies']) if self.metrics['energies'] else 0,
            'total_estimated_cost': sum(self.metrics['cost_estimates']),
            'avg_estimated_cost': np.mean(self.metrics['cost_estimates']) if self.metrics['cost_estimates'] else 0,
            'sampler_type': type(self.active_sampler).__name__ if self.active_sampler else 'None'
        }
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = {
            'qpu_calls': 0,
            'successful_calls': 0,
            'total_qpu_time': 0.0,
            'total_wall_time': 0.0,
            'chain_breaks': 0,
            'embedding_retries': 0,
            'problem_sizes': [],
            'energies': [],
            'num_occurrences': [],
            'cost_estimates': [],
            'actual_costs': []
        }
        self.logger.info("Performance metrics reset")
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to D-Wave services and return system information.
        
        Returns:
            Dictionary with connection status and system information
        """
        try:
            if not DWAVE_SYSTEM_AVAILABLE:
                return {
                    'status': 'error',
                    'message': 'dwave-system not available',
                    'simulator_available': DIMOD_AVAILABLE
                }
            
            # Test simulator
            test_bqm = BinaryQuadraticModel({'x': 1, 'y': -1}, {('x', 'y'): 2}, 'BINARY')
            sim_result = self.sim_sampler.sample(test_bqm, num_reads=10)
            
            result = {
                'status': 'success',
                'simulator_available': True,
                'simulator_test': 'passed'
            }
            
            # Test QPU if available
            if self.config.use_real_qpu:
                try:
                    qpu_sampler = DWaveSampler()
                    solver_info = {
                        'name': qpu_sampler.solver.name,
                        'properties': dict(qpu_sampler.solver.properties),
                        'parameters': list(qpu_sampler.solver.parameters.keys())
                    }
                    qpu_sampler.close()
                    
                    result.update({
                        'qpu_available': True,
                        'qpu_test': 'passed',
                        'solver_info': solver_info
                    })
                    
                except Exception as e:
                    result.update({
                        'qpu_available': False,
                        'qpu_test': 'failed',
                        'qpu_error': str(e)
                    })
            else:
                result['qpu_available'] = False
                result['qpu_test'] = 'not_configured'
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'simulator_available': False
            }


# Convenience functions for integration with existing framework

def solve_benders_master_with_dwave(f_coeffs: np.ndarray,
                                   D_matrix: np.ndarray,
                                   d_vector: np.ndarray,
                                   optimality_cuts: List,
                                   feasibility_cuts: List,
                                   B_matrix: Optional[np.ndarray] = None,
                                   b_vector: Optional[np.ndarray] = None,
                                   Ny: int = None,
                                   config: Optional[Dict] = None,
                                   dwave_config: Optional[DWaveConfig] = None,
                                   logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Convenience function to solve Benders master problem with D-Wave.
    
    This function provides a direct interface compatible with existing QAOA solvers
    in the optimization framework.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        adapter = DWaveQPUAdapter(config=dwave_config, logger=logger)
        return adapter.solve_benders_master_with_dwave(
            f_coeffs, D_matrix, d_vector, optimality_cuts, feasibility_cuts,
            B_matrix, b_vector, Ny, config
        )
    except Exception as e:
        logger.error(f"Error in D-Wave solver: {e}")
        return {"error": str(e)}


def create_dwave_food_optimizer(optimizer: Any, 
                               dwave_config: Optional[DWaveConfig] = None) -> DWaveQPUAdapter:
    """
    Create a D-Wave adapter for an existing food production optimizer.
    
    Args:
        optimizer: FoodProductionOptimizer instance
        dwave_config: D-Wave configuration parameters
        
    Returns:
        DWaveQPUAdapter instance configured for the optimizer
    """
    if hasattr(optimizer, 'logger'):
        logger = optimizer.logger
    else:
        logger = logging.getLogger(__name__)
    
    return DWaveQPUAdapter(config=dwave_config, logger=logger)


def get_free_dwave_analysis(max_budget_usd: float = 100.0,
                           use_real_qpu: bool = False,
                           logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Get a complete free analysis of D-Wave capabilities for different complexity levels.
    
    Args:
        max_budget_usd: Maximum budget for QPU usage
        use_real_qpu: Whether to configure for real QPU (still won't use it for estimation)
        logger: Logger instance
        
    Returns:
        Comprehensive analysis report
    """
    config = DWaveConfig(
        use_real_qpu=use_real_qpu,
        max_budget_usd=max_budget_usd,
        estimate_cost_only=False  # Allow full analysis
    )
    
    adapter = DWaveQPUAdapter(config=config, logger=logger)
    return adapter.run_free_complexity_analysis()


def estimate_dwave_cost_for_problem(num_farms: int,
                                   num_foods: int,
                                   num_reads: int = 1000,
                                   budget_usd: float = 100.0) -> Dict[str, Any]:
    """
    Estimate D-Wave cost for a specific problem size.
    
    Args:
        num_farms: Number of farms
        num_foods: Number of food types
        num_reads: Number of annealing cycles
        budget_usd: Available budget
        
    Returns:
        Cost estimation and recommendations
    """
    config = DWaveConfig(num_reads=num_reads, max_budget_usd=budget_usd)
    adapter = DWaveQPUAdapter(config=config)
    
    return adapter.get_complexity_recommendation(num_farms, num_foods, budget_usd)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test D-Wave connection
    config = DWaveConfig(use_real_qpu=False)  # Use simulator for testing
    adapter = DWaveQPUAdapter(config=config, logger=logger)
    
    connection_test = adapter.test_connection()
    print("D-Wave Connection Test:")
    print(f"Status: {connection_test}")
    
    # Test simple QUBO problem
    if connection_test.get('simulator_available'):
        print("\nTesting simple QUBO problem...")
        
        # Create a small test QUBO
        Q = np.array([[1, -2], [-2, 1]])
        bqm = adapter.create_bqm_from_qubo(Q)
        
        result = adapter._solve_bqm(bqm)
        print(f"Test result: {result}")
        
        print(f"Performance metrics: {adapter._get_performance_metrics()}")
