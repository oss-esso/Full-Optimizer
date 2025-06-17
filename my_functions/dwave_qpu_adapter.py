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
import sys
import os
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
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logging.warning("matplotlib not available. Plotting functionality will be limited.")
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:
    from dwave.system import DWaveSampler, EmbeddingComposite, LazyFixedEmbeddingComposite
    from dwave.system.composites import FixedEmbeddingComposite
    from dwave.cloud.exceptions import SolverNotFoundError
    # Add hybrid solver imports
    try:
        from dwave.system import LeapHybridSampler, LeapHybridBQMSampler, LeapHybridCQMSampler
        HYBRID_AVAILABLE = True
    except ImportError:
        LeapHybridSampler = None
        LeapHybridBQMSampler = None
        LeapHybridCQMSampler = None
        HYBRID_AVAILABLE = False
    DWAVE_SYSTEM_AVAILABLE = True
except ImportError:
    logging.warning("dwave-system not available. Real QPU access will be disabled.")
    DWaveSampler = None
    EmbeddingComposite = None
    LazyFixedEmbeddingComposite = None
    FixedEmbeddingComposite = None
    SolverNotFoundError = None
    LeapHybridSampler = None
    LeapHybridBQMSampler = None
    LeapHybridCQMSampler = None
    DWAVE_SYSTEM_AVAILABLE = False
    HYBRID_AVAILABLE = False

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
    
    # Solver configuration
    use_real_qpu: bool = False  # Whether to use real QPU or simulator
    use_hybrid: bool = True     # Whether to use hybrid solvers (recommended)
    solver_name: Optional[str] = None  # Specific solver name
    solver_type: str = 'auto'   # 'qpu', 'hybrid', 'simulator', 'auto'
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
    
    # Hybrid solver parameters - FIXED
    time_limit: float = 3.0    # Time limit for hybrid solvers (seconds) - minimum 3s for hybrid
    qpu_time_limit: float = 0.01  # QPU time limit (10ms = 0.01s) for real QPU


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
        'micro': ComplexityLevel(
            name='micro',
            num_farms=2,
            num_foods=3,
            num_nutrients=3,
            num_constraints=5,
            description='Micro test case: 2 farms, 3 crops (6 variables)'
        ),
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
        self.hybrid_sampler = None
        self.active_sampler = None
        self.sampler_type = None
        
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
        """Initialize D-Wave samplers (simulator, hybrid, and QPU)."""
        try:
            # Always initialize simulated annealing sampler (CPU-based)
            self.sim_sampler = SimulatedAnnealingSampler()
            self.logger.info("✓ Initialized simulated annealing sampler (CPU)")
            
            # Initialize hybrid solver if available
            if HYBRID_AVAILABLE and self.config.solver_type in ['hybrid', 'auto']:
                try:
                    self.hybrid_sampler = LeapHybridBQMSampler()
                    self.logger.info("✓ Initialized D-Wave Leap Hybrid BQM sampler")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize hybrid sampler: {e}")
            
            # Try to initialize real QPU sampler
            if self.config.use_real_qpu and DWAVE_SYSTEM_AVAILABLE:
                try:
                    if self.config.solver_name:
                        self.qpu_sampler = DWaveSampler(solver=self.config.solver_name)
                    else:
                        self.qpu_sampler = DWaveSampler()
                    
                    # Wrap with embedding composite
                    self.qpu_sampler = EmbeddingComposite(self.qpu_sampler)
                    
                    self.logger.info(f"✓ Initialized D-Wave QPU sampler: {self.qpu_sampler.solver.name}")
                    
                except (SolverNotFoundError, Exception) as e:
                    self.logger.warning(f"Failed to initialize D-Wave QPU: {e}")
                    self.qpu_sampler = None
            
            # Select active sampler based on configuration and availability
            self._select_active_sampler()
                
        except Exception as e:
            self.logger.error(f"Error initializing samplers: {e}")
            # Fallback to simulator
            self.active_sampler = self.sim_sampler
            self.sampler_type = 'simulator'
    
    def _select_active_sampler(self):
        """Select the active sampler based on configuration and availability."""
        if self.config.solver_type == 'qpu' and self.qpu_sampler:
            self.active_sampler = self.qpu_sampler
            self.sampler_type = 'qpu'
        elif self.config.solver_type == 'hybrid' and self.hybrid_sampler:
            self.active_sampler = self.hybrid_sampler
            self.sampler_type = 'hybrid'
        elif self.config.solver_type == 'simulator':
            self.active_sampler = self.sim_sampler
            self.sampler_type = 'simulator'
        elif self.config.solver_type == 'auto':
            # Auto-select: prefer hybrid > simulator > qpu
            if self.hybrid_sampler:
                self.active_sampler = self.hybrid_sampler
                self.sampler_type = 'hybrid'
            elif self.sim_sampler:
                self.active_sampler = self.sim_sampler
                self.sampler_type = 'simulator'
            elif self.qpu_sampler:
                self.active_sampler = self.qpu_sampler
                self.sampler_type = 'qpu'
        else:
            # Fallback to simulator
            self.active_sampler = self.sim_sampler
            self.sampler_type = 'simulator'
        
        self.logger.info(f"Selected sampler: {self.sampler_type}")
    def switch_solver(self, solver_type: str):
        """Switch to a different solver type."""
        old_type = self.sampler_type
        self.config.solver_type = solver_type
        self._select_active_sampler()
        self.logger.info(f"Switched solver from {old_type} to {self.sampler_type}")
    
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
        if not DIMOD_AVAILABLE:
            raise ImportError("dimod library required for BQM creation")
        
        n_vars = Q_matrix.shape[0]
        
        if variable_labels is None:
            variable_labels = list(range(n_vars))
        elif len(variable_labels) != n_vars:
            raise ValueError(f"Number of labels ({len(variable_labels)}) must match matrix size ({n_vars})")
        
        # Create BQM with proper variable type specification
        bqm = BinaryQuadraticModel(vartype='BINARY')
        
        # Add linear terms (diagonal elements)
        for i in range(n_vars):
            if abs(Q_matrix[i, i]) > 1e-10:  # Only add non-zero terms
                bqm.add_variable(variable_labels[i], Q_matrix[i, i])
        
        # Add quadratic terms (off-diagonal elements)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if abs(Q_matrix[i, j]) > 1e-10:  # Only add non-zero terms
                    bqm.add_interaction(variable_labels[i], variable_labels[j], Q_matrix[i, j])
        
        # Add offset
        bqm.offset = offset
          # Validate BQM
        if len(bqm.variables) == 0:
            self.logger.warning("Created BQM has no variables - check QUBO matrix")
        
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
        
        This method converts the Benders master problem to QUBO format
        and solves it using the configured D-Wave sampler.
        
        Args:
            f_coeffs: Objective coefficients
            D_matrix: Constraint matrix  
            d_vector: Constraint RHS
            optimality_cuts: List of optimality cuts
            feasibility_cuts: List of feasibility cuts
            B_matrix: Additional constraint matrix
            b_vector: Additional constraint RHS
            Ny: Number of variables
            config: QUBO conversion configuration
            
        Returns:
            Dictionary with solution and metadata
        """
        try:
            wall_start = time.time()
            
            # Import QUBO converter
            if convert_benders_master_to_qubo is None:
                return {"error": "QUBO converter not available"}
            
            # Default configuration for QUBO conversion
            default_config = {
                "eta_min": -100.0,
                "eta_max": 100.0,
                "eta_num_bits": 4,
                "penalty_coefficient": 1000.0,
                "penalty_slack_num_bits": 3
            }
            
            if config:
                default_config.update(config)
            
            # Convert Benders master to QUBO
            qubo_result = convert_benders_master_to_qubo(
                f_coeffs=f_coeffs,
                D_matrix=D_matrix,
                d_vector=d_vector,
                optimality_cuts=optimality_cuts,
                feasibility_cuts=feasibility_cuts,
                B_matrix=B_matrix,
                b_vector=b_vector,
                Ny=Ny,
                **default_config
            )
            
            if qubo_result.get('error'):
                return qubo_result
            
            Q_full = qubo_result['Q']
            variable_mapping = qubo_result.get('variable_mapping', {})
            
            # Create BQM from QUBO matrix
            bqm = self.create_bqm_from_qubo(Q_full)
            
            self.logger.info(f"Solving Benders master problem with {len(bqm.variables)} variables")
            
            # Estimate cost before solving
            if not self.config.estimate_cost_only:
                cost_estimation = self.estimate_qpu_cost(len(bqm.variables))
                self.logger.info(f"Estimated cost: ${cost_estimation.estimated_cost_usd:.6f}")
                
                if cost_estimation.estimated_cost_usd > self.config.max_budget_usd:
                    return {
                        "error": f"Estimated cost ${cost_estimation.estimated_cost_usd:.6f} exceeds budget ${self.config.max_budget_usd}",
                        "cost_estimation": cost_estimation
                    }
            
            # Solve using D-Wave
            result = self._solve_bqm(bqm)
            
            if result.get('error'):
                return {"error": f"D-Wave solving failed: {result['error']}"}
            
            # Extract solution
            sample = result['sample']
            
            # Convert BQM solution back to original variables
            solution = np.zeros(Ny if Ny else len(f_coeffs))
            
            for var, value in sample.items():
                if isinstance(var, int) and var < len(solution):
                    solution[var] = value
                elif str(var).startswith('y_') and var in variable_mapping:
                    orig_idx = variable_mapping[var]
                    if orig_idx < len(solution):
                        solution[orig_idx] = value
            
            wall_time = time.time() - wall_start
            
            # Calculate objective value
            objective = float(np.dot(f_coeffs.flatten(), solution))
            
            self.metrics['successful_calls'] += 1
            self.metrics['total_wall_time'] += wall_time
            self.metrics['problem_sizes'].append(len(bqm.variables))
            self.metrics['energies'].append(result['energy'])
            
            return {
                'solution': solution,
                'objective': objective,
                'energy': result['energy'],
                'num_occurrences': result['num_occurrences'],
                'chain_break_fraction': result.get('chain_break_fraction', 0.0),
                'timing': result.get('timing', {}),
                'wall_time': wall_time,
                'problem_size': len(bqm.variables),
                'qubo_matrix': Q_full,
                'bqm_info': {
                    'num_variables': len(bqm.variables),
                    'num_interactions': len(bqm.quadratic),
                    'offset': bqm.offset
                },
                'metrics': self._get_performance_metrics()
            }
            
        except Exception as e:
            self.logger.error(f"Error solving Benders master with D-Wave: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}
        
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
        """Solve a BinaryQuadraticModel using the active D-Wave sampler."""
        if not DIMOD_AVAILABLE:
            return {'error': 'dimod library not available'}
        
        self.metrics['qpu_calls'] += 1
        
        try:
            # Validate BQM before solving
            if len(bqm.variables) == 0:
                return {'error': 'BQM has no variables'}
            
            # Prepare sampler parameters based on sampler type
            if self.sampler_type == 'hybrid':
                # Hybrid BQM sampler parameters
                sampler_params = {
                    'time_limit': max(self.config.time_limit, 3.0),  # Minimum 3 seconds for hybrid
                    'label': f'Food-Optimization-{int(time.time())}'
                }
                
                # Optional hybrid parameters
                if hasattr(self.config, 'hybrid_solver_params'):
                    sampler_params.update(self.config.hybrid_solver_params)
                
            elif self.sampler_type == 'qpu':
                # QPU-specific parameters
                sampler_params = {
                    'num_reads': self.config.num_reads,
                    'annealing_time': self.config.annealing_time,
                    'programming_thermalization': self.config.programming_thermalization,
                    'readout_thermalization': self.config.readout_thermalization,
                    'reduce_intersample_correlation': self.config.reduce_intersample_correlation,
                    'reinitialize_state': self.config.reinitialize_state,
                    'postprocess': self.config.postprocess
                }
                
                # Add QPU time limit (10ms)
                total_qpu_time_us = (
                    self.config.annealing_time + 
                    self.config.programming_thermalization + 
                    self.config.readout_thermalization
                ) * self.config.num_reads
                
                # Limit to 10ms total QPU time if specified
                if hasattr(self.config, 'qpu_time_limit'):
                    max_qpu_time_us = self.config.qpu_time_limit * 1_000_000  # Convert to microseconds
                    if total_qpu_time_us > max_qpu_time_us:
                        # Reduce num_reads to fit time limit
                        max_reads = int(max_qpu_time_us / (
                            self.config.annealing_time + 
                            self.config.programming_thermalization + 
                            self.config.readout_thermalization
                        ))
                        sampler_params['num_reads'] = max(1, max_reads)
                        self.logger.warning(f"Reduced num_reads to {sampler_params['num_reads']} to fit {self.config.qpu_time_limit*1000:.1f}ms time limit")
                
                # Chain strength and embedding parameters
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
                    
            else:  # simulator
                # Simulated annealing parameters
                sampler_params = {
                    'num_reads': min(self.config.num_reads, 1000),  # Limit for faster simulation
                    'beta_range': [0.1, 10.0],
                    'num_sweeps': 1000,
                    'seed': None
                }
            
            # Log the submission
            self.logger.info(f"Submitting BQM to {self.sampler_type} sampler:")
            self.logger.info(f"  Variables: {len(bqm.variables)}")
            self.logger.info(f"  Interactions: {len(bqm.quadratic)}")
            self.logger.info(f"  Parameters: {sampler_params}")
            
            qpu_start = time.time()
            
            # Submit to sampler
            sampleset = self.active_sampler.sample(bqm, **sampler_params)
            
            qpu_time = time.time() - qpu_start
            self.metrics['total_qpu_time'] += qpu_time
            
            # Validate sampleset
            if len(sampleset) == 0:
                return {'error': 'No samples returned from solver'}
            
            # Extract best solution
            best_sample = sampleset.first
            sample_dict = dict(best_sample.sample)
            energy = best_sample.energy
            num_occurrences = best_sample.num_occurrences
            
            # Extract chain break information (QPU only)
            chain_break_fraction = 0.0
            if (hasattr(sampleset, 'data_vectors') and 
                'chain_break_fraction' in sampleset.data_vectors):
                chain_break_fractions = sampleset.data_vectors['chain_break_fraction']
                if len(chain_break_fractions) > 0:
                    chain_break_fraction = chain_break_fractions[0]
                    self.metrics['chain_breaks'] += int(chain_break_fraction > 0)
            
            # Extract timing information
            timing_info = {}
            if hasattr(sampleset, 'info') and 'timing' in sampleset.info:
                timing_info = sampleset.info['timing']
                # Log actual QPU time if available
                if 'qpu_access_time' in timing_info:
                    actual_qpu_time_us = timing_info['qpu_access_time']
                    self.logger.info(f"Actual QPU time: {actual_qpu_time_us/1000:.2f}ms")
            
            # Log results
            self.logger.info(f"{self.sampler_type.title()} sampling completed:")
            self.logger.info(f"  Wall time: {qpu_time:.3f}s")
            self.logger.info(f"  Energy: {energy:.6f}")
            self.logger.info(f"  Occurrences: {num_occurrences}")
            
            if chain_break_fraction > 0:
                self.logger.warning(f"Chain breaks detected: {chain_break_fraction:.3f}")
            
            return {
                'sample': sample_dict,
                'energy': energy,
                'num_occurrences': num_occurrences,
                'chain_break_fraction': chain_break_fraction,
                'timing': timing_info,
                'qpu_time': qpu_time,
                'sampler_type': self.sampler_type,
                'sampleset_info': sampleset.info if hasattr(sampleset, 'info') else {},
                'bqm_stats': {
                    'num_variables': len(bqm.variables),
                    'num_interactions': len(bqm.quadratic),
                    'offset': bqm.offset
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error during {self.sampler_type} sampling: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
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
                         complexity_level: Optional[str] = None,
                         time_limit_seconds: Optional[float] = None) -> CostEstimation:
        """
        Estimate the cost of running a problem on D-Wave QPU with proper time limits.
        
        Args:
            problem_size: Number of variables in the problem
            num_reads: Number of annealing cycles (uses config default if None)
            complexity_level: Name of complexity level (for reference)
            time_limit_seconds: Optional time limit in seconds (e.g., 0.01 for 10ms)
            
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
        
        # Check feasibility based on QPU limits
        max_qubits = 5000  # Current D-Wave Advantage systems
        is_feasible = estimated_qubits <= max_qubits
        
        if not is_feasible:
            warnings.append(f"Problem may exceed QPU capacity ({estimated_qubits} > {max_qubits} qubits)")
            recommendations.append("Consider problem decomposition or use classical algorithms")
        
        # Calculate QPU time requirements
        base_annealing_time = self.config.annealing_time  # microseconds (default: 20μs)
        programming_time = self.config.programming_thermalization  # microseconds (default: 1000μs)
        readout_time = self.config.readout_thermalization  # microseconds (default: 1000μs)
        
        total_time_per_read = base_annealing_time + programming_time + readout_time
        
        # Apply time limit if specified (e.g., 10ms = 0.01s)
        if time_limit_seconds is not None:
            max_qpu_time_us = time_limit_seconds * 1_000_000  # Convert to microseconds
            
            # Calculate maximum reads that fit within time limit
            max_reads_in_limit = int(max_qpu_time_us / total_time_per_read)
            
            if num_reads > max_reads_in_limit:
                warnings.append(f"Requested {num_reads} reads exceeds {time_limit_seconds*1000:.1f}ms limit")
                recommendations.append(f"Reduce num_reads to {max_reads_in_limit} to fit time limit")
                # Use the limited number of reads for cost calculation
                effective_num_reads = max_reads_in_limit
            else:
                effective_num_reads = num_reads
        else:
            effective_num_reads = num_reads
        
        # Calculate total QPU time
        total_qpu_time = total_time_per_read * effective_num_reads
        
        # Apply overhead factor for programming and setup
        total_qpu_time *= self.DWAVE_PRICING['overhead_factor']
        
        # Convert to seconds and calculate cost
        total_qpu_time_seconds = total_qpu_time / 1_000_000  # Convert microseconds to seconds
        
        # Updated pricing model (more accurate for 2025)
        cost_per_second = 2.5  # $2.50 per second of QPU time (realistic estimate)
        base_cost = total_qpu_time_seconds * cost_per_second
        
        # Apply minimum charge
        cost_usd = max(base_cost, self.DWAVE_PRICING['minimum_charge'])
        
        # Add time limit specific recommendations
        if time_limit_seconds is not None:
            if time_limit_seconds <= 0.01:  # 10ms or less
                recommendations.append("Using 10ms limit - very cost-effective for testing")
                recommendations.append(f"Effective num_reads: {effective_num_reads}")
            elif time_limit_seconds <= 0.1:  # 100ms or less
                recommendations.append("Short time limit - good for budget-conscious testing")
            
            # Calculate time utilization
            time_utilization = (total_qpu_time_seconds / time_limit_seconds) * 100 if time_limit_seconds > 0 else 0
            if time_utilization < 50:
                recommendations.append(f"Low time utilization ({time_utilization:.1f}%) - can increase num_reads")
        
        # Add cost warnings and recommendations
        if cost_usd > self.config.warn_cost_threshold:
            warnings.append(f"Estimated cost (${cost_usd:.6f}) exceeds warning threshold")
            
        if cost_usd > self.config.max_budget_usd:
            warnings.append(f"Estimated cost (${cost_usd:.6f}) exceeds budget (${self.config.max_budget_usd})")
            recommendations.append("Reduce num_reads, use time limit, or use simulated annealing")
        
        # Size-based recommendations
        if problem_size < 50:
            recommendations.append("Small problem - classical solvers may be faster")
        elif problem_size < 200:
            recommendations.append("Good candidate for quantum annealing")
        else:
            recommendations.append("Large problem - monitor embedding quality and chain breaks")
        
        # QPU vs Hybrid recommendations
        if effective_num_reads < 50:
            recommendations.append("Low num_reads - consider hybrid solver for better results")
        
        return CostEstimation(
            complexity_level=complexity_level or 'custom',
            num_variables=problem_size,
            estimated_qubits=estimated_qubits,
            estimated_qpu_time_us=total_qpu_time,
            estimated_cost_usd=cost_usd,
            num_reads=effective_num_reads,  # Use effective reads (after time limit applied)
            is_feasible=is_feasible,
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
            num_foods: Number of food types        budget_usd: Available budget (uses config default if None)
            
        Returns:
            Dictionary with recommendation and analysis
        """
        if budget_usd is None:
            budget_usd = self.config.max_budget_usd
        
        problem_size = num_farms * num_foods
        estimation = self.estimate_qpu_cost(problem_size, complexity_level='custom')
        
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
        
        recommendation['affordable_levels'] = affordable_levels
        
        return recommendation
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to D-Wave services and return system information."""
        try:
            result = {
                'status': 'success',
                'simulator_available': DIMOD_AVAILABLE,
                'hybrid_available': HYBRID_AVAILABLE and self.hybrid_sampler is not None,
                'qpu_available': self.qpu_sampler is not None,
                'dwave_system_available': DWAVE_SYSTEM_AVAILABLE
            }
            
            # Test simulator (always available)
            if DIMOD_AVAILABLE:
                test_bqm = BinaryQuadraticModel({'x': 1, 'y': -1}, {('x', 'y'): 2}, 'BINARY')
                sim_result = self.sim_sampler.sample(test_bqm, num_reads=10)
                result['simulator_test'] = 'passed'
            
            # Test hybrid solver
            if result['hybrid_available']:
                try:
                    # Simple test for hybrid solver
                    test_bqm = BinaryQuadraticModel({'x': 1, 'y': -1}, {('x', 'y'): 2}, 'BINARY')
                    hybrid_result = self.hybrid_sampler.sample(test_bqm, time_limit=3)
                    result['hybrid_test'] = 'passed'
                except Exception as e:
                    result['hybrid_test'] = f'failed: {e}'
            
            # Test QPU if available
            if self.qpu_sampler:
                try:
                    # Get solver info without actually solving
                    if hasattr(self.qpu_sampler, 'child'):
                        solver_info = {
                            'name': self.qpu_sampler.child.solver.name,
                            'properties': dict(self.qpu_sampler.child.solver.properties),
                            'parameters': list(self.qpu_sampler.child.solver.parameters.keys())
                        }
                    else:
                        solver_info = {'name': 'Unknown QPU'}
                    
                    result.update({
                        'qpu_test': 'connection_ok',
                        'solver_info': solver_info
                    })
                    
                except Exception as e:
                    result.update({
                        'qpu_test': f'failed: {e}',
                        'qpu_available': False
                    })
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'simulator_available': False,
                'hybrid_available': False,
                'qpu_available': False
            }
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            'qpu_calls': self.metrics['qpu_calls'],
            'successful_calls': self.metrics['successful_calls'],
            'total_qpu_time': self.metrics['total_qpu_time'],
            'total_wall_time': self.metrics['total_wall_time'],
            'avg_qpu_time': self.metrics['total_qpu_time'] / max(self.metrics['qpu_calls'], 1),
            'chain_breaks': self.metrics['chain_breaks'],
            'embedding_retries': self.metrics['embedding_retries'],
            'problem_sizes': self.metrics['problem_sizes'][-10:],  # Last 10 problem sizes
            'energies': self.metrics['energies'][-10:],  # Last 10 energies
            'success_rate': self.metrics['successful_calls'] / max(self.metrics['qpu_calls'], 1)
        }

    def random_subqubo(self, bqm: BinaryQuadraticModel, num_vars: int) -> BinaryQuadraticModel:
        """
        Extract a random sub-BQM of specified size from a larger BQM.
        
        Args:
            bqm: Source BinaryQuadraticModel
            num_vars: Number of variables in the sub-BQM
            
        Returns:
            Random sub-BQM with num_vars variables
        """
        if not DIMOD_AVAILABLE:
            raise ImportError("dimod library required for BQM operations")
        
        all_vars = list(bqm.variables)
        if num_vars >= len(all_vars):
            return bqm.copy()
        
        # Randomly select variables
        np.random.seed(int(time.time() * 1000) % 2**32)  # Time-based seed
        selected_vars = np.random.choice(all_vars, size=num_vars, replace=False)
        
        # Create sub-BQM with only selected variables
        sub_bqm = bqm.copy()
        vars_to_remove = set(all_vars) - set(selected_vars)
        
        for var in vars_to_remove:
            sub_bqm.remove_variable(var)
        
        return sub_bqm
    
    def benchmark_sampler(self, sampler: Any, bqm: BinaryQuadraticModel, num_reads: int) -> float:
        """
        Benchmark a sampler on a BQM and return elapsed time.
        
        Args:
            sampler: D-Wave sampler instance
            bqm: BinaryQuadraticModel to solve
            num_reads: Number of reads for the sampler
            
        Returns:
            Elapsed time in seconds
        """
        if not DIMOD_AVAILABLE:
            raise ImportError("dimod library required for sampling")
        
        # Prepare sampler parameters based on type
        if hasattr(sampler, 'parameters') and 'num_reads' in sampler.parameters:
            sample_params = {'num_reads': num_reads}
        else:
            sample_params = {}
        
        # Add other common parameters
        if hasattr(sampler, 'parameters'):
            if 'num_sweeps' in sampler.parameters:
                sample_params['num_sweeps'] = 1000
            if 'beta_range' in sampler.parameters:
                sample_params['beta_range'] = [0.1, 10.0]
        
        start_time = time.time()
        try:
            sampleset = sampler.sample(bqm, **sample_params)
            # Ensure we actually get the results
            _ = sampleset.first
        except Exception as e:
            self.logger.warning(f"Sampling failed: {e}")
            return float('inf')
        
        elapsed_time = time.time() - start_time
        return elapsed_time
    
    def fit_scaling_curve(self, sizes: List[int], times: List[float]) -> Tuple[float, float]:
        """
        Fit a power-law scaling curve to size vs time data using log-log linear regression.
        
        Args:
            sizes: List of problem sizes
            times: List of corresponding solve times
            
        Returns:
            Tuple of (exponent, intercept) for the power law: time = intercept * size^exponent
        """
        if len(sizes) < 2 or len(times) < 2:
            self.logger.warning("Need at least 2 data points for curve fitting")
            return 1.0, 1.0
        
        # Filter out zero or negative times
        valid_data = [(s, t) for s, t in zip(sizes, times) if t > 0 and s > 0]
        if len(valid_data) < 2:
            self.logger.warning("Not enough valid data points for curve fitting")
            return 1.0, 1.0
        
        sizes_valid, times_valid = zip(*valid_data)
        
        # Log-log linear regression
        log_sizes = np.log(sizes_valid)
        log_times = np.log(times_valid)
        
        # Fit: log(time) = log(intercept) + exponent * log(size)
        coeffs = np.polyfit(log_sizes, log_times, 1)
        exponent = coeffs[0]
        log_intercept = coeffs[1]
        intercept = np.exp(log_intercept)
        
        return exponent, intercept
    
    def estimate_full_time(self, 
                          bqm: BinaryQuadraticModel,
                          sampler: Any,
                          sample_sizes: List[int],
                          num_reads: int = 100,
                          plot_results: bool = True) -> Dict[str, Any]:
        """
        Estimate solve time for full BQM by benchmarking on smaller sub-problems.
        
        Args:
            bqm: Full BinaryQuadraticModel to estimate
            sampler: D-Wave sampler to use for benchmarking
            sample_sizes: List of sub-problem sizes to benchmark
            num_reads: Number of reads for each benchmark
            plot_results: Whether to generate scaling plots
            
        Returns:
            Dictionary with scaling analysis results
        """
        if not DIMOD_AVAILABLE:
            raise ImportError("dimod library required for scaling analysis")
        
        full_size = len(bqm.variables)
        self.logger.info(f"Starting scaling analysis for BQM with {full_size} variables")
        self.logger.info(f"Sample sizes: {sample_sizes}")
        self.logger.info(f"Num reads per benchmark: {num_reads}")
        
        # Benchmark each sample size
        benchmark_results = []
        
        for size in sample_sizes:
            if size >= full_size:
                self.logger.warning(f"Sample size {size} >= full size {full_size}, using full BQM")
                sub_bqm = bqm
                actual_size = full_size
            else:
                sub_bqm = self.random_subqubo(bqm, size)
                actual_size = len(sub_bqm.variables)
            
            self.logger.info(f"Benchmarking sub-problem with {actual_size} variables...")
            
            # Run benchmark
            elapsed_time = self.benchmark_sampler(sampler, sub_bqm, num_reads)
            
            if elapsed_time == float('inf'):
                self.logger.warning(f"Benchmark failed for size {actual_size}")
                continue
            
            benchmark_results.append((actual_size, elapsed_time))
            print(f"  Size {actual_size}: {elapsed_time:.4f} seconds")
        
        if len(benchmark_results) < 2:
            return {"error": "Not enough successful benchmarks for scaling analysis"}
        
        # Extract sizes and times
        sizes, times = zip(*benchmark_results)
        
        # Fit scaling curve
        exponent, intercept = self.fit_scaling_curve(list(sizes), list(times))
        
        # Predict full problem time - FIX: Add this missing calculation
        predicted_time = intercept * (full_size ** exponent)
        
        print(f"\nScaling Analysis Results:")
        print(f"  Fitted exponent: {exponent:.3f}")
        print(f"  Intercept: {intercept:.6f}")
        print(f"  Power law: time = {intercept:.6f} * size^{exponent:.3f}")
        print(f"  Predicted time for full problem ({full_size} vars): {predicted_time:.4f} seconds")
        
        # Estimate costs if using QPU
        estimated_cost = 0.0
        if hasattr(sampler, 'solver') or 'qpu' in str(type(sampler)).lower():
            # Rough QPU cost estimation
            qpu_time_us = predicted_time * 1000000  # Convert to microseconds
            cost_per_us = 0.00015  # Approximate cost per microsecond
            estimated_cost = qpu_time_us * cost_per_us
            print(f"  Estimated QPU cost: ${estimated_cost:.4f}")
        
        # Generate plot if requested
        plot_path = None
        if plot_results and MATPLOTLIB_AVAILABLE:
            plot_path = self._plot_scaling_results(sizes, times, exponent, intercept, full_size, predicted_time)
        
        return {
            'benchmark_results': benchmark_results,
            'exponent': exponent,
            'intercept': intercept,
            'full_size': full_size,
            'predicted_time': predicted_time,
            'estimated_cost': estimated_cost,
            'plot_path': plot_path,
            'sampler_type': str(type(sampler).__name__)
        }
    
    def _plot_scaling_results(self, 
                             sizes: List[int], 
                             times: List[float],
                             exponent: float,
                             intercept: float,
                             full_size: int,
                             predicted_time: float) -> Optional[str]:
        """Generate scaling analysis plot."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot benchmark data
            ax.loglog(sizes, times, 'bo', markersize=8, label='Benchmark Data')
            
            # Plot fitted curve
            size_range = np.logspace(np.log10(min(sizes)), np.log10(max(sizes) * 2), 100)
            fitted_times = intercept * (size_range ** exponent)
            ax.loglog(size_range, fitted_times, 'r-', linewidth=2, 
                     label=f'Fitted: t = {intercept:.3e} × n^{exponent:.3f}')
            
            # Plot prediction for full problem
            ax.loglog([full_size], [predicted_time], 'rs', markersize=12, 
                     label=f'Prediction ({full_size} vars): {predicted_time:.3f}s')
            
            ax.set_xlabel('Problem Size (Number of Variables)', fontsize=12)
            ax.set_ylabel('Solve Time (Seconds)', fontsize=12)
            ax.set_title('D-Wave Sampler Scaling Analysis\nFood Production Optimization', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
            
            # Add text box with scaling info
            textstr = f'Scaling Exponent: {exponent:.3f}\n'
            if exponent < 1.5:
                textstr += 'Better than quadratic scaling'
            elif exponent < 2.5:
                textstr += 'Approximately quadratic scaling'  
            else:
                textstr += 'Worse than quadratic scaling'
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            
            # Save plot
            results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Results')
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, f'dwave_scaling_analysis_{int(time.time())}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate plot: {e}")
            return None

def create_simple_food_problem() -> Dict[str, Any]:
    """
    Create a simple 2-farm, 3-crop optimization problem for testing.
    
    Returns:
        Dictionary with problem data compatible with food optimization framework
    """
    farms = ['Farm_A', 'Farm_B']
    
    foods = {
        'Wheat': {
            'nutritional_value': 0.7,
            'nutrient_density': 0.6,
            'affordability': 0.9,
            'sustainability': 0.8,
            'environmental_impact': 0.3,
            'calories_per_kg': 3000,
            'protein_g_per_kg': 120,
            'cost_per_kg': 2.5
        },
        'Corn': {
            'nutritional_value': 0.6,
            'nutrient_density': 0.5,
            'affordability': 0.8,
            'sustainability': 0.7,
            'environmental_impact': 0.4,
            'calories_per_kg': 3600,
            'protein_g_per_kg': 90,
            'cost_per_kg': 3.0
        },
        'Beans': {
            'nutritional_value': 0.9,
            'nutrient_density': 0.8,
            'affordability': 0.7,
            'sustainability': 0.9,
            'environmental_impact': 0.2,
            'calories_per_kg': 1400,
            'protein_g_per_kg': 220,
            'cost_per_kg': 4.0
        }
    }
    
    food_groups = {
        'grains': ['Wheat', 'Corn'],
        'legumes': ['Beans']
    }
    
    config = {
        'land_availability': {'Farm_A': 100, 'Farm_B': 80},
        'min_production': {'Wheat': 50, 'Corn': 30, 'Beans': 20},
        'max_production': {'Wheat': 200, 'Corn': 150, 'Beans': 100},
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.20,
            'affordability': 0.20,
            'sustainability': 0.20,
            'environmental_impact': 0.15
        }
    }
    
    return {
        'farms': farms,
        'foods': foods,
        'food_groups': food_groups,
        'config': config,
        'complexity': 'micro',
        'description': 'Simple 2-farm, 3-crop test case for D-Wave validation'
    }


def get_free_dwave_analysis(budget_usd: float = 100.0) -> Dict[str, Any]:
    """
    Perform free D-Wave cost analysis without using any QPU time.
    
    Args:
        budget_usd: Budget constraint for analysis
        
    Returns:
        Dictionary with comprehensive cost analysis
    """
    from datetime import datetime
    
    # Create adapter with estimation-only configuration
    config = DWaveConfig(estimate_cost_only=True, max_budget_usd=budget_usd)
    adapter = DWaveQPUAdapter(config=config)
    
    start_time = time.time()
    
    # Get cost estimations for all complexity levels
    estimations = adapter.estimate_all_complexity_levels()
    
    # Analyze sampler availability
    connection_test = adapter.test_connection()
    
    # Calculate summary statistics
    costs = [est.estimated_cost_usd for est in estimations.values()]
    feasible_levels = [name for name, est in estimations.items() if est.is_feasible]
    affordable_levels = [name for name, est in estimations.items() if est.estimated_cost_usd <= budget_usd]
    
    analysis_duration = time.time() - start_time
    
    return {
        'timestamp': datetime.now().isoformat(),
        'analysis_duration': analysis_duration,
        'budget_usd': budget_usd,
        'estimations': estimations,
        'sampler_info': {
            'simulator_available': connection_test.get('simulator_available', False),
            'hybrid_available': connection_test.get('hybrid_available', False),
            'qpu_configured': connection_test.get('qpu_available', False),
            'hybrid_test': connection_test.get('hybrid_test', 'not_tested')
        },
        'summary': {
            'total_levels': len(estimations),
            'feasible_levels': feasible_levels,
            'affordable_levels': affordable_levels,
            'cost_range': {
                'min': min(costs),
                'max': max(costs),
                'avg': sum(costs) / len(costs)
            }
        },
        'recommendations': _generate_analysis_recommendations(estimations, budget_usd, connection_test)
    }


def estimate_dwave_cost_for_problem(num_farms: int, num_foods: int, 
                                   time_limit_seconds: Optional[float] = None) -> CostEstimation:
    """
    Estimate D-Wave cost for a specific problem size with optional time limit.
    
    Args:
        num_farms: Number of farms
        num_foods: Number of food types  
        time_limit_seconds: Optional time limit (e.g., 0.01 for 10ms)
        
    Returns:
        CostEstimation object
    """
    config = DWaveConfig(estimate_cost_only=True)
    
    if time_limit_seconds is not None:
        config.qpu_time_limit = time_limit_seconds
    
    adapter = DWaveQPUAdapter(config=config)
    problem_size = num_farms * num_foods
    
    return adapter.estimate_qpu_cost(
        problem_size=problem_size, 
        complexity_level='custom',
        time_limit_seconds=time_limit_seconds
    )


def _generate_analysis_recommendations(estimations: Dict[str, CostEstimation], 
                                     budget_usd: float,
                                     connection_test: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on cost analysis."""
    recommendations = []
    
    affordable_count = sum(1 for est in estimations.values() if est.estimated_cost_usd <= budget_usd)
    
    if affordable_count == 0:
        recommendations.append("No complexity levels fit within budget - consider increasing budget or using classical methods")
    elif affordable_count < len(estimations) // 2:
        recommendations.append("Limited complexity levels affordable - focus on smaller problems initially")
    else:
        recommendations.append("Most complexity levels are affordable - good budget for experimentation")
    
    if connection_test.get('hybrid_available'):
        recommendations.append("Hybrid solver available - recommended for most problems")
    else:
        recommendations.append("Hybrid solver not available - check D-Wave Leap access")
    
    if connection_test.get('qpu_available'):
        recommendations.append("QPU access configured - use time limits for cost control")
    else:
        recommendations.append("QPU not configured - hybrid and simulator available")
    
        return recommendations
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
        
        recommendation['affordable_levels'] = affordable_levels
        
        return recommendation
