"""
QUBO Converter Module

This module provides functionality to convert various optimization problems to QUBO
(Quadratic Unconstrained Binary Optimization) form. It supports:

1. Automatic conversion of linear constraints to quadratic penalties
2. Handling of different variable types (binary, integer, continuous)
3. Integration with existing optimization infrastructure
4. Support for both classical and quantum solvers
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import logging
from enum import Enum
import networkx as nx
from my_functions.optimization import FoodProductionOptimizer

class VariableType(Enum):
    """Types of variables in the optimization problem."""
    BINARY = "binary"
    INTEGER = "integer"
    CONTINUOUS = "continuous"

@dataclass
class Variable:
    """Representation of a variable in the optimization problem."""
    name: str
    type: VariableType
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    bits: Optional[int] = None  # Number of bits for integer/continuous encoding

@dataclass
class Constraint:
    """Representation of a constraint in the optimization problem."""
    coefficients: Dict[str, float]  # Maps variable names to their coefficients
    sense: str  # "<=", ">=", or "=="
    rhs: float  # Right-hand side value
    penalty_weight: Optional[float] = None  # Weight for QUBO penalty term

@dataclass
class QUBOModel:
    """Representation of a problem in QUBO form."""
    Q: np.ndarray  # Quadratic terms matrix
    c: np.ndarray  # Linear terms vector
    offset: float  # Constant offset
    variable_map: Dict[str, int]  # Maps variable names to matrix indices
    reverse_map: Dict[int, str]  # Maps matrix indices to variable names
    encoding_info: Dict[str, Dict]  # Information about variable encodings

class QUBOConverter:
    """
    Converts optimization problems to QUBO form.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize the QUBO converter.
        
        Args:
            log_file: Optional file to write logs to
        """
        self.variables: Dict[str, Variable] = {}
        self.objective_terms: Dict[Tuple[str, str], float] = {}  # (var1, var2) -> coeff
        self.constraints: List[Constraint] = []
        self.log_file = log_file
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for the converter."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file) if self.log_file else logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def add_variable(self, variable: Variable):
        """Add a variable to the problem."""
        if variable.name in self.variables:
            raise ValueError(f"Variable {variable.name} already exists")
        
        # Set default bounds if not provided
        if variable.type == VariableType.BINARY:
            variable.lower_bound = variable.lower_bound or 0
            variable.upper_bound = variable.upper_bound or 1
            variable.bits = 1
        elif variable.type == VariableType.INTEGER:
            if variable.lower_bound is None or variable.upper_bound is None:
                raise ValueError("Integer variables must have bounds")
            # Calculate required bits for encoding
            range_size = variable.upper_bound - variable.lower_bound
            variable.bits = int(np.ceil(np.log2(range_size + 1)))
        elif variable.type == VariableType.CONTINUOUS:
            if variable.lower_bound is None or variable.upper_bound is None:
                raise ValueError("Continuous variables must have bounds")
            # Default to 8 bits if not specified
            variable.bits = variable.bits or 8
        
        self.variables[variable.name] = variable
    
    def add_objective_term(self, var1: str, var2: str, coefficient: float):
        """Add a term to the objective function."""
        if var1 not in self.variables or var2 not in self.variables:
            raise ValueError("Variables must be added before using in objective")
        self.objective_terms[(var1, var2)] = coefficient
    
    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the problem."""
        for var in constraint.coefficients:
            if var not in self.variables:
                raise ValueError(f"Variable {var} in constraint not defined")
        self.constraints.append(constraint)
    
    def _encode_integer_variable(self, var: Variable) -> Dict[str, Dict]:
        """
        Encode an integer variable using binary variables.
        Returns mapping of new binary variables and their weights.
        """
        encoding = {}
        base = 1
        for i in range(var.bits):
            bin_var_name = f"{var.name}_bit_{i}"
            encoding[bin_var_name] = {
                "weight": base,
                "original_var": var.name
            }
            base *= 2
        return encoding
    
    def _convert_constraint_to_penalty(self, constraint: Constraint) -> Dict[Tuple[str, str], float]:
        """
        Convert a constraint to penalty terms in the QUBO.
        
        Args:
            constraint: Constraint to convert
            
        Returns:
            Dictionary of penalty terms
        """
        penalty_terms = {}
        
        # Get unique slack variable name
        slack_var = f"slack_{len(self.variables)}"
        while slack_var in self.variables:
            slack_var = f"slack_{len(self.variables)}"
            
        # Add slack variable
        self.add_variable(Variable(slack_var, VariableType.BINARY))
        
        # Convert constraint to equality
        if constraint.sense == "<=":
            # a^T x <= b -> a^T x + s = b
            penalty_terms[(slack_var, slack_var)] = constraint.penalty_weight
            for var, coeff in constraint.coefficients.items():
                penalty_terms[(var, slack_var)] = -2 * coeff * constraint.penalty_weight
                penalty_terms[(var, var)] = coeff * coeff * constraint.penalty_weight
                
        elif constraint.sense == ">=":
            # a^T x >= b -> a^T x - s = b
            penalty_terms[(slack_var, slack_var)] = constraint.penalty_weight
            for var, coeff in constraint.coefficients.items():
                penalty_terms[(var, slack_var)] = 2 * coeff * constraint.penalty_weight
                penalty_terms[(var, var)] = coeff * coeff * constraint.penalty_weight
                
        return penalty_terms
    
    def convert_to_qubo(self) -> QUBOModel:
        """
        Convert the problem to QUBO form.
        
        Returns:
            QUBOModel containing the QUBO matrix and variable mapping
        """
        # First, encode all non-binary variables
        encoded_vars = {}
        var_index_map = {}
        reverse_map = {}
        current_index = 0
        
        # Process original variables
        for var_name, var in self.variables.items():
            if var.type == VariableType.BINARY:
                var_index_map[var_name] = current_index
                reverse_map[current_index] = var_name
                current_index += 1
            else:
                # Encode integer/continuous variables
                encoded = self._encode_integer_variable(var)
                encoded_vars.update(encoded)
                # Add binary variables to index map
                for bin_var in encoded:
                    var_index_map[bin_var] = current_index
                    reverse_map[current_index] = bin_var
                    current_index += 1
        
        n = len(var_index_map)
        Q = np.zeros((n, n))
        c = np.zeros(n)
        offset = 0.0
        
        # Add objective terms
        for (var1, var2), coef in self.objective_terms.items():
            if var1 in var_index_map and var2 in var_index_map:
                i, j = var_index_map[var1], var_index_map[var2]
                if i == j:
                    c[i] += coef
                else:
                    Q[i, j] += coef / 2
                    Q[j, i] += coef / 2
            else:
                # Handle encoded variables
                self._add_encoded_objective_term(var1, var2, coef, encoded_vars, var_index_map, Q, c)
        
        # Add constraint penalties
        for constraint in self.constraints:
            penalty_terms = self._convert_constraint_to_penalty(constraint)
            for (var1, var2), coef in penalty_terms.items():
                if var1 == "offset" and var2 == "offset":
                    offset += coef
                    continue
                    
                i = var_index_map.get(var1)
                j = var_index_map.get(var2)
                
                if i is not None and j is not None:
                    if i == j:
                        c[i] += coef
                    else:
                        Q[i, j] += coef / 2
                        Q[j, i] += coef / 2
        
        return QUBOModel(
            Q=Q,
            c=c,
            offset=offset,
            variable_map=var_index_map,
            reverse_map=reverse_map,
            encoding_info=encoded_vars
        )
    
    def _add_encoded_objective_term(
        self,
        var1: str,
        var2: str,
        coef: float,
        encoded_vars: Dict[str, Dict],
        var_index_map: Dict[str, int],
        Q: np.ndarray,
        c: np.ndarray
    ):
        """Add objective terms for encoded variables."""
        # Handle cases where variables are encoded
        if var1 in encoded_vars or var2 in encoded_vars:
            # Expand encoded variables and add appropriate terms
            if var1 in encoded_vars:
                for bin_var, info in encoded_vars.items():
                    if info["original_var"] == var1:
                        weight = info["weight"]
                        i = var_index_map[bin_var]
                        if var2 in var_index_map:
                            j = var_index_map[var2]
                            Q[i, j] += coef * weight / 2
                            Q[j, i] += coef * weight / 2
                        elif var2 in encoded_vars:
                            for bin_var2, info2 in encoded_vars.items():
                                if info2["original_var"] == var2:
                                    j = var_index_map[bin_var2]
                                    weight2 = info2["weight"]
                                    Q[i, j] += coef * weight * weight2 / 2
                                    Q[j, i] += coef * weight * weight2 / 2
            
            if var2 in encoded_vars:
                for bin_var, info in encoded_vars.items():
                    if info["original_var"] == var2:
                        weight = info["weight"]
                        j = var_index_map[bin_var]
                        if var1 in var_index_map:
                            i = var_index_map[var1]
                            Q[i, j] += coef * weight / 2
                            Q[j, i] += coef * weight / 2

def convert_food_optimization_to_qubo(optimizer: 'FoodProductionOptimizer') -> QUBOModel:
    """
    Convert a food production optimization problem to QUBO form.
    
    Args:
        optimizer: Instance of FoodProductionOptimizer
        
    Returns:
        QUBOModel representing the problem in QUBO form
    """
    converter = QUBOConverter()
    
    # Add variables for each farm-food combination
    for farm in optimizer.farms:
        for food in optimizer.foods:
            var_name = f"x_{farm}_{food}"
            converter.add_variable(Variable(
                name=var_name,
                type=VariableType.BINARY
            ))
    
    # Add objective terms
    weights = optimizer.parameters['objective_weights']
    for farm in optimizer.farms:
        for food in optimizer.foods:
            var_name = f"x_{farm}_{food}"
            # Add weighted objective terms
            for obj in optimizer.foods[food]:
                weight = weights[obj]
                score = optimizer.foods[food][obj]
                seasonal_factor = optimizer.parameters['seasonal_factors'][food]
                converter.add_objective_term(
                    var_name,
                    var_name,
                    -weight * score * seasonal_factor  # Negative because we're minimizing
                )
    
    # Add constraints
    # Land availability constraints
    for farm in optimizer.farms:
        coeffs = {
            f"x_{farm}_{food}": 1.0
            for food in optimizer.foods
        }
        converter.add_constraint(Constraint(
            coefficients=coeffs,
            sense="<=",
            rhs=optimizer.parameters['land_availability'][farm],
            penalty_weight=1.0
        ))
    
    # Minimum planting area constraints
    for food in optimizer.foods:
        min_area = optimizer.parameters['minimum_planting_area'][food]
        for farm in optimizer.farms:
            var_name = f"x_{farm}_{food}"
            converter.add_constraint(Constraint(
                coefficients={var_name: 1.0},
                sense=">=",
                rhs=min_area,
                penalty_weight=1.0
            ))
    
    # Market demand constraints
    for food in optimizer.foods:
        coeffs = {
            f"x_{farm}_{food}": 1.0
            for farm in optimizer.farms
        }
        converter.add_constraint(Constraint(
            coefficients=coeffs,
            sense=">=",
            rhs=optimizer.parameters['market_demand'][food],
            penalty_weight=1.0
        ))
    
    return converter.convert_to_qubo() 

# --- New functions for Benders Master to QUBO conversion ---

def _add_squared_penalty_to_converter(
    converter: 'QUBOConverter',
    expr_terms: Dict[str, float],  # var_name -> coeff in expr
    const_term: float,
    penalty_P: float
):
    """
    Adds P * (sum(coeff_i * x_i) + const_term)^2 to the QUBO terms.
    x_i are binary variables.
    Modifies converter.objective_terms and converter.overall_offset.
    """
    if not hasattr(converter, 'overall_offset'):
        converter.overall_offset = 0.0

    # Constant part of penalty: P * const_term^2
    converter.overall_offset += penalty_P * (const_term**2)

    # Linear terms from penalty: P * (coeff_i^2 * x_i) + P * (2 * const_term * coeff_i * x_i)
    # Since x_i is binary, x_i^2 = x_i. So, P * coeff_i^2 * x_i is linear.
    for var_i, coeff_i in expr_terms.items():
        linear_coeff_from_penalty = penalty_P * (coeff_i**2 + 2 * const_term * coeff_i)
        
        term_key = tuple(sorted((var_i, var_i))) # Canonical key for linear term
        if term_key not in converter.objective_terms:
            converter.objective_terms[term_key] = 0.0
        converter.objective_terms[term_key] += linear_coeff_from_penalty

    # Quadratic terms from penalty: P * (2 * coeff_i * coeff_j * x_i * x_j) for i!=j
    var_names_in_expr = list(expr_terms.keys())
    for i_idx in range(len(var_names_in_expr)):
        for j_idx in range(i_idx + 1, len(var_names_in_expr)):  # Ensures i_idx < j_idx
            var_i = var_names_in_expr[i_idx]
            var_j = var_names_in_expr[j_idx]
            coeff_i = expr_terms[var_i]
            coeff_j = expr_terms[var_j]
            
            # Full coefficient for x_i * x_j is P * 2 * coeff_i * coeff_j
            quadratic_coeff_for_pair = penalty_P * 2 * coeff_i * coeff_j
            
            term_key = tuple(sorted((var_i, var_j))) # Canonical key
            if term_key not in converter.objective_terms:
                converter.objective_terms[term_key] = 0.0
            converter.objective_terms[term_key] += quadratic_coeff_for_pair


def _finalize_qubo_from_converter_state(converter: 'QUBOConverter', config: Dict) -> QUBOModel:
    """
    Builds Q, c, offset from converter.variables, converter.objective_terms, 
    and converter.overall_offset.
    Assumes all variables in converter.variables are binary.
    """
    var_index_map: Dict[str, int] = {}
    reverse_map: Dict[int, str] = {}
    current_index = 0
    
    for var_name, var_obj in converter.variables.items():
        if var_obj.type != VariableType.BINARY:
            raise ValueError(f"Variable {var_name} is not binary in finalization step for Benders QUBO.")
        var_index_map[var_name] = current_index
        reverse_map[current_index] = var_name
        current_index += 1
        
    n = len(var_index_map)
    Q_matrix = np.zeros((n, n))
    c_vector = np.zeros(n)
    
    for (v1, v2), coeff in converter.objective_terms.items():
        if v1 not in var_index_map or v2 not in var_index_map:
            raise ValueError(f"Unknown variable in objective_terms: {v1} or {v2}")
            
        idx1 = var_index_map[v1]
        idx2 = var_index_map[v2]
        
        if idx1 == idx2:  # Linear term
            c_vector[idx1] += coeff
        else:  # Quadratic term
            # objective_terms stores the full quadratic coefficient for x_i*x_j
            # So, Q_ij = Q_ji = coeff / 2
            Q_matrix[idx1, idx2] += coeff / 2.0
            Q_matrix[idx2, idx1] += coeff / 2.0
            
    final_offset = getattr(converter, 'overall_offset', 0.0)
    
    # Add eta_min to the offset if eta was part of the objective function
    # Objective was f^T y + eta. eta_approx = eta_min + sum(binary_eta_terms...)
    # The sum(binary_eta_terms...) and f^T y are in c_vector. eta_min is a constant.
    if config.get("eta_num_bits", 0) > 0: # Check if eta was actually used
        final_offset += config.get("eta_min", 0.0)

    return QUBOModel(
        Q=Q_matrix,
        c=c_vector,
        offset=final_offset,
        variable_map=var_index_map,
        reverse_map=reverse_map,
        encoding_info={} # All variables are directly binary in this context
    )

def convert_benders_master_to_qubo(
    f_coeffs: np.ndarray,        # Ny x 1 vector for f^T y
    D_matrix: np.ndarray,        # n_dy x Ny matrix for Dy >= d
    d_vector: np.ndarray,        # n_dy x 1 vector
    optimality_cuts: List[np.ndarray],  # List of pi vectors (m x 1)
    feasibility_cuts: List[np.ndarray], # List of pi_ray vectors (m x 1)
    B_matrix: np.ndarray,        # m x Ny matrix (complicating constraint part for y)
    b_vector: np.ndarray,        # m x 1 vector (complicating constraint RHS)
    Ny: int,                     # Number of y variables
    config: Dict,                # QUBO conversion params
    logger: Optional[logging.Logger] = None
) -> QUBOModel:
    """
    Converts a Benders master problem definition to QUBO form.

    The Benders master problem is approximately:
    min_{y, eta}  f^T y + eta
    s.t.
        D y >= d
        eta + (pi_opt^T B) y >= pi_opt^T b  (for each optimality cut pi_opt)
        (pi_feas^T B) y <= pi_feas^T b      (for each feasibility cut pi_feas)
        y binary
    
    Args:
        f_coeffs: Coefficients for y in the objective.
        D_matrix, d_vector: Constraints involving only y variables.
        optimality_cuts: List of dual vectors for optimality cuts.
        feasibility_cuts: List of dual ray vectors for feasibility cuts.
        B_matrix: Matrix B linking y to complicating constraints.
        b_vector: RHS vector b for complicating constraints.
        Ny: Number of binary y variables.
        config: Dictionary with QUBO parameters:
            - eta_min, eta_max, eta_num_bits: for discretizing eta.
            - penalty_slack_num_bits: for slack variables in inequality constraints.
            - penalty_coefficient: base value for penalty terms.
        logger: Optional logger instance.

    Returns:
        A QUBOModel object.
    """
    
    converter = QUBOConverter()
    if logger:
        converter.logger = logger
    # Store config in converter if _finalize_qubo_from_converter_state needs it
    converter.config = config 
    converter.overall_offset = 0.0 # Initialize offset

    # --- 1. Define y variables (binary) ---
    y_var_names = [f"y_{i}" for i in range(Ny)]
    for y_name in y_var_names:
        converter.add_variable(Variable(name=y_name, type=VariableType.BINARY))

    # --- 2. Define eta and its binary representation ---
    eta_min = config.get("eta_min", -100.0) # Example default, must be tuned
    eta_max = config.get("eta_max", 100.0)   # Example default, must be tuned
    eta_num_bits = config.get("eta_num_bits", 8) # Example default
    
    eta_binary_vars: List[str] = []
    eta_step_size = 0.0
    if eta_num_bits > 0:
        eta_range = eta_max - eta_min
        if eta_range <= 0 and eta_num_bits > 0:
            if converter.logger: converter.logger.warning("eta_max <= eta_min with eta_num_bits > 0. Setting eta_step_size to 0.")
            eta_step_size = 0.0
        elif (2**eta_num_bits - 1) == 0 and eta_num_bits ==1 : # handles single bit case, range is just eta_step_size
             eta_step_size = eta_range
        elif (2**eta_num_bits - 1) == 0 and eta_num_bits > 1: # should not happen if eta_num_bits is int > 0
             if converter.logger: converter.logger.warning("Invalid state for eta_step_size calculation.")
             eta_step_size = 0.0 # Or raise error
        else:
            eta_step_size = eta_range / (2**eta_num_bits - 1)


        for j in range(eta_num_bits):
            eta_bit_name = f"eta_bit_{j}"
            converter.add_variable(Variable(name=eta_bit_name, type=VariableType.BINARY))
            eta_binary_vars.append(eta_bit_name)

    # --- 3. Add objective function terms: f^T * y + eta ---
    # f^T * y terms (linear terms for y_i)
    if f_coeffs is not None:
        for i in range(Ny):
            if i < f_coeffs.shape[0]:
                coeff_val = f_coeffs[i, 0]
                if coeff_val != 0:
                    term_key = tuple(sorted((y_var_names[i], y_var_names[i])))
                    if term_key not in converter.objective_terms: converter.objective_terms[term_key] = 0.0
                    converter.objective_terms[term_key] += coeff_val
    
    # eta terms from objective: sum(eta_bit_j * 2^j * eta_step_size)
    # The constant eta_min is added to offset in _finalize_qubo_from_converter_state
    if eta_num_bits > 0:
        for j in range(eta_num_bits):
            eta_bit_name = eta_binary_vars[j]
            coeff_val = eta_step_size * (2**j)
            if coeff_val != 0:
                term_key = tuple(sorted((eta_bit_name, eta_bit_name)))
                if term_key not in converter.objective_terms: converter.objective_terms[term_key] = 0.0
                converter.objective_terms[term_key] += coeff_val
                
    # --- Penalty Configuration ---
    penalty_P = config.get("penalty_coefficient", 1000.0) # Must be tuned
    num_slack_bits = config.get("penalty_slack_num_bits", 5) # Must be tuned

    # --- 4. Constraints: D*y >= d  =>  D*y - d - Slack_sum = 0 ---
    # Slack_sum = sum(s_k * 2^k) >= 0
    if D_matrix is not None and d_vector is not None:
        for i in range(D_matrix.shape[0]): # For each constraint
            expr_terms_Dy: Dict[str, float] = {}
            # D_i*y part
            for j in range(Ny):
                if D_matrix[i, j] != 0:
                    expr_terms_Dy[y_var_names[j]] = D_matrix[i, j]
            
            # - Slack_sum part
            for k in range(num_slack_bits):
                slack_name = f"slack_Dy_i{i}_k{k}"
                converter.add_variable(Variable(name=slack_name, type=VariableType.BINARY))
                expr_terms_Dy[slack_name] = -(2**k) 
            
            const_term_Dy = -d_vector[i, 0]
            _add_squared_penalty_to_converter(converter, expr_terms_Dy, const_term_Dy, penalty_P)

    # --- 5. Optimality cuts: eta + (pi^T B) y - pi^T b >= 0 ---
    #    => eta_approx + (pi^T B) y - pi^T b - Slack_sum = 0
    for idx, pi in enumerate(optimality_cuts):
        expr_terms_opt: Dict[str, float] = {}
        
        # eta_approx terms
        if eta_num_bits > 0:
            for j in range(eta_num_bits):
                expr_terms_opt[eta_binary_vars[j]] = eta_step_size * (2**j)
        
        # (pi^T B) y terms
        pi_T_B = pi.T @ B_matrix  # (1 x Ny)
        for j in range(Ny):
            coeff = pi_T_B[0, j]
            if coeff != 0:
                if y_var_names[j] not in expr_terms_opt: expr_terms_opt[y_var_names[j]] = 0.0
                expr_terms_opt[y_var_names[j]] += coeff
                
        # - Slack_sum part
        for k in range(num_slack_bits):
            slack_name = f"slack_opt_c{idx}_k{k}"
            converter.add_variable(Variable(name=slack_name, type=VariableType.BINARY))
            expr_terms_opt[slack_name] = -(2**k)

        # Constant part: eta_min (from eta_approx) - pi^T b
        const_term_opt = (eta_min if eta_num_bits > 0 else 0.0) - (pi.T @ b_vector)[0, 0]
        _add_squared_penalty_to_converter(converter, expr_terms_opt, const_term_opt, penalty_P)

    # --- 6. Feasibility cuts: (pi_ray^T B) y - pi_ray^T b <= 0 ---
    #    => (pi_ray^T B) y - pi_ray^T b + Slack_sum = 0
    for idx, pi_ray in enumerate(feasibility_cuts):
        expr_terms_feas: Dict[str, float] = {}
        
        # (pi_ray^T B) y terms
        pi_ray_T_B = pi_ray.T @ B_matrix  # (1 x Ny)
        for j in range(Ny):
            coeff = pi_ray_T_B[0, j]
            if coeff != 0:
                if y_var_names[j] not in expr_terms_feas: expr_terms_feas[y_var_names[j]] = 0.0
                expr_terms_feas[y_var_names[j]] += coeff
        
        # + Slack_sum part
        for k in range(num_slack_bits):
            slack_name = f"slack_feas_c{idx}_k{k}"
            converter.add_variable(Variable(name=slack_name, type=VariableType.BINARY))
            expr_terms_feas[slack_name] = (2**k) # Positive as we want Expr + Slack = 0 for Expr <= 0
            
        # Constant part: - pi_ray^T b
        const_term_feas = -(pi_ray.T @ b_vector)[0, 0]
        _add_squared_penalty_to_converter(converter, expr_terms_feas, const_term_feas, penalty_P)
        
    # --- Finalize QUBO model ---
    return _finalize_qubo_from_converter_state(converter, config) 

# --- Integration with Mean-Field solver ---

def solve_benders_master_with_mean_field(
    f_coeffs: np.ndarray,        # Ny x 1 vector for f^T y
    D_matrix: np.ndarray,        # n_dy x Ny matrix for Dy >= d
    d_vector: np.ndarray,        # n_dy x 1 vector
    optimality_cuts: List[np.ndarray],  # List of pi vectors (m x 1)
    feasibility_cuts: List[np.ndarray], # List of pi_ray vectors (m x 1)
    B_matrix: np.ndarray,        # m x Ny matrix (complicating constraint part for y)
    b_vector: np.ndarray,        # m x 1 vector (complicating constraint RHS)
    Ny: int,                     # Number of y variables
    config: Dict,                # QUBO conversion params
    mean_field_params: Optional[Dict] = None,  # Parameters for mean field algorithm
    logger: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, float]:
    """
    Solves the Benders master problem directly using QUBO conversion and 
    Mean-Field Approximate Optimization Algorithm.
    
    This function provides a complete workflow:
    1. Converts the Benders master problem to QUBO form
    2. Converts QUBO to Ising model parameters
    3. Solves with Mean-Field algorithm
    4. Converts solution back to binary form for the original problem
    
    Args:
        f_coeffs, D_matrix, d_vector: Parameters describing the master problem
        optimality_cuts, feasibility_cuts: Benders cuts from previous iterations
        B_matrix, b_vector: Parameters for linking constraints
        Ny: Number of y variables
        config: Dictionary with QUBO parameters
        mean_field_params: Parameters for mean field algorithm (optional)
        logger: Optional logger
        
    Returns:
        Tuple[np.ndarray, float]: (y_solution, objective_value)
        where y_solution is a Ny x 1 column vector of binary values
    """
    # Import mean field solver here to avoid circular imports
    from my_functions.mean_field_base import mean_field_aoa, qubo_to_ising
    
    # Default mean field parameters
    if mean_field_params is None:
        mean_field_params = {
            "p": 1000,            # Number of Trotter steps
            "tau": 0.5,           # Time step size
            "tau_decay": 0.99,    # Tau decay rate for adaptive annealing
            "beta_init": 0.5,     # Initial inverse temperature
            "beta_final": 20.0,   # Final inverse temperature
            "restart_count": 1,   # Number of random restarts
            "flip_improve": True  # Whether to perform improvement with bit flips
        }
    
    # 1. Convert master problem to QUBO
    if logger:
        logger.info("Converting Benders master problem to QUBO...")
    
    qubo_model = convert_benders_master_to_qubo(
        f_coeffs, D_matrix, d_vector, optimality_cuts, feasibility_cuts,
        B_matrix, b_vector, Ny, config, logger
    )
    
    # 2. Extract Q matrix and convert to Ising model
    Q_matrix = qubo_model.Q
    c_vector = qubo_model.c
    offset = qubo_model.offset
    
    # Incorporate linear terms into Q matrix diagonal (standard form for QUBO)
    Q_full = Q_matrix.copy()
    np.fill_diagonal(Q_full, np.diag(Q_full) + c_vector)
    
    # Print information about problem size
    num_variables = Q_full.shape[0]
    num_qubits_qaoa = num_variables
    
    if logger:
        logger.info(f"QUBO matrix size: {Q_full.shape}")
        logger.info(f"QUBO offset constant: {offset}")
        logger.info(f"Number of binary variables: {num_variables}")
        logger.info(f"Estimated qubits needed for QAOA: {num_qubits_qaoa}")
    else:
        print(f"QUBO matrix size: {Q_full.shape}")
        print(f"Number of binary variables: {num_variables}")
        print(f"Estimated qubits needed for QAOA: {num_qubits_qaoa}")
    
    # 3. Convert QUBO to Ising
    h, J = qubo_to_ising(Q_full)
    
    # Print information about Ising model size
    num_spins = len(h)
    
    if logger:
        logger.info("Converting QUBO to Ising model...")
        logger.info(f"Ising model size: h:{h.shape}, J:{J.shape}")
        logger.info(f"Number of spins for mean-field algorithm: {num_spins}")
    else:
        print(f"Ising model size: h:{h.shape}, J:{J.shape}")
        print(f"Number of spins for mean-field algorithm: {num_spins}")
    
    # 4. Solve with Mean-Field algorithm with multiple restarts to find best solution
    if logger:
        logger.info("Solving with Mean-Field AOA algorithm with multiple restarts...")
    
    best_sigma = None
    best_ising_cost = float('inf')
    restart_count = mean_field_params.get("restart_count", 1)
    
    for restart in range(restart_count):
        if logger and restart > 0:
            logger.info(f"Starting Mean-Field restart {restart+1}/{restart_count}")
        
        # Extract parameters for mean field algorithm
        p = int(mean_field_params.get("p", 1000))  # Ensure p is an integer
        tau = mean_field_params.get("tau", 0.5)
        tau_decay = mean_field_params.get("tau_decay", 0.99)
        beta_init = mean_field_params.get("beta_init", 0.5)
        beta_final = mean_field_params.get("beta_final", 20.0)
        flip_improve = mean_field_params.get("flip_improve", True)
        
        # Call mean field algorithm with extended parameters
        sigma, ising_cost = mean_field_aoa(
            h, J, 
            p=p,
            tau=tau,
            tau_decay=tau_decay,
            beta_init=beta_init,
            beta_final=beta_final,
            flip_improve=flip_improve
        )
        
        if logger:
            logger.info(f"Restart {restart+1} solution, ising cost: {ising_cost}")
        
        if ising_cost < best_ising_cost:
            best_sigma = sigma
            best_ising_cost = ising_cost
            if logger:
                logger.info(f"New best solution found, ising cost: {best_ising_cost}")
    
    # Use the best solution found
    sigma = best_sigma
    ising_cost = best_ising_cost
    
    # 5. Convert Ising solution (-1/+1) back to binary (0/1)
    x_solution = (sigma + 1) // 2
    
    # 6. Extract y values from the solution
    # Need to map variables from QUBO variables back to original y variables
    y_solution = np.zeros((Ny, 1))
    
    for i in range(Ny):
        var_name = f"y_{i}"
        if var_name in qubo_model.variable_map:
            idx = qubo_model.variable_map[var_name]
            y_solution[i, 0] = x_solution[idx]
    
    # 7. Compute objective value for the original problem
    # Original objective was f^T y + eta
    # We need to approximate eta from binary representation
    if config.get("eta_num_bits", 0) > 0:
        eta_min = config.get("eta_min", -100.0)
        eta_max = config.get("eta_max", 100.0)
        eta_num_bits = config.get("eta_num_bits", 8)
        eta_range = eta_max - eta_min
        
        if eta_range <= 0:
            eta_step_size = 0.0
        elif (2**eta_num_bits - 1) == 0:
            eta_step_size = eta_range
        else:
            eta_step_size = eta_range / (2**eta_num_bits - 1)
        
        # Reconstruct eta from binary bits
        eta_value = eta_min
        for j in range(eta_num_bits):
            eta_bit_name = f"eta_bit_{j}"
            if eta_bit_name in qubo_model.variable_map:
                idx = qubo_model.variable_map[eta_bit_name]
                eta_value += eta_step_size * (2**j) * x_solution[idx]
    else:
        # If eta was not part of the QUBO, use a default or fallback value
        eta_value = 0.0
    
    # Compute f^T y
    f_y_value = 0.0
    if f_coeffs is not None:
        f_y_value = float(f_coeffs.T @ y_solution)
    
    # Full objective value
    obj_value = f_y_value + eta_value
    
    if logger:
        logger.info(f"Solution found with objective value: {obj_value}")
        logger.info(f"Solution y vector sum: {np.sum(y_solution)}")
        logger.info(f"Non-zero y variables: {np.sum(y_solution > 0.5)}/{Ny}")

    # Return y solution as Ny x 1 column vector and objective value
    return y_solution, obj_value 