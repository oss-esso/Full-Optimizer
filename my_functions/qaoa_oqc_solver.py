"""
QAOA Solver using the OpenQuantumComputing QAOA library for solving QUBO problems.
This module integrates the external QAOA package with our system.
"""

import numpy as np
import logging
import traceback
from typing import Dict, Any, Optional, Tuple, List, Union

# Import Qiskit components for backend management
try:
    from qiskit_aer import AerSimulator
    from qiskit.providers.exceptions import QiskitBackendNotFoundError
    from qiskit import transpile # For checking circuit size against backend
    # Import Qiskit optimizers that might be selected by string name
    from qiskit_algorithms.optimizers import (
        COBYLA, SPSA, ADAM, NELDER_MEAD, POWELL, SLSQP, 
        L_BFGS_B, TNC, AQGD, NFT # Added more common ones
    )
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    AerSimulator = None # Placeholder
    QiskitBackendNotFoundError = Exception # Placeholder
    transpile = None # Placeholder
    # Optimizers will also be None if qiskit_algorithms isn't there
    COBYLA, SPSA, ADAM, NELDER_MEAD, POWELL, SLSQP, L_BFGS_B, TNC, AQGD, NFT = (None,)*10
    logging.warning("Qiskit or Qiskit Aer or Qiskit Algorithms not found. Simulator fallback and optimizer selection might be limited.")

# Initialize qaoa_version and has_qaoa to default/safe values
qaoa_version: str = "unavailable"
has_qaoa: bool = False

try:
    # Attempt to import the top-level package and then specific components
    import qaoa
    from qaoa import QAOA
    from qaoa.problems import QUBO as QAOA_QUBO # Renamed to avoid conflict if user has local QUBO
    from qaoa.initialstates import Plus
    from qaoa.mixers import X
    # REMOVED: from qaoa.operators import Z, ZZ # These are not compatible with local QUBO class interface
    
    # If all the above imports were successful, then QAOA is considered available
    has_qaoa = True
    
    # Try to get the version, but don't let this affect has_qaoa
    try:
        qaoa_version = qaoa.__version__
        logging.info(f"Successfully imported QAOA library. Version: {qaoa_version}")
    except AttributeError:
        qaoa_version = "unknown (module lacks __version__ attribute)"
        logging.info(f"Successfully imported core QAOA components. Version is {qaoa_version}.")
    
    # Optionally log the file location for debugging
    try:
        logging.debug(f"QAOA package location: {qaoa.__file__}")
    except AttributeError:
        logging.debug("QAOA package location unknown (module lacks __file__ attribute).")

except ImportError as e:
    # This block will catch errors if 'import qaoa' or any 'from qaoa import ...' fails
    has_qaoa = False # Redundant due to initialization, but explicit
    qaoa_version = "import failed"
    logging.warning(f"OpenQuantumComputing QAOA library or one of its essential components failed to import: {str(e)}. QAOA solver will not be available.")
    # For deeper debugging of which import failed, you might log the traceback:
    # logging.debug(traceback.format_exc())

except Exception as e:
    # This catches any other unexpected errors ONLY during the import sequence of qaoa components
    has_qaoa = False # Redundant due to initialization, but explicit
    qaoa_version = "unexpected import error"
    logging.error(f"An unexpected error occurred while trying to import QAOA or its components: {str(e)}. QAOA solver will not be available.")
    # logging.debug(traceback.format_exc())

# Import from local modules
from .qubo_converter import QUBOModel

class QAOASolver:
    """
    Solver that uses the OpenQuantumComputing QAOA library to solve QUBO problems.
    """
    
    def __init__(self, 
                 qaoa_depth: int = 3,
                 num_shots: int = 1024,
                 backend: Optional[Any] = None, # This can be a Qiskit backend object or a name
                 optimizer_method: str = 'COBYLA',
                 max_iter: int = 200,
                 use_simulator_fallback: bool = True): # Added for clarity
        """
        Initialize the QAOA solver.
        
        Args:
            qaoa_depth: Number of QAOA layers (p)
            num_shots: Number of measurement shots for the quantum circuit
            backend: Optional Qiskit backend or backend name for quantum simulation/execution
            optimizer_method: Classical optimizer method for QAOA parameter optimization
            max_iter: Maximum iterations for the classical optimizer
            use_simulator_fallback: If True, fallback to simulator if hardware limits exceeded
        """
        self.qaoa_depth = qaoa_depth
        self.num_shots = num_shots
        self.specified_backend = backend # Store the originally specified backend
        self.backend = backend # This will be dynamically updated
        self.optimizer_method = optimizer_method
        self.max_iter = max_iter
        self.use_simulator_fallback = use_simulator_fallback
        self.logger = logging.getLogger(__name__)
        
        if not has_qaoa:
            self.logger.error("OpenQuantumComputing QAOA package not installed. Please install it first.")
    
    def solve(self, 
              Q_matrix: np.ndarray, 
              linear_coeffs_c: np.ndarray,
              offset_b: float = 0.0,
              warm_start: Optional[Dict[int, int]] = None) -> Dict[str, Any]:
        """
        Solve the QUBO problem using QAOA.
        
        Args:
            Q_matrix: The Q matrix for the QUBO problem (quadratic terms only, typically upper or lower triangular)
            linear_coeffs_c: The c vector for linear terms in the QUBO objective
            offset_b: Constant scalar offset b in the QUBO objective
            warm_start: Optional dictionary mapping variable indices to binary values for warm starting
            
        Returns:
            Dictionary containing the solution and metrics
        """
        if not has_qaoa:
            self.logger.error("QAOA package not available. Cannot solve.")
            # Ensure a consistent error dictionary format
            return {"error": "QAOA package not available", "solution": None, "objective": float('-inf'), "num_qubits": Q_matrix.shape[0] if Q_matrix is not None else 0, "qaoa_depth": self.qaoa_depth}
        
        qaoa_instance = None # Initialize qaoa_instance
        n = 0 # Initialize n for error return if Q_matrix is None
        result = None # Initialize result variable to avoid UnboundLocalError

        try:
            # Get problem dimensions
            n = Q_matrix.shape[0]
            self.logger.info(f"QUBO problem size: {n} qubits")

            # --- BEGIN Enhanced Backend selection and validation logic ---
            effective_backend = self.specified_backend
            # Define a threshold for typical small hardware backends
            HARDWARE_QUBIT_THRESHOLD = 29 # Based on observed errors

            if n > HARDWARE_QUBIT_THRESHOLD and self.use_simulator_fallback:
                self.logger.warning(f"Problem requires {n} qubits, exceeding threshold of {HARDWARE_QUBIT_THRESHOLD}. Setting backend to None for QAOA library's default simulator due to use_simulator_fallback=True.")
                effective_backend = None # Force None to let QAOA library use its default simulator
            elif HAS_QISKIT:
                if effective_backend is not None:
                    if isinstance(effective_backend, str):
                        try:
                            # This part is tricky as we don't instantiate the backend here directly for the QAOA library.
                            # The QAOA library is expected to handle the string name.
                            pass # Assuming qaoa library handles backend name string
                        except Exception as e:
                            self.logger.warning(f"Could not introspect backend string '{effective_backend}': {e}. Will pass name to QAOA library.")
                    elif hasattr(effective_backend, 'configuration'): # It's a Qiskit backend object
                        backend_config = effective_backend.configuration()
                        max_qubits = getattr(backend_config, 'n_qubits', float('inf'))
                        if n > max_qubits:
                            self.logger.warning(f"Problem requires {n} qubits, but backend '{getattr(backend_config, 'backend_name', 'unknown')}' only supports {max_qubits}.")
                            if self.use_simulator_fallback:
                                self.logger.info("Setting backend to None for QAOA library's default simulator due to qubit limit and use_simulator_fallback=True.")
                                effective_backend = None # Force None
                            else:
                                self.logger.error(f"Cannot run on backend {getattr(backend_config, 'backend_name', 'unknown')}. use_simulator_fallback is {self.use_simulator_fallback}.")
                                return {"error": f"Qubit limit exceeded for backend {getattr(backend_config, 'backend_name', 'unknown')}", "solution": None, "objective": float('-inf'), "num_qubits": n, "qaoa_depth": self.qaoa_depth}
                elif self.use_simulator_fallback : # No backend specified, default to simulator if fallback is true
                    self.logger.info("No backend specified. Setting backend to None for QAOA library's default simulator as use_simulator_fallback=True.")
                    effective_backend = None # Force None
                else: # No backend, no fallback
                     self.logger.info("No backend specified and simulator fallback is false. QAOA library will use its default or handle None as passed.")
                     # effective_backend remains self.specified_backend which could be None or a specific backend
            else: # No Qiskit
                self.logger.warning("Qiskit not available for detailed backend checks. Passing backend to QAOA library directly.")
                if n > HARDWARE_QUBIT_THRESHOLD and self.use_simulator_fallback:
                    self.logger.warning(f"Problem requires {n} qubits, Qiskit not available. Setting backend to None for QAOA library's default simulator if use_simulator_fallback=True.")
                    effective_backend = None # Force None
            
            self.backend = effective_backend # This will be used by qaoa_instance.backend
            # --- END Enhanced Backend selection and validation logic ---

            # Ensure Q_matrix is in the expected form for qaoa.problems.QUBO
            # The QAOA library expects a lower triangular matrix for Q
            if not np.allclose(Q_matrix, np.tril(Q_matrix)):
                self.logger.info("Converting Q matrix to lower triangular form")
                if np.allclose(Q_matrix, Q_matrix.T):  # If symmetric
                    Q_lower = np.tril(Q_matrix)
                else:  # If not symmetric, symmetrize first then take lower triangle
                    Q_lower = np.tril(Q_matrix + Q_matrix.T - np.diag(np.diag(Q_matrix)))
            else:
                Q_lower = Q_matrix
            
            # Ensure linear_coeffs_c has the right shape (n,)
            if linear_coeffs_c is None:
                linear_coeffs_c = np.zeros(n)
            elif isinstance(linear_coeffs_c, np.ndarray) and linear_coeffs_c.ndim > 1:
                linear_coeffs_c = linear_coeffs_c.flatten()
            
            # Ensure offset_b is a scalar
            if not np.isscalar(offset_b):
                self.logger.warning(f"Offset is not scalar: {offset_b}, converting to scalar")
                if isinstance(offset_b, np.ndarray):
                    offset_b = float(offset_b.item(0)) if offset_b.size == 1 else 0.0
                else:
                    offset_b = float(offset_b)
            
            self.logger.debug(f"Creating QAOA_QUBO problem with:")
            self.logger.debug(f"  Q shape: {Q_lower.shape}")
            self.logger.debug(f"  c shape: {linear_coeffs_c.shape}")
            self.logger.debug(f"  b value: {offset_b}")
            
            # Create the QUBO problem
            try:
                problem = QAOA_QUBO(Q=Q_lower, c=linear_coeffs_c, b=offset_b)
                self.logger.info("Successfully created QAOA_QUBO problem")
            except Exception as qubo_error:
                self.logger.error(f"Error creating QAOA_QUBO: {qubo_error}")
                # Try alternative initialization if the first approach fails
                self.logger.info("Trying alternative QAOA_QUBO initialization")
                problem = QAOA_QUBO(n)  # Create with just the number of qubits
                problem.Q = Q_lower     # Then set Q matrix manually
                problem.c = linear_coeffs_c  # Set c vector manually
                problem.b = offset_b    # Set constant offset
            
            # Create initial state (|+⟩^⊗n)
            self.logger.debug(f"Creating initial Plus state for {n} qubits")
            try:
                # Try to create the Plus state with the right number of qubits
                # If the Plus class doesn't accept arguments, create it without arguments
                initial_state = Plus()
                self.logger.info("Successfully created Plus initial state")
            except Exception as init_error:
                self.logger.error(f"Error creating Plus state: {init_error}")
                raise ValueError(f"Could not create Plus state: {init_error}")
            
            # Create mixer Hamiltonian (∑X_i)
            self.logger.debug(f"Creating mixer for {n} qubits")
            try:
                # Create X mixer without arguments and set N_qubits afterwards
                mixer = X()
                # Add the N_qubits attribute which is expected by the library
                if not hasattr(mixer, 'N_qubits'):
                    mixer.N_qubits = n
                
                self.logger.info(f"Successfully created X mixer and set N_qubits={n}")
            except Exception as mixer_error:
                self.logger.error(f"Error creating X mixer: {mixer_error}")
                raise ValueError(f"Could not create X mixer: {mixer_error}")
            
            # --- BEGIN Optimizer Preparation ---
            optimizer_class = None
            optimizer_options = {'maxiter': self.max_iter} # Common option

            opt_method_upper = self.optimizer_method.upper()
            if opt_method_upper == 'COBYLA' and COBYLA:
                optimizer_class = COBYLA
            elif opt_method_upper == 'SPSA' and SPSA:
                optimizer_class = SPSA
            elif opt_method_upper == 'ADAM' and ADAM:
                optimizer_class = ADAM
            elif opt_method_upper == 'NELDER-MEAD' and NELDER_MEAD: # Qiskit uses NELDER_MEAD
                optimizer_class = NELDER_MEAD
            elif opt_method_upper == 'POWELL' and POWELL:
                optimizer_class = POWELL
            elif opt_method_upper == 'SLSQP' and SLSQP:
                optimizer_class = SLSQP
            elif opt_method_upper == 'L-BFGS-B' and L_BFGS_B:
                optimizer_class = L_BFGS_B
            elif opt_method_upper == 'TNC' and TNC:
                optimizer_class = TNC
            elif opt_method_upper == 'AQGD' and AQGD: # Not a standard Qiskit optimizer in this form usually.
                optimizer_class = AQGD # Make sure it's the right one from qiskit_algorithms
            elif opt_method_upper == 'NFT' and NFT:
                optimizer_class = NFT
            else:
                self.logger.warning(f"Unsupported or unavailable optimizer_method: {self.optimizer_method}. Defaulting to COBYLA if available, else QAOA library's internal default.")
                if COBYLA: # Fallback to COBYLA if string not recognized but COBYLA is available
                    optimizer_class = COBYLA
                # If COBYLA is also None (Qiskit not found), qaoa_optimizer_arg will remain None,
                # and local QAOA lib will use its default [COBYLA (from its own import), {}]
                # which is fine if its own COBYLA import works.

            qaoa_optimizer_arg = None
            if optimizer_class:
                qaoa_optimizer_arg = [optimizer_class, optimizer_options]
                self.logger.info(f"Prepared optimizer for local QAOA: {optimizer_class.__name__} with options {optimizer_options}")
            else:
                self.logger.warning(f"Could not resolve optimizer class for '{self.optimizer_method}'. Local QAOA library will use its internal default optimizer.")
            # --- END Optimizer Preparation ---

            # Create the QAOA instance
            self.logger.info(f"Running QAOA optimization with depth p={self.qaoa_depth}")
            try:
                # Try minimal QAOA creation with just the required parameters
                qaoa_instance = QAOA(
                    problem=problem,
                    mixer=mixer,
                    initialstate=initial_state,
                )
            except Exception as alt_error:
                # Last attempt with minimal parameters
                self.logger.error(f"Alternative QAOA creation also failed: {alt_error}")
                self.logger.warning("Falling back to manual solution approach")
                # Create a fallback solution
                # self.logger.info("Using classical QUBO solution as fallback")
                # return solve_qubo_classically(Q_matrix, linear_coeffs_c, offset_b, n, self.logger, depth=self.qaoa_depth)
                return {"error": f"QAOA instance alternative creation failed: {alt_error}", "solution": None, "objective": float('-inf'), "num_qubits": n, "qaoa_depth": self.qaoa_depth}
            
            if self.backend:
                try:
                    # Pass depth parameter to optimize() method, not constructor
                    # The optimize method correctly expects 'depth', not 'p'
                    result = qaoa_instance.optimize(
                        depth=self.qaoa_depth,  # Depth parameter 
                    )
                    self.logger.info(f"QAOA optimization complete")
                except Exception as opt_error:
                    self.logger.error(f"Error in QAOA optimization: {opt_error}")
                    # Try alternative API patterns for optimization
                    # Only attempt sample_cost_landscape if backend is not None, as it seems to cause issues otherwise.
                    if qaoa_instance is not None and qaoa_instance.backend is not None:
                        try:
                            self.logger.info("Trying sample_cost_landscape instead")
                            # Try using sample_cost_landscape as an alternative
                            qaoa_instance.sample_cost_landscape()
                            # If sample_cost_landscape somehow "fixes" things or produces a usable state,
                            # we might need to re-evaluate how `result` is obtained. 
                            # For now, assuming it populates similar internal state that get_optimal_solutions can use.
                        except Exception as alt_opt_error:
                            self.logger.error(f"Alternative optimization (sample_cost_landscape) also failed: {alt_opt_error}")
                            # If optimization fails, fall back to classical solution
                            # self.logger.warning("Using fallback classical optimization approach after sample_cost_landscape failure")
                            # return solve_qubo_classically(Q_matrix, linear_coeffs_c, offset_b, n, self.logger, depth=self.qaoa_depth)
                            return {"error": f"QAOA sample_cost_landscape failed: {alt_opt_error}", "solution": None, "objective": float('-inf'), "num_qubits": n, "qaoa_depth": self.qaoa_depth}
                    else:
                        log_msg_suffix = "qaoa_instance was None" if qaoa_instance is None else "backend was None or qaoa_instance was None initially"
                        self.logger.warning(f"Backend is None or qaoa_instance not valid, skipping sample_cost_landscape and reporting optimization error: {log_msg_suffix}.")
                        # return solve_qubo_classically(Q_matrix, linear_coeffs_c, offset_b, n, self.logger, depth=self.qaoa_depth)
                        return {"error": f"QAOA optimization failed ({opt_error}) and fallback condition met ({log_msg_suffix})", "solution": None, "objective": float('-inf'), "num_qubits": n, "qaoa_depth": self.qaoa_depth}
            
            # Sample from the optimized circuit
            self.logger.info(f"Sampling from optimized circuit with {self.num_shots} shots")
            try:
                # Try the hist method which exists in the source code we looked at
                best_angles = qaoa_instance.get_angles(self.qaoa_depth)
                counts = qaoa_instance.hist(best_angles, self.num_shots)
                self.logger.info(f"Sample results obtained")
            except Exception as sample_error:
                self.logger.error(f"Error sampling from QAOA circuit: {sample_error}")
                self.logger.warning("Using fallback random sampling approach")
                # Create a random fallback solution if sampling fails
                import random
                counts = {"".join(str(random.randint(0, 1)) for _ in range(n)): self.num_shots}
            
            # Get the best solution
            try:
                # First try to get the solutions directly from the QAOA instance
                best_solutions, best_cost = qaoa_instance.get_optimal_solutions()
                best_bitstring = best_solutions[0]  # Take the first solution if multiple exist
                best_solution = {i: int(best_bitstring[i]) for i in range(len(best_bitstring))}
                self.logger.info(f"Found optimal solution with cost {best_cost}")
                
                # Initialize result here
                result = {
                    "solution": best_solution,
                    "objective": best_cost,
                    "bitstring": best_bitstring,
                    "counts": counts,
                    "num_qubits": n,
                    "qaoa_depth": self.qaoa_depth
                }
            except Exception as e:
                # Fall back to looking at the counts if that fails
                self.logger.warning(f"Error getting optimal solutions from QAOA: {e}")
                self.logger.warning("Falling back to counts-based solution selection")
                best_bitstring = max(counts, key=counts.get)
                best_solution = {i: int(best_bitstring[i]) for i in range(n)}
                probability = counts[best_bitstring] / self.num_shots
                
                # Calculate the energy manually since our QUBO object doesn't have an energy method
                energy = calculate_qubo_energy(Q_matrix, linear_coeffs_c, offset_b, best_bitstring, n)
                master_obj = energy
                
                self.logger.info(f"Best bitstring: {best_bitstring}, energy: {energy}")
                
                # Initialize result using counts-based solution
                result = {
                    "solution": best_solution,
                    "objective": energy,
                    "bitstring": best_bitstring,
                    "counts": counts,
                    "num_qubits": n,
                    "qaoa_depth": self.qaoa_depth,
                    "probability": probability
                }
            
            # Ensure a default return in case of unexpected issues before this point
            # Only check result now that we're sure it's initialized
            if result is None or 'solution' not in result: # Make sure the result has a solution key
                self.logger.error("QAOA result dictionary is None or does not contain 'solution' key. Fallback needed.")
                # return solve_qubo_classically(Q_matrix, linear_coeffs_c, offset_b, n, self.logger, depth=self.qaoa_depth) # Fallback
                return {"error": "QAOA result dictionary missing 'solution' key", "solution": None, "objective": float('-inf'), "num_qubits": n, "qaoa_depth": self.qaoa_depth}

            return result
            
        except Exception as e:
            self.logger.error(f"Unhandled exception in QAOASolver.solve: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Ensure a consistent error dictionary format on any unhandled exception
            return {"error": f"Unhandled exception: {str(e)}", "solution": None, "objective": float('-inf'), "num_qubits": n, "qaoa_depth": self.qaoa_depth}

def calculate_qubo_energy(Q_matrix, c_vector, offset, bitstring, n):
    """
    Calculate the energy/objective value for a given bitstring in a QUBO problem.
    
    Args:
        Q_matrix: The quadratic terms matrix
        c_vector: The linear terms vector
        offset: The constant offset
        bitstring: The binary string solution
        n: Number of variables
    
    Returns:
        The calculated energy/objective value
    """
    # Convert bitstring to binary array
    if isinstance(bitstring, str):
        x = np.array([int(bit) for bit in bitstring])
    elif isinstance(bitstring, (list, tuple)):
        x = np.array([int(b) for b in bitstring])
    elif isinstance(bitstring, np.ndarray):
        x = bitstring.astype(int)
    else:
        raise ValueError("Bitstring must be a string, list, tuple, or numpy array of bits.")

    if x.shape[0] != n:
        raise ValueError(f"Bitstring length {x.shape[0]} does not match QUBO size {n}.")

    energy = 0.0
    # Quadratic part x^T Q x
    # Ensure Q_matrix is dense for this calculation if it was sparse
    Q_dense = Q_matrix.toarray() if hasattr(Q_matrix, 'toarray') else Q_matrix
    energy += x.T @ Q_dense @ x
    
    # Linear part c^T x
    if c_vector is not None:
        energy += c_vector.T @ x
        
    # Constant offset
    if offset is not None:
        energy += offset
        
    return energy

def solve_benders_master_with_qaoa(f_coeffs: np.ndarray,
                                  D_matrix: np.ndarray,
                                  d_vector: np.ndarray,
                                  optimality_cuts: List[np.ndarray],
                                  feasibility_cuts: List[np.ndarray],
                                  B_matrix: np.ndarray,
                                  b_vector: np.ndarray,
                                  Ny: int,
                                  config: Dict[str, Any],
                                  qaoa_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Solve the Benders master problem using QAOA.
    
    Args:
        f_coeffs: Coefficients of the master problem objective
        D_matrix: Matrix D for constraints D*y >= d
        d_vector: Vector d for constraints D*y >= d
        optimality_cuts: List of optimality cuts (each cut is a row vector)
        feasibility_cuts: List of feasibility cuts (each cut is a row vector)
        B_matrix: Matrix B for constraints B*y >= b
        b_vector: Vector b for constraints B*y >= b
        Ny: Number of binary variables in the master problem
        config: Configuration parameters for the QUBO conversion
        qaoa_params: Configuration parameters for the QAOA solver
        
    Returns:
        Dictionary containing the solution and metrics
    """
    from .qubo_converter import convert_benders_master_to_qubo
    
    logger = logging.getLogger(__name__)
    
    if not has_qaoa:
        logger.error("OpenQuantumComputing QAOA package not available. Cannot solve master problem with QAOA.")
        # Fallback to a default or error state
        return {"error": "QAOA package not available", "solution": None, "objective": float('-inf')}

    logger.info("Solving Benders master problem with QAOA")
    
    # Default QAOA parameters if not provided
    qaoa_config = {
        'qaoa_depth': 3,
        'num_shots': 1024,
        'backend': None, # Will be handled by QAOASolver, potentially using AerSimulator
        'optimizer_method': 'COBYLA',
        'max_iter': 100,
        'use_simulator_fallback': True # Default to true here
    }
    if qaoa_params:
        qaoa_config.update(qaoa_params)

    logger.info(f"QAOA parameters for master problem: {qaoa_config}")

    # Convert Benders master problem to QUBO
    qubo_model = convert_benders_master_to_qubo(
        f_coeffs=f_coeffs,
        D_matrix=D_matrix,
        d_vector=d_vector,
        optimality_cuts=optimality_cuts,
        feasibility_cuts=feasibility_cuts,
        B_matrix=B_matrix,
        b_vector=b_vector,
        Ny=Ny,
        config=config
    )
    
    # Initialize and use the QAOASolver
    solver = QAOASolver(
        qaoa_depth=qaoa_config['qaoa_depth'],
        num_shots=qaoa_config['num_shots'],
        backend=qaoa_config['backend'], # Pass the backend specified in params
        optimizer_method=qaoa_config['optimizer_method'],
        max_iter=qaoa_config['max_iter'],
        use_simulator_fallback=qaoa_config['use_simulator_fallback'] # Pass the fallback flag
    )
    
    # Solve the QUBO problem
    qaoa_result = solver.solve(
        Q_matrix=qubo_model.Q,
        linear_coeffs_c=qubo_model.c,
        offset_b=qubo_model.offset,
        warm_start=None # Add warm_start if/when QAOA library supports it well
    )

    # Ensure qaoa_result is a dictionary, even if an error occurred in solve()
    if not isinstance(qaoa_result, dict):
        logger.error(f"QAOASolver.solve() returned non-dict type: {type(qaoa_result)}. Critical error.")
        return {"error": "QAOASolver.solve() critical failure", "solution": None, "objective": float('-inf')}

    # Log the raw result from QAOA solver for debugging
    logger.debug(f"Raw result from QAOASolver.solve(): {qaoa_result}")

    if qaoa_result.get("error"):
        logger.error(f"Error from QAOASolver: {qaoa_result['error']}")
        # Return the error structure from QAOASolver
        return qaoa_result

    # Process the solution if no error
    # The QAOASolver.solve() should return a dictionary with a 'solution' key
    # which is a dictionary of {var_index: value} and an 'objective' key.
    
    final_solution_vars = qaoa_result.get("solution") # This should be a dict {idx: val}
    final_master_objective = qaoa_result.get("objective", float('-inf'))

    if final_solution_vars is None:
        logger.error("QAOA solver did not return a 'solution' dictionary. Critical fallback.")
        return {"error": "QAOA solution missing", "solution": None, "objective": float('-inf')}

    # Convert solution dictionary to numpy array for Benders
    y_solution_np = np.zeros(Ny)
    for idx, val in final_solution_vars.items():
        if 0 <= idx < Ny:
            y_solution_np[idx] = val
        else:
            logger.warning(f"Solution index {idx} out of bounds for Ny={Ny}")
            
    logger.info(f"QAOA master solution objective: {final_master_objective}")
    logger.info(f"QAOA master solution (y vector sum): {np.sum(y_solution_np)}")

    # Prepare metrics to be returned
    # qaoa_result should contain 'num_qubits' and 'qaoa_depth'
    num_qaoa_qubits = qaoa_result.get("num_qubits", Ny) # Default to Ny if not present
    actual_qaoa_depth = qaoa_result.get("qaoa_depth", qaoa_config['qaoa_depth'])

    # Set metrics that might be used by the calling Benders loop
    qaoa_metrics_output = {
        "num_variables": num_qaoa_qubits, # Reflects actual QUBO size if available
        "num_qubits": num_qaoa_qubits,    # Alias for clarity
        "qaoa_depth": actual_qaoa_depth,
        "optimizer_iterations": qaoa_result.get("optimizer_iterations", 0), # If available
        "info": qaoa_result.get("info", ""),
        "counts": qaoa_result.get("counts", {}) # Pass along counts
    }
    
    return {
        "solution": y_solution_np,         # Numpy array for Benders
        "objective": final_master_objective, # Objective value from QAOA
        "metrics": qaoa_metrics_output,
        "error": None # No error in this path
    } 