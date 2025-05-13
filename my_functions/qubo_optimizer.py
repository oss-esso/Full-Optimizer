import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import time
import dimod
from dimod import BinaryQuadraticModel

@dataclass
class QUBOResult:
    """Class for storing QUBO optimization results."""
    problem: BinaryQuadraticModel
    status: str
    objective_value: float
    solution: Dict[str, int]
    metrics: Dict[str, float]
    runtime: float

class QUBOOptimizer:
    def __init__(self, 
                 variables: List[str],
                 linear_terms: Dict[str, float],
                 quadratic_terms: Dict[Tuple[str, str], float],
                 config: Optional[Dict] = None):
        """
        Initialize the QUBO optimization model
        
        Args:
            variables (List[str]): List of binary variable names
            linear_terms (Dict[str, float]): Linear coefficients for each variable
            quadratic_terms (Dict[Tuple[str, str], float]): Quadratic coefficients for variable pairs
            config (Dict): Optional configuration dictionary
        """
        # Logging configuration
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Core configuration
        self.variables = variables
        self.linear_terms = linear_terms
        self.quadratic_terms = quadratic_terms
        self.config = config or {}

        # Validation
        self._validate_inputs()
        
        # Model parameters
        self.parameters = self._generate_model_parameters()
        
        # Initialize results storage
        self.results = []

    def _validate_inputs(self):
        """Validate input configurations and raise informative errors."""
        if not self.variables:
            raise ValueError("At least one variable must be provided")
        
        # Validate linear terms
        for var in self.variables:
            if var not in self.linear_terms:
                raise ValueError(f"Missing linear coefficient for variable {var}")
        
        # Validate quadratic terms
        for (var1, var2) in self.quadratic_terms:
            if var1 not in self.variables or var2 not in self.variables:
                raise ValueError(f"Invalid variable pair in quadratic terms: ({var1}, {var2})")

    def _generate_model_parameters(self) -> Dict:
        """Generate model parameters with default values."""
        config = self.config.get('parameters', {})
        
        return {
            'num_reads': config.get('num_reads', 100),
            'timeout': config.get('timeout', None),
            'verbose': config.get('verbose', True)
        }

    def solve_optimization_problem(self) -> QUBOResult:
        """Solve the QUBO optimization problem."""
        try:
            start_time = time.time()
            
            # Create the QUBO model
            bqm = self._create_qubo_model()
            
            # Solve using classical solver (Simulated Annealing)
            sampler = dimod.SimulatedAnnealingSampler()
            response = sampler.sample(
                bqm,
                num_reads=self.parameters['num_reads']
            )
            
            # Get the best solution
            best_sample = response.first.sample
            best_energy = response.first.energy
            
            # Convert solution to dictionary format
            solution = {
                var: int(best_sample[var])
                for var in self.variables
            }
            
            # Create result object
            result = QUBOResult(
                problem=bqm,
                status="Optimal",
                objective_value=best_energy,
                solution=solution,
                metrics={
                    'num_occurrences': response.first.num_occurrences,
                    'chain_break_fraction': response.first.chain_break_fraction
                },
                runtime=time.time() - start_time
            )
            
            self.results.append(result)
            return result
        
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise

    def _create_qubo_model(self) -> BinaryQuadraticModel:
        """Create the QUBO model."""
        # Initialize the model
        bqm = BinaryQuadraticModel('BINARY')
        
        # Add linear terms
        for var, coeff in self.linear_terms.items():
            bqm.add_variable(var, coeff)
        
        # Add quadratic terms
        for (var1, var2), coeff in self.quadratic_terms.items():
            bqm.add_interaction(var1, var2, coeff)
        
        return bqm

    def get_qubo_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the QUBO matrix representation.
        Returns:
            Tuple[np.ndarray, np.ndarray]: (Q matrix, linear terms)
        """
        n = len(self.variables)
        Q = np.zeros((n, n))
        linear = np.zeros(n)
        
        # Fill linear terms
        for i, var in enumerate(self.variables):
            linear[i] = self.linear_terms[var]
        
        # Fill quadratic terms
        for (var1, var2), coeff in self.quadratic_terms.items():
            i = self.variables.index(var1)
            j = self.variables.index(var2)
            Q[i, j] = coeff
            Q[j, i] = coeff
        
        return Q, linear 