"""
QUBO-QAOA Adapter Module

This module provides utilities to convert QUBO problems to a format suitable for QAOA
and interpret the results back in terms of the original problem variables.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import networkx as nx
from my_functions.qubo_converter import QUBOModel
from my_functions.qaoa_solver import QAOACircuit

@dataclass
class QAOAResult:
    """Results from running QAOA on a QUBO problem."""
    original_variables: Dict[str, Union[bool, int, float]]
    qubo_solution: Dict[str, bool]
    energy: float
    success_probability: float
    circuit_depth: int
    runtime: float

class QUBOQAOAAdapter:
    """
    Adapter class to interface between QUBO problems and QAOA solver.
    """
    
    def __init__(self, qaoa_solver: QAOACircuit):
        """
        Initialize the adapter.
        
        Args:
            qaoa_solver: Instance of QAOACircuit to use
        """
        self.qaoa_solver = qaoa_solver
    
    def qubo_to_ising(self, qubo: QUBOModel) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Convert QUBO problem to Ising form.
        
        The conversion follows the relation:
            x_i ∈ {0,1} -> (s_i + 1)/2, s_i ∈ {-1,1}
        
        Args:
            qubo: QUBOModel instance
            
        Returns:
            Tuple of (J, h, offset) where:
                J: Coupling matrix for Ising model
                h: Local field vector
                offset: Energy offset
        """
        n = len(qubo.variable_map)
        Q, c = qubo.Q, qubo.c
        offset = qubo.offset
        
        # Initialize Ising parameters
        J = np.zeros((n, n))
        h = np.zeros(n)
        
        # Convert quadratic terms: x_i x_j -> (s_i + 1)(s_j + 1)/4
        for i in range(n):
            for j in range(n):
                if i != j:
                    J[i,j] = Q[i,j]/4
                    offset += Q[i,j]/4
        
        # Convert linear terms: x_i -> (s_i + 1)/2
        for i in range(n):
            h[i] = c[i]/2 + sum(Q[i,j]/2 for j in range(n))
            offset += c[i]/2
        
        return J, h, offset
    
    def solve_qubo(self, 
                  qubo: QUBOModel, 
                  p: int = 4,
                  shots: int = 1000) -> QAOAResult:
        """
        Solve a QUBO problem using QAOA.
        
        Args:
            qubo: QUBOModel instance
            p: Number of QAOA layers
            shots: Number of measurement shots
            
        Returns:
            QAOAResult containing the solution and metrics
        """
        # Convert QUBO to Ising
        J, h, offset = self.qubo_to_ising(qubo)
        
        # Convert J matrix to dictionary format
        J_dict = {}
        n = len(qubo.variable_map)
        for i in range(n):
            for j in range(i+1, n):
                if abs(J[i,j]) > 1e-10:  # Add edge if coupling is non-zero
                    J_dict[(i,j)] = J[i,j]
        
        # Solve using QAOA
        solution, energy = self.qaoa_solver.solve_ising(h, J_dict, p=p)
        
        # Convert Ising solution back to QUBO variables
        qubo_solution = {}
        for idx, spin in enumerate(solution):
            var_name = qubo.reverse_map[idx]
            qubo_solution[var_name] = (spin + 1) // 2  # Convert from {-1,1} to {0,1}
        
        # Decode original variables
        original_vars = self._decode_solution(qubo, qubo_solution)
        
        return QAOAResult(
            original_variables=original_vars,
            qubo_solution=qubo_solution,
            energy=energy + offset,
            success_probability=1.0,  # TODO: Get actual success probability from QAOACircuit
            circuit_depth=0,  # TODO: Get actual circuit depth from QAOACircuit
            runtime=0.0  # TODO: Get actual runtime from QAOACircuit
        )
    
    def _decode_solution(self, 
                        qubo: QUBOModel, 
                        binary_solution: Dict[str, bool]) -> Dict[str, Union[bool, int, float]]:
        """
        Decode binary solution back to original variable values.
        
        Args:
            qubo: QUBOModel instance
            binary_solution: Dictionary mapping binary variable names to their values
            
        Returns:
            Dictionary mapping original variable names to their decoded values
        """
        original_vars = {}
        
        # Process encoded variables
        encoded_vars = {}
        for var_name, value in binary_solution.items():
            if var_name in qubo.encoding_info:
                info = qubo.encoding_info[var_name]
                orig_var = info['original_var']
                if orig_var not in encoded_vars:
                    encoded_vars[orig_var] = 0
                if value:
                    encoded_vars[orig_var] += info['weight']
        
        # Add decoded values to result
        original_vars.update(encoded_vars)
        
        # Add binary variables that weren't encoded
        for var_name, value in binary_solution.items():
            if var_name not in qubo.encoding_info:
                # Skip slack variables
                if not var_name.startswith('slack_'):
                    original_vars[var_name] = value
        
        return original_vars 