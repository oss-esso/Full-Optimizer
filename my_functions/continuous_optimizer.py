import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from typing import Dict, List, Tuple, Optional
import logging

class ContinuousVariableConverter:
    """
    A class to handle conversion between continuous and binary variables
    for quantum optimization problems.
    """
    
    def __init__(self, num_bits: int = 8, scale_factor: float = 100.0):
        """
        Initialize the converter with parameters for binary encoding.
        
        Args:
            num_bits (int): Number of bits to use for binary encoding
            scale_factor (float): Scaling factor for continuous values
        """
        self.num_bits = num_bits
        self.scale_factor = scale_factor
        self.logger = logging.getLogger(__name__)
        
    def continuous_to_binary(self, value: float, lower_bound: float, upper_bound: float) -> List[int]:
        """
        Convert a continuous value to binary representation.
        
        Args:
            value (float): The continuous value to convert
            lower_bound (float): Lower bound of the variable
            upper_bound (float): Upper bound of the variable
            
        Returns:
            List[int]: Binary representation of the value
        """
        # Scale the value to [0, 1]
        scaled_value = (value - lower_bound) / (upper_bound - lower_bound)
        
        # Convert to binary
        binary_value = int(scaled_value * (2**self.num_bits - 1))
        binary_string = format(binary_value, f'0{self.num_bits}b')
        
        return [int(bit) for bit in binary_string]
    
    def binary_to_continuous(self, binary: List[int], lower_bound: float, upper_bound: float) -> float:
        """
        Convert a binary representation back to a continuous value.
        
        Args:
            binary (List[int]): Binary representation
            lower_bound (float): Lower bound of the variable
            upper_bound (float): Upper bound of the variable
            
        Returns:
            float: The continuous value
        """
        # Convert binary to integer
        binary_string = ''.join(str(bit) for bit in binary)
        int_value = int(binary_string, 2)
        
        # Scale back to original range
        scaled_value = int_value / (2**self.num_bits - 1)
        return lower_bound + scaled_value * (upper_bound - lower_bound)
    
    def convert_problem(self, problem: QuadraticProgram) -> Tuple[QuadraticProgram, Dict]:
        """
        Convert a problem with continuous variables to a binary problem.
        
        Args:
            problem (QuadraticProgram): The original problem with continuous variables
            
        Returns:
            Tuple[QuadraticProgram, Dict]: The converted problem and variable mapping
        """
        # Create new problem
        binary_problem = QuadraticProgram()
        
        # Variable mapping
        var_mapping = {}
        
        # Convert continuous variables
        for var in problem.variables:
            if var.vartype == 'CONTINUOUS':
                # Get bounds
                lower_bound = var.lowerbound if var.lowerbound is not None else 0
                upper_bound = var.upperbound if var.upperbound is not None else float('inf')
                
                # Create binary variables
                binary_vars = []
                for i in range(self.num_bits):
                    binary_name = f"{var.name}_bit_{i}"
                    binary_problem.binary_var(name=binary_name)
                    binary_vars.append(binary_name)
                
                var_mapping[var.name] = {
                    'binary_vars': binary_vars,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
            else:
                # Copy non-continuous variables as is
                if var.vartype == 'BINARY':
                    binary_problem.binary_var(name=var.name)
                elif var.vartype == 'INTEGER':
                    binary_problem.integer_var(
                        name=var.name,
                        lowerbound=var.lowerbound,
                        upperbound=var.upperbound
                    )
                var_mapping[var.name] = {'original': var.name}
        
        # Convert objective
        self._convert_objective(problem, binary_problem, var_mapping)
        
        # Convert constraints
        self._convert_constraints(problem, binary_problem, var_mapping)
        
        return binary_problem, var_mapping
    
    def _convert_objective(self, original: QuadraticProgram, binary: QuadraticProgram, mapping: Dict):
        """Convert the objective function to use binary variables."""
        # Get original objective
        linear = original.objective.linear.to_dict()
        quadratic = original.objective.quadratic.to_dict()
        constant = original.objective.constant
        
        # Initialize new objective
        new_linear = {}
        new_quadratic = {}
        new_constant = constant
        
        # Convert linear terms
        for var, coeff in linear.items():
            if var in mapping and 'binary_vars' in mapping[var]:
                # Convert continuous variable to binary representation
                binary_vars = mapping[var]['binary_vars']
                for i, bit in enumerate(binary_vars):
                    weight = 2**i / (2**self.num_bits - 1)
                    new_linear[bit] = coeff * weight
            else:
                new_linear[var] = coeff
        
        # Convert quadratic terms
        for (var1, var2), coeff in quadratic.items():
            if var1 in mapping and 'binary_vars' in mapping[var1]:
                # Handle continuous variable in quadratic term
                binary_vars1 = mapping[var1]['binary_vars']
                for i, bit1 in enumerate(binary_vars1):
                    weight1 = 2**i / (2**self.num_bits - 1)
                    if var2 in mapping and 'binary_vars' in mapping[var2]:
                        binary_vars2 = mapping[var2]['binary_vars']
                        for j, bit2 in enumerate(binary_vars2):
                            weight2 = 2**j / (2**self.num_bits - 1)
                            new_quadratic[(bit1, bit2)] = coeff * weight1 * weight2
                    else:
                        new_quadratic[(bit1, var2)] = coeff * weight1
            elif var2 in mapping and 'binary_vars' in mapping[var2]:
                binary_vars2 = mapping[var2]['binary_vars']
                for j, bit2 in enumerate(binary_vars2):
                    weight2 = 2**j / (2**self.num_bits - 1)
                    new_quadratic[(var1, bit2)] = coeff * weight2
            else:
                new_quadratic[(var1, var2)] = coeff
        
        # Set the new objective
        binary.minimize(
            linear=new_linear,
            quadratic=new_quadratic,
            constant=new_constant
        )
    
    def _convert_constraints(self, original: QuadraticProgram, binary: QuadraticProgram, mapping: Dict):
        """Convert constraints to use binary variables."""
        for constraint in original.linear_constraints:
            # Get constraint coefficients
            linear = constraint.linear.to_dict()
            rhs = constraint.rhs
            
            # Convert to binary representation
            new_linear = {}
            for var, coeff in linear.items():
                if var in mapping and 'binary_vars' in mapping[var]:
                    binary_vars = mapping[var]['binary_vars']
                    for i, bit in enumerate(binary_vars):
                        weight = 2**i / (2**self.num_bits - 1)
                        new_linear[bit] = coeff * weight
                else:
                    new_linear[var] = coeff
            
            # Add the constraint
            binary.linear_constraint(
                linear=new_linear,
                sense=constraint.sense,
                rhs=rhs,
                name=constraint.name
            )
    
    def interpret_solution(self, solution: Dict, mapping: Dict) -> Dict:
        """
        Convert a binary solution back to continuous values.
        
        Args:
            solution (Dict): The binary solution
            mapping (Dict): The variable mapping
            
        Returns:
            Dict: The solution with continuous values
        """
        interpreted_solution = {}
        
        for var_name, var_info in mapping.items():
            if 'binary_vars' in var_info:
                # Convert binary variables back to continuous
                binary_values = [solution[bit] for bit in var_info['binary_vars']]
                continuous_value = self.binary_to_continuous(
                    binary_values,
                    var_info['lower_bound'],
                    var_info['upper_bound']
                )
                interpreted_solution[var_name] = continuous_value
            else:
                # Copy non-continuous variables as is
                interpreted_solution[var_name] = solution[var_info['original']]
        
        return interpreted_solution 