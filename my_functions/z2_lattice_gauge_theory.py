"""
Z2 Lattice Gauge Theory Implementation for High-Order Unconstrained Binary Optimization

This module implements the Z2 Lattice Gauge Theory approach for solving high-order
unconstrained binary optimization (HUBO) problems as described in the paper:
"Speedup of high-order unconstrained binary optimization using quantum Z2 lattice gauge theory"

The implementation includes:
1. Mapping from HUBO to QZ2LGT (Quantum Z2 Lattice Gauge Theory)
2. Gauged Local Quantum Annealing (gLQA) algorithm
3. Quantum-inspired classical algorithm
4. Utility functions for graph representation and visualization
5. Support for both {0,1} and {-1,1} binary variables
6. Multiple solving methods including brute force and simulated annealing
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Tuple, Set, Optional, Union, Callable
import time
from itertools import combinations, product
from collections import defaultdict

class HUBOProblem:
    """
    Class representing a High-Order Unconstrained Binary Optimization (HUBO) problem.
    
    A HUBO problem is defined as minimizing a polynomial function of binary variables:
    H(s) = sum_i J_i s_i + sum_{i<j} J_{ij} s_i s_j + sum_{i<j<k} J_{ijk} s_i s_j s_k + ...
    where s_i can be either in {0, 1} or {-1, 1} and J's are real-number coefficients.
    """
    
    def __init__(self, binary_type: str = "spin"):
        """
        Initialize an empty HUBO problem.
        
        Args:
            binary_type: Type of binary variables ("spin" for {-1,1} or "binary" for {0,1})
        """
        self.terms = {}  # Dictionary to store terms and their coefficients
        self.num_variables = 0
        self.binary_type = binary_type  # "spin" for {-1,1}, "binary" for {0,1}
    
    def add_term(self, indices: Tuple[int, ...], coefficient: float):
        """
        Add a term to the HUBO problem.
        
        Args:
            indices: Tuple of variable indices involved in the term
            coefficient: Coefficient of the term
        """
        if not indices:
            raise ValueError("Indices tuple cannot be empty")
        
        # Sort indices to ensure consistent representation
        sorted_indices = tuple(sorted(set(indices)))
        
        # Skip if coefficient is zero
        if abs(coefficient) < 1e-10:
            return
        
        # Update the term or add a new one
        if sorted_indices in self.terms:
            self.terms[sorted_indices] += coefficient
        else:
            self.terms[sorted_indices] = coefficient
        
        # Update the number of variables if necessary
        self.num_variables = max(self.num_variables, max(indices) + 1)
        
        # Remove the term if coefficient becomes effectively zero
        if abs(self.terms[sorted_indices]) < 1e-10:
            del self.terms[sorted_indices]
    
    def evaluate(self, state: np.ndarray) -> float:
        """
        Evaluate the HUBO objective function for a given state.
        
        Args:
            state: Binary state vector (in {-1,1} or {0,1} depending on binary_type)
            
        Returns:
            Value of the objective function
        """
        if len(state) < self.num_variables:
            raise ValueError(f"State vector length {len(state)} is less than the number of variables {self.num_variables}")
        
        energy = 0.0
        for indices, coefficient in self.terms.items():
            term_value = coefficient
            for idx in indices:
                term_value *= state[idx]
            energy += term_value
        
        return energy
    
    def get_degree(self) -> int:
        """Return the maximum degree of the HUBO problem."""
        if not self.terms:
            return 0
        return max(len(variables) for variables in self.terms.keys())
    
    def get_hubo_graph(self) -> nx.Graph:
        """
        Create a graph representation of the HUBO problem.
        
        In this graph, each vertex represents a variable, and edges represent interactions
        between variables. For higher-order terms, we create a hypergraph-like structure.
        
        Returns:
            NetworkX graph representing the HUBO problem
        """
        G = nx.Graph()
        
        # Add nodes
        for i in range(self.num_variables):
            G.add_node(i, variable=i)
        
        # Add edges for pairwise interactions
        for indices, coefficient in self.terms.items():
            if len(indices) == 2:
                i, j = indices
                G.add_edge(i, j, weight=coefficient)
            elif len(indices) > 2:
                # For higher-order terms, add edges between all pairs of variables
                for i in indices:
                    for j in indices:
                        if i < j:
                            if G.has_edge(i, j):
                                # If edge already exists, we don't modify its weight
                                pass
                            else:
                                G.add_edge(i, j, weight=0.0, higher_order=True)
        
        return G
    
    def convert_to_spin(self) -> 'HUBOProblem':
        """
        Convert the problem from {0,1} to {-1,1} representation.
        
        Returns:
            New HUBOProblem instance with spin variables
        """
        if self.binary_type == "spin":
            return self
        
        spin_hubo = HUBOProblem(binary_type="spin")
        
        for indices, coefficient in self.terms.items():
            # Convert each term using the transformation x = (1 + s)/2
            # where x is in {0,1} and s is in {-1,1}
            new_coefficient = coefficient * (0.5 ** len(indices))
            spin_hubo.add_term(indices, new_coefficient)
            
            # Add all lower-order terms from the expansion
            for k in range(1, len(indices) + 1):
                for sub_indices in combinations(indices, k):
                    sub_coefficient = coefficient * (0.5 ** len(indices))
                    spin_hubo.add_term(sub_indices, sub_coefficient)
        
        return spin_hubo
    
    def convert_to_binary(self) -> 'HUBOProblem':
        """
        Convert the problem from {-1,1} to {0,1} representation.
        
        Returns:
            New HUBOProblem instance with binary variables
        """
        if self.binary_type == "binary":
            return self
        
        binary_hubo = HUBOProblem(binary_type="binary")
        
        for indices, coefficient in self.terms.items():
            # Convert each term using the transformation s = 2x - 1
            # where x is in {0,1} and s is in {-1,1}
            new_coefficient = coefficient * (2 ** len(indices))
            binary_hubo.add_term(indices, new_coefficient)
            
            # Add all lower-order terms from the expansion
            for k in range(1, len(indices) + 1):
                for sub_indices in combinations(indices, k):
                    sub_coefficient = coefficient * (-1) ** (len(indices) - k) * (2 ** k)
                    binary_hubo.add_term(sub_indices, sub_coefficient)
        
        return binary_hubo
    
    def __str__(self) -> str:
        """String representation of the HUBO problem."""
        terms_str = []
        for indices, coefficient in sorted(self.terms.items(), key=lambda x: (len(x[0]), x[0])):
            if self.binary_type == "spin":
                var_str = " ".join([f"s_{i}" for i in indices])
            else:
                var_str = " * ".join([f"x_{i}" for i in indices])
            terms_str.append(f"{coefficient:.4f} * {var_str}")
        
        if not terms_str:
            return "0"
        
        return " + ".join(terms_str)


class QZ2LGT:
    """
    Quantum Z2 Lattice Gauge Theory implementation for solving HUBO problems.
    
    This class implements the mapping from HUBO to QZ2LGT and provides methods
    for solving the problem using various approaches including:
    1. Gauged Local Quantum Annealing (gLQA)
    2. Brute force search
    3. Simulated annealing
    4. Quantum-inspired classical algorithm
    """
    
    def __init__(self, hubo: HUBOProblem, penalty_strength: float = 10.0):
        """
        Initialize the QZ2LGT solver with a HUBO problem.
        
        Args:
            hubo: The HUBO problem to solve
            penalty_strength: Strength of the penalty terms for constraints
        """
        self.hubo = hubo
        self.penalty_strength = penalty_strength
        self.hubo_graph = hubo.get_hubo_graph()
        self.g_graph = None  # Will be initialized in map_to_qz2lgt
        self.plaquettes = []  # List of plaquettes in the G-graph
        self.gauge_operators = {}  # Dictionary of gauge operators
        self.auxiliary_vars = {}  # Maps higher-order terms to auxiliary variables
        self.gauge_constraints = []  # List of gauge constraints
        self.var_mapping = {}  # Maps original variables to transformed variables
        self.reverse_mapping = {}  # Maps transformed variables to original variables
        
        # Map the HUBO problem to QZ2LGT
        self.map_to_qz2lgt()
    
    def map_to_qz2lgt(self):
        """
        Map the HUBO problem to QZ2LGT.
        
        This method creates the dual graph (G-graph) and identifies the plaquettes
        and gauge operators.
        """
        # Step 1: Create the dual graph (G-graph)
        self.g_graph = nx.Graph()
        
        # For each edge in the HUBO graph, create a node in the G-graph
        edge_to_node = {}
        node_counter = 0
        
        for u, v in self.hubo_graph.edges():
            self.g_graph.add_node(node_counter, edge=(u, v))
            edge_to_node[(u, v)] = node_counter
            edge_to_node[(v, u)] = node_counter  # Add both directions
            node_counter += 1
        
        # For each vertex in the HUBO graph, create a plaquette in the G-graph
        for v in self.hubo_graph.nodes():
            plaquette = []
            neighbors = list(self.hubo_graph.neighbors(v))
            
            # If the vertex has less than 2 neighbors, it doesn't form a plaquette
            if len(neighbors) < 2:
                continue
            
            # Create edges between nodes in the G-graph that correspond to adjacent edges in the HUBO graph
            for i in range(len(neighbors)):
                u1 = neighbors[i]
                u2 = neighbors[(i + 1) % len(neighbors)]
                
                node1 = edge_to_node[(v, u1)]
                node2 = edge_to_node[(v, u2)]
                
                self.g_graph.add_edge(node1, node2)
                plaquette.append(node1)
            
            self.plaquettes.append(plaquette)
        
        # Step 2: Identify gauge operators
        for v in self.hubo_graph.nodes():
            # The gauge operator for vertex v is the product of sigma_z operators
            # on all links connected to v
            gauge_op = []
            for u in self.hubo_graph.neighbors(v):
                gauge_op.append(edge_to_node[(v, u)])
            
            if gauge_op:  # Only add if the gauge operator is not empty
                self.gauge_operators[v] = gauge_op
    
    def find_cycles(self) -> List[List[int]]:
        """
        Find all cycles in the G-graph using a closed-loop search algorithm.
        
        Returns:
            List of cycles, where each cycle is a list of node indices
        """
        cycles = []
        visited_edges = set()
        
        def dfs(node, path, start_node):
            if node == start_node and len(path) > 2:
                # Found a cycle
                cycles.append(path.copy())
                return
            
            for neighbor in self.g_graph.neighbors(node):
                edge = tuple(sorted([node, neighbor]))
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    path.append(neighbor)
                    dfs(neighbor, path, start_node)
                    path.pop()
                    visited_edges.remove(edge)
        
        # Start DFS from each node
        for node in self.g_graph.nodes():
            dfs(node, [node], node)
        
        # Remove duplicate cycles (cycles that contain the same set of nodes)
        unique_cycles = []
        cycle_sets = []
        
        for cycle in cycles:
            cycle_set = frozenset(cycle)
            if cycle_set not in cycle_sets:
                cycle_sets.append(cycle_set)
                unique_cycles.append(cycle)
        
        return unique_cycles
    
    def find_efficient_cycles(self) -> List[List[int]]:
        """
        Find efficient cycles in the G-graph that cannot be further decomposed.
        
        Returns:
            List of efficient cycles, where each cycle is a list of node indices
        """
        all_cycles = self.find_cycles()
        efficient_cycles = []
        
        # Check each cycle to see if it can be decomposed into smaller cycles
        for cycle in all_cycles:
            can_decompose = False
            cycle_set = set(cycle)
            
            for other_cycle in all_cycles:
                if len(other_cycle) < len(cycle) and set(other_cycle).issubset(cycle_set):
                    can_decompose = True
                    break
            
            if not can_decompose:
                efficient_cycles.append(cycle)
        
        return efficient_cycles
    
    def local_quantum_annealing(self, num_iterations: int = 1000, temperature: float = 0.1) -> np.ndarray:
        """
        Perform Local Quantum Annealing (LQA) to find the ground state of the HUBO problem.
        
        Args:
            num_iterations: Number of annealing iterations
            temperature: Temperature parameter for the annealing process
            
        Returns:
            Binary state vector representing the solution
        """
        # Initialize random state
        state = np.random.choice([-1, 1], size=self.hubo.num_variables)
        current_energy = self.hubo.evaluate(state)
        
        for _ in range(num_iterations):
            # Randomly select a variable to flip
            idx = np.random.randint(0, self.hubo.num_variables)
            
            # Flip the variable
            new_state = state.copy()
            new_state[idx] *= -1
            
            # Calculate the new energy
            new_energy = self.hubo.evaluate(new_state)
            
            # Accept or reject the flip based on the energy difference
            delta_energy = new_energy - current_energy
            if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / temperature):
                state = new_state
                current_energy = new_energy
        
        return state
    
    def gauged_local_quantum_annealing(self, num_iterations: int = 1000, temperature: float = 0.1) -> np.ndarray:
        """
        Perform Gauged Local Quantum Annealing (gLQA) to find the ground state of the HUBO problem.
        
        This method uses gauge symmetry to improve the performance of LQA.
        
        Args:
            num_iterations: Number of annealing iterations
            temperature: Temperature parameter for the annealing process
            
        Returns:
            Binary state vector representing the solution
        """
        # Initialize random state
        state = np.random.choice([-1, 1], size=self.hubo.num_variables)
        current_energy = self.hubo.evaluate(state)
        
        # Find efficient cycles
        efficient_cycles = self.find_efficient_cycles()
        
        for iteration in range(num_iterations):
            # With some probability, apply a gauge transformation
            if np.random.random() < 0.3:  # 30% chance to apply gauge transformation
                # Randomly select a gauge operator
                if self.gauge_operators:
                    vertex = random.choice(list(self.gauge_operators.keys()))
                    gauge_op = self.gauge_operators[vertex]
                    
                    # Apply the gauge transformation
                    for edge_idx in gauge_op:
                        edge = self.g_graph.nodes[edge_idx]['edge']
                        u, v = edge
                        
                        # Flip both variables connected by the edge
                        state[u] *= -1
                        state[v] *= -1
                    
                    # Recalculate energy after gauge transformation
                    current_energy = self.hubo.evaluate(state)
            
            # Randomly select a variable to flip
            idx = np.random.randint(0, self.hubo.num_variables)
            
            # Flip the variable
            new_state = state.copy()
            new_state[idx] *= -1
            
            # Calculate the new energy
            new_energy = self.hubo.evaluate(new_state)
            
            # Accept or reject the flip based on the energy difference
            delta_energy = new_energy - current_energy
            if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / temperature):
                state = new_state
                current_energy = new_energy
            
            # Annealing schedule: reduce temperature over time
            temperature = max(0.01, temperature * 0.999)
        
        return state
    
    def quantum_inspired_classical_algorithm(self, num_iterations: int = 1000, initial_temperature: float = 1.0) -> np.ndarray:
        """
        Implement a quantum-inspired classical algorithm for solving the HUBO problem.
        
        This algorithm leverages the gauge symmetry to improve the performance of
        classical optimization methods.
        
        Args:
            num_iterations: Number of iterations
            initial_temperature: Initial temperature for annealing schedule
            
        Returns:
            Binary state vector representing the solution
        """
        # Initialize random state
        state = np.random.choice([-1, 1], size=self.hubo.num_variables)
        current_energy = self.hubo.evaluate(state)
        best_state = state.copy()
        best_energy = current_energy
        temperature = initial_temperature
        
        # Find efficient cycles
        efficient_cycles = self.find_efficient_cycles()
        
        for iteration in range(num_iterations):
            # With some probability, apply a gauge transformation
            if np.random.random() < 0.2:  # 20% chance to apply gauge transformation
                # Randomly select a gauge operator
                if self.gauge_operators:
                    vertex = random.choice(list(self.gauge_operators.keys()))
                    gauge_op = self.gauge_operators[vertex]
                    
                    # Apply the gauge transformation
                    for edge_idx in gauge_op:
                        edge = self.g_graph.nodes[edge_idx]['edge']
                        u, v = edge
                        
                        # Flip both variables connected by the edge
                        state[u] *= -1
                        state[v] *= -1
                    
                    # Recalculate energy after gauge transformation
                    current_energy = self.hubo.evaluate(state)
            
            # Try to improve the solution by flipping variables
            for idx in range(self.hubo.num_variables):
                # Flip the variable
                state[idx] *= -1
                
                # Calculate the new energy
                new_energy = self.hubo.evaluate(state)
                
                if new_energy < best_energy:
                    best_state = state.copy()
                    best_energy = new_energy
            
            # Annealing schedule: reduce temperature over time
            temperature = max(0.01, temperature * 0.999)
        
        return best_state
    
    def transform_to_qubo(self) -> HUBOProblem:
        """
        Transform the HUBO problem to a QUBO problem using Z2 lattice gauge theory.
        
        Returns:
            A transformed HUBO problem with at most quadratic terms
        """
        transformed_hubo = HUBOProblem(binary_type=self.hubo.binary_type)
        next_aux_var = self.hubo.num_variables
        
        # Map original variables to themselves first
        for i in range(self.hubo.num_variables):
            self.var_mapping[i] = i
            self.reverse_mapping[i] = i
            
        # Process the original HUBO terms
        for variables, coefficient in self.hubo.terms.items():
            if len(variables) <= 2:
                # Keep linear and quadratic terms as they are
                transformed_hubo.add_term(variables, coefficient)
            else:
                # Apply Z2 lattice gauge reduction for higher-order terms
                transformed_vars, aux_vars, constraints = self._reduce_higher_order_term(
                    variables, next_aux_var
                )
                next_aux_var += len(aux_vars)
                
                # Add the transformed term to the new HUBO
                transformed_hubo.add_term(transformed_vars, coefficient)
                
                # Record the auxiliary variables
                for i, aux_var in enumerate(aux_vars):
                    self.auxiliary_vars.setdefault(variables, []).append(aux_var)
                    self.reverse_mapping[aux_var] = f"aux_{variables}_{i}"
                
                # Add the gauge constraints
                for constraint in constraints:
                    self.gauge_constraints.append(constraint)
                    
                    # Add penalty terms for each constraint (a = b * c => (a - b * c)^2)
                    a, b, c = constraint
                    if self.hubo.binary_type == "binary":
                        # For binary variables: (a - b * c)^2 = a - 2*a*b*c + b*c
                        transformed_hubo.add_term((a,), 1.0)
                        transformed_hubo.add_term((a, b, c), -2.0)
                        transformed_hubo.add_term((b, c), 1.0)
                    else:
                        # For spin variables: (a - b * c)^2 = 1 - a*b*c
                        transformed_hubo.add_term((a, b, c), -1.0)
                        transformed_hubo.add_term((), 1.0)
        
        # Apply additional reduction for remaining higher-order terms
        while transformed_hubo.get_degree() > 2:
            new_hubo = HUBOProblem(binary_type=self.hubo.binary_type)
            next_aux_var = max(self.reverse_mapping.keys()) + 1
            
            for variables, coefficient in transformed_hubo.terms.items():
                if len(variables) <= 2:
                    new_hubo.add_term(variables, coefficient)
                else:
                    transformed_vars, aux_vars, constraints = self._reduce_higher_order_term(
                        variables, next_aux_var
                    )
                    next_aux_var += len(aux_vars)
                    
                    new_hubo.add_term(transformed_vars, coefficient)
                    
                    for i, aux_var in enumerate(aux_vars):
                        self.auxiliary_vars.setdefault(variables, []).append(aux_var)
                        self.reverse_mapping[aux_var] = f"aux_{variables}_{i}"
                    
                    for constraint in constraints:
                        self.gauge_constraints.append(constraint)
                        
                        a, b, c = constraint
                        if self.hubo.binary_type == "binary":
                            new_hubo.add_term((a,), 1.0)
                            new_hubo.add_term((a, b), -1.0)
                            new_hubo.add_term((a, c), -1.0)
                            new_hubo.add_term((b, c), 1.0)
                        else:
                            new_hubo.add_term((a, b, c), -1.0)
                            new_hubo.add_term((), 1.0)
            
            transformed_hubo = new_hubo
        
        return transformed_hubo
    
    def _reduce_higher_order_term(self, variables: Tuple[int, ...], next_aux_var: int) -> Tuple:
        """
        Reduce a higher-order term using the Z2 lattice gauge transformation.
        
        Args:
            variables: The variables in the higher-order term
            next_aux_var: The next available auxiliary variable index
            
        Returns:
            Tuple containing:
            - Transformed variables (quadratic or lower)
            - New auxiliary variables
            - List of gauge constraints
        """
        if len(variables) <= 2:
            return variables, [], []
        
        # For a term with k variables, we need k-2 auxiliary variables
        aux_vars = list(range(next_aux_var, next_aux_var + len(variables) - 2))
        constraints = []
        
        # Create a binary tree reduction
        nodes = list(variables)
        while len(nodes) > 1:
            new_nodes = []
            for i in range(0, len(nodes), 2):
                if i + 1 < len(nodes):
                    # If we have a pair, create an auxiliary variable
                    a, b = nodes[i], nodes[i + 1]
                    aux_var = aux_vars.pop(0) if aux_vars else next_aux_var
                    next_aux_var += 1
                    new_nodes.append(aux_var)
                    
                    # Add the constraint: aux = a * b
                    constraints.append((aux_var, a, b))
                else:
                    # If we have a leftover node, just pass it to the next level
                    new_nodes.append(nodes[i])
            nodes = new_nodes
        
        # The final node is the transformed term
        transformed_vars = (nodes[0],)
        
        return transformed_vars, aux_vars, constraints
    
    def map_solution(self, qubo_solution: np.ndarray) -> np.ndarray:
        """
        Map a solution of the transformed problem back to the original problem.
        
        Args:
            qubo_solution: Solution of the transformed QUBO problem
            
        Returns:
            Solution of the original HUBO problem
        """
        hubo_solution = np.zeros(self.hubo.num_variables, dtype=int)
        
        for original_var, transformed_var in self.var_mapping.items():
            hubo_solution[original_var] = qubo_solution[transformed_var]
            
        return hubo_solution
    
    def solve(self, method: str = "gLQA", **kwargs) -> Tuple[np.ndarray, float]:
        """
        Solve the HUBO problem using the specified method.
        
        Args:
            method: Solving method ("gLQA", "brute_force", "simulated_annealing", "quantum_inspired")
            **kwargs: Additional arguments for the specific solving method
            
        Returns:
            Tuple containing:
            - Optimal binary assignment
            - Optimal energy value
        """
        if method == "gLQA":
            solution = self.gauged_local_quantum_annealing(**kwargs)
        elif method == "brute_force":
            solution = self._solve_brute_force()
        elif method == "simulated_annealing":
            solution = self._solve_simulated_annealing(**kwargs)
        elif method == "quantum_inspired":
            solution = self.quantum_inspired_classical_algorithm(**kwargs)
        else:
            raise ValueError(f"Unknown solving method: {method}")
        
        energy = self.hubo.evaluate(solution)
        return solution, energy
    
    def _solve_brute_force(self) -> np.ndarray:
        """
        Solve the HUBO problem using brute force search.
        
        Returns:
            Optimal binary assignment
        """
        n_vars = self.hubo.num_variables
        best_energy = float('inf')
        best_assignment = None
        
        # Try all possible binary assignments
        for i in range(2**n_vars):
            # Convert integer to binary array
            if self.hubo.binary_type == "binary":
                assignment = np.array([int(b) for b in format(i, f'0{n_vars}b')], dtype=int)
            else:
                assignment = np.array([1 if b == '1' else -1 for b in format(i, f'0{n_vars}b')], dtype=int)
            
            # Evaluate the energy
            energy = self.hubo.evaluate(assignment)
            
            # Update if we found a better solution
            if energy < best_energy:
                best_energy = energy
                best_assignment = assignment
        
        return best_assignment
    
    def _solve_simulated_annealing(self, steps: int = 1000, temp_init: float = 10.0, 
                                  temp_final: float = 0.1) -> np.ndarray:
        """
        Solve the HUBO problem using simulated annealing.
        
        Args:
            steps: Number of annealing steps
            temp_init: Initial temperature
            temp_final: Final temperature
            
        Returns:
            Optimal binary assignment
        """
        n_vars = self.hubo.num_variables
        
        # Initialize random state
        if self.hubo.binary_type == "binary":
            state = np.random.randint(0, 2, size=n_vars)
        else:
            state = np.random.choice([-1, 1], size=n_vars)
        
        energy = self.hubo.evaluate(state)
        best_state = state.copy()
        best_energy = energy
        
        # Annealing schedule
        temp_schedule = np.geomspace(temp_init, temp_final, steps)
        
        for step in range(steps):
            temp = temp_schedule[step]
            
            # Choose a random bit to flip
            flip_idx = np.random.randint(0, n_vars)
            new_state = state.copy()
            if self.hubo.binary_type == "binary":
                new_state[flip_idx] = 1 - new_state[flip_idx]
            else:
                new_state[flip_idx] *= -1
            
            # Calculate new energy
            new_energy = self.hubo.evaluate(new_state)
            
            # Metropolis acceptance criterion
            delta_energy = new_energy - energy
            if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / temp):
                state = new_state
                energy = new_energy
                
                # Update best solution if needed
                if energy < best_energy:
                    best_energy = energy
                    best_state = state.copy()
        
        return best_state

def create_random_hubo(n_variables: int, max_degree: int, n_terms: int, 
                      coef_range: Tuple[float, float] = (-1.0, 1.0),
                      binary_type: str = "spin") -> HUBOProblem:
    """
    Create a random HUBO problem for testing.
    
    Args:
        n_variables: Number of variables
        max_degree: Maximum degree of the terms
        n_terms: Number of terms to generate
        coef_range: Range for random coefficients
        binary_type: Type of binary variables ("spin" for {-1,1} or "binary" for {0,1})
        
    Returns:
        A randomly generated HUBO problem
    """
    hubo = HUBOProblem(binary_type=binary_type)
    
    for _ in range(n_terms):
        # Choose random degree between 1 and max_degree
        degree = np.random.randint(1, max_degree + 1)
        
        # Choose random variables
        variables = tuple(sorted(np.random.choice(n_variables, size=degree, replace=False)))
        
        # Choose random coefficient
        coefficient = np.random.uniform(*coef_range)
        
        hubo.add_term(variables, coefficient)
    
    return hubo

def visualize_qubo_graph(qubo: HUBOProblem, ax=None):
    """
    Visualize the QUBO problem as a graph.
    
    Args:
        qubo: The QUBO problem to visualize
        ax: Matplotlib axis for plotting
    """
    if qubo.get_degree() > 2:
        print("Warning: The problem has higher than quadratic terms, only visualizing quadratic terms.")
    
    G = nx.Graph()
    
    # Add nodes
    for i in range(qubo.num_variables):
        G.add_node(i)
    
    # Add edges for quadratic terms
    for variables, coefficient in qubo.terms.items():
        if len(variables) == 2:
            i, j = variables
            G.add_edge(i, j, weight=coefficient)
    
    # Add self-loops for linear terms
    for variables, coefficient in qubo.terms.items():
        if len(variables) == 1:
            i = variables[0]
            G.nodes[i]['weight'] = coefficient
    
    # Determine edge colors based on coefficients
    edge_colors = []
    for u, v, data in G.edges(data=True):
        if data['weight'] > 0:
            edge_colors.append('red')
        else:
            edge_colors.append('blue')
    
    # Determine node colors based on linear coefficients
    node_colors = []
    for i in range(qubo.num_variables):
        weight = G.nodes[i].get('weight', 0)
        if weight > 0:
            node_colors.append('red')
        elif weight < 0:
            node_colors.append('blue')
        else:
            node_colors.append('gray')
    
    # Plot the graph
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    
    # Add edge weights as labels
    edge_labels = {(u, v): f"{data['weight']:.2f}" for u, v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)
    
    ax.set_title("QUBO Graph Visualization")
    ax.set_axis_off()
    
    return ax

def benchmark_solvers(n_vars_list: List[int], max_degree: int = 3, n_terms_factor: int = 2,
                     binary_type: str = "spin", methods: List[str] = None):
    """
    Benchmark different solving methods on HUBO problems of various sizes.
    
    Args:
        n_vars_list: List of numbers of variables to test
        max_degree: Maximum degree of the terms
        n_terms_factor: Factor to multiply by number of variables to get number of terms
        binary_type: Type of binary variables ("spin" for {-1,1} or "binary" for {0,1})
        methods: List of solving methods to benchmark
    """
    if methods is None:
        methods = ["gLQA", "simulated_annealing", "quantum_inspired"]
    
    results = {}
    
    for n_vars in n_vars_list:
        print(f"\nSolving HUBO with {n_vars} variables...")
        n_terms = n_vars * n_terms_factor
        
        # Create random HUBO problem
        hubo = create_random_hubo(n_variables=n_vars, max_degree=max_degree,
                                n_terms=n_terms, binary_type=binary_type)
        
        # Initialize solver
        solver = QZ2LGT(hubo)
        
        # Try each method
        for method in methods:
            if method == "brute_force" and n_vars > 15:
                print(f"  Skipping brute force for {n_vars} variables (too large)")
                continue
            
            print(f"  Using {method}...")
            start_time = time.time()
            solution, energy = solver.solve(method=method)
            runtime = time.time() - start_time
            
            results.setdefault(method, []).append({
                'n_vars': n_vars,
                'energy': energy,
                'runtime': runtime
            })
            
            print(f"    Energy: {energy:.4f}")
            print(f"    Runtime: {runtime:.4f}s")
    
    return results

# Example usage
if __name__ == "__main__":
    print("Demonstrating Quantum Z2 Lattice Gauge Theory for HUBO problems")
    
    # Create a simple HUBO problem
    hubo = HUBOProblem(binary_type="spin")
    hubo.add_term((0, 1, 2), 1.0)  # s_0 * s_1 * s_2
    hubo.add_term((0, 1), -0.5)    # -0.5 * s_0 * s_1
    hubo.add_term((1,), 0.3)       # 0.3 * s_1
    
    print("Original HUBO problem:")
    print(hubo)
    
    # Transform to QUBO
    solver = QZ2LGT(hubo)
    qubo = solver.transform_to_qubo()
    
    print("\nTransformed QUBO problem:")
    print(qubo)
    
    # Solve using different methods
    print("\nSolving with different methods:")
    for method in ["gLQA", "brute_force", "simulated_annealing"]:
        solution, energy = solver.solve(method=method)
        print(f"\n{method}:")
        print(f"  Solution: {solution}")
        print(f"  Energy: {energy}")
    
    # Create a random HUBO problem
    print("\nCreating a random HUBO problem...")
    random_hubo = create_random_hubo(n_variables=5, max_degree=4, n_terms=10)
    
    print("Random HUBO problem:")
    print(random_hubo)
    
    # Benchmark different methods
    print("\nBenchmarking different methods...")
    results = benchmark_solvers(n_vars_list=[10, 15, 20])
    
    # Visualize a QUBO graph
    plt.figure(figsize=(10, 8))
    visualize_qubo_graph(qubo)
    plt.tight_layout()
    plt.show()