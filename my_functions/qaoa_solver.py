"""
Quantum Approximate Optimization Algorithm (QAOA) Implementation

This module provides a comprehensive implementation of QAOA that combines:
1. General optimization capabilities (QUBO and Ising)
2. MaxCut-specific implementations
3. Both state vector and quantum circuit simulations
4. Extensive analysis and visualization tools
5. Multiple parameter optimization strategies
6. Noise modeling and hardware considerations
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_algorithms import QAOA as QiskitQAOA, VQE
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo, IntegerToBinary
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.translators.docplex_mp import from_docplex_mp
from typing import Union, List, Dict, Tuple, Optional, Callable
import itertools
from tqdm import tqdm
import time


# Import from new modules
from my_functions.graph_utils import GraphPartitioner
from my_functions.optimization import ClassicalOptimizer, SimulatedAnnealing

def create_random_graph(n_nodes: int, edge_probability: float = 0.5, 
                       weight_range: Tuple[float, float] = (0.1, 1.0)) -> nx.Graph:
    """
    Create a random weighted graph for MaxCut problems.
    
    Args:
        n_nodes: Number of nodes in the graph
        edge_probability: Probability of an edge existing between any two nodes
        weight_range: Tuple of (min_weight, max_weight) for edge weights
        
    Returns:
        NetworkX graph with random edges and weights
    """
    # Create an empty graph
    graph = nx.Graph()
    
    # Add nodes
    graph.add_nodes_from(range(n_nodes))
    
    # Add edges with random weights
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.random() < edge_probability:
                weight = np.random.uniform(weight_range[0], weight_range[1])
                graph.add_edge(i, j, weight=weight)
    
    return graph

class QAOABase:
    """Base class for QAOA implementations with common functionality"""
    
    def __init__(self, p: int = 1, shots: int = 1024):
        """
        Initialize QAOA base class
        
        Args:
            p: Number of QAOA layers
            shots: Number of measurement shots
        """
        self.p = p
        self.shots = shots
        self.optimal_params = None
        self.optimal_value = None
        self.optimal_p = None
        self.sampler = Sampler()
        
    def _heuristic_initial_point(self, p: int) -> np.ndarray:
        """Generate heuristic initial parameters"""
        if p == 1:
            return np.array([0.7 * np.pi, 0.5 * np.pi])
        return np.concatenate([
            np.linspace(0.1 * np.pi, 0.7 * np.pi, p),
            np.linspace(0.1 * np.pi, 0.5 * np.pi, p)
        ])
    
    def _fourier_initialization(self, prev_params: np.ndarray, prev_p: int, new_p: int) -> np.ndarray:
        """Initialize parameters using Fourier heuristic"""
        prev_gammas = prev_params[:prev_p]
        prev_betas = prev_params[prev_p:]
        
        gammas = np.zeros(new_p)
        betas = np.zeros(new_p)
        
        gammas[:prev_p] = prev_gammas
        betas[:prev_p] = prev_betas
        
        return np.concatenate((gammas, betas))

class QAOACircuit(QAOABase):
    """QAOA implementation using Qiskit quantum circuits"""
    
    def __init__(self, p: int = 1, shots: int = 1024):
        """
        Initialize QAOA circuit implementation
        
        Args:
            p: Number of QAOA layers
            shots: Number of measurement shots
        """
        super().__init__(p, shots)
        self.sampler = Sampler()
        self.circuit = None
        self.n_qubits = None
        
    def build_circuit(self, hamiltonian):
        """Build the QAOA circuit for the given Hamiltonian."""
        # Determine number of qubits needed
        max_qubit = 0
        for term in hamiltonian:
            if len(term) == 2:
                i, j, _ = term
                max_qubit = max(max_qubit, i, j)
            else:
                i, _ = term
                max_qubit = max(max_qubit, i)
        self.n_qubits = max_qubit + 1
        
        # Create quantum circuit
        self.circuit = QuantumCircuit(self.n_qubits)
        
        # Initial state preparation
        for i in range(self.n_qubits):
            self.circuit.h(i)
            
        # QAOA layers
        for layer in range(self.p):
            # Problem unitary
            for term in hamiltonian:
                if len(term) == 2:  # Two-qubit term
                    i, j, weight = term
                    self.circuit.rzz(2 * weight, i, j)
                else:  # Single-qubit term
                    i, weight = term
                    self.circuit.rz(2 * weight, i)
                    
            # Mixer unitary
            for i in range(self.n_qubits):
                self.circuit.rx(2, i)
                
        # Measurement
        self.circuit.measure_all()
        
    def optimize(self, hamiltonian, shots=1024):
        """Optimize the QAOA circuit parameters."""
        try:
            # Build the circuit
            self.build_circuit(hamiltonian)
            
            # Initialize parameters
            self.parameters = np.random.rand(2 * self.p)
            
            # Define objective function
            def objective(params):
                try:
                    # Set parameters in circuit
                    param_idx = 0
                    for layer in range(self.p):
                        for term in hamiltonian:
                            if len(term) == 2:
                                i, j, weight = term
                                self.circuit.rzz(2 * weight * params[param_idx], i, j)
                            else:
                                i, weight = term
                                self.circuit.rz(2 * weight * params[param_idx], i)
                        param_idx += 1
                        
                        for i in range(self.n_qubits):
                            self.circuit.rx(2 * params[param_idx], i)
                        param_idx += 1
                    
                    # Run circuit
                    backend = AerSimulator(method='statevector')
                    result = backend.run(self.circuit, shots=shots).result()
                    counts = result.get_counts()
                    
                    # Calculate expectation value
                    expectation = 0
                    for bitstring, count in counts.items():
                        state = [int(b) for b in bitstring]
                        energy = 0
                        for term in hamiltonian:
                            if len(term) == 2:
                                i, j, weight = term
                                energy += weight * (1 - 2 * state[i]) * (1 - 2 * state[j])
                            else:
                                i, weight = term
                                energy += weight * (1 - 2 * state[i])
                        expectation += energy * count / shots
                        
                    return expectation
                except Exception as e:
                    print(f"Error in objective function: {e}")
                    return float('inf')
            
            # Optimize parameters
            result = minimize(objective, self.parameters, method='COBYLA')
            self.parameters = result.x
            
            # Get final solution
            backend = AerSimulator(method='statevector')
            result = backend.run(self.circuit, shots=shots).result()
            counts = result.get_counts()
            
            # Find best solution
            best_energy = float('inf')
            best_solution = None
            for bitstring, count in counts.items():
                state = [int(b) for b in bitstring]
                energy = 0
                for term in hamiltonian:
                    if len(term) == 2:
                        i, j, weight = term
                        energy += weight * (1 - 2 * state[i]) * (1 - 2 * state[j])
                    else:
                        i, weight = term
                        energy += weight * (1 - 2 * state[i])
                if energy < best_energy:
                    best_energy = energy
                    best_solution = state
                    
            if best_solution is None:
                raise ValueError("No valid solution found")
            
            return np.array(best_solution), best_energy
            
        except Exception as e:
            print(f"Error in QAOA optimization: {e}")
            return None, None

    def solve_qubo(self, Q: np.ndarray, p: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        Solve a QUBO problem using QAOA
        
        Args:
            Q: QUBO matrix
            p: Optional override for number of layers
            
        Returns:
            Tuple of (solution, energy)
        """
        p_value = p if p is not None else self.p
        
        # Create quadratic program
        n = Q.shape[0]
        qp = QuadraticProgram()
        for i in range(n):
            qp.binary_var(name=f'x{i}')
            
        # Set objective
        linear = np.diag(Q).copy()
        quadratic = {}
        for i in range(n):
            for j in range(i+1, n):
                if Q[i, j] != 0:
                    quadratic[(i, j)] = Q[i, j] + Q[j, i]
                    
        qp.minimize(linear=linear, quadratic=quadratic)
        
        # Convert to QUBO and solve
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)
        
        qaoa = QiskitQAOA(
            optimizer=COBYLA(maxiter=1000),
            reps=p_value,
            sampler=self.sampler
        )
        
        optimizer = MinimumEigenOptimizer(qaoa)
        result = optimizer.solve(qubo)
        
        return np.array(result.x), result.fval
        
    def solve_ising(self, h: np.ndarray, J: Dict[Tuple[int, int], float], 
                   p: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        Solve an Ising model problem using QAOA
        
        Args:
            h: Linear coefficients
            J: Dictionary of coupling terms
            p: Optional override for number of layers
            
        Returns:
            Tuple of (solution, energy)
        """
        # Convert Ising to QUBO
        n = len(h)
        Q = np.zeros((n, n))
        
        # Set diagonal terms
        for i in range(n):
            Q[i, i] = 2 * h[i]
            for j in J:
                if i in j:
                    other = j[0] if j[1] == i else j[1]
                    Q[i, i] += 2 * J[j]
        
        # Set off-diagonal terms
        for (i, j), coupling in J.items():
            Q[i, j] = 4 * coupling
            Q[j, i] = 4 * coupling
        
        # Calculate offset
        offset = -np.sum(h) - sum(J.values())
        
        # Solve QUBO
        binary_sol, qubo_energy = self.solve_qubo(Q, p)
        
        # Convert to spin
        spin_sol = 2 * binary_sol - 1
        ising_energy = qubo_energy + offset
        
        return spin_sol, ising_energy

class QAOAMaxCut(QAOABase):
    """QAOA implementation specifically for MaxCut problems"""
    
    def __init__(self, graph: Union[nx.Graph, np.ndarray], p: int = 1, shots: int = 1024, timeout: Optional[float] = None):
        """
        Initialize QAOA for MaxCut.
        
        Args:
            graph: Input graph (NetworkX or adjacency matrix)
            p: Number of QAOA layers
            shots: Number of measurement shots
            timeout: Maximum time in seconds for optimization (optional)
        """
        super().__init__(p=p, shots=shots)
        self.graph = graph if isinstance(graph, nx.Graph) else nx.from_numpy_array(graph)
        self.n_qubits = len(self.graph)
        self.best_solution = None
        self.best_energy = float('inf')
        self.start_time = None
        self.timeout = timeout
        
        # Initialize backend with specific options
        self.backend = AerSimulator(
            method='statevector',
            max_parallel_threads=1,
            max_parallel_experiments=1,
            max_parallel_shots=1
        )
        self.sampler = Sampler()
        
    def _build_circuit(self, params: np.ndarray, p: int) -> QuantumCircuit:
        """
        Build QAOA circuit for MaxCut.
        
        Args:
            params: QAOA parameters
            p: Number of QAOA layers
            
        Returns:
            QuantumCircuit: QAOA circuit
        """
        try:
            # Create quantum circuit
            qc = QuantumCircuit(self.n_qubits)
            
            # Apply Hadamard gates to all qubits
            for i in range(self.n_qubits):
                qc.h(i)
                
            # Apply QAOA layers
            for layer in range(p):
                # Cost Hamiltonian
                for u, v, data in self.graph.edges(data=True):
                    weight = data.get('weight', 1.0)
                    # Skip self-loops and duplicate edges
                    if u != v:
                        qc.cx(u, v)
                        qc.rz(2 * weight * params[2 * layer], v)
                        qc.cx(u, v)
                
                # Mixer Hamiltonian
                for i in range(self.n_qubits):
                    qc.rx(2 * params[2 * layer + 1], i)
                    
            return qc
            
        except Exception as e:
            raise RuntimeError(f"Failed to build QAOA circuit: {e}")
            
    def _run_circuit(self, circuit: QuantumCircuit) -> dict:
        """
        Run the quantum circuit and return measurement counts.
        
        Args:
            circuit: Quantum circuit to run
            
        Returns:
            dict: Measurement counts
        """
        try:
            # Add measurement gates
            circuit.measure_all()
            
            # Transpile circuit with specific options
            transpiled = transpile(
                circuit,
                self.backend,
                optimization_level=0,  # Minimal optimization
                basis_gates=['u1', 'u2', 'u3', 'cx'],  # Basic gate set
                coupling_map=None,  # No specific coupling map
                seed_transpiler=42  # Fixed seed for reproducibility
            )
            
            # Run circuit with timeout
            if self.timeout is not None:
                start_time = time.time()
                while time.time() - start_time < self.timeout:
                    try:
                        job = self.backend.run(transpiled, shots=self.shots)
                        result = job.result()
                        return result.get_counts()
                    except Exception:
                        continue
                raise TimeoutError("Circuit execution timed out")
            else:
                job = self.backend.run(transpiled, shots=self.shots)
                result = job.result()
                return result.get_counts()
            
        except Exception as e:
            raise RuntimeError(f"Failed to run quantum circuit: {e}")
            
    def _compute_expectation(self, counts: dict) -> float:
        """
        Compute expectation value from measurement counts.
        
        Args:
            counts: Measurement counts
            
        Returns:
            float: Expectation value
        """
        try:
            total = 0
            for bitstring, count in counts.items():
                # Convert bitstring to solution
                solution = np.array([int(bit) for bit in bitstring])
                
                # Compute cut value
                cut_value = 0
                for u, v, data in self.graph.edges(data=True):
                    weight = data.get('weight', 1.0)
                    if solution[u] != solution[v]:
                        cut_value += weight
                        
                total += cut_value * count
                
            return total / self.shots
            
        except Exception as e:
            raise RuntimeError(f"Failed to compute expectation value: {e}")
            
    def optimize(self, p: Optional[int] = None, method: str = 'COBYLA', 
                init_method: str = 'heuristic', maxiter: int = 100,
                timeout: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        Optimize QAOA parameters for MaxCut.
        
        Args:
            p: Number of QAOA layers (overrides initialization if provided)
            method: Optimization method ('COBYLA', 'SPSA', etc.)
            init_method: Parameter initialization method ('heuristic' or 'fourier')
            maxiter: Maximum number of iterations
            timeout: Maximum time in seconds for optimization (overrides initialization if provided)
            
        Returns:
            Tuple of (best solution, best energy)
        """
        try:
            if p is not None:
                self.p = p
                
            # Use provided timeout or fall back to instance timeout
            self.timeout = timeout if timeout is not None else self.timeout
                
            # Initialize parameters
            if init_method == 'heuristic':
                params = self._heuristic_initial_point(self.p)
            else:
                params = self._fourier_initialization(np.zeros(2), 1, self.p)
                
            # Set up optimizer
            if method == 'COBYLA':
                optimizer = COBYLA(maxiter=maxiter, tol=1e-4)
            elif method == 'SPSA':
                optimizer = SPSA(maxiter=maxiter, learning_rate=0.01, perturbation=0.01)
            else:
                raise ValueError(f"Unsupported optimization method: {method}")
                
            # Start timing
            self.start_time = time.time()
            
            def objective(params):
                # Check timeout
                if self.timeout is not None and (time.time() - self.start_time) > self.timeout:
                    raise TimeoutError("QAOA optimization timed out")
                    
                try:
                    # Build and run circuit
                    circuit = self._build_circuit(params, self.p)
                    counts = self._run_circuit(circuit)
                    
                    # Compute expectation value
                    energy = self._compute_expectation(counts)
                    
                    # Update best solution
                    if energy < self.best_energy:
                        self.best_energy = energy
                        self.best_solution = self._get_solution_from_counts(counts)
                        
                    return energy
                except Exception as e:
                    print(f"Error in objective function: {e}")
                    return float('inf')
                
            try:
                result = optimizer.minimize(objective, params)
                return self.best_solution, self.best_energy
            except TimeoutError:
                if self.best_solution is not None:
                    return self.best_solution, self.best_energy
                raise
            except Exception as e:
                raise RuntimeError(f"Optimization failed: {e}")
                
        except Exception as e:
            raise RuntimeError(f"QAOA optimization failed: {e}")
        
    def _get_solution_from_counts(self, counts: dict) -> np.ndarray:
        """Extract the most frequent solution from measurement counts."""
        max_count = 0
        best_solution = None
        for bitstring, count in counts.items():
            if count > max_count:
                max_count = count
                best_solution = np.array([int(b) for b in bitstring])
        return best_solution
        
    def get_solution(self) -> Tuple[str, float]:
        """Get the best MaxCut solution"""
        if self.best_solution is None:
            raise ValueError("Run optimize() first")
            
        bitstring = ''.join(str(int(bit)) for bit in self.best_solution)
        cut_value = 0
        for u, v, data in self.graph.edges(data=True):
            weight = data.get('weight', 1.0)
            if self.best_solution[u] != self.best_solution[v]:
                cut_value += weight
                
        return bitstring, cut_value
        
    def visualize_solution(self):
        """Visualize the MaxCut solution"""
        if self.best_solution is None:
            raise ValueError("Run optimize() first")
            
        bitstring, cut_value = self.get_solution()
        solution = [1 if bit == '0' else -1 for bit in bitstring]
        
        # Create node colors
        node_colors = ['red' if bit == 1 else 'blue' for bit in solution]
        
        # Create edge colors
        edge_colors = []
        for u, v, _ in self.graph.edges():
            if solution[u] != solution[v]:
                edge_colors.append('green')
            else:
                edge_colors.append('black')
                
        # Draw graph
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph)
        nx.draw_networkx(
            self.graph, 
            pos, 
            node_color=node_colors,
            edge_color=edge_colors,
            with_labels=True
        )
        
        plt.title(f"MaxCut Solution: {cut_value}")
        plt.axis('off')
        plt.show()

class QAOAAnalyzer:
    """Class for analyzing and comparing QAOA implementations"""
    
    def __init__(self, graph: nx.Graph):
        """
        Initialize QAOA analyzer
        
        Args:
            graph: NetworkX graph to analyze
        """
        self.graph = graph
        self.n_nodes = graph.number_of_nodes()
        
    def brute_force_maxcut(self) -> Tuple[np.ndarray, float]:
        """Calculate exact MaxCut solution using brute force"""
        best_cut = 0
        best_solution = None
        
        for b in itertools.product([-1, 1], repeat=self.n_nodes):
            cut = 0
            for u, v in self.graph.edges():
                if b[u] != b[v]:
                    cut += 1
            if cut > best_cut:
                best_cut = cut
                best_solution = b
                
        return np.array(best_solution), best_cut
        
    def compare_optimization_methods(self, p_max: int, methods: List[str], 
                                   noise_model: Optional[NoiseModel] = None) -> Dict:
        """
        Compare different optimization methods
        
        Args:
            p_max: Maximum number of layers to test
            methods: List of initialization methods to compare
            noise_model: Optional noise model
            
        Returns:
            Dictionary of results
        """
        results = {method: [] for method in methods}
        exact_solution, exact_value = self.brute_force_maxcut()
        
        for p in range(1, p_max + 1):
            for method in methods:
                qaoa = QAOAMaxCut(self.graph, p=p)
                params, value = qaoa.optimize(method=method)
                
                results[method].append({
                    'p': p,
                    'params': params,
                    'value': value,
                    'approx_ratio': value / exact_value
                })
                
        return results, exact_value
        
    def plot_results(self, results: Dict, exact_value: float):
        """Plot comparison results"""
        methods = list(results.keys())
        p_values = range(1, len(results[methods[0]]) + 1)
        
        # Plot approximation ratios
        plt.figure(figsize=(10, 6))
        for method in methods:
            ratios = [result['approx_ratio'] for result in results[method]]
            plt.plot(p_values, ratios, 'o-', label=method)
            
        plt.axhline(y=1.0, color='r', linestyle='--', label='Exact solution')
        plt.xlabel('Number of layers (p)')
        plt.ylabel('Approximation ratio')
        plt.title('QAOA Performance vs Circuit Depth')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        # Plot parameter values
        for method in methods:
            plt.figure(figsize=(12, 8))
            for p_idx, result in enumerate(results[method]):
                p = p_idx + 1
                gamma = result['params'][:p]
                beta = result['params'][p:]
                
                plt.subplot(2, 1, 1)
                plt.plot(range(1, p + 1), gamma, 'o-', label=f'p={p}')
                plt.title(f'{method} - Gamma Parameters')
                plt.ylabel('Gamma value')
                plt.grid(True)
                
                plt.subplot(2, 1, 2)
                plt.plot(range(1, p + 1), beta, 'o-', label=f'p={p}')
                plt.title(f'{method} - Beta Parameters')
                plt.xlabel('Layer index')
                plt.ylabel('Beta value')
                plt.grid(True)
                
            plt.tight_layout()
            plt.legend()
            plt.show()

class QuadraticProgramConverter:
    """Class for converting and solving quadratic optimization problems"""
    
    def __init__(self, problem_size: int = 10):
        """
        Initialize quadratic program converter
        
        Args:
            problem_size: Size of the quadratic optimization problem
        """
        self.problem_size = problem_size
        self.sampler = Sampler()
        
    def create_sample_problem(self) -> QuadraticProgram:
        """
        Create a sample quadratic optimization problem
        
        Returns:
            QuadraticProgram: Sample optimization problem
        """
        problem = QuadraticProgram()
        
        # Add continuous variables
        for i in range(self.problem_size):
            problem.continuous_var(
                f'x_{i}', 
                lowerbound=0, 
                upperbound=1
            )
        
        # Set quadratic objective
        linear_coeffs = np.random.uniform(-1, 1, self.problem_size)
        quadratic_coeffs = np.random.uniform(-0.5, 0.5, 
                                           (self.problem_size, self.problem_size))
        
        problem.minimize(
            linear=linear_coeffs,
            quadratic=quadratic_coeffs
        )
        
        # Add constraints
        for _ in range(3):
            constraint_coeffs = np.random.uniform(-1, 1, self.problem_size)
            problem.linear_constraint(
                linear=constraint_coeffs, 
                sense='<=', 
                rhs=np.random.uniform(0, self.problem_size)
            )
        
        return problem
    
    def convert_to_qubo(self, problem: QuadraticProgram) -> QuadraticProgram:
        """
        Convert quadratic program to QUBO form
        
        Args:
            problem: Quadratic program to convert
            
        Returns:
            QuadraticProgram: QUBO-converted problem
        """
        converter = QuadraticProgramToQubo()
        return converter.convert(problem)
    
    def convert_integer_to_binary(self, problem: QuadraticProgram) -> QuadraticProgram:
        """
        Convert integer variables to binary
        
        Args:
            problem: Quadratic program to convert
            
        Returns:
            QuadraticProgram: Binary-converted problem
        """
        converter = IntegerToBinary()
        return converter.convert(problem)
    
    def solve_with_multiple_methods(self, problem: QuadraticProgram) -> Dict[str, Dict]:
        """
        Solve problem using multiple quantum methods
        
        Args:
            problem: Problem to solve
            
        Returns:
            Dictionary of results for each method
        """
        results = {}
        
        # Define quantum methods
        quantum_methods = [
            ('QAOA', QiskitQAOA(sampler=self.sampler, optimizer=COBYLA(maxiter=100), reps=2)),
            ('VQE', VQE(sampler=self.sampler, optimizer=SPSA(maxiter=100)))
        ]
        
        for method_name, algorithm in quantum_methods:
            try:
                min_eigen_optimizer = MinimumEigenOptimizer(algorithm)
                result = min_eigen_optimizer.solve(problem)
                
                results[method_name] = {
                    'optimal_value': result.fval,
                    'optimal_point': result.x,
                    'status': 'success'
                }
            except Exception as e:
                results[method_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def create_docplex_problem(self) -> QuadraticProgram:
        """
        Create a problem using DOcplex modeling
        
        Returns:
            QuadraticProgram: DOcplex-converted problem
        """
        from docplex.mp.model import Model
        
        mdl = Model('Supply Chain Optimization')
        x = {i: mdl.continuous_var(name=f'x_{i}') for i in range(self.problem_size)}
        
        mdl.minimize(mdl.sum(x[i]*i for i in range(self.problem_size)))
        mdl.add_constraint(mdl.sum(x[i] for i in range(self.problem_size)) <= 100)
        
        return from_docplex_mp(mdl)


# Example usage
if __name__ == "__main__":
    # Create different types of graphs for testing
    print("Creating test graphs...")
    G_regular = nx.random_regular_graph(3, 8)  # 3-regular graph with 8 nodes
    G_erdos = nx.erdos_renyi_graph(8, 0.5)    # Random graph with 8 nodes and 0.5 edge probability
    G_weighted = nx.Graph()
    G_weighted.add_weighted_edges_from([(0,1,2.0), (1,2,1.5), (2,3,1.0), (3,0,0.5)])
    
    # Example 1: Solve QUBO problem using circuit implementation
    print("\nExample 1: Solving QUBO problem")
    qaoa_circuit = QAOACircuit(p=2, shots=2048)
    Q = nx.adjacency_matrix(G_regular).todense()
    solution, energy = qaoa_circuit.solve_qubo(Q)
    print(f"Circuit solution: {solution}")
    print(f"Energy: {energy}")
    
    # Example 2: Solve MaxCut on different graph types
    print("\nExample 2: Solving MaxCut on different graphs")
    for graph, name in [(G_regular, "Regular"), (G_erdos, "Erdos-Renyi"), (G_weighted, "Weighted")]:
        print(f"\nSolving {name} graph:")
        qaoa_maxcut = QAOAMaxCut(graph, p=2, shots=2048)
        params, value = qaoa_maxcut.optimize(method='heuristic', maxiter=200)
        bitstring, cut_value = qaoa_maxcut.get_solution()
        print(f"Optimal parameters: {params}")
        print(f"Solution bitstring: {bitstring}")
        print(f"Cut value: {cut_value}")
        qaoa_maxcut.visualize_solution()
    
    # Example 3: Compare optimization methods
    print("\nExample 3: Comparing optimization methods")
    analyzer = QAOAAnalyzer(G_regular)
    results, exact_value = analyzer.compare_optimization_methods(
        p_max=3,
        methods=['heuristic', 'fourier']
    )
    print(f"Exact MaxCut value: {exact_value}")
    analyzer.plot_results(results, exact_value)
    
    # Example 4: Solve Ising model
    print("\nExample 4: Solving Ising model")
    h = np.array([1.0, -1.0, 0.5, -0.5])  # Local fields
    J = {(0,1): 1.0, (1,2): -0.5, (2,3): 0.5, (3,0): -1.0}  # Coupling terms
    spin_sol, ising_energy = qaoa_circuit.solve_ising(h, J, 2)
    print(f"Spin solution: {spin_sol}")
    print(f"Ising energy: {ising_energy}")
    
    # Example 5: Quadratic program conversion and solving
    print("\nExample 5: Quadratic program conversion and solving")
    converter = QuadraticProgramConverter(problem_size=4)
    problem = converter.create_sample_problem()
    qubo_problem = converter.convert_to_qubo(problem)
    results = converter.solve_with_multiple_methods(qubo_problem)
    print("Solution results:")
    for method, result in results.items():
        print(f"{method}: {result}") 
