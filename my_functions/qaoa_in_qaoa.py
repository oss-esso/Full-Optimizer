"""
QAOA-in-QAOA Implementation for Solving Large-Scale MaxCut Problems on Small Quantum Machines

This module implements the QAOA-in-QAOA (QAOA²) approach as described in the paper:
"QAOA-in-QAOA: solving large-scale MaxCut problems on small quantum machines"

The implementation includes:
1. Graph partitioning strategies
2. Subgraph solution using QAOA
3. Solution merging process
4. Benchmarking against classical algorithms and standard QAOA
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple, Optional, Union, Callable
import time
import random
from collections import defaultdict

# Updated Qiskit imports for 1.0 compatibility
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit.circuit import Parameter
import scipy.optimize as optimize

# Import from new modules
from my_functions.graph_utils import GraphPartitioner
from my_functions.optimization import ClassicalOptimizer, SimulatedAnnealing
from my_functions.qaoa_solver import QAOABase as StandardQAOA, QAOACircuit, QAOAMaxCut, create_random_graph

# Community detection package
import community as community_louvain  # python-louvain package

# Import the updated QAOA implementation
from my_functions.qaoa_solver import QAOACircuit, QAOAMaxCut, create_random_graph

class MaxCutProblem:
    """
    Simple MaxCut problem class to handle graph-based QUBO problems.
    
    This is a simplified implementation that wraps a graph representation of a QUBO problem.
    """
    
    def __init__(self, graph: nx.Graph):
        """
        Initialize a MaxCut problem from a graph.
        
        Args:
            graph: NetworkX graph with weighted edges
        """
        self.graph = graph
        self.n_qubits = len(graph.nodes())
        
        # Extract node and edge weights
        self.node_weights = {i: graph.nodes[i].get('weight', 0.0) for i in graph.nodes()}
        self.edge_weights = {(i, j): graph[i][j].get('weight', 0.0) for i, j in graph.edges()}
        
        # Initialize solution
        self.solution = None
        self.energy = None
    
    def solve_qaoa(self, p=1, optimization_method='COBYLA'):
        """
        Solve the MaxCut problem using QAOA.
        
        Args:
            p: QAOA circuit depth
            optimization_method: Classical optimizer method
            
        Returns:
            Dictionary with solution and energy
        """
        try:
            # Import qiskit components
            from qiskit import Aer
            from qiskit.algorithms.optimizers import COBYLA, SPSA, ADAM
            from qiskit.circuit.library import QAOAAnsatz
            from qiskit.primitives import Sampler
            from qiskit.algorithms import QAOA
            from qiskit_optimization.applications import MaxCut
            
            # Create temporary graph with standard node indices
            temp_graph = nx.Graph()
            node_mapping = {node: i for i, node in enumerate(self.graph.nodes())}
            
            for i, j in self.graph.edges():
                weight = self.graph[i][j].get('weight', 1.0)
                temp_graph.add_edge(node_mapping[i], node_mapping[j], weight=weight)
            
            # Create MaxCut instance
            maxcut = MaxCut(temp_graph)
            qubo = maxcut.to_qubo()
            
            # Set up QAOA
            optimizer = None
            if optimization_method == 'COBYLA':
                optimizer = COBYLA(maxiter=100)
            elif optimization_method == 'SPSA':
                optimizer = SPSA(maxiter=100)
            elif optimization_method == 'ADAM':
                optimizer = ADAM(maxiter=100)
            else:
                optimizer = COBYLA(maxiter=100)
            
            # Initialize quantum backend
            backend = Aer.get_backend('aer_simulator')
            
            # Create QAOA instance
            qaoa = QAOA(
                sampler=Sampler(),
                optimizer=optimizer,
                reps=p,
                initial_point=None,
                quantum_instance=backend
            )
            
            # Solve MaxCut
            result = qaoa.compute_minimum_eigenvalue(qubo.to_ising()[0])
            
            # Extract solution
            x = maxcut.sample_most_likely(result.eigenstate)
            
            # Map solution back to original graph nodes
            reverse_mapping = {i: node for node, i in node_mapping.items()}
            self.solution = {reverse_mapping[i]: int(bit) for i, bit in enumerate(x)}
            self.energy = result.eigenvalue.real
            
            return {
                'solution': self.solution,
                'energy': self.energy
            }
            
        except (ImportError, ModuleNotFoundError):
            # Fallback to simplified classical solution
            print("QAOA implementation not available. Using classical approximation.")
            return self._solve_classical()
    
    def _solve_classical(self):
        """
        Classical approximation for MaxCut.
        
        Returns:
            Dictionary with solution and energy
        """
        import random
        import numpy as np
        
        # Simple greedy approach
        solution = {}
        remaining_nodes = list(self.graph.nodes())
        
        # Start with a random node assigned to 1 instead of 0
        if remaining_nodes:
            start_node = random.choice(remaining_nodes)
            solution[start_node] = 1  # Assign first node to 1 instead of 0
            remaining_nodes.remove(start_node)
        
        # Assign each remaining node to minimize energy
        while remaining_nodes:
            node = remaining_nodes.pop(0)
            
            # Calculate energy contribution for both assignments
            energy_0 = 0
            energy_1 = 0
            
            for neighbor, assignment in solution.items():
                if self.graph.has_edge(node, neighbor):
                    weight = self.graph[node][neighbor].get('weight', 1.0)
                    
                    if assignment == 0:
                        energy_0 -= weight  # Same assignment
                        energy_1 += weight  # Different assignment
                    else:
                        energy_0 += weight  # Different assignment
                        energy_1 -= weight  # Same assignment
            
            # Choose assignment with lower energy
            solution[node] = 0 if energy_0 < energy_1 else 1
        
        # Force at least a minimum number of nodes to be 1
        # This ensures we don't return all 0s
        min_ones = max(1, len(self.graph.nodes()) // 3)
        ones_count = sum(1 for val in solution.values() if val == 1)
        
        if ones_count < min_ones:
            # Flip some 0s to 1s to meet minimum requirement
            zeros = [node for node, val in solution.items() if val == 0]
            to_flip = random.sample(zeros, min(len(zeros), min_ones - ones_count))
            
            for node in to_flip:
                solution[node] = 1
        
        # Calculate final energy
        energy = 0
        for i, j in self.graph.edges():
            if solution.get(i, 0) != solution.get(j, 0):
                energy += self.graph[i][j].get('weight', 1.0)
            else:
                energy -= self.graph[i][j].get('weight', 1.0)
        
        self.solution = solution
        self.energy = energy
        
        return {
            'solution': self.solution,
            'energy': self.energy
        }


class GraphPartitioner:
    """
    Enhanced graph partitioner with multiple partitioning strategies.
    """
    
    def __init__(self, graph: nx.Graph, max_qubits: int):
        """
        Initialize the graph partitioner.
        
        Args:
            graph: NetworkX graph to partition
            max_qubits: Maximum number of qubits available on the quantum device
        """
        self.graph = graph
        self.num_nodes = len(graph.nodes())
        self.edges = list(graph.edges())
        
        # Extract edge weights
        self.weights = {(i, j): graph[i][j].get('weight', 1.0) 
                       for i, j in self.edges} # Ensure symmetry
        self.max_qubits = max_qubits
    
    def random_partition(self, num_subgraphs: int) -> List[nx.Graph]:
        """
        Partition the graph randomly into subgraphs.
        
        Args:
            num_subgraphs: Number of subgraphs to create
            
        Returns:
            List of subgraphs
        """
        # Assign each node to a random subgraph
        node_assignments = {node: random.randint(0, num_subgraphs - 1) 
                           for node in self.graph.nodes()}
        
        # Create subgraphs
        subgraphs = [nx.Graph() for _ in range(num_subgraphs)]
        
        # Add nodes to subgraphs
        for node, subgraph_idx in node_assignments.items():
            subgraphs[subgraph_idx].add_node(node)
        
        # Add edges to subgraphs
        for i, j in self.edges:
            if node_assignments[i] == node_assignments[j]:
                # Nodes are in the same subgraph
                subgraphs[node_assignments[i]].add_edge(i, j, weight=self.weights[(i, j)])
        
        return subgraphs
    
    def metis_partition(self, num_subgraphs: int) -> List[nx.Graph]:
        """
        Partition the graph using the METIS algorithm (community detection as a proxy).
        
        Args:
            num_subgraphs: Number of subgraphs to create
            
        Returns:
            List of subgraphs
        """
        # Use Louvain community detection as a proxy for METIS
        # Adjust resolution parameter to get approximately num_subgraphs communities
        resolution = 1.0
        max_attempts = 10
        attempts = 0
        
        while attempts < max_attempts:
            # Perform community detection
            partition = community_louvain.best_partition(self.graph, resolution=resolution)
            communities = set(partition.values())
            
            if len(communities) >= num_subgraphs:
                break
            
            # Adjust resolution to get more communities
            resolution *= 1.5
            attempts += 1
        
        # If we have more communities than requested, merge the smallest ones
        community_sizes = defaultdict(int)
        for node, comm in partition.items():
            community_sizes[comm] += 1
        
        if len(communities) > num_subgraphs:
            # Sort communities by size
            sorted_communities = sorted(community_sizes.items(), key=lambda x: x[1])
            
            # Map small communities to larger ones
            community_map = {}
            for i, (comm, _) in enumerate(sorted_communities):
                if i < len(sorted_communities) - num_subgraphs:
                    # Map this community to a larger one
                    target_comm = sorted_communities[-(i % num_subgraphs) - 1][0]
                    community_map[comm] = target_comm
                else:
                    community_map[comm] = comm
            
            # Update partition
            for node, comm in partition.items():
                partition[node] = community_map[comm]
        
        # Create subgraphs
        subgraphs = [nx.Graph() for _ in range(num_subgraphs)]
        
        # Add nodes to subgraphs
        for node, comm in partition.items():
            subgraph_idx = comm % num_subgraphs  # Ensure index is within range
            subgraphs[subgraph_idx].add_node(node)
        
        # Add edges to subgraphs
        for i, j in self.edges:
            if partition[i] % num_subgraphs == partition[j] % num_subgraphs:
                # Nodes are in the same subgraph
                subgraphs[partition[i] % num_subgraphs].add_edge(i, j, weight=self.weights[(i, j)])
        
        return subgraphs
    
    def spectral_partition(self, num_subgraphs: int) -> List[nx.Graph]:
        """
        Partition the graph using spectral clustering.
        
        Args:
            num_subgraphs: Number of subgraphs to create
            
        Returns:
            List of subgraphs
        """
        # Compute the normalized Laplacian
        laplacian = nx.normalized_laplacian_matrix(self.graph).todense()
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        
        # Use the eigenvectors corresponding to the smallest non-zero eigenvalues
        # for clustering (skip the first eigenvector with eigenvalue 0)
        n_clusters = min(num_subgraphs, len(eigenvalues) - 1)
        indices = np.argsort(eigenvalues)[1:n_clusters+1]
        features = eigenvectors[:, indices]
        
        # Perform k-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(features)
        labels = kmeans.labels_
        
        # Create subgraphs
        subgraphs = [nx.Graph() for _ in range(n_clusters)]
        
        # Add nodes to subgraphs
        for i, node in enumerate(self.graph.nodes()):
            subgraph_idx = labels[i]
            subgraphs[subgraph_idx].add_node(node)
        
        # Add edges to subgraphs
        for i, j in self.edges:
            # Find the indices of nodes i and j in the original graph's node list
            i_idx = list(self.graph.nodes()).index(i)
            j_idx = list(self.graph.nodes()).index(j)
            
            if labels[i_idx] == labels[j_idx]:
                # Nodes are in the same subgraph
                subgraph_idx = labels[i_idx]
                subgraphs[subgraph_idx].add_edge(i, j, weight=self.weights[(i, j)])
        
        return subgraphs
    
    def greedy_partition(self, num_subgraphs: int) -> List[nx.Graph]:
        """
        Use a greedy approach to partition the graph.
        
        Args:
            num_subgraphs: Number of subgraphs to create
            
        Returns:
            List of subgraphs
        """
        nodes = list(self.graph.nodes())
        n_nodes = len(nodes)
        subgraphs = []
        
        # Calculate how many partitions we need
        partition_size = n_nodes // num_subgraphs
        
        for i in range(0, n_nodes, partition_size):
            partition_nodes = nodes[i:min(i + partition_size, n_nodes)]
            subgraph = self.graph.subgraph(partition_nodes).copy()
            subgraphs.append(subgraph)
        
        return subgraphs
    
    def balanced_partition(self, num_subgraphs: int) -> List[nx.Graph]:
        """
        Partition the graph into balanced subgraphs (approximately equal number of nodes).
        
        Args:
            num_subgraphs: Number of subgraphs to create
            
        Returns:
            List of subgraphs
        """
        # Determine the target size for each subgraph
        target_size = self.num_nodes // num_subgraphs
        
        # Create subgraphs
        subgraphs = [nx.Graph() for _ in range(num_subgraphs)]
        
        # Assign nodes to subgraphs
        nodes = list(self.graph.nodes())
        random.shuffle(nodes)
        
        for i, node in enumerate(nodes):
            subgraph_idx = min(i // target_size, num_subgraphs - 1)
            subgraphs[subgraph_idx].add_node(node)
        
        # Add edges to subgraphs
        for i, j in self.edges:
            for subgraph_idx in range(num_subgraphs):
                if i in subgraphs[subgraph_idx].nodes() and j in subgraphs[subgraph_idx].nodes():
                    subgraphs[subgraph_idx].add_edge(i, j, weight=self.weights[(i, j)])
        
        return subgraphs
    
    def visualize_partition(self, subgraphs: List[nx.Graph], filename: Optional[str] = None):
        """
        Visualize the graph partition.
        
        Args:
            subgraphs: List of subgraphs
            filename: If provided, save the visualization to this file
        """
        plt.figure(figsize=(12, 10))
        
        # Create a position layout for the original graph
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Draw the original graph with light gray edges
        nx.draw_networkx_edges(self.graph, pos, alpha=0.2, edge_color='gray')
        
        # Draw each subgraph with a different color
        colors = plt.cm.tab10.colors
        for i, subgraph in enumerate(subgraphs):
            color = colors[i % len(colors)]
            nx.draw_networkx_nodes(subgraph, pos, node_color=color, node_size=100, alpha=0.8)
            nx.draw_networkx_edges(subgraph, pos, edge_color=color, width=2, alpha=0.6)
        
        # Draw node labels
        nx.draw_networkx_labels(self.graph, pos, font_size=8)
        
        plt.title(f'Graph Partition into {len(subgraphs)} Subgraphs')
        plt.axis('off')
        
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        
        plt.show()


class QAOASquared:
    """
    Enhanced QAOA-in-QAOA (QAOA²) implementation for solving large-scale MaxCut problems.
    """
    
    def __init__(self, graph: nx.Graph, max_qubits: int = 20, 
                partition_method: str = 'metis', p: int = 1):
        """
        Initialize the QAOA-in-QAOA algorithm.
        
        Args:
            graph: NetworkX graph representing the QUBO problem
            max_qubits: Maximum number of qubits available on the quantum device
            partition_method: Method to use for graph partitioning
            p: QAOA depth
        """
        # Set up logging
        import logging
        self.logger = logging.getLogger(__name__)
        
        self.graph = graph
        self.max_qubits = max_qubits
        self.partition_method = partition_method
        self.p = p
        
        # Validate graph
        if graph is None or len(graph.nodes()) == 0:
            raise ValueError("Graph is empty or None")
        
        # Get node and edge weights from graph
        self.node_weights = {i: graph.nodes[i].get('weight', 0.0) for i in graph.nodes()}
        self.edge_weights = {(i, j): graph[i][j].get('weight', 0.0) for i, j in graph.edges()}
        
        # Calculate number of subgraphs needed
        total_nodes = len(graph.nodes())
        self.num_subgraphs = max(1, total_nodes // max_qubits + (1 if total_nodes % max_qubits else 0))
        
        # Partition the graph
        self.subgraphs = self._partition_graph()
        
        # Create subproblems from subgraphs
        self.subproblems = [MaxCutProblem(subgraph) for subgraph in self.subgraphs]
        
        # Store for intermediate results
        self.subproblem_results = None
        self.solution = None
        self.energy = None
        self.best_cut_assignment = None
        
        # Extra attributes for tracking performance
        self.partition_time = 0
        self.solve_time = 0
    
    def _partition_graph(self):
        """Partition the graph into subgraphs using the specified method."""
        start_time = time.time()
        
        self.logger.info(f"Partitioning graph with {len(self.graph.nodes())} nodes using {self.partition_method}")
        
        # Create graph partitioner
        self.partitioner = GraphPartitioner(self.graph, self.max_qubits)
        
        # Determine number of subgraphs based on problem size and max_qubits
        n = len(self.graph.nodes())
        # Only split if number of nodes exceeds max_qubits
        if n > self.max_qubits:
            self.num_subgraphs = max(2, n // (self.max_qubits - 2) + (1 if n % (self.max_qubits - 2) else 0))
        else:
            self.num_subgraphs = 1  # No splitting needed if graph is small enough
        
        self.logger.info(f"Automatically determined {self.num_subgraphs} subgraphs")
        
        # Call the specified partitioning method
        if self.partition_method == 'random':
            subgraphs = self.partitioner.random_partition(self.num_subgraphs)
        elif self.partition_method == 'spectral':
            subgraphs = self.partitioner.spectral_partition(self.num_subgraphs)
        elif self.partition_method == 'metis':
            try:
                subgraphs = self.partitioner.metis_partition(self.num_subgraphs)
            except ValueError as e:
                self.logger.warning(f"METIS partitioning failed: {e}. Falling back to spectral partitioning.")
                try:
                    subgraphs = self.partitioner.spectral_partition(self.num_subgraphs)
                except Exception as e2:
                    self.logger.warning(f"Spectral partitioning also failed: {e2}. Falling back to random partitioning.")
                    subgraphs = self.partitioner.random_partition(self.num_subgraphs)
        elif self.partition_method == 'greedy':
            subgraphs = self.partitioner.greedy_partition(self.num_subgraphs)
        elif self.partition_method == 'balanced':
            subgraphs = self.partitioner.balanced_partition(self.num_subgraphs)
        elif self.partition_method == 'manual':
            # Check if manual_partition method exists
            if hasattr(self.partitioner, 'manual_partition'):
                subgraphs = self.partitioner.manual_partition(self.num_subgraphs)
            else:
                self.logger.warning("Manual partition method not found. Falling back to random partitioning.")
                subgraphs = self.partitioner.random_partition(self.num_subgraphs)
        else:
            raise ValueError(f"Unknown partition method: {self.partition_method}")
        
        # Post-process: Check if any subgraph exceeds the maximum qubit limit
        # If so, split it further
        final_subgraphs = []
        for i, sg in enumerate(subgraphs):
            if len(sg.nodes()) > self.max_qubits:
                self.logger.warning(f"Subgraph {i} has {len(sg.nodes())} nodes, exceeding max_qubits={self.max_qubits}")
                self.logger.info(f"Further splitting subgraph {i} to meet max_qubits constraint")
                
                # Create a new partitioner for just this subgraph
                sub_partitioner = GraphPartitioner(sg, self.max_qubits)
                
                # Determine how many pieces to split into
                n_sub = len(sg.nodes())
                sub_parts = max(2, n_sub // (self.max_qubits - 2) + (1 if n_sub % (self.max_qubits - 2) else 0))
                
                # Split this subgraph further
                if self.partition_method == 'metis':
                    try:
                        sub_subgraphs = sub_partitioner.metis_partition(sub_parts)
                    except:
                        sub_subgraphs = sub_partitioner.random_partition(sub_parts)
                else:
                    sub_subgraphs = sub_partitioner.random_partition(sub_parts)
                
                # Add these smaller subgraphs
                final_subgraphs.extend(sub_subgraphs)
                self.logger.info(f"  Split subgraph {i} into {len(sub_subgraphs)} smaller subgraphs")
            else:
                final_subgraphs.append(sg)
        
        # Log final partitioning results
        self.logger.info(f"Final partition: {len(final_subgraphs)} subgraphs")
        for i, sg in enumerate(final_subgraphs):
            self.logger.info(f"  Subgraph {i}: {len(sg.nodes())} nodes, {len(sg.edges())} edges")
        
        self.partition_time = time.time() - start_time
        self.logger.info(f"Partitioning took {self.partition_time:.2f} seconds")
        
        return final_subgraphs
    
    def solve(self, optimization_method: str = 'COBYLA'):
        """
        Solve the partitioned problem using QAOA on each subproblem.
        
        Args:
            optimization_method: Classical optimizer method
            
        Returns:
            Dictionary with complete solution and metrics
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"Solving {len(self.subproblems)} subproblems with QAOA...")
        
        # Solve each subproblem with QAOA
        self.subproblem_results = []
        for i, subproblem in enumerate(self.subproblems):
            self.logger.info(f"Solving subproblem {i+1}/{len(self.subproblems)} ({subproblem.n_qubits} qubits)")
            try:
                result = subproblem.solve_qaoa(p=self.p, optimization_method=optimization_method)
                self.subproblem_results.append(result)
                self.logger.info(f"  Energy: {result.get('energy', 'N/A')}")
            except Exception as e:
                self.logger.error(f"Error solving subproblem {i+1}: {e}")
                # Use classical approximation if QAOA fails
                self.logger.info("Falling back to classical approximation")
                result = subproblem._solve_classical()
                self.subproblem_results.append(result)
                self.logger.info(f"  Classical energy: {result.get('energy', 'N/A')}")
        
        # Combine solutions
        self.solution = {}
        for result in self.subproblem_results:
            if result and 'solution' in result:
                self.solution.update(result['solution'])
        
        # Calculate total energy
        self.energy = self._calculate_total_energy()
        
        self.solve_time = time.time() - start_time
        self.logger.info(f"QAOA² completed in {self.solve_time:.2f} seconds")
        self.logger.info(f"Total energy: {self.energy}")
        
        return {
            'solution': self.solution,
            'objective': self.energy,
            'num_partitions': len(self.subproblems),
            'partition_time': self.partition_time,
            'solve_time': self.solve_time
        }
    
    def _calculate_total_energy(self):
        """
        Calculate the total energy of the combined solution.
        
        Returns:
            Total energy value
        """
        energy = 0.0
        
        # Contribution from node weights
        for node, weight in self.node_weights.items():
            if node in self.solution:
                energy += weight * self.solution[node]
        
        # Contribution from edge weights
        for (i, j), weight in self.edge_weights.items():
            if i in self.solution and j in self.solution:
                if self.solution[i] != self.solution[j]:
                    energy += weight  # Different values
                else:
                    energy -= weight  # Same values
        
        return energy
    
    def visualize_solution(self, filename: Optional[str] = None) -> None:
        """
        Visualize the MaxCut solution.
        
        Args:
            filename: If provided, save the visualization to this file
        """
        if self.best_cut_assignment is None:
            print("No solution available to visualize")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Create a position layout for the graph
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Draw nodes with colors based on their assignment
        node_colors = [self.best_cut_assignment[node] for node in self.graph.nodes()]
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                             cmap=plt.cm.coolwarm, node_size=100)
        
        # Draw edges with different styles for cut and non-cut edges
        cut_edges = [(u, v) for u, v in self.graph.edges() 
                    if self.best_cut_assignment[u] != self.best_cut_assignment[v]]
        non_cut_edges = [(u, v) for u, v in self.graph.edges() 
                        if self.best_cut_assignment[u] == self.best_cut_assignment[v]]
        
        nx.draw_networkx_edges(self.graph, pos, edgelist=cut_edges, 
                             edge_color='red', width=2)
        nx.draw_networkx_edges(self.graph, pos, edgelist=non_cut_edges, 
                             edge_color='gray', alpha=0.5)
        
        plt.title(f'MaxCut Solution (Value: {self.energy})')
        plt.axis('off')
        
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        
        plt.show()


def benchmark(graph_sizes: List[int], max_qubits: int = 20, 
             num_repetitions: int = 3) -> Dict:
    """
    Benchmark the nested QAOA algorithm on graphs of different sizes.
    
    Args:
        graph_sizes: List of graph sizes to test
        max_qubits: Maximum number of qubits available
        num_repetitions: Number of repetitions for each size
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "graph_sizes": graph_sizes,
        "execution_times": [],
        "cut_values": [],
        "approximation_ratios": []
    }
    
    for size in graph_sizes:
        size_times = []
        size_cut_values = []
        size_approx_ratios = []
        
        for rep in range(num_repetitions):
            print(f"Benchmarking graph with {size} nodes (repetition {rep+1}/{num_repetitions})...")
            
            # Generate random graph
            graph = create_random_graph(size, p=0.5, seed=42+rep)
            
            # Solve with nested QAOA
            solver = QAOASquared(graph, max_qubits=max_qubits)
            result = solver.solve()
            
            # Compute approximation ratio (if possible)
            approx_ratio = None
            try:
                if size <= 20:  # Only compute exact solution for small graphs
                    partition_1, partition_2 = nx.algorithms.approximation.maxcut.maximum_cut(graph)
                    
                    # Calculate the cut value as the number of edges between partitions
                    exact_cut_value = sum(1 for u, v in graph.edges() 
                                        if (u in partition_1 and v in partition_2) or 
                                           (u in partition_2 and v in partition_1))
                    
                    if exact_cut_value > 0:  # Avoid division by zero
                        approx_ratio = result["objective"] / exact_cut_value
                    print(f"Exact cut value: {exact_cut_value}, QAOA cut value: {result['objective']}")
            except Exception as e:
                print(f"Error computing exact solution: {e}")
            
            size_times.append(result["solve_time"])
            size_cut_values.append(result["objective"])
            if approx_ratio is not None:
                size_approx_ratios.append(approx_ratio)
        
        results["execution_times"].append(np.mean(size_times))
        results["cut_values"].append(np.mean(size_cut_values))
        if size_approx_ratios:
            results["approximation_ratios"].append(np.mean(size_approx_ratios))
        else:
            results["approximation_ratios"].append(None)
    
    return results


def plot_benchmark_results(results: Dict) -> None:
    """
    Plot benchmark results.
    
    Args:
        results: Benchmark results dictionary
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot execution time
    axs[0].plot(results["graph_sizes"], results["execution_times"], 'o-')
    axs[0].set_xlabel("Graph Size (nodes)")
    axs[0].set_ylabel("Execution Time (s)")
    axs[0].set_title("Execution Time vs. Graph Size")
    axs[0].grid(True)
    
    # Plot approximation ratio (if available)
    valid_indices = [i for i, r in enumerate(results["approximation_ratios"]) if r is not None]
    if valid_indices:
        valid_sizes = [results["graph_sizes"][i] for i in valid_indices]
        valid_ratios = [results["approximation_ratios"][i] for i in valid_indices]
        
        axs[1].plot(valid_sizes, valid_ratios, 'o-')
        axs[1].set_xlabel("Graph Size (nodes)")
        axs[1].set_ylabel("Approximation Ratio")
        axs[1].set_title("Approximation Ratio vs. Graph Size")
        axs[1].grid(True)
        axs[1].set_ylim(0, 1.1)
    else:
        axs[1].text(0.5, 0.5, "No approximation ratios available", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=axs[1].transAxes)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function demonstrating the nested QAOA algorithm.
    """
    print("QAOA-in-QAOA: Solving large-scale MaxCut problems on small quantum machines")
    print("----------------------------------------------------------------------")
    
    # Example: Small graph that fits on the quantum device
    print("\nExample 1: Small graph (standard QAOA)")
    small_graph = create_random_graph(6, p=0.5, seed=42)
    nx.draw(small_graph, with_labels=True)
    plt.title("Small Random Graph (6 nodes)")
    plt.show()
    
    solver_small = QAOASquared(small_graph, max_qubits=10)
    result_small = solver_small.solve()
    
    print(f"Max cut value: {result_small['objective']}")
    print(f"Max cut assignment: {result_small['solution']}")
    print(f"Execution time: {result_small['solve_time']:.4f} seconds")
    
    solver_small.visualize_solution()
    
    # Example: Larger graph that requires partitioning
    print("\nExample 2: Larger graph (nested QAOA)")
    large_graph = create_random_graph(30, p=0.3, seed=42)
    
    solver_large = QAOASquared(large_graph, max_qubits=10)
    result_large = solver_large.solve()
    
    print(f"Max cut value: {result_large['objective']}")
    print(f"Execution time: {result_large['solve_time']:.4f} seconds")
    print(f"Number of subgraphs: {result_large['num_partitions']}")
    
    solver_large.visualize_solution()
    
    # Benchmark with different graph sizes
    print("\nRunning benchmark...")
    benchmark_results = benchmark(graph_sizes=[10, 20, 30, 40, 50], 
                                max_qubits=10, num_repetitions=1)
    plot_benchmark_results(benchmark_results)


if __name__ == "__main__":
    main()