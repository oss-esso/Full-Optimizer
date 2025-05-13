import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import community  # python-louvain
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.problems import QuadraticProgram
from dataclasses import dataclass

@dataclass
class TreeDecomposition:
    """Class representing a tree decomposition."""
    bags: Dict[int, Set[int]]
    tree: nx.Graph
    width: int

class TreeDecomposer:
    """Tree decomposition implementation with quantum computing capabilities."""
    
    def __init__(self, graph: nx.Graph):
        """Initialize with a graph representation of the problem.
        
        Args:
            graph: NetworkX graph where nodes represent variables and 
                  edges represent interactions/constraints
        """
        self.graph = graph
        self.bags = {}  # Will store the decomposition bags
        self.tree = nx.Graph()  # The tree connecting the bags
        self.communities = None
        self.boundary_nodes = None
        self.reduced_graph = None
        self.n = len(graph)
        
    def decompose(self, method: str = 'min_fill') -> TreeDecomposition:
        """
        Perform tree decomposition using specified method.
        
        Args:
            method: Decomposition method ('min_fill' or 'min_degree')
            
        Returns:
            TreeDecomposition object containing bags and tree
        """
        try:
            # Create copy of graph for decomposition
            G = self.graph.copy()
            
            # Initialize bags and tree
            bags = {}
            tree = nx.Graph()
            elimination_order = []
            
            # Get elimination order based on method
            if method == 'min_fill':
                elimination_order = self._min_fill_ordering(G)
            elif method == 'min_degree':
                elimination_order = self._min_degree_ordering(G)
            else:
                raise ValueError(f"Unknown decomposition method: {method}")
            
            # Process elimination order
            for i, node in enumerate(elimination_order):
                # Get neighbors
                neighbors = set(G.neighbors(node))
                
                # Create bag
                bag = neighbors.union({node})
                bags[i] = bag
                
                # Add edges to tree
                for j in range(i):
                    if bags[j].intersection(bag):
                        tree.add_edge(j, i)
                
                # Add fill edges and remove node
                for u in neighbors:
                    for v in neighbors:
                        if u != v and not G.has_edge(u, v):
                            G.add_edge(u, v)
                G.remove_node(node)
            
            # Calculate treewidth
            width = max(len(bag) - 1 for bag in bags.values())
            
            return TreeDecomposition(bags=bags, tree=tree, width=width)
            
        except Exception as e:
            raise RuntimeError(f"Tree decomposition failed: {e}")
    
    def find_communities(self) -> List[List[int]]:
        """Find communities using Louvain method."""
        partition = community.best_partition(self.graph)
        communities = {}
        for node, comm_id in partition.items():
            communities[comm_id] = communities.get(comm_id, []) + [node]
        self.communities = list(communities.values())
        return self.communities
    
    def identify_boundary_nodes(self) -> List[int]:
        """Identify boundary nodes between communities."""
        if self.communities is None:
            self.find_communities()
            
        boundary = set()
        for comm in self.communities:
            for node in comm:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in comm:
                        boundary.add(node)
                        break
        self.boundary_nodes = list(boundary)
        return self.boundary_nodes
    
    def reduce_problem(self) -> nx.Graph:
        """Create reduced PUBO problem (simplified)."""
        if self.boundary_nodes is None:
            self.identify_boundary_nodes()
            
        # Create mapping of boundary nodes to consecutive indices
        boundary_map = {node: i for i, node in enumerate(self.boundary_nodes)}
        
        reduced_edges = []
        for u, v in self.graph.edges():
            if u in self.boundary_nodes and v in self.boundary_nodes:
                reduced_edges.append((boundary_map[u], boundary_map[v]))
        
        self.reduced_graph = nx.Graph(reduced_edges)
        return self.reduced_graph
    
    def qaoa_circuit(self, graph: nx.Graph, params: np.ndarray, p: int) -> QuantumCircuit:
        """Create QAOA circuit for MaxCut problem."""
        n = len(graph.nodes)
        qc = QuantumCircuit(n)
        
        # Initial state |+>^n
        qc.h(range(n))
        
        gamma = params[:p]
        beta = params[p:]
        
        for layer in range(p):
            # Cost Hamiltonian for MaxCut
            for u, v in graph.edges():
                qc.cx(u, v)
                qc.rz(gamma[layer], v)
                qc.cx(u, v)
            
            # Mixer Hamiltonian
            for qubit in range(n):
                qc.rx(2 * beta[layer], qubit)
        
        return qc
    
    def run_qaoa(self, graph: nx.Graph, p: int, params: np.ndarray) -> float:
        """Run QAOA for MaxCut and return expectation value."""
        # Create quadratic program for MaxCut
        qp = QuadraticProgram()
        for i in range(len(graph.nodes)):
            qp.binary_var(name=f'x{i}')
            
        # Set objective for MaxCut
        for u, v in graph.edges():
            qp.minimize(quadratic={('x'+str(u), 'x'+str(v)): 1})
            
        # Convert to QUBO
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)
        
        # Create QAOA instance
        qaoa = QAOA(
            optimizer=COBYLA(maxiter=1000),
            reps=p,
            quantum_instance=AerSimulator()
        )
        
        # Solve using QAOA
        optimizer = MinimumEigenOptimizer(qaoa)
        result = optimizer.solve(qubo)
        
        # Compute expectation value
        energy = result.fval
        
        return energy
    
    def visualize(self, show_communities: bool = True) -> None:
        """Visualize the graph and its decomposition."""
        plt.figure(figsize=(12, 5))
        
        if show_communities and self.communities is not None:
            plt.subplot(1, 2, 1)
            pos = nx.spring_layout(self.graph, seed=42)
            colors = []
            for node in self.graph.nodes():
                for i, comm in enumerate(self.communities):
                    if node in comm:
                        colors.append(i)
                        break
            nx.draw(self.graph, pos, with_labels=True, node_color=colors, cmap=plt.cm.rainbow)
            plt.title("Original Graph with Communities")
            
            if self.reduced_graph is not None:
                plt.subplot(1, 2, 2)
                nx.draw(self.reduced_graph, with_labels=True, node_color='lightblue')
                plt.title("Reduced Graph (Boundary Nodes)")
        else:
            nx.draw(self.graph, with_labels=True)
            plt.title("Graph Structure")
        
        plt.tight_layout()
        plt.show()
    
    def _min_fill_ordering(self, G: nx.Graph) -> List[int]:
        """Get elimination order using minimum fill heuristic."""
        order = []
        H = G.copy()
        
        while H.number_of_nodes() > 0:
            # Find node with minimum fill
            min_fill = float('inf')
            best_node = None
            
            for node in H.nodes():
                neighbors = list(H.neighbors(node))
                fill = 0
                
                # Count fill edges needed
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        if not H.has_edge(neighbors[i], neighbors[j]):
                            fill += 1
                
                if fill < min_fill:
                    min_fill = fill
                    best_node = node
            
            if best_node is None:
                break
                
            # Add fill edges and remove node
            neighbors = list(H.neighbors(best_node))
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if not H.has_edge(neighbors[i], neighbors[j]):
                        H.add_edge(neighbors[i], neighbors[j])
            
            H.remove_node(best_node)
            order.append(best_node)
        
        return order
    
    def _min_degree_ordering(self, G: nx.Graph) -> List[int]:
        """Get elimination order using minimum degree heuristic."""
        order = []
        H = G.copy()
        
        while H.number_of_nodes() > 0:
            # Find node with minimum degree
            min_degree = float('inf')
            best_node = None
            
            for node in H.nodes():
                degree = H.degree(node)
                if degree < min_degree:
                    min_degree = degree
                    best_node = node
            
            if best_node is None:
                break
                
            # Add fill edges and remove node
            neighbors = list(H.neighbors(best_node))
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if not H.has_edge(neighbors[i], neighbors[j]):
                        H.add_edge(neighbors[i], neighbors[j])
            
            H.remove_node(best_node)
            order.append(best_node)
        
        return order
    
    def _maximum_cardinality_search(self) -> List[int]:
        """Generate elimination order using maximum cardinality search."""
        G = self.graph.copy()
        ordering = []
        
        while G.number_of_nodes() > 0:
            # Find node with maximum degree
            max_degree = -1
            max_node = None
            
            for node in G.nodes():
                degree = G.degree(node)
                if degree > max_degree:
                    max_degree = degree
                    max_node = node
            
            ordering.append(max_node)
            
            # Remove the node
            G.remove_node(max_node)
            
        return ordering
    
    def _count_fill_edges(self, G: nx.Graph, node: int) -> int:
        """Count how many edges would be added if node is eliminated."""
        neighbors = list(G.neighbors(node))
        fill_edges = 0
        
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                if neighbors[j] not in G[neighbors[i]]:
                    fill_edges += 1
                    
        return fill_edges
    
    def _build_tree_decomposition(self, elimination_order: List[int]) -> nx.Graph:
        """Build tree decomposition from elimination ordering."""
        G = self.graph.copy()
        bags = {}
        tree = nx.Graph()
        
        for i, node in enumerate(elimination_order):
            # Create a bag containing the node and its neighbors
            bag = set([node] + list(G.neighbors(node)))
            bags[i] = bag
            
            # Connect all neighbors
            neighbors = list(G.neighbors(node))
            for u in range(len(neighbors)):
                for v in range(u+1, len(neighbors)):
                    if neighbors[v] not in G[neighbors[u]]:
                        G.add_edge(neighbors[u], neighbors[v])
            
            G.remove_node(node)
        
        # Connect bags to form a tree
        for i in range(len(bags)):
            for j in range(i+1, len(bags)):
                if bags[i].intersection(bags[j]):
                    tree.add_edge(i, j)
        
        self.bags = bags
        self.tree = tree
        return tree
    
    def get_treewidth(self) -> int:
        """Calculate the treewidth of the decomposition."""
        if not self.bags:
            raise ValueError("Tree decomposition not performed yet")
        return max(len(bag) - 1 for bag in self.bags.values())
    
    def get_subproblems(self) -> Dict[int, Set[int]]:
        """Return the subproblems as sets of variables."""
        return self.bags
