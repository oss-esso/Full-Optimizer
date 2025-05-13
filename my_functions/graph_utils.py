"""
Graph Utilities Module

This module provides graph partitioning and manipulation utilities used across the library.
It includes various partitioning strategies and graph analysis tools.
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import community as community_louvain
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

class GraphPartitioner:
    """
    Enhanced graph partitioner with multiple partitioning strategies.
    """
    
    def __init__(self, graph: nx.Graph):
        """
        Initialize the graph partitioner.
        
        Args:
            graph: NetworkX graph to partition
        """
        self.graph = graph
        self.num_nodes = len(graph.nodes())
        self.edges = list(graph.edges())
        
    def random_partition(self, num_partitions: int) -> List[List[int]]:
        """
        Randomly partition the graph into k parts.
        
        Args:
            num_partitions: Number of partitions to create
            
        Returns:
            List of node lists, each representing a partition
        """
        nodes = list(self.graph.nodes())
        np.random.shuffle(nodes)
        return np.array_split(nodes, num_partitions)
    
    def louvain_partition(self) -> Dict[int, int]:
        """
        Partition graph using the Louvain community detection algorithm.
        
        Returns:
            Dictionary mapping nodes to their partition indices
        """
        return community_louvain.best_partition(self.graph)
    
    def spectral_partition(self, num_partitions: int) -> List[List[int]]:
        """
        Partition graph using spectral clustering.
        
        Args:
            num_partitions: Number of partitions to create
            
        Returns:
            List of node lists, each representing a partition
        """
        L = laplacian(nx.adjacency_matrix(self.graph))
        _, eigenvectors = eigsh(L, k=num_partitions, which='SM')
        kmeans = KMeans(n_clusters=num_partitions)
        labels = kmeans.fit_predict(eigenvectors)
        
        partitions = [[] for _ in range(num_partitions)]
        for node, label in enumerate(labels):
            partitions[label].append(node)
            
        return partitions
    
    def get_subgraph(self, nodes: List[int]) -> nx.Graph:
        """
        Extract subgraph induced by given nodes.
        
        Args:
            nodes: List of nodes to include in subgraph
            
        Returns:
            Induced subgraph
        """
        return self.graph.subgraph(nodes)
    
    def merge_solutions(self, subgraphs: List[nx.Graph], 
                       solutions: List[Dict[int, int]]) -> Dict[int, int]:
        """
        Merge solutions from subgraphs into a complete solution.
        
        Args:
            subgraphs: List of subgraphs
            solutions: List of solutions for each subgraph
            
        Returns:
            Merged solution for the complete graph
        """
        merged_solution = {}
        for subgraph, solution in zip(subgraphs, solutions):
            for node in subgraph.nodes():
                merged_solution[node] = solution[node]
        return merged_solution 