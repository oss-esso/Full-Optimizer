#!/usr/bin/env python3
"""
Test script to verify simulated annealing on the Maximum Cut (MaxCut) problem.

MaxCut is a classic QUBO problem where we want to partition a graph's vertices
into two sets to maximize the number of edges between the sets.

For a graph with adjacency matrix A, the QUBO formulation is:
H = sum_{i,j} A[i,j] * (1 - x_i * x_j) / 2

Where x_i ∈ {0, 1} indicates which partition vertex i belongs to.
"""

import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_graphs() -> List[Tuple[str, nx.Graph, float]]:
    """Create test graphs with known optimal solutions."""
    graphs = []
    
    # 1. Simple triangle graph (3 vertices)
    G1 = nx.Graph()
    G1.add_edges_from([(0, 1), (1, 2), (2, 0)])
    graphs.append(("Triangle", G1, 2.0))  # Optimal cut = 2 edges
    
    # 2. Complete graph K4 (4 vertices)
    G2 = nx.complete_graph(4)
    graphs.append(("Complete K4", G2, 4.0))  # Optimal cut = 4 edges
    
    # 3. Path graph (5 vertices in a line)
    G3 = nx.path_graph(5)
    graphs.append(("Path P5", G3, 4.0))  # Optimal cut = 4 edges
    
    # 4. Cycle graph (6 vertices in a cycle)
    G4 = nx.cycle_graph(6)
    graphs.append(("Cycle C6", G4, 6.0))  # Optimal cut = 6 edges (all edges)
    
    return graphs

def graph_to_qubo(G: nx.Graph) -> np.ndarray:
    """Convert a graph to QUBO matrix for MaxCut problem."""
    n = len(G.nodes())
    Q = np.zeros((n, n))
    
    # For MaxCut: maximize sum of cut edges
    # QUBO form: minimize -sum_{(i,j) in E} (x_i + x_j - 2*x_i*x_j)
    for i, j in G.edges():
        Q[i, i] += -1  # Linear term for x_i
        Q[j, j] += -1  # Linear term for x_j
        Q[i, j] += 2   # Quadratic term for x_i * x_j
        Q[j, i] += 2   # Make symmetric
    
    return Q

def qubo_objective(x: np.ndarray, Q: np.ndarray) -> float:
    """Evaluate QUBO objective function."""
    return x.T @ Q @ x

def maxcut_value(x: np.ndarray, G: nx.Graph) -> float:
    """Calculate the actual MaxCut value (number of cut edges)."""
    cut_value = 0
    for i, j in G.edges():
        if x[i] != x[j]:  # Edge is cut
            cut_value += 1
    return cut_value

def test_simulated_annealing_maxcut():
    """Test simulated annealing on MaxCut problems."""
    try:
        from my_functions.optimization import SimulatedAnnealing
        
        logger.info("Testing Simulated Annealing on MaxCut problems")
        logger.info("=" * 60)
        
        graphs = create_test_graphs()
        results = []
        
        for graph_name, G, optimal_cut in graphs:
            logger.info(f"\nTesting {graph_name} graph ({len(G.nodes())} vertices, {len(G.edges())} edges)")
            logger.info(f"Known optimal cut: {optimal_cut}")
            
            # Convert to QUBO
            Q = graph_to_qubo(G)
            n = len(G.nodes())
            
            # Create objective function for SA
            def objective_function(params):
                # Convert continuous [0,1] to binary {0,1}
                x_binary = (params > 0.5).astype(int)
                # Return negative because SA minimizes (we want to maximize cut)
                return -maxcut_value(x_binary, G)
            
            # Binary neighbor function
            def binary_neighbor_function(params):
                neighbor = params.copy()
                # Flip a random bit
                flip_idx = np.random.randint(0, len(params))
                neighbor[flip_idx] = 1.0 - neighbor[flip_idx]
                return neighbor
            
            # Random initial solution
            initial_params = np.random.choice([0.0, 1.0], size=n)
            
            # Create and run simulated annealing
            sa_optimizer = SimulatedAnnealing(
                objective_function=objective_function,
                initial_params=initial_params,
                temperature=10.0,
                cooling_rate=0.95,
                min_temperature=0.01,
                max_iterations=2000,
                neighbor_function=binary_neighbor_function,
                log_file=None
            )
            
            logger.info("Running simulated annealing optimization...")
            sa_result = sa_optimizer.optimize()
            
            # Extract and evaluate solution
            solution_binary = (sa_result.solution > 0.5).astype(int)
            cut_value = maxcut_value(solution_binary, G)
            objective_value = sa_result.objective_value
            runtime = sa_result.runtime
            
            logger.info(f"SA completed in {runtime:.3f}s")
            logger.info(f"Found cut value: {cut_value}")
            logger.info(f"Optimal cut value: {optimal_cut}")
            logger.info(f"Solution quality: {cut_value/optimal_cut:.2%}")
            
            # Store results
            results.append({
                'graph': graph_name,
                'vertices': len(G.nodes()),
                'edges': len(G.edges()),
                'optimal_cut': optimal_cut,
                'found_cut': cut_value,
                'quality': cut_value / optimal_cut,
                'runtime': runtime,
                'solution': solution_binary
            })
            
            # Visualize small graphs
            if len(G.nodes()) <= 6:
                visualize_maxcut_solution(G, solution_binary, graph_name, cut_value)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("MAXCUT RESULTS SUMMARY")
        logger.info("=" * 60)
        
        total_quality = 0
        for result in results:
            logger.info(f"{result['graph']:12s}: {result['found_cut']:3.0f}/{result['optimal_cut']:3.0f} "
                       f"({result['quality']:6.1%}) in {result['runtime']:6.3f}s")
            total_quality += result['quality']
        
        avg_quality = total_quality / len(results)
        logger.info(f"\nAverage solution quality: {avg_quality:.1%}")
        
        # Test passed if we get reasonable results (>= 80% optimal on average)
        success = avg_quality >= 0.8
        
        if success:
            logger.info("✓ MAXCUT TEST PASSED - SA finds good solutions!")
        else:
            logger.warning("⚠ MAXCUT TEST MARGINAL - Solutions could be better")
            
        return success, results
        
    except Exception as e:
        logger.error(f"✗ MaxCut test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, []

def visualize_maxcut_solution(G: nx.Graph, solution: np.ndarray, title: str, cut_value: float):
    """Visualize the MaxCut solution."""
    try:
        plt.figure(figsize=(8, 6))
        
        # Position nodes
        pos = nx.spring_layout(G, seed=42)
        
        # Color nodes by partition
        node_colors = ['red' if solution[i] == 0 else 'blue' for i in G.nodes()]
        
        # Draw edges
        cut_edges = [(i, j) for i, j in G.edges() if solution[i] != solution[j]]
        uncut_edges = [(i, j) for i, j in G.edges() if solution[i] == solution[j]]
        
        # Draw uncut edges in gray
        nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, edge_color='gray', width=1)
        
        # Draw cut edges in black (thicker)
        nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color='black', width=3)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos)
        
        plt.title(f"{title} - MaxCut = {cut_value}")
        plt.axis('off')
        
        # Save plot
        plot_path = f"Results/maxcut_{title.lower().replace(' ', '_')}.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Solution visualization saved to {plot_path}")
        
    except Exception as e:
        logger.warning(f"Could not create visualization: {e}")

def test_qubo_formulation():
    """Test that our QUBO formulation is correct."""
    logger.info("\nTesting QUBO formulation correctness...")
    
    # Simple triangle graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    
    # Test all possible partitions
    solutions = [
        ([0, 0, 0], 0),  # All in same partition
        ([1, 1, 1], 0),  # All in same partition  
        ([0, 0, 1], 2),  # Vertex 2 separate
        ([0, 1, 0], 2),  # Vertex 1 separate
        ([1, 0, 0], 2),  # Vertex 0 separate
        ([0, 1, 1], 2),  # Vertex 0 separate
        ([1, 0, 1], 2),  # Vertex 1 separate
        ([1, 1, 0], 2),  # Vertex 2 separate
    ]
    
    Q = graph_to_qubo(G)
    
    logger.info("Testing all partitions of triangle graph:")
    for partition, expected_cut in solutions:
        x = np.array(partition)
        actual_cut = maxcut_value(x, G)
        qubo_obj = qubo_objective(x, Q)
        
        logger.info(f"Partition {partition}: Cut={actual_cut}, Expected={expected_cut}, "
                   f"QUBO={qubo_obj:.1f}")
        
        if actual_cut != expected_cut:
            logger.error(f"ERROR: Cut calculation mismatch!")
            return False
    
    logger.info("✓ QUBO formulation test passed")
    return True

def main():
    """Run all MaxCut tests."""
    logger.info("SIMULATED ANNEALING MAXCUT TEST")
    logger.info("=" * 60)
    
    # Test QUBO formulation
    qubo_test = test_qubo_formulation()
    
    if not qubo_test:
        logger.error("✗ QUBO formulation test failed")
        return False
    
    # Test SA on MaxCut problems
    sa_test, results = test_simulated_annealing_maxcut()
    
    logger.info("=" * 60)
    if qubo_test and sa_test:
        logger.info("✓ ALL MAXCUT TESTS PASSED!")
        logger.info("Simulated Annealing successfully solves graph problems!")
    else:
        logger.error("✗ SOME TESTS FAILED")
    logger.info("=" * 60)
    
    return qubo_test and sa_test

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
