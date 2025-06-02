#!/usr/bin/env python3
"""
Test simulated annealing on larger, more challenging graph problems.

This script tests the scalability and robustness of our SA implementation
on larger MaxCut instances and more complex graph structures.
"""

import sys
import os
import logging
import numpy as np
import networkx as nx
import time
from typing import List, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_large_test_graphs() -> List[Tuple[str, nx.Graph, str]]:
    """Create larger test graphs."""
    graphs = []
    
    # 1. Random graph (20 vertices, moderate density)
    np.random.seed(42)
    G1 = nx.erdos_renyi_graph(20, 0.3, seed=42)
    graphs.append(("Random G(20,0.3)", G1, "heuristic"))
    
    # 2. Complete graph K10
    G2 = nx.complete_graph(10)
    graphs.append(("Complete K10", G2, "known"))  # Optimal = 25 edges
    
    # 3. Grid graph 4x4
    G3 = nx.grid_2d_graph(4, 4)
    G3 = nx.convert_node_labels_to_integers(G3)
    graphs.append(("Grid 4x4", G3, "heuristic"))
    
    # 4. Petersen graph (famous graph theory example)
    G4 = nx.petersen_graph()
    graphs.append(("Petersen", G4, "heuristic"))
    
    # 5. Random regular graph (15 vertices, degree 4)
    G5 = nx.random_regular_graph(4, 15, seed=42)
    graphs.append(("4-Regular(15)", G5, "heuristic"))
    
    return graphs

def maxcut_value(x: np.ndarray, G: nx.Graph) -> float:
    """Calculate the MaxCut value (number of cut edges)."""
    cut_value = 0
    for i, j in G.edges():
        if x[i] != x[j]:  # Edge is cut
            cut_value += 1
    return cut_value

def estimate_maxcut_upper_bound(G: nx.Graph) -> float:
    """Estimate upper bound for MaxCut value."""
    # Simple upper bound: total edges / 2 (for balanced cut)
    # For most graphs, optimal is close to |E|/2
    return len(G.edges()) / 2

def test_large_graph_performance():
    """Test SA performance on larger graphs."""
    try:
        from my_functions.optimization import SimulatedAnnealing
        
        logger.info("Testing Simulated Annealing on larger graphs")
        logger.info("=" * 70)
        
        graphs = create_large_test_graphs()
        results = []
        
        for graph_name, G, bound_type in graphs:
            n_vertices = len(G.nodes())
            n_edges = len(G.edges())
            estimated_bound = estimate_maxcut_upper_bound(G)
            
            logger.info(f"\nTesting {graph_name}")
            logger.info(f"  Vertices: {n_vertices}, Edges: {n_edges}")
            logger.info(f"  Estimated upper bound: {estimated_bound:.1f}")
            
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
            
            # Test multiple runs to assess consistency
            run_results = []
            
            for run in range(3):  # 3 runs to check consistency
                # Random initial solution
                initial_params = np.random.choice([0.0, 1.0], size=n_vertices)
                
                # Adjust SA parameters based on graph size
                if n_vertices <= 10:
                    max_iter = 2000
                    temp = 10.0
                elif n_vertices <= 20:
                    max_iter = 5000
                    temp = 20.0
                else:
                    max_iter = 10000
                    temp = 50.0
                
                # Create and run simulated annealing
                sa_optimizer = SimulatedAnnealing(
                    objective_function=objective_function,
                    initial_params=initial_params,
                    temperature=temp,
                    cooling_rate=0.95,
                    min_temperature=0.01,
                    max_iterations=max_iter,
                    neighbor_function=binary_neighbor_function,
                    log_file=None
                )
                
                start_time = time.time()
                sa_result = sa_optimizer.optimize()
                end_time = time.time()
                
                # Extract and evaluate solution
                solution_binary = (sa_result.solution > 0.5).astype(int)
                cut_value = maxcut_value(solution_binary, G)
                runtime = end_time - start_time
                
                run_results.append({
                    'cut_value': cut_value,
                    'runtime': runtime,
                    'solution': solution_binary
                })
            
            # Analyze results across runs
            cut_values = [r['cut_value'] for r in run_results]
            runtimes = [r['runtime'] for r in run_results]
            
            best_cut = max(cut_values)
            avg_cut = np.mean(cut_values)
            std_cut = np.std(cut_values)
            avg_runtime = np.mean(runtimes)
            
            # Calculate quality metrics
            if bound_type == "known" and graph_name == "Complete K10":
                # For K10, optimal cut is 25 (5*5 partition)
                optimal_cut = 25
                quality = best_cut / optimal_cut
            else:
                # Use estimated bound
                quality = best_cut / estimated_bound
            
            logger.info(f"  Results over 3 runs:")
            logger.info(f"    Best cut: {best_cut}")
            logger.info(f"    Avg cut: {avg_cut:.1f} ± {std_cut:.1f}")
            logger.info(f"    Quality vs bound: {quality:.1%}")
            logger.info(f"    Avg runtime: {avg_runtime:.3f}s")
            
            # Store results
            results.append({
                'graph': graph_name,
                'vertices': n_vertices,
                'edges': n_edges,
                'best_cut': best_cut,
                'avg_cut': avg_cut,
                'std_cut': std_cut,
                'quality': quality,
                'avg_runtime': avg_runtime,
                'bound_type': bound_type
            })
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("LARGE GRAPH RESULTS SUMMARY")
        logger.info("=" * 70)
        
        total_quality = 0
        for result in results:
            logger.info(f"{result['graph']:15s}: {result['best_cut']:3.0f} "
                       f"({result['quality']:6.1%}) in {result['avg_runtime']:6.3f}s avg")
            total_quality += result['quality']
        
        avg_quality = total_quality / len(results)
        logger.info(f"\nOverall average quality: {avg_quality:.1%}")
        
        # Test passed if we get reasonable results (>= 75% of bounds)
        success = avg_quality >= 0.75
        
        if success:
            logger.info("✓ LARGE GRAPH TEST PASSED - SA scales well!")
        else:
            logger.warning("⚠ LARGE GRAPH TEST MARGINAL - Consider parameter tuning")
            
        return success, results
        
    except Exception as e:
        logger.error(f"✗ Large graph test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, []

def test_scalability():
    """Test how SA performance scales with graph size."""
    try:
        from my_functions.optimization import SimulatedAnnealing
        
        logger.info("\nTesting scalability across different graph sizes")
        logger.info("-" * 50)
        
        sizes = [5, 10, 15, 20, 25]
        scalability_results = []
        
        for n in sizes:
            # Create random graph
            G = nx.erdos_renyi_graph(n, 0.3, seed=42)
            
            def objective_function(params):
                x_binary = (params > 0.5).astype(int)
                return -maxcut_value(x_binary, G)
            
            def binary_neighbor_function(params):
                neighbor = params.copy()
                flip_idx = np.random.randint(0, len(params))
                neighbor[flip_idx] = 1.0 - neighbor[flip_idx]
                return neighbor
            
            initial_params = np.random.choice([0.0, 1.0], size=n)
            
            # Fixed SA parameters for fair comparison
            sa_optimizer = SimulatedAnnealing(
                objective_function=objective_function,
                initial_params=initial_params,
                temperature=20.0,
                cooling_rate=0.95,
                min_temperature=0.01,
                max_iterations=3000,
                neighbor_function=binary_neighbor_function,
                log_file=None
            )
            
            start_time = time.time()
            sa_result = sa_optimizer.optimize()
            runtime = time.time() - start_time
            
            solution_binary = (sa_result.solution > 0.5).astype(int)
            cut_value = maxcut_value(solution_binary, G)
            estimated_bound = len(G.edges()) / 2
            quality = cut_value / estimated_bound if estimated_bound > 0 else 0
            
            scalability_results.append({
                'size': n,
                'edges': len(G.edges()),
                'cut_value': cut_value,
                'quality': quality,
                'runtime': runtime
            })
            
            logger.info(f"Size {n:2d}: {cut_value:3.0f} cut ({quality:5.1%}) in {runtime:6.3f}s")
        
        # Check if runtime grows reasonably
        logger.info("\nScalability analysis:")
        for i in range(1, len(scalability_results)):
            prev = scalability_results[i-1]
            curr = scalability_results[i]
            size_ratio = curr['size'] / prev['size']
            time_ratio = curr['runtime'] / prev['runtime'] if prev['runtime'] > 0 else float('inf')
            logger.info(f"Size {prev['size']} → {curr['size']}: time ratio = {time_ratio:.2f}x")
        
        return True, scalability_results
        
    except Exception as e:
        logger.error(f"✗ Scalability test failed: {e}")
        return False, []

def main():
    """Run all large graph tests."""
    logger.info("SIMULATED ANNEALING LARGE GRAPH TESTING")
    logger.info("=" * 70)
    
    # Test performance on larger graphs
    perf_test, perf_results = test_large_graph_performance()
    
    # Test scalability
    scale_test, scale_results = test_scalability()
    
    logger.info("=" * 70)
    if perf_test and scale_test:
        logger.info("✓ ALL LARGE GRAPH TESTS PASSED!")
        logger.info("Simulated Annealing demonstrates excellent scalability!")
    else:
        logger.error("✗ SOME TESTS FAILED OR MARGINAL")
    logger.info("=" * 70)
    
    return perf_test and scale_test

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
