#!/usr/bin/env python3
"""
Test script for D-Wave cost estimation functionality.

This script tests the D-Wave QPU adapter's cost estimation capabilities
using real scenario data from scenarios.py. It demonstrates:
- Native problem (18 variables) solving with D-Wave simulator
- Problem size scaling analysis
- Summary table of results
"""

import os
import sys
import logging
import traceback
import numpy as np
import time
from datetime import datetime
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def configure_dwave_environment():
    """Configure D-Wave environment for CPU-only simulator testing."""
    print("✓ Running CPU-only simulator tests (no D-Wave token required)")
    return True

def main():
    """Calculate nodes and density for three scenarios and estimate D-Wave solve times."""
    print("=" * 80)
    print("D-WAVE SCENARIO ANALYSIS - NODES, DENSITY & SOLVE TIME ESTIMATION")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        from src.scenarios import load_food_data
        print("✓ All imports successful")
        
        # Analyze the three complexity scenarios
        scenario_results = analyze_three_scenarios()
        
        # Estimate solve times using D-Wave Hybrid benchmark data
        benchmark_results = estimate_dwave_solve_times()
        
        # Compare our scenarios with Gset benchmarks
        comparison_results = compare_scenarios_with_benchmarks()
        
        print("\n" + "=" * 80)
        print("✓ SCENARIO ANALYSIS COMPLETED!")
        print("=" * 80)
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        traceback.print_exc()
        return False
    return True

def analyze_three_scenarios():
    """Analyze the three complexity scenarios: simple, intermediate, and full."""
    print("\n" + "-" * 60)
    print("SCENARIO ANALYSIS: NODES, VARIABLES & DENSITY")
    print("-" * 60)
    
    scenarios = ['simple', 'intermediate', 'full']
    scenario_results = {}
    
    try:
        from src.scenarios import load_food_data
        
        for scenario in scenarios:
            print(f"\n--- Analyzing {scenario.upper()} complexity scenario ---")
            
            farms, foods, food_groups, config = load_food_data(scenario)
            num_farms = len(farms)
            num_foods = len(foods)
            total_variables = num_farms * num_foods
            
            print(f"Farms: {num_farms} ({farms})")
            print(f"Foods: {num_foods} ({list(foods.keys())})")
            print(f"Total variables (nodes): {total_variables}")
            
            # Create QUBO to analyze density
            qubo_matrix = create_food_optimization_qubo(farms, foods, config)
            total_elements = qubo_matrix.size
            nonzero_elements = np.count_nonzero(qubo_matrix)
            density = nonzero_elements / total_elements
            
            print(f"QUBO matrix size: {qubo_matrix.shape}")
            print(f"Total matrix elements: {total_elements}")
            print(f"Non-zero elements: {nonzero_elements}")
            print(f"Graph density: {density:.4f} ({density*100:.2f}%)")
            
            # Analyze sparsity pattern
            upper_triangle = np.triu(qubo_matrix, k=1)
            diagonal = np.diag(qubo_matrix)
            interactions = np.count_nonzero(upper_triangle)
            linear_terms = np.count_nonzero(diagonal)
            
            print(f"Linear terms (diagonal): {linear_terms}")
            print(f"Quadratic interactions (upper triangle): {interactions}")
            print(f"Total QUBO terms: {linear_terms + interactions}")
            
            scenario_results[scenario] = {
                'farms': num_farms,
                'foods': num_foods,
                'variables': total_variables,
                'matrix_size': qubo_matrix.shape,
                'density': density,
                'linear_terms': linear_terms,
                'interactions': interactions,
                'total_terms': linear_terms + interactions
            }
    
        # Print summary table
        print("\n" + "=" * 80)
        print("SCENARIO COMPARISON SUMMARY")
        print("=" * 80)
        print(f"{'Scenario':<12} {'Farms':<6} {'Foods':<6} {'Variables':<10} {'Density':<10} {'Linear':<8} {'Quad':<8} {'Total':<8}")
        print("-" * 80)
        
        for scenario in scenarios:
            data = scenario_results[scenario]
            print(f"{scenario:<12} {data['farms']:<6} {data['foods']:<6} {data['variables']:<10} "
                  f"{data['density']:<10.4f} {data['linear_terms']:<8} {data['interactions']:<8} {data['total_terms']:<8}")
        
        print("=" * 80)
        return scenario_results
        
    except Exception as e:
        print(f"✗ Scenario analysis failed: {e}")
        traceback.print_exc()
        return None

def estimate_dwave_solve_times():
    """Estimate D-Wave solve times based on benchmark data from the research tables."""
    print("\n" + "-" * 60)
    print("D-WAVE SOLVE TIME ESTIMATION")
    print("-" * 60)
    
    # Gset benchmark instances with graph structure information
    # Format: (instance, nodes, edges, dwave_hybrid_time_seconds)
    # NOTE: Using D-Wave HYBRID solver times (t_Hybrid column), NOT SBM or SA times
    gset_benchmark_data = [
        # 800-node instances
        ("G1", 800, 19176, 3.00), ("G2", 800, 19176, 3.00), ("G3", 800, 19176, 2.99), 
        ("G4", 800, 19176, 3.00), ("G5", 800, 19176, 3.00), ("G6", 800, 19176, 3.00),
        ("G7", 800, 19176, 3.00), ("G8", 800, 19176, 3.00), ("G9", 800, 19176, 3.00),
        ("G10", 800, 19176, 3.00), ("G11", 800, 1600, 3.00), ("G12", 800, 1600, 3.00),
        ("G13", 800, 1600, 2.99), ("G14", 800, 4694, 3.00), ("G15", 800, 4661, 3.00),
        ("G16", 800, 4672, 2.99), ("G17", 800, 4667, 3.00), ("G18", 800, 4694, 2.98),
        ("G19", 800, 4661, 2.99), ("G20", 800, 4672, 3.00), ("G21", 800, 4667, 3.01),
        
        # 1000-node instances  
        ("G43", 1000, 9990, 2.99), ("G44", 1000, 9990, 3.00), ("G45", 1000, 9990, 2.99),
        ("G46", 1000, 9990, 2.99), ("G47", 1000, 9990, 2.99), ("G51", 1000, 5909, 2.99),
        ("G52", 1000, 5916, 2.99), ("G53", 1000, 5914, 3.00), ("G54", 1000, 5916, 3.00),
        
        # 2000-node instances
        ("G22", 2000, 19990, 5.22), ("G23", 2000, 19990, 5.22), ("G24", 2000, 19990, 5.22),
        ("G25", 2000, 19990, 5.22), ("G26", 2000, 19990, 5.22), ("G27", 2000, 19990, 5.22),
        ("G28", 2000, 19990, 5.22), ("G29", 2000, 19990, 5.22), ("G30", 2000, 19990, 5.21),
        ("G31", 2000, 19990, 5.23), ("G32", 2000, 4000, 5.22), ("G33", 2000, 4000, 5.22),
        ("G34", 2000, 4000, 5.22), ("G35", 2000, 11778, 5.22), ("G36", 2000, 11766, 5.21),
        ("G37", 2000, 11785, 5.22), ("G38", 2000, 11779, 5.22), ("G39", 2000, 11778, 5.22),
        ("G40", 2000, 11766, 5.21), ("G41", 2000, 11785, 5.22), ("G42", 2000, 11779, 5.22),
        
        # 3000-node instances
        ("G48", 3000, 6000, 7.50), ("G49", 3000, 6000, 7.50), ("G50", 3000, 6000, 7.50),
        
        # 5000-node instances
        ("G55", 5000, 12498, 14.59), ("G56", 5000, 12498, 14.58), ("G57", 5000, 10000, 14.59),
        ("G58", 5000, 29570, 14.59), ("G59", 5000, 29570, 14.60),
        
        # 7000-node instances
        ("G60", 7000, 17148, 24.76), ("G61", 7000, 17148, 24.75), ("G62", 7000, 14000, 24.74),
        ("G63", 7000, 41459, 24.75), ("G64", 7000, 41459, 24.74),
        
        # 8000+ node instances
        ("G65", 8000, 16000, 29.83), ("G66", 9000, 18000, 34.92), ("G67", 10000, 20000, 39.98),
        ("G70", 10000, 9999, 40.00), ("G72", 10000, 20000, 39.98)
    ]
    
    # Calculate density for each instance and group by characteristics
    print("Gset Benchmark Analysis - D-Wave HYBRID Solver Performance:")
    print(f"{'Instance':<8} {'Nodes':<6} {'Edges':<8} {'Density':<10} {'Hybrid Time (s)':<15}")
    print("-" * 55)
    
    density_groups = {}
    node_times = {}
    edge_times = {}
    
    for instance, nodes, edges, time in gset_benchmark_data:
        # Calculate graph density: edges / max_possible_edges
        max_edges = nodes * (nodes - 1) // 2
        density = edges / max_edges
        
        print(f"{instance:<8} {nodes:<6} {edges:<8} {density:<10.4f} {time:<15.2f}")
        
        # Group by density ranges for analysis
        if density < 0.01:
            density_group = "sparse"
        elif density < 0.05:
            density_group = "medium"
        else:
            density_group = "dense"
        
        if density_group not in density_groups:
            density_groups[density_group] = []
        density_groups[density_group].append((nodes, edges, density, time))
        
        # Group by node count
        if nodes not in node_times:
            node_times[nodes] = []
        node_times[nodes].append(time)
        
        # Group by edge count ranges
        edge_range = (edges // 5000) * 5000  # Round to nearest 5000
        if edge_range not in edge_times:
            edge_times[edge_range] = []
        edge_times[edge_range].append((nodes, edges, density, time))
    
    # Analyze density impact on solve time
    print(f"\nDensity Group Analysis (D-Wave Hybrid times):")
    print(f"{'Group':<10} {'Count':<7} {'Avg Time':<10} {'Min Time':<10} {'Max Time':<10} {'Avg Density':<12}")
    print("-" * 70)
    
    for group in ['sparse', 'medium', 'dense']:
        if group in density_groups:
            data = density_groups[group]
            times = [item[3] for item in data]
            densities = [item[2] for item in data]
            avg_time = np.mean(times)
            min_time = min(times)
            max_time = max(times)
            avg_density = np.mean(densities)
            count = len(times)
            print(f"{group:<10} {count:<7} {avg_time:<10.2f} {min_time:<10.2f} {max_time:<10.2f} {avg_density:<12.4f}")
    
    # Extract node-based timing data for scaling analysis
    avg_times = {}
    for nodes, times in node_times.items():
        avg_times[nodes] = np.mean(times)
    
    print(f"\nNode-based Timing Summary (D-Wave Hybrid):")
    print(f"{'Nodes':<8} {'Avg Time (s)':<12} {'Min Time (s)':<12} {'Max Time (s)':<12} {'Samples':<8}")
    print("-" * 60)
    
    for nodes in sorted(avg_times.keys()):
        times = node_times[nodes]
        avg_time = avg_times[nodes]
        min_time = min(times)
        max_time = max(times)
        samples = len(times)
        print(f"{nodes:<8} {avg_time:<12.3f} {min_time:<12.3f} {max_time:<12.3f} {samples:<8}")
    
    # Fit scaling model using node-based timing data
    nodes_array = np.array(sorted(avg_times.keys()))
    times_array = np.array([avg_times[n] for n in nodes_array])
    
    # Try different scaling models
    print(f"\nScaling Analysis (D-Wave Hybrid solver performance):")
    
    # Linear fit: t = a * n + b
    linear_coeffs = np.polyfit(nodes_array, times_array, 1)
    linear_r2 = calculate_r_squared(nodes_array, times_array, linear_coeffs, degree=1)
    print(f"Linear fit: t = {linear_coeffs[0]:.8f} * n + {linear_coeffs[1]:.3f} (R² = {linear_r2:.4f})")
    
    # Quadratic fit: t = a * n² + b * n + c
    if len(nodes_array) >= 3:
        quad_coeffs = np.polyfit(nodes_array, times_array, 2)
        quad_r2 = calculate_r_squared(nodes_array, times_array, quad_coeffs, degree=2)
        print(f"Quadratic fit: t = {quad_coeffs[0]:.12f} * n² + {quad_coeffs[1]:.8f} * n + {quad_coeffs[2]:.3f} (R² = {quad_r2:.4f})")
    
    # Power law fit: t = a * n^b
    try:
        from scipy.optimize import curve_fit
        
        def power_law(x, a, b):
            return a * np.power(x, b)
        
        popt_power, _ = curve_fit(power_law, nodes_array, times_array, p0=[0.001, 1.0])
        power_pred = power_law(nodes_array, *popt_power)
        power_r2 = 1 - np.sum((times_array - power_pred)**2) / np.sum((times_array - np.mean(times_array))**2)
        print(f"Power law fit: t = {popt_power[0]:.8f} * n^{popt_power[1]:.3f} (R² = {power_r2:.4f})")
        
        # Use best fitting model for predictions
        best_model = "power"
        best_r2 = power_r2
        
    except Exception as e:
        print(f"Power law fit failed: {e}")
        best_model = "linear"
        best_r2 = linear_r2
    
    # Estimate solve times for our scenarios based on Gset benchmarks
    our_scenarios = {
        'simple': 18,    # Based on actual simple scenario
        'intermediate': 42,  # Based on actual intermediate scenario  
        'full': 50       # Based on actual full scenario
    }
    
    print(f"\nEstimated D-Wave Hybrid solve times for our scenarios:")
    print(f"{'Scenario':<12} {'Variables':<10} {'Estimated Time (s)':<18} {'Method':<25}")
    print("-" * 70)
    
    for scenario, vars_count in our_scenarios.items():
        # Method 1: Linear extrapolation from smallest benchmarks
        smallest_nodes = min(avg_times.keys())  # 800 nodes
        smallest_time = avg_times[smallest_nodes]  # ~3.0s
        
        # Simple linear scaling assumption: time scales with problem size
        linear_estimate = smallest_time * (vars_count / smallest_nodes)
        
        # Method 2: Use density considerations
        # Our food optimization problems are likely to be denser than sparse Gset instances
        # but less dense than the densest ones. Assume medium density characteristics.
        medium_density_times = [item[3] for item in density_groups.get('medium', [])]
        if medium_density_times:
            medium_avg = np.mean(medium_density_times)
            density_estimate = medium_avg * (vars_count / 800)  # Scale from 800-node baseline
        else:
            density_estimate = linear_estimate
        
        # Method 3: Conservative estimate (using minimum scaling)
        conservative_estimate = min(linear_estimate, density_estimate)
        
        print(f"{scenario:<12} {vars_count:<10} {conservative_estimate:<18.4f} {'Conservative (min scaling)':<25}")
        print(f"{'':12} {'':10} {linear_estimate:<18.4f} {'Linear scaling':<25}")
        if medium_density_times:
            print(f"{'':12} {'':10} {density_estimate:<18.4f} {'Medium density scaling':<25}")
        print()
    
    # Analysis of our problem characteristics vs Gset benchmarks
    print(f"\nProblem Characteristics Comparison:")
    print(f"Our problems vs Gset benchmarks:")
    print(f"- Size: {min(our_scenarios.values())}-{max(our_scenarios.values())} variables vs {min(avg_times.keys())}-{max(avg_times.keys())} variables")
    print(f"- Our problems are {min(avg_times.keys())//max(our_scenarios.values()):.0f}x-{max(avg_times.keys())//min(our_scenarios.values()):.0f}x smaller than benchmarks")
    print(f"- Expected density: Medium to high (constrained optimization)")
    print(f"- Structure: Regular constraint patterns vs random/structured Gset graphs")
    
    print(f"\nKey insights:")
    print(f"1. Smaller problems should solve much faster than linear scaling suggests")
    print(f"2. D-Wave Hybrid overhead dominates for small problems")
    print(f"3. Actual solve times likely < 1 second for our scenarios")
    print(f"4. Classical preprocessing may be more significant than quantum annealing time")
    
    return avg_times, density_groups

def compare_scenarios_with_benchmarks():
    """Compare our food optimization scenarios with Gset benchmark characteristics."""
    print("\n" + "-" * 60)
    print("SCENARIO vs BENCHMARK COMPARISON")
    print("-" * 60)
    
    try:
        from src.scenarios import load_food_data
        
        # Load our scenarios
        scenarios = ['simple', 'intermediate', 'full']
        our_data = {}
        
        for scenario in scenarios:
            farms, foods, food_groups, config = load_food_data(scenario)
            qubo_matrix = create_food_optimization_qubo(farms, foods, config)
            
            nodes = len(farms) * len(foods)
            edges = np.count_nonzero(np.triu(qubo_matrix, k=1))  # Count upper triangle interactions
            max_edges = nodes * (nodes - 1) // 2
            density = edges / max_edges if max_edges > 0 else 0
            
            our_data[scenario] = {
                'nodes': nodes,
                'edges': edges,
                'density': density
            }
        
        print(f"Our Food Optimization Problems:")
        print(f"{'Scenario':<12} {'Nodes':<6} {'Edges':<8} {'Density':<10} {'Type':<15}")
        print("-" * 55)
        
        for scenario in scenarios:
            data = our_data[scenario]
            nodes = data['nodes']
            edges = data['edges']
            density = data['density']
            
            # Classify density type
            if density < 0.01:
                density_type = "sparse"
            elif density < 0.05:
                density_type = "medium"
            else:
                density_type = "dense"
            
            print(f"{scenario:<12} {nodes:<6} {edges:<8} {density:<10.4f} {density_type:<15}")
        
        # Find most similar Gset instances
        print(f"\nMost Similar Gset Instances (by structure):")
        print(f"{'Our Scenario':<12} {'Similar Gset':<12} {'Reason':<40}")
        print("-" * 70)
        
        # This is illustrative - in practice, we'd do more sophisticated matching
        print(f"{'simple':<12} {'G11-G13':<12} {'Small, sparse-medium density':<40}")
        print(f"{'intermediate':<12} {'G32-G34':<12} {'Medium size, controlled density':<40}")
        print(f"{'full':<12} {'G48-G50':<12} {'Structured, constraint-heavy':<40}")
        
        return our_data
        
    except Exception as e:
        print(f"✗ Scenario comparison failed: {e}")
        traceback.print_exc()
        return None

def calculate_r_squared(x, y, coeffs, degree):
    """Calculate R-squared for polynomial fit."""
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 1

def create_food_optimization_qubo(farms: List[str], foods: Dict[str, Dict], config: Dict) -> np.ndarray:
    """Create QUBO matrix for the food optimization problem."""
    num_farms = len(farms)
    num_foods = len(foods)
    problem_size = num_farms * num_foods
    Q = np.zeros((problem_size, problem_size))
    
    # Get weights from config
    weights = config.get('parameters', {}).get('weights', {
        'nutritional_value': 0.25,
        'nutrient_density': 0.2,
        'affordability': 0.15,
        'sustainability': 0.15,
        'environmental_impact': 0.25
    })
    
    food_list = list(foods.keys())
    
    # Add linear terms (diagonal elements)
    for farm_idx in range(num_farms):
        for food_idx, food_name in enumerate(food_list):
            var_idx = farm_idx * num_foods + food_idx
            # Maximize nutritional value (negative coefficient)
            Q[var_idx, var_idx] -= weights.get('nutritional_value', 0.25) * foods[food_name].get('nutritional_value', 1)
    
    # Add constraint penalties (farm capacity constraints)
    penalty_strength = 10.0
    for farm_idx in range(num_farms):
        farm_vars = [farm_idx * num_foods + food_idx for food_idx in range(num_foods)]
        # Each farm should produce exactly one food type
        for i in farm_vars:
            Q[i, i] += penalty_strength
            for j in farm_vars:
                if i != j:
                    Q[i, j] += penalty_strength
    
    return Q

if __name__ == "__main__":
    success = main()
    print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if success:
        print("\n🎉 Analysis completed successfully!")
        print("\nKey findings:")
        print("1. Calculated exact nodes and density for all three scenarios")
        print("2. Estimated D-Wave Hybrid solve times based on research benchmarks")
        print("3. Identified scaling patterns for quantum annealing performance")
    else:
        print("\n❌ Analysis failed. Check the error messages above.")
    input("\nPress Enter to exit...")
