"""
Example script demonstrating food production optimization with the original configuration
from OQI_benders_vs_pulp.py, comparing all four solution methods.
"""
import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Set, Any, Union

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s: %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Ensure paths are correctly set up
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Full Optimizer directory
sys.path.insert(0, parent_dir)  # Add parent directory to path

# Import optimizer
try:
    from src.optimizer import FoodProductionOptimizer, SimpleFoodOptimizer
    from src.data_models import OptimizationObjective, OptimizationResult
except ImportError as e:
    print(f"Error importing optimizer components: {e}")
    print("Please ensure 'src' directory is in the Python path and contains the necessary files.")
    sys.exit(1)

def log_optimization_results(result: OptimizationResult, 
                           farms: List[str], 
                           parameters: Dict,
                           method_name: str,
                           solver_name: str,
                           log_file: str = None):
    """Log optimization results to a file."""
    if log_file is None:
        log_file = os.path.join("Results", f"{method_name}_results.txt")
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f" OPTIMIZATION RESULTS - {solver_name}\n")
        f.write("="*80 + "\n\n")
        
        if not result.solution:
            f.write("WARNING: No valid solution found.\n")
            return
        
        # Log configuration
        f.write("CONFIGURATION:\n")
        f.write("-"*40 + "\n")
        f.write(f"Farms: {farms}\n")
        f.write(f"Objective Value: {result.objective_value:.4f}\n")
        f.write(f"Runtime: {result.runtime:.2f} seconds\n\n")
        
        # Log solution
        f.write("SOLUTION:\n")
        f.write("-"*40 + "\n")
        for (farm, food), area in result.solution.items():
            f.write(f"Farm: {farm}, Food: {food}, Area: {area:.2f} hectares\n")
        f.write("\n")
        
        # Log metrics
        f.write("METRICS:\n")
        f.write("-"*40 + "\n")
        for metric, value in result.metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("\n")
        
        # Log farm utilization
        f.write("FARM UTILIZATION:\n")
        f.write("-"*40 + "\n")
        for farm in farms:
            farm_area = sum(area for (f, _), area in result.solution.items() if f == farm)
            farm_utilization = farm_area / parameters['land_availability'][farm] * 100
            f.write(f"{farm}: {farm_area:.2f} hectares ({farm_utilization:.1f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(" END OF RESULTS\n")
        f.write("="*80 + "\n")

def plot_convergence_comparison(results, save_path=None):
    """Plot a comparison of convergence between all Benders variants."""
    plt.figure(figsize=(14, 10))
    
    colors = {
        'Benders': ('b', 'r'), 
        'Quantum-Inspired': ('g', 'm'),
        'Quantum-Enhanced': ('c', 'y'),
        'PuLP': ('k', '#8c564b')
    }
    
    # Plot all bounds (convert to maximization by negating)
    for method, (lb_color, ub_color) in colors.items():
        if method in results and 'benders_data' in results[method]:
            benders_data = results[method]['benders_data']
            
            if 'lower_bounds' in benders_data and benders_data['lower_bounds']:
                lb = [-val for val in benders_data['lower_bounds']]
                plt.plot(lb, f'{lb_color}-o', label=f'Lower Bound ({method})', markersize=4)
                
            if 'upper_bounds' in benders_data and benders_data['upper_bounds']:
                ub = [-val for val in benders_data['upper_bounds']]
                plt.plot(ub, f'{ub_color}-o', label=f'Upper Bound ({method})', markersize=4)
    
    # Add labels and styling
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Objective Value (maximization)', fontsize=12)
    plt.title('Convergence Comparison: All Benders Methods', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    if save_path:
        results_dir = os.path.dirname(save_path)
        if results_dir and not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Convergence plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_method_comparison(results, save_path=None):
    """Plot a comparison of results from different methods."""
    # Extract data for comparison
    methods = list(results.keys())
    objective_values = [results[m]['objective_value'] for m in methods]
    runtimes = [results[m]['runtime'] for m in methods]
    
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot objective values
    bars1 = ax1.bar(methods, objective_values, color=['royalblue', 'forestgreen', 'darkorange', 'crimson'])
    ax1.set_title('Objective Values', fontsize=16)
    ax1.set_ylabel('Objective Value', fontsize=14)
    ax1.set_ylim(0, max(objective_values) * 1.1)
    
    # Add value labels to bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                f'{height:.4f}', ha='center', va='bottom', fontsize=12)
    
    # Plot runtimes
    bars2 = ax2.bar(methods, runtimes, color=['lightblue', 'lightgreen', 'moccasin', 'lightcoral'])
    ax2.set_title('Runtimes', fontsize=16)
    ax2.set_ylabel('Time (seconds)', fontsize=14)
    ax2.set_ylim(0, max(runtimes) * 1.1)
    
    # Add value labels to bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=12)
    
    # Add speedup percentage for quantum methods
    if 'Benders' in results and 'Quantum-Inspired' in results:
        benders_time = results['Benders']['runtime']
        quantum_inspired_time = results['Quantum-Inspired']['runtime']
        if benders_time > 0:
            speedup = (benders_time - quantum_inspired_time) / benders_time * 100
            if speedup > 0:
                ax2.text(1, quantum_inspired_time * 0.5, 
                        f"{speedup:.1f}% faster", ha='center', fontsize=12, color='darkgreen')
    
    # Add a title for the entire figure
    plt.suptitle('Comparison of Optimization Methods', fontsize=18, y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        results_dir = os.path.dirname(save_path)
        if results_dir and not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        logger.info(f"Methods comparison plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def main():
    """Run optimization using the original configuration from OQI_benders_vs_pulp.py."""
    print("\n" + "=" * 80)
    print(" FOOD PRODUCTION OPTIMIZATION - ORIGINAL CONFIGURATION")
    print("=" * 80)
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(current_dir), "Results")
    os.makedirs(results_dir, exist_ok=True)

    optimizer = SimpleFoodOptimizer()
    optimizer.load_food_data()
    
    
    
    # Create the optimizer
    print("Creating food production optimizer with original configuration...")
    
    
    # Dictionary to store results from all methods
    all_results = {}
    
    # 1. Solve with classical Benders decomposition
    print("\n" + "=" * 80)
    print(" METHOD 1: CLASSICAL BENDERS DECOMPOSITION")
    print("=" * 80)
    
    start_time = time.time()
    benders_result = optimizer.optimize_with_benders()
    runtime = time.time() - start_time
    benders_result.runtime = runtime
    
    print(f"Status: {benders_result.status}")
    print(f"Objective Value: {benders_result.objective_value:.4f}")
    print(f"Runtime: {runtime:.2f} seconds")
    
    # Print solution summary
    print("\nLand Allocation:")
    for (farm, food), area in benders_result.solution.items():
        print(f"{farm}: {food} - {area:.2f} hectares")
    
    # Store results and log to file
    all_results['Benders'] = {
        'result': benders_result,
        'objective_value': benders_result.objective_value,
        'runtime': runtime,
        'benders_data': benders_result.benders_data
    }
    
    log_file = os.path.join(results_dir, "benders_results.txt")
    log_optimization_results(benders_result, optimizer.farms, optimizer.parameters, 
                           'benders', 'Benders Decomposition', log_file)
    
    # 2. Solve with PuLP
    print("\n" + "=" * 80)
    print(" METHOD 2: DIRECT PULP SOLVER")
    print("=" * 80)
    
    start_time = time.time()
    pulp_result = optimizer.optimize_with_pulp()
    runtime = time.time() - start_time
    pulp_result.runtime = runtime
    
    print(f"Status: {pulp_result.status}")
    print(f"Objective Value: {pulp_result.objective_value:.4f}")
    print(f"Runtime: {runtime:.2f} seconds")
    
    # Print solution summary
    print("\nLand Allocation:")
    for (farm, food), area in pulp_result.solution.items():
        print(f"{farm}: {food} - {area:.2f} hectares")
    
    # Store results and log to file
    all_results['PuLP'] = {
        'result': pulp_result,
        'objective_value': pulp_result.objective_value,
        'runtime': runtime
    }
    
    log_file = os.path.join(results_dir, "pulp_results.txt")
    log_optimization_results(pulp_result, optimizer.farms, optimizer.parameters, 
                           'pulp', 'Direct PuLP Solver', log_file)
    
    # 3. Solve with quantum-inspired Benders
    print("\n" + "=" * 80)
    print(" METHOD 3: QUANTUM-INSPIRED BENDERS")
    print("=" * 80)
    
    start_time = time.time()
    qi_result = optimizer.optimize_with_quantum_inspired_benders()
    runtime = time.time() - start_time
    qi_result.runtime = runtime
    
    print(f"Status: {qi_result.status}")
    print(f"Objective Value: {qi_result.objective_value:.4f}")
    print(f"Runtime: {runtime:.2f} seconds")
    
    # Print solution summary
    print("\nLand Allocation:")
    for (farm, food), area in qi_result.solution.items():
        print(f"{farm}: {food} - {area:.2f} hectares")
    
    # Store results and log to file
    all_results['Quantum-Inspired'] = {
        'result': qi_result,
        'objective_value': qi_result.objective_value,
        'runtime': runtime,
        'benders_data': qi_result.benders_data
    }
    
    log_file = os.path.join(results_dir, "quantum_inspired_results.txt")
    log_optimization_results(qi_result, optimizer.farms, optimizer.parameters, 
                           'quantum_inspired', 'Quantum-Inspired Benders', log_file)
    
    # Print quantum metrics if available
    if qi_result.benders_data and 'quantum_metrics' in qi_result.benders_data:
        metrics = qi_result.benders_data['quantum_metrics']
        print("\nQuantum-Inspired Metrics:")
        print(f"QUBO Variables: {metrics.get('num_variables', 'N/A')}")
        print(f"QUBO Matrix Density: {metrics.get('matrix_density', 'N/A')}%")
        print(f"Original binary variables: {metrics.get('original_vars', 'N/A')}")
    
    # 4. Solve with quantum-enhanced Benders (this might fall back to quantum-inspired)
    print("\n" + "=" * 80)
    print(" METHOD 4: QUANTUM-ENHANCED BENDERS")
    print("=" * 80)
    
    start_time = time.time()
    qe_result = optimizer.optimize_with_quantum_benders()
    runtime = time.time() - start_time
    qe_result.runtime = runtime
    
    print(f"Status: {qe_result.status}")
    print(f"Objective Value: {qe_result.objective_value:.4f}")
    print(f"Runtime: {runtime:.2f} seconds")
    
    # Print solution summary
    print("\nLand Allocation:")
    for (farm, food), area in qe_result.solution.items():
        print(f"{farm}: {food} - {area:.2f} hectares")
    
    # Store results and log to file
    all_results['Quantum-Enhanced'] = {
        'result': qe_result,
        'objective_value': qe_result.objective_value,
        'runtime': runtime,
        'benders_data': qe_result.benders_data
    }
    
    log_file = os.path.join(results_dir, "quantum_enhanced_results.txt")
    log_optimization_results(qe_result, optimizer.farms, optimizer.parameters, 
                           'quantum_enhanced', 'Quantum-Enhanced Benders', log_file)
    
    # Print summary comparing all methods
    print("\n" + "=" * 80)
    print(" COMPARISON OF ALL OPTIMIZATION METHODS")
    print("=" * 80)
    
    # Print objective values
    print(f"Objective values (maximization):")
    print(f"  Classical Benders:      {all_results['Benders']['objective_value']:.4f}")
    print(f"  PuLP:                   {all_results['PuLP']['objective_value']:.4f}")
    print(f"  Quantum-Inspired:       {all_results['Quantum-Inspired']['objective_value']:.4f}")
    print(f"  Quantum-Enhanced:       {all_results['Quantum-Enhanced']['objective_value']:.4f}")
    
    # Print runtimes
    print(f"\nRuntimes:")
    print(f"  Classical Benders:      {all_results['Benders']['runtime']:.2f} seconds")
    print(f"  PuLP:                   {all_results['PuLP']['runtime']:.2f} seconds")
    print(f"  Quantum-Inspired:       {all_results['Quantum-Inspired']['runtime']:.2f} seconds")
    print(f"  Quantum-Enhanced:       {all_results['Quantum-Enhanced']['runtime']:.2f} seconds")
    
    # Generate visualization comparing all methods
    method_comparison_path = os.path.join(results_dir, "method_comparison.png")
    plot_method_comparison(all_results, method_comparison_path)
    
    # Generate convergence comparison for Benders variants
    convergence_comparison_path = os.path.join(results_dir, "convergence_comparison.png")
    plot_convergence_comparison(all_results, convergence_comparison_path)
    
    print(f"\nResults have been saved to: {results_dir}")
    print(f"Method comparison plot: {method_comparison_path}")
    print(f"Convergence comparison plot: {convergence_comparison_path}")
    print("\nDone!")
    

if __name__ == "__main__":
    main() 
