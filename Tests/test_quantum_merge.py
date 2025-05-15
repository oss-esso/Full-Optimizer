"""
Test script to compare the quantum-enhanced-merge method with the original quantum-enhanced method.
"""
import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import gc
from typing import Dict, List, Tuple
from dataclasses import dataclass

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
    from src.optimizer import SimpleFoodOptimizer
    from src.data_models import OptimizationResult
except ImportError as e:
    print(f"Error importing optimizer components: {e}")
    print("Please ensure 'src' directory is in the Python path and contains the necessary files.")
    sys.exit(1)

@dataclass
class SolverResult:
    """Class to store results from a single solver run."""
    objective_value: float
    runtime: float
    status: str
    method: str
    solution: Dict[Tuple[str, str], float]
    memory_peak: float
    gap: float = None
    max_violation: float = None
    inf_norm_x: float = None
    one_norm_y: float = None

def run_single_test(method: str) -> SolverResult:
    """Run a single test with the specified solver method."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    optimizer = SimpleFoodOptimizer()
    optimizer.load_food_data()
    
    start_time = time.time()
    
    if method == 'Quantum-Enhanced':
        result = optimizer.optimize_with_quantum_benders()
    elif method == 'Quantum-Enhanced-Merge':
        result = optimizer.optimize_with_quantum_benders_merge()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    runtime = time.time() - start_time
    peak_memory = process.memory_info().rss - initial_memory
    
    solver_result = SolverResult(
        objective_value=result.objective_value,
        runtime=runtime,
        status=result.status,
        method=method,
        solution=result.solution,
        memory_peak=peak_memory
    )
    
    # Clean up
    del optimizer
    gc.collect()
    
    return solver_result

def generate_report(results: Dict[str, List[SolverResult]]) -> str:
    """Generate a report of the test results."""
    report_lines = []
    report_lines.append("# Quantum Enhanced Methods Comparison")
    report_lines.append("")
    
    # Calculate statistics
    stats = {}
    for method, method_results in results.items():
        obj_values = [r.objective_value for r in method_results]
        runtimes = [r.runtime for r in method_results]
        success_rates = [1 if r.status.lower() == 'optimal' else 0 for r in method_results]
        memory_peaks = [r.memory_peak / 1024 / 1024 for r in method_results]  # Convert to MB
        
        stats[method] = {
            'obj_mean': np.mean(obj_values),
            'obj_std': np.std(obj_values),
            'runtime_mean': np.mean(runtimes),
            'runtime_std': np.std(runtimes),
            'success_rate': np.mean(success_rates) * 100,
            'memory_mean': np.mean(memory_peaks),
            'memory_std': np.std(memory_peaks)
        }
    
    # Add results to report
    for method, method_stats in stats.items():
        report_lines.append(f"## {method}")
        report_lines.append(f"- **Objective Value**: {method_stats['obj_mean']:.4f} ± {method_stats['obj_std']:.4f}")
        report_lines.append(f"- **Runtime**: {method_stats['runtime_mean']:.2f} ± {method_stats['runtime_std']:.2f} seconds")
        report_lines.append(f"- **Success Rate**: {method_stats['success_rate']:.1f}%")
        report_lines.append(f"- **Memory Peak**: {method_stats['memory_mean']:.2f} ± {method_stats['memory_std']:.2f} MB")
        report_lines.append("")
    
    # Print a summary table for console
    report_lines.append("## Summary Table")
    report_lines.append("")
    report_lines.append("| Method | Objective Value | Runtime (s) | Success Rate | Memory Peak (MB) |")
    report_lines.append("|--------|----------------|-------------|--------------|------------------|")
    
    for method, method_stats in stats.items():
        report_lines.append(
            f"| {method} | {method_stats['obj_mean']:.4f} ± {method_stats['obj_std']:.4f} | "
            f"{method_stats['runtime_mean']:.2f} ± {method_stats['runtime_std']:.2f} | "
            f"{method_stats['success_rate']:.1f}% | "
            f"{method_stats['memory_mean']:.2f} ± {method_stats['memory_std']:.2f} |"
        )
    
    report_lines.append("")
    report_lines.append("## Individual Run Results")
    report_lines.append("")
    
    for method, method_results in results.items():
        report_lines.append(f"### {method}")
        report_lines.append("")
        report_lines.append("| Run | Objective Value | Runtime (s) | Status | Memory Peak (MB) |")
        report_lines.append("|-----|----------------|-------------|--------|------------------|")
        
        for i, result in enumerate(method_results):
            report_lines.append(
                f"| {i+1} | {result.objective_value:.4f} | {result.runtime:.2f} | {result.status} | "
                f"{result.memory_peak/1024/1024:.2f} |"
            )
        
        report_lines.append("")
    
    return "\n".join(report_lines)

def main():
    """Run tests with quantum-enhanced and quantum-enhanced-merge methods."""
    print("\n" + "=" * 80)
    print(" QUANTUM-ENHANCED METHODS COMPARISON")
    print("=" * 80)
    
    # Methods to test
    methods = ['Quantum-Enhanced', 'Quantum-Enhanced-Merge']
    
    # Number of runs per method
    n_runs = 5
    
    # Store results
    results = {method: [] for method in methods}
    
    # Run tests
    for method in methods:
        print(f"\nRunning {n_runs} tests for {method}...")
        for i in range(n_runs):
            print(f"  Run {i+1}/{n_runs}...")
            try:
                result = run_single_test(method)
                results[method].append(result)
                print(f"  Objective: {result.objective_value:.4f}, Runtime: {result.runtime:.2f}s")
            except Exception as e:
                print(f"  ERROR: {str(e)}")
    
    # Generate report
    print("\nGenerating report...")
    report = generate_report(results)
    
    # Print the report
    print("\n" + "=" * 80)
    print(" RESULTS")
    print("=" * 80)
    print(report)
    
    # Save report to file
    report_path = os.path.join(parent_dir, "Results", "quantum_methods_comparison.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nDetailed report saved to: {report_path}")

if __name__ == "__main__":
    main() 