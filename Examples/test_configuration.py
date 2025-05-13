"""
Test script that runs the original configuration multiple times to gather statistics
about the performance of different solvers.
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
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

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

def calculate_solution_metrics(result: SolverResult, pulp_result: SolverResult) -> SolverResult:
    """Calculate additional metrics comparing with PuLP solution."""
    if pulp_result is None or not pulp_result.solution:
        return result
    
    # Calculate gap
    if pulp_result.objective_value != 0:
        result.gap = abs(result.objective_value - pulp_result.objective_value) / abs(pulp_result.objective_value)
    
    # Calculate solution differences
    if result.solution and pulp_result.solution:
        # Maximum violation of constraints
        result.max_violation = max(abs(result.solution.get(k, 0) - pulp_result.solution.get(k, 0)) 
                                 for k in set(result.solution.keys()) | set(pulp_result.solution.keys()))
        
        # Infinity norm of x differences
        x_diffs = [abs(result.solution.get(k, 0) - pulp_result.solution.get(k, 0)) 
                  for k in set(result.solution.keys()) | set(pulp_result.solution.keys())]
        result.inf_norm_x = max(x_diffs) if x_diffs else 0
        
        # One norm of y differences
        result.one_norm_y = sum(x_diffs)
    
    return result

def run_single_test(method: str, pulp_result: SolverResult = None) -> SolverResult:
    """Run a single test with the specified solver method."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    optimizer = SimpleFoodOptimizer()
    optimizer.load_food_data()
    
    start_time = time.time()
    
    if method == 'Benders':
        result = optimizer.optimize_with_benders()
    elif method == 'PuLP':
        result = optimizer.optimize_with_pulp()
    elif method == 'Quantum-Inspired':
        result = optimizer.optimize_with_quantum_inspired_benders()
    elif method == 'Quantum-Enhanced':
        result = optimizer.optimize_with_quantum_benders()
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
    
    # Calculate additional metrics if we have PuLP result for comparison
    if method != 'PuLP' and pulp_result is not None:
        solver_result = calculate_solution_metrics(solver_result, pulp_result)
    
    # Clean up
    del optimizer
    gc.collect()
    
    return solver_result

def run_parallel_tests(n_runs: int = 50) -> Dict[str, List[SolverResult]]:
    """Run tests in parallel for all methods."""
    methods = ['PuLP', 'Benders', 'Quantum-Inspired', 'Quantum-Enhanced']  # PuLP first for comparison
    results = {method: [] for method in methods}
    
    # Use ProcessPoolExecutor to run tests in parallel
    n_cores = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
    logger.info(f"Running tests using {n_cores} cores")
    
    # First run PuLP to get reference solutions
    logger.info(f"Starting {n_runs} runs for PuLP")
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        pulp_futures = [executor.submit(run_single_test, 'PuLP') for _ in range(n_runs)]
        for future in pulp_futures:
            results['PuLP'].append(future.result())
    
    # Then run other methods with PuLP results for comparison
    for method in methods[1:]:  # Skip PuLP as it's already done
        logger.info(f"Starting {n_runs} runs for {method}")
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = []
            for i in range(n_runs):
                pulp_result = results['PuLP'][i]
                futures.append(executor.submit(run_single_test, method, pulp_result))
            for future in futures:
                results[method].append(future.result())
    
    return results

def analyze_results(results: Dict[str, List[SolverResult]]) -> pd.DataFrame:
    """Analyze the results and return a DataFrame with statistics."""
    # Prepare data for analysis
    data = []
    for method, method_results in results.items():
        for result in method_results:
            row = {
                'Method': method,
                'obj_value': result.objective_value,
                'runtime': result.runtime,
                'status': result.status,
                'memory_peak': result.memory_peak,
                'gap': result.gap if result.gap is not None else np.nan,
                'max_violation': result.max_violation if result.max_violation is not None else np.nan,
                'inf_norm_x': result.inf_norm_x if result.inf_norm_x is not None else np.nan,
                'one_norm_y': result.one_norm_y if result.one_norm_y is not None else np.nan
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Calculate statistics
    stats = []
    for method in results.keys():
        method_data = df[df['Method'] == method]
        
        stats.append({
            'Method': method,
            'Mean Objective': method_data['obj_value'].mean(),
            'Std Objective': method_data['obj_value'].std(),
            'Mean Runtime': method_data['runtime'].mean(),
            'Std Runtime': method_data['runtime'].std(),
            'Success Rate': (method_data['status'] == 'optimal').mean(),
            'Min Objective': method_data['obj_value'].min(),
            'Max Objective': method_data['obj_value'].max(),
            'Mean Gap': method_data['gap'].mean(),
            'Std Gap': method_data['gap'].std(),
            'Mean Max Violation': method_data['max_violation'].mean(),
            'Std Max Violation': method_data['max_violation'].std(),
            'Mean Inf Norm X': method_data['inf_norm_x'].mean(),
            'Std Inf Norm X': method_data['inf_norm_x'].std(),
            'Mean One Norm Y': method_data['one_norm_y'].mean(),
            'Std One Norm Y': method_data['one_norm_y'].std(),
            'Mean Memory Peak': method_data['memory_peak'].mean(),
            'Std Memory Peak': method_data['memory_peak'].std()
        })
    
    return pd.DataFrame(stats)

def generate_markdown_report(stats_df: pd.DataFrame, save_dir: str):
    """Generate a comprehensive markdown report."""
    report_lines = []
    report_lines.append('# Optimization Methods Evaluation Report')
    report_lines.append('')
    report_lines.append('## Solution Quality Metrics')
    report_lines.append('')
    
    # Objective values
    report_lines.append('### Objective Values')
    for _, row in stats_df.iterrows():
        report_lines.append(f"#### {row['Method']}")
        report_lines.append(f"- Mean: {row['Mean Objective']:.4f} ± {row['Std Objective']:.4f}")
        report_lines.append(f"- Range: [{row['Min Objective']:.4f}, {row['Max Objective']:.4f}]")
        report_lines.append(f"- Success Rate: {row['Success Rate']*100:.1f}%")
        report_lines.append('')
    
    # Gaps and violations
    report_lines.append('### Solution Differences (compared to PuLP)')
    for _, row in stats_df.iterrows():
        if row['Method'] != 'PuLP':
            report_lines.append(f"#### {row['Method']}")
            report_lines.append(f"- Gap: {row['Mean Gap']:.4f} ± {row['Std Gap']:.4f}")
            report_lines.append(f"- Max Violation: {row['Mean Max Violation']:.4f} ± {row['Std Max Violation']:.4f}")
            report_lines.append(f"- Infinity Norm X: {row['Mean Inf Norm X']:.4f} ± {row['Std Inf Norm X']:.4f}")
            report_lines.append(f"- One Norm Y: {row['Mean One Norm Y']:.4f} ± {row['Std One Norm Y']:.4f}")
            report_lines.append('')
    
    # Performance metrics
    report_lines.append('## Performance Metrics')
    report_lines.append('')
    report_lines.append('### Runtime')
    for _, row in stats_df.iterrows():
        report_lines.append(f"#### {row['Method']}")
        report_lines.append(f"- Mean: {row['Mean Runtime']:.2f} ± {row['Std Runtime']:.2f} seconds")
        report_lines.append('')
    
    # Resource utilization
    report_lines.append('### Resource Utilization')
    for _, row in stats_df.iterrows():
        report_lines.append(f"#### {row['Method']}")
        report_lines.append(f"- Memory Peak: {row['Mean Memory Peak']/1024/1024:.2f} ± {row['Std Memory Peak']/1024/1024:.2f} MB")
        report_lines.append('')
    
    # Save report
    report_path = os.path.join(save_dir, 'evaluation_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    logger.info(f"Markdown report generated: {report_path}")

def plot_results(results: Dict[str, List[SolverResult]], save_dir: str):
    """Create visualizations of the results."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data for plotting
    data = []
    for method, method_results in results.items():
        for result in method_results:
            data.append({
                'Method': method,
                'Objective Value': result.objective_value,
                'Runtime': result.runtime,
                'Memory Peak (MB)': result.memory_peak / 1024 / 1024,
                'Gap': result.gap if result.gap is not None else np.nan,
                'Max Violation': result.max_violation if result.max_violation is not None else np.nan
            })
    df = pd.DataFrame(data)
    
    # Plot 1: Box plot of objective values
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Method', y='Objective Value')
    plt.title('Distribution of Objective Values by Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'objective_distribution.png'))
    plt.close()
    
    # Plot 2: Box plot of runtimes
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Method', y='Runtime')
    plt.title('Distribution of Runtimes by Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'runtime_distribution.png'))
    plt.close()
    
    # Plot 3: Scatter plot of objective vs runtime
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Runtime', y='Objective Value', hue='Method', alpha=0.6)
    plt.title('Objective Value vs Runtime by Method')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'objective_vs_runtime.png'))
    plt.close()
    
    # Plot 4: Box plot of gaps
    plt.figure(figsize=(12, 6))
    gap_df = df[df['Method'] != 'PuLP'].copy()  # Exclude PuLP
    sns.boxplot(data=gap_df, x='Method', y='Gap')
    plt.title('Distribution of Solution Gaps (compared to PuLP)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gap_distribution.png'))
    plt.close()
    
    # Plot 5: Box plot of memory usage
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Method', y='Memory Peak (MB)')
    plt.title('Distribution of Memory Usage by Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'memory_distribution.png'))
    plt.close()

def main():
    """Run multiple tests and analyze the results."""
    print("\n" + "=" * 80)
    print(" MULTIPLE RUN TEST OF OPTIMIZATION METHODS")
    print("=" * 80)
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(current_dir), "Results", "multiple_runs")
    os.makedirs(results_dir, exist_ok=True)
    
    # Run tests
    n_runs = 50
    logger.info(f"Starting {n_runs} runs for each method")
    results = run_parallel_tests(n_runs)
    
    # Analyze results
    stats_df = analyze_results(results)
    
    # Save statistics to CSV
    stats_file = os.path.join(results_dir, "statistics.csv")
    stats_df.to_csv(stats_file, index=False)
    
    # Generate markdown report
    generate_markdown_report(stats_df, results_dir)
    
    # Create visualizations
    plot_results(results, results_dir)
    print(f"\nResults saved to: {results_dir}")
    
    # Print summary
    print("\nSummary:")
    print("=" * 80)
    for _, row in stats_df.iterrows():
        print(f"\n{row['Method']}:")
        print(f"  Objective Value: {row['Mean Objective']:.4f} ± {row['Std Objective']:.4f}")
        print(f"  Runtime: {row['Mean Runtime']:.2f} ± {row['Std Runtime']:.2f} seconds")
        print(f"  Success Rate: {row['Success Rate']*100:.1f}%")
        if row['Method'] != 'PuLP':
            print(f"  Gap: {row['Mean Gap']:.4f} ± {row['Std Gap']:.4f}")
            print(f"  Max Violation: {row['Mean Max Violation']:.4f} ± {row['Std Max Violation']:.4f}")
        print(f"  Memory Peak: {row['Mean Memory Peak']/1024/1024:.2f} ± {row['Std Memory Peak']/1024/1024:.2f} MB")

if __name__ == "__main__":
    main() 