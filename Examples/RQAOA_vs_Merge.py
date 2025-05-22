"""
Example script demonstrating food production optimization with Quantum-Enhanced-Merge and Recursive QAOA methods.
"""
import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')  # Set the backend to Agg for PDF generation
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s: %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Define OptimizationObjective enum to match what's expected in optimizer.py
class OptimizationObjective(Enum):
    """Types of optimization objectives."""
    NUTRITIONAL_VALUE = "nutritional_value"
    NUTRIENT_DENSITY = "nutrient_density"
    ENVIRONMENTAL_IMPACT = "environmental_impact"
    AFFORDABILITY = "affordability"
    SUSTAINABILITY = "sustainability"

# Ensure paths are correctly set up
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Full Optimizer directory
sys.path.insert(0, parent_dir)  # Add parent directory to path

# First, patch the OptimizationObjective into the src.optimizer module
import src.optimizer
src.optimizer.OptimizationObjective = OptimizationObjective

# Import the optimizer class
from src.optimizer import SimpleFoodOptimizer

# Also patch the parameter generation method to use our provided config directly
def patched_generate_parameters(self):
    """Patched method that just returns the parameters from config instead of generating them"""
    return self.config.get('parameters', {})

# Add a compatibility wrapper for calculate_metrics
def calculate_metrics(self, solution):
    """Compatibility wrapper for _calculate_metrics"""
    if hasattr(self, '_calculate_metrics'):
        # Make sure parameters has the right keys for _calculate_metrics
        if 'weights' in self.parameters and 'objective_weights' not in self.parameters:
            self.parameters['objective_weights'] = self.parameters['weights']
        
        return self._calculate_metrics(solution)
    
    # Fallback implementation if _calculate_metrics doesn't exist
    metrics = {}
    
    # Calculate objective contributions
    if hasattr(self, 'parameters'):
        # Get weights from either 'objective_weights' or 'weights'
        weights = self.parameters.get('objective_weights', self.parameters.get('weights', {}))
        
        for obj_name in ['nutritional_value', 'nutrient_density', 'environmental_impact', 'affordability', 'sustainability']:
            metrics[obj_name] = 0
            if obj_name in weights:
                for (farm, food), area in solution.items():
                    if food in self.foods and obj_name in self.foods[food]:
                        value = self.foods[food][obj_name] * area * weights.get(obj_name, 0)
                        if obj_name == 'environmental_impact':
                            metrics[obj_name] -= value  # Negative for impact to minimize
                        else:
                            metrics[obj_name] += value
    
    # Calculate total area
    total_area = sum(solution.values())
    metrics['total_area'] = total_area
    
    # Calculate farm utilization
    if hasattr(self, 'parameters') and 'land_availability' in self.parameters:
        for farm in self.farms:
            if farm in self.parameters['land_availability']:
                farm_area = sum(area for (f, _), area in solution.items() if f == farm)
                metrics[f'utilization_{farm}'] = farm_area / self.parameters['land_availability'][farm]
    
    return metrics

# Add the methods to the FoodProductionOptimizer class
src.optimizer.FoodProductionOptimizer._generate_model_parameters = patched_generate_parameters
src.optimizer.FoodProductionOptimizer.calculate_metrics = calculate_metrics

# Now import FoodProductionOptimizer which will use our patched enum and method
from src.optimizer import FoodProductionOptimizer
from src.data_models import OptimizationResult

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
        
        # Log configuration
        f.write("CONFIGURATION:\n")
        f.write("-"*40 + "\n")
        f.write(f"Farms: {farms}\n")
        f.write(f"Objective Value: {result.objective_value if result.objective_value else 0.0:.4f}\n")
        f.write(f"Runtime: {result.runtime:.2f} seconds\n\n")
        
        if not result.solution:
            f.write("WARNING: No valid solution found in model, but the following objective value was reported.\n")
            return
        
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

def plot_method_comparison(results, save_path=None):
    """Plot a comparison of the results from the optimization methods."""
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    plt.subplots_adjust(hspace=0.4)
    
    # List of methods
    methods = ['Quantum-Enhanced-Merge', 'Recursive-QAOA']
    
    # Extract objective values
    objective_values = [results[method]['objective_value'] if method in results else 0 for method in methods]
    runtimes = [results[method]['runtime'] if method in results else 0 for method in methods]

    # Colors for bars
    colors = ['plum', 'lightblue']
    
    # Plot objective values
    plt.subplot(2, 1, 1)
    bars = plt.bar(methods, objective_values, color=colors)
    plt.title('Objective Values Comparison (Maximize)', fontsize=14)
    plt.xlabel('Optimization Method', fontsize=12)
    plt.ylabel('Objective Value', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Plot runtimes
    plt.subplot(2, 1, 2)
    bars = plt.bar(methods, runtimes, color=colors)
    plt.title('Runtime Comparison', fontsize=14)
    plt.xlabel('Optimization Method', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
    
    # Save figure
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig) # Close the figure

def plot_solution(result: OptimizationResult, 
                 farms: List[str], 
                 foods: Dict[str, Dict[str, float]], 
                 parameters: Dict,  
                 save_path: Optional[str] = None):
    """Plot the optimization solution with enhanced styling."""
    plot_logger = logging.getLogger(__name__) # Define logger for this function

    # === Start: Added Detailed Debugging ===
    plot_logger.info(f"DEBUG: PLOT_SOLUTION called. Target save_path: {os.path.basename(save_path) if save_path else 'None'}")
    if result:
        plot_logger.info(f"DEBUG: PLOT_SOLUTION - result object received.")
        if hasattr(result, 'solution'):
            plot_logger.info(f"DEBUG: PLOT_SOLUTION - result has 'solution' attribute.")
            plot_logger.info(f"DEBUG: PLOT_SOLUTION - result.solution type: {type(result.solution)}")
            if result.solution is not None:
                plot_logger.info(f"DEBUG: PLOT_SOLUTION - result.solution is not None. Length: {len(result.solution)}")
                plot_logger.info(f"DEBUG: PLOT_SOLUTION - result.solution content (first 50 chars): {str(result.solution)[:50]}")
                if not result.solution: # Check if it's empty
                    plot_logger.warning("DEBUG: PLOT_SOLUTION - result.solution is an empty dictionary (falsy).")
            else:
                plot_logger.warning("DEBUG: PLOT_SOLUTION - result.solution is None.")
        else:
            plot_logger.warning("DEBUG: PLOT_SOLUTION - result does NOT have 'solution' attribute.")
    else:
        plot_logger.warning("DEBUG: PLOT_SOLUTION - result object is None.")
    # === End: Added Detailed Debugging ===

    # Set style and context
    sns.set_style("whitegrid")
    sns.set_context("talk")
    
    # Create figure with adjusted size
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)

    # Check if there's a solution to plot
    solution_to_use = {} 
    if result and hasattr(result, 'solution') and result.solution:
        solution_to_use = result.solution
    
    has_solution = len(solution_to_use) > 0
    
    if not has_solution:
        plot_logger.warning("No valid solution found in result. Creating placeholder visualization.")
        # Create placeholder visualization
        for i in range(2):
            for j in range(2):
                ax = axes[i, j]
                ax.text(0.5, 0.5, 'No solution available for this solver',
                      horizontalalignment='center', verticalalignment='center',
                      transform=ax.transAxes, fontsize=14, color='red')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False)
        
        plt.suptitle('No Valid Solution Found', fontsize=20, fontweight='bold', y=1.02)
        
        if save_path:
            # Create Results directory if it doesn't exist
            results_dir = os.path.dirname(save_path)
            if results_dir and not os.path.exists(results_dir):
                os.makedirs(results_dir, exist_ok=True)
            
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plot_logger.info(f"Placeholder plot saved to {save_path}")
        else:
            plt.show()
        plt.close(fig) # Close the figure
        return
    
    # Color palette configuration
    palette = sns.color_palette("husl", n_colors=len(foods))
    title_style = {'fontsize': 16, 'fontweight': 'bold', 'color': '#2d3436'}

    # Convert solution to DataFrame for easier plotting
    solution_data = []
    for (farm, food), area in solution_to_use.items():
        solution_data.append({'Farm': farm, 'Food': food, 'Area': area})
    
    df = pd.DataFrame(solution_data)
    
    # Plot 1: Area distribution by farm (Pie Chart)
    if df.empty:
        axes[0,0].text(0.5, 0.5, 'No solution data available',
                      horizontalalignment='center', verticalalignment='center',
                      transform=axes[0,0].transAxes)
    else:
        farm_areas = df.groupby('Farm')['Area'].sum()
        
        wedges, texts, autotexts = axes[0,0].pie(
            farm_areas,
            labels=farm_areas.index,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
            textprops={'fontsize': 10},
            colors=sns.color_palette("pastel", n_colors=len(farm_areas))
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        axes[0,0].add_artist(centre_circle)
        axes[0,0].set_title('Area Distribution by Farm', **title_style)

    # Plot 2: Food distribution (Bar Plot)
    if df.empty:
        axes[0,1].text(0.5, 0.5, 'No solution data available',
                      horizontalalignment='center', verticalalignment='center',
                      transform=axes[0,1].transAxes)
    else:
        food_areas = df.groupby('Food')['Area'].sum().sort_values(ascending=False)
        
        sns.barplot(x=food_areas.values, y=food_areas.index, ax=axes[0,1],
                    color=sns.color_palette("viridis")[0],
                    edgecolor='black')
        axes[0,1].set_xlabel('Total Area', fontweight='bold')
        axes[0,1].set_ylabel('')
        if len(axes[0,1].containers) > 0:
            axes[0,1].bar_label(axes[0,1].containers[0], fmt='%.1f',
                           padding=2, fontsize=9)
        axes[0,1].set_title('Crop Distribution by Area', **title_style)

    # Plot 3: Objective contributions (Horizontal Bar Chart)
    metrics = {}
    if result and hasattr(result, 'metrics') and result.metrics:
        metrics = result.metrics
    
    weights = parameters.get('objective_weights', parameters.get('weights', {}))  # Get weights from parameters
    
    obj_components = {
        k: v for k, v in metrics.items() 
        if k in ['nutritional_value', 'nutrient_density', 'affordability', 
                  'sustainability', 'environmental_impact'] 
        and weights.get(k, 0) != 0  # Filter out if weight is 0
    }
    
    if obj_components:
        objectives = list(obj_components.keys())
        contributions = list(obj_components.values())
        # Sort by contribution
        sorted_idx = np.argsort(contributions)
        
        sns.barplot(x=np.array(contributions)[sorted_idx], 
                    y=[o.replace('_', ' ').title() for o in np.array(objectives)[sorted_idx]], 
                    ax=axes[1,0], color=sns.color_palette("rocket")[0], edgecolor='black')
        axes[1,0].set_xlabel('Contribution Score', fontweight='bold')
        if len(axes[1,0].containers) > 0:
            axes[1,0].bar_label(axes[1,0].containers[0], fmt='%.2f',
                           padding=2, fontsize=9)
        axes[1,0].set_title('Objective Contributions', **title_style)
    else:
        axes[1,0].text(0.5, 0.5, 'No metrics data available',
                      horizontalalignment='center', verticalalignment='center',
                      transform=axes[1,0].transAxes)

    # Plot 4: Farm utilization (Bar Plot)
    utilizations = {}
    for farm in farms:
        util_key = f'utilization_{farm}'
        if util_key in metrics:
            utilizations[farm] = metrics[util_key] * 100
    
    if utilizations:
        sns.barplot(x=list(utilizations.keys()), y=list(utilizations.values()),
                    ax=axes[1,1], color=sns.color_palette("mako")[0], edgecolor='black')
        axes[1,1].set_ylabel('Utilization (%)', fontweight='bold')
        axes[1,1].set_ylim(0, 100)
        if len(axes[1,1].containers) > 0:
            axes[1,1].bar_label(axes[1,1].containers[0], fmt='%.1f%%',
                           padding=2, fontsize=9)
        axes[1,1].set_title('Farm Land Utilization', **title_style)
    else:
        axes[1,1].text(0.5, 0.5, 'No utilization data available',
                      horizontalalignment='center', verticalalignment='center',
                      transform=axes[1,1].transAxes)

    # Final adjustments
    for ax in axes.flat:
        ax.tick_params(axis='both', which='major', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('#b2bec3')
            spine.set_linewidth(0.5)

    plt.tight_layout()
    
    # Add title for the entire figure
    obj_value = result.objective_value if result and hasattr(result, 'objective_value') else 0.0
    solver_status = result.status if result and hasattr(result, 'status') else "unknown"
    fig.suptitle(f'Optimization Results (Objective: {obj_value:.2f}, Status: {solver_status})', 
                 fontsize=20, fontweight='bold', y=1.02)
    
    if save_path:
        # Create Results directory if it doesn't exist
        results_dir = os.path.dirname(save_path)
        if results_dir and not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plot_logger.info(f"Solution plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig) # Close the figure

def main():
    """Run optimization using only Quantum-Enhanced-Merge and Recursive QAOA."""
    print("\n" + "=" * 80)
    print(" FOOD PRODUCTION OPTIMIZATION - QUANTUM METHODS COMPARISON")
    print("=" * 80)

    complexity_level = 'full'
    
    # Create results directory for text files
    results_dir = os.path.join(os.path.dirname(current_dir), "Results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create plots directory for PDFs
    plots_dir = os.path.join(os.path.dirname(current_dir), "Plots for PPT")
    os.makedirs(plots_dir, exist_ok=True)

    # Initialize solution paths
    solution_path_qem = os.path.join(plots_dir, "quantum_enhanced_merge_solution.png")
    solution_path_rqaoa = os.path.join(plots_dir, "recursive_qaoa_solution.png")

    # Load food data from Excel
    print("\nLoading food data from Excel...")
    
    # Import the Qoptimizer at the beginning to use for both methods
    from src.Qoptimizer import SimpleFoodOptimizer as Qoptimizer
    
    # 1. Solve with Quantum-Enhanced Benders with Merge
    print("\n" + "=" * 80)
    print(" METHOD 1: QUANTUM-ENHANCED BENDERS WITH MERGE")
    print("=" * 80)
    
    # Create the optimizer with quantum capabilities
    optimizer = Qoptimizer(complexity_level=complexity_level)
    optimizer.load_food_data()
    
    start_time = time.time()
    qem_result = optimizer.optimize_with_quantum_benders_merge()
    runtime = time.time() - start_time
    qem_result.runtime = runtime
    
    print(f"Status: {qem_result.status}")
    print(f"Objective Value: {qem_result.objective_value:.4f}")
    print(f"Runtime: {runtime:.2f} seconds")
    
    # Print solution summary
    if qem_result.solution:
        print("\nLand Allocation:")
        for (farm, food), area in qem_result.solution.items():
            print(f"{farm}: {food} - {area:.2f} hectares")
    else:
        print("\nWARNING: No valid solution found.")
    
    # Store results and log to file
    all_results = {}
    all_results['Quantum-Enhanced-Merge'] = {
        'result': qem_result,
        'objective_value': qem_result.objective_value if qem_result.objective_value else 0.0,
        'runtime': runtime,
        'benders_data': qem_result.benders_data
    }
    
    log_file = os.path.join(results_dir, "quantum_enhanced_merge_results.txt")
    log_optimization_results(qem_result, optimizer.farms, optimizer.parameters, 
                           'quantum_enhanced_merge', 'Quantum-Enhanced Benders with Merge', log_file)
    
    # Plot Quantum-Enhanced-Merge solution
    try:
        plot_solution(qem_result, optimizer.farms, optimizer.foods, optimizer.parameters, save_path=solution_path_qem)
        print(f"Quantum-Enhanced-Merge solution visualization saved to: {solution_path_qem}")
    except Exception as e:
        print(f"WARNING: Failed to generate Quantum-Enhanced-Merge solution visualization: {str(e)}")
    
    # 2. Solve with Recursive QAOA
    print("\n" + "=" * 80)
    print(" METHOD 2: RECURSIVE QAOA")
    print("=" * 80)
    
    # Set recursion threshold
    recursion_threshold = 10  # Adjust based on expected problem size
    
    try:
        start_time = time.time()
        # Use our new recursive_qaoa_merge method in the Qoptimizer class
        rqaoa_result = optimizer.optimize_with_recursive_qaoa_merge(recursion_threshold=recursion_threshold)
        runtime = time.time() - start_time
        rqaoa_result.runtime = runtime
        
        print(f"Status: {rqaoa_result.status}")
        print(f"Objective Value: {rqaoa_result.objective_value:.4f}")
        print(f"Runtime: {runtime:.2f} seconds")
        
        # Print solution summary
        if rqaoa_result.solution:
            print("\nLand Allocation:")
            for (farm, food), area in rqaoa_result.solution.items():
                print(f"{farm}: {food} - {area:.2f} hectares")
        else:
            print("\nWARNING: No valid solution found.")
    except AttributeError as e:
        if "optimize_with_recursive_qaoa_merge" in str(e):
            print(f"\nERROR: The optimizer does not have the optimize_with_recursive_qaoa_merge method.")
            print("Make sure you're using the latest version of the Qoptimizer class with the recursive QAOA implementation.")
            # Create a dummy result for consistent processing
            from src.data_models import OptimizationResult
            rqaoa_result = OptimizationResult(
                solution={},
                objective_value=0.0,
                status="error",
                runtime=0.0,
                metrics={}
            )
            runtime = 0.0
        else:
            raise
    except Exception as e:
        print(f"\nERROR: Failed to run Recursive QAOA: {str(e)}")
        # Create a dummy result for consistent processing
        from src.data_models import OptimizationResult
        rqaoa_result = OptimizationResult(
            solution={},
            objective_value=0.0,
            status="error",
            runtime=0.0,
            metrics={}
        )
        runtime = 0.0
    
    # Store results and log to file
    all_results['Recursive-QAOA'] = {
        'result': rqaoa_result,
        'objective_value': rqaoa_result.objective_value if rqaoa_result.objective_value else 0.0,
        'runtime': runtime
    }
    
    log_file = os.path.join(results_dir, "recursive_qaoa_results.txt")
    log_optimization_results(rqaoa_result, optimizer.farms, optimizer.parameters, 
                           'recursive_qaoa', 'Recursive QAOA', log_file)
    
    # Plot Recursive QAOA solution
    try:
        plot_solution(rqaoa_result, optimizer.farms, optimizer.foods, optimizer.parameters, save_path=solution_path_rqaoa)
        print(f"Recursive QAOA solution visualization saved to: {solution_path_rqaoa}")
    except Exception as e:
        print(f"WARNING: Failed to generate Recursive QAOA solution visualization: {str(e)}")
    
    # Print summary comparing all methods
    print("\n" + "=" * 80)
    print(" COMPARISON OF QUANTUM OPTIMIZATION METHODS")
    print("=" * 80)
    
    # Print objective values
    print(f"Objective values (maximization):")
    print(f"  Quantum-Enhanced-Merge:  {all_results['Quantum-Enhanced-Merge']['objective_value']:.4f}")
    print(f"  Recursive-QAOA:          {all_results['Recursive-QAOA']['objective_value']:.4f}")
    
    # Print runtimes
    print(f"\nRuntimes:")
    print(f"  Quantum-Enhanced-Merge:  {all_results['Quantum-Enhanced-Merge']['runtime']:.2f} seconds")
    print(f"  Recursive-QAOA:          {all_results['Recursive-QAOA']['runtime']:.2f} seconds")
    
    # Generate visualization comparing methods
    method_comparison_path = os.path.join(plots_dir, "method_comparison.png")
    plot_method_comparison(all_results, method_comparison_path)
    
    print(f"\nResults have been saved to: {results_dir}")
    print(f"Method comparison plot: {method_comparison_path}")
    print(f"Individual solution plots:")
    print(f"  Quantum-Enhanced-Merge: {solution_path_qem}")
    print(f"  Recursive-QAOA: {solution_path_rqaoa}")
    print("\nDone!")

if __name__ == "__main__":
    main()
