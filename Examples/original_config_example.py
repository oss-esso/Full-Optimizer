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

def plot_convergence_comparison(results, save_path=None):
    """Plot a comparison of convergence between all Benders variants."""
    fig = plt.figure(figsize=(14, 10))
    
    colors = {
        'Benders': ('b', 'r'), 
        'Quantum-Enhanced': ('c', 'y'),
        'Quantum-Enhanced-Merge': ('g', 'm'),
        'PuLP': ('k', '#8c564b')
    }
    
    # Plot all bounds (convert to maximization by negating)
    for method, (lb_color, ub_color) in colors.items():
        if method in results and 'benders_data' in results[method]:
            benders_data = results[method]['benders_data']
            
            if 'lower_bounds' in benders_data and benders_data['lower_bounds']:
                lb = [-x for x in benders_data['lower_bounds']]
                plt.plot(range(1, len(lb) + 1), lb, color=lb_color, 
                         marker='o', markersize=5, label=f'{method} Lower Bound')
            
            if 'upper_bounds' in benders_data and benders_data['upper_bounds']:
                ub = [-x for x in benders_data['upper_bounds']]
                plt.plot(range(1, len(ub) + 1), ub, color=ub_color, 
                         marker='s', markersize=5, label=f'{method} Upper Bound')
    
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Objective Value (Maximized)', fontsize=14)
    plt.title('Convergence Comparison of Benders Variants', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig) # Close the figure

def plot_method_comparison(results, save_path=None):
    """Plot a comparison of the results from all optimization methods."""
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    plt.subplots_adjust(hspace=0.4)
    
    # List of all methods
    methods = ['PuLP', 'Benders', 'Quantum-Enhanced', 'Quantum-Enhanced-Merge']
    
    # Extract objective values
    objective_values = [results[method]['objective_value'] if method in results else 0 for method in methods]
    runtimes = [results[method]['runtime'] if method in results else 0 for method in methods]

    # Colors for bars
    colors = ['skyblue', 'lightgreen', 'lightsalmon', 'plum']
    
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
            
            # Specific debug for PuLP plot saving
            is_pulp_plot_placeholder = "pulp_solution" in os.path.basename(save_path)
            if is_pulp_plot_placeholder:
                plot_logger.info(f"DEBUG: PLOT_SOLUTION Attempting to save PuLP PLACEHOLDER plot to {save_path}")
            
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
            if is_pulp_plot_placeholder:
                plot_logger.info(f"DEBUG: PLOT_SOLUTION Successfully saved PuLP PLACEHOLDER plot to {save_path}")
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
        # if len(axes[1,1].containers) > 0:
        #     axes[1,1].bar_label(axes[1,1].containers[0], fmt='%.1f%%',
        #                    padding=2, fontsize=9)
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

        is_pulp_plot = "pulp_solution" in os.path.basename(save_path)
        if is_pulp_plot:
            plot_logger.info(f"DEBUG: PLOT_SOLUTION Attempting to save ACTUAL PuLP plot to {save_path}")
        
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        if is_pulp_plot:
            plot_logger.info(f"DEBUG: PLOT_SOLUTION Successfully saved ACTUAL PuLP plot to {save_path}")
        plot_logger.info(f"Solution plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig) # Close the figure

def main():
    """Run optimization using the original configuration from OQI_benders_vs_pulp.py."""
    print("\n" + "=" * 80)
    print(" FOOD PRODUCTION OPTIMIZATION - ORIGINAL CONFIGURATION")
    print("=" * 80)

    complexity_level = 'intermediate'
    
    # Create results directory for text files
    results_dir = os.path.join(os.path.dirname(current_dir), "Results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create plots directory for PDFs
    plots_dir = os.path.join(os.path.dirname(current_dir), "Plots for PPT")
    os.makedirs(plots_dir, exist_ok=True)

    # Initialize solution paths
    solution_path_pulp = os.path.join(plots_dir, "pulp_solution.png")
    solution_path_benders = os.path.join(plots_dir, "benders_solution.png")
    solution_path_qe = os.path.join(plots_dir, "quantum_enhanced_solution.png")
    solution_path_qem = os.path.join(plots_dir, "quantum_enhanced_merge_solution.png")

    # Load food data from Excel
    print("\nLoading food data from Excel...")
    
    # Create the optimizer with the loaded data
    print("\nCreating food production optimizer with original configuration...")
    optimizer = SimpleFoodOptimizer(complexity_level=complexity_level)
    optimizer.load_food_data()
    # Dictionary to store results from all methods
    all_results = {}
    
    # 1. Solve with PuLP (using consistent naming)
    print("\n" + "=" * 80)
    print(" METHOD 1: PULP DIRECT SOLVER")
    print("=" * 80)
    
    
    # Solve with PuLP
    start_time_pulp = time.time()
    pulp_result = optimizer.solve('pulp')
    runtime_pulp = time.time() - start_time_pulp
    
    # Update runtime
    pulp_result.runtime = runtime_pulp
    
    # Save PuLP results
    log_file_pulp = os.path.join(plots_dir, "pulp_results.txt")
    log_optimization_results(pulp_result, optimizer.farms, optimizer.parameters, 
                           'pulp', 'PuLP Direct Solver', log_file_pulp)
    
    # Plot PuLP solution
    plot_solution(pulp_result, optimizer.farms, optimizer.foods, optimizer.parameters, save_path=solution_path_pulp)
    
    # Store results and log to file
    all_results['PuLP'] = {
        'result': pulp_result,
        'objective_value': pulp_result.objective_value if pulp_result.objective_value else 0.0,
        'runtime': runtime_pulp
    }
    
    # 2. Solve with classical Benders decomposition
    print("\n" + "=" * 80)
    print(" METHOD 2: CLASSICAL BENDERS DECOMPOSITION")
    print("=" * 80)
    
    start_time = time.time()
    benders_result = optimizer.solve('benders')
    runtime = time.time() - start_time
    benders_result.runtime = runtime
    
    print(f"Status: {benders_result.status}")
    print(f"Objective Value: {benders_result.objective_value:.4f}")
    print(f"Runtime: {runtime:.2f} seconds")
    
    # Print solution summary
    if benders_result.solution:
        print("\nLand Allocation:")
        for (farm, food), area in benders_result.solution.items():
            print(f"{farm}: {food} - {area:.2f} hectares")
    else:
        print("\nWARNING: No valid solution found.")
    
    # Store results and log to file
    all_results['Benders'] = {
        'result': benders_result,
        'objective_value': benders_result.objective_value if benders_result.objective_value else 0.0,
        'runtime': runtime,
        'benders_data': benders_result.benders_data if hasattr(benders_result, 'benders_data') else {}
    }
    
    log_file = os.path.join(results_dir, "benders_results.txt")
    log_optimization_results(benders_result, optimizer.farms, optimizer.parameters, 
                           'benders', 'Benders Decomposition', log_file)
    
    # Plot Benders solution
    solution_path_benders = os.path.join(plots_dir, "benders_solution.png")
    try:
        plot_solution(benders_result, optimizer.farms, optimizer.foods, optimizer.parameters, save_path=solution_path_benders)
        print(f"Benders solution visualization saved to: {solution_path_benders}")
    except Exception as e:
        print(f"WARNING: Failed to generate Benders solution visualization: {str(e)}")


    from src.Qoptimizer import SimpleFoodOptimizer as Qoptimizer

    optimizer = Qoptimizer(complexity_level=complexity_level)
    optimizer.load_food_data()

    
    # 3. Solve with quantum-enhanced Benders
    print("\n" + "=" * 80)
    print(" METHOD 3: QUANTUM-ENHANCED BENDERS")
    print("=" * 80)
    
    start_time = time.time()
    qe_result = optimizer.optimize_with_quantum_benders()
    runtime = time.time() - start_time
    qe_result.runtime = runtime
    
    print(f"Status: {qe_result.status}")
    print(f"Objective Value: {qe_result.objective_value:.4f}")
    print(f"Runtime: {runtime:.2f} seconds")
    
    # Print solution summary
    if qe_result.solution:
        print("\nLand Allocation:")
        for (farm, food), area in qe_result.solution.items():
            print(f"{farm}: {food} - {area:.2f} hectares")
    else:
        print("\nWARNING: No valid solution found.")
    
    # Store results and log to file
    all_results['Quantum-Enhanced'] = {
        'result': qe_result,
        'objective_value': qe_result.objective_value if qe_result.objective_value else 0.0,
        'runtime': runtime,
        'benders_data': qe_result.benders_data
    }
    
    log_file = os.path.join(results_dir, "quantum_enhanced_results.txt")
    log_optimization_results(qe_result, optimizer.farms, optimizer.parameters, 
                           'quantum_enhanced', 'Quantum-Enhanced Benders', log_file)
    
    # Plot Quantum-Enhanced solution
    solution_path_qe = os.path.join(plots_dir, "quantum_enhanced_solution.png")
    try:
        plot_solution(qe_result, optimizer.farms, optimizer.foods, optimizer.parameters, save_path=solution_path_qe)
        print(f"Quantum-Enhanced solution visualization saved to: {solution_path_qe}")
    except Exception as e:
        print(f"WARNING: Failed to generate Quantum-Enhanced solution visualization: {str(e)}")
    
    # Print quantum metrics if available
    if qe_result.benders_data and 'quantum_metrics' in qe_result.benders_data:
        metrics = qe_result.benders_data['quantum_metrics']
        print("\nQuantum-Enhanced Metrics:")
        print(f"QUBO Variables: {metrics.get('num_variables', 'N/A')}")
        print(f"QUBO Matrix Density: {metrics.get('matrix_density', 'N/A')}%")
        print(f"Original binary variables: {metrics.get('original_vars', 'N/A')}")
    
    # 4. Solve with quantum-enhanced merge Benders
    print("\n" + "=" * 80)
    print(" METHOD 4: QUANTUM-ENHANCED BENDERS WITH MERGE")
    print("=" * 80)
    
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
    solution_path_qem = os.path.join(plots_dir, "quantum_enhanced_merge_solution.png")
    try:
        plot_solution(qem_result, optimizer.farms, optimizer.foods, optimizer.parameters, save_path=solution_path_qem)
        print(f"Quantum-Enhanced-Merge solution visualization saved to: {solution_path_qem}")
    except Exception as e:
        print(f"WARNING: Failed to generate Quantum-Enhanced-Merge solution visualization: {str(e)}")
    
    # Print summary comparing all methods
    print("\n" + "=" * 80)
    print(" COMPARISON OF ALL OPTIMIZATION METHODS")
    print("=" * 80)
    
    # Print objective values
    print(f"Objective values (maximization):")
    print(f"  PuLP:                   {all_results['PuLP']['objective_value']:.4f}")
    print(f"  Classical Benders:       {all_results['Benders']['objective_value']:.4f}")
    print(f"  Quantum-Enhanced:        {all_results['Quantum-Enhanced']['objective_value']:.4f}")
    print(f"  Quantum-Enhanced-Merge:  {all_results['Quantum-Enhanced-Merge']['objective_value']:.4f}")
    
    # Print runtimes
    print(f"\nRuntimes:")
    print(f"  PuLP:                   {all_results['PuLP']['runtime']:.2f} seconds")
    print(f"  Classical Benders:       {all_results['Benders']['runtime']:.2f} seconds")
    print(f"  Quantum-Enhanced:        {all_results['Quantum-Enhanced']['runtime']:.2f} seconds")
    print(f"  Quantum-Enhanced-Merge:  {all_results['Quantum-Enhanced-Merge']['runtime']:.2f} seconds")
    
    # Generate visualization comparing all methods
    method_comparison_path = os.path.join(plots_dir, "method_comparison.png")
    plot_method_comparison(all_results, method_comparison_path)
    
    # Generate convergence comparison for Benders variants
    convergence_comparison_path = os.path.join(plots_dir, "convergence_comparison.png")
    plot_convergence_comparison(all_results, convergence_comparison_path)
    
    print(f"\nResults have been saved to: {results_dir}")
    print(f"Method comparison plot: {method_comparison_path}")
    print(f"Convergence comparison plot: {convergence_comparison_path}")
    print(f"Individual solution plots:")
    print(f"  PuLP: {solution_path_pulp}")
    print(f"  Classical Benders: {solution_path_benders}")
    print(f"  Quantum-Enhanced: {solution_path_qe}")
    print(f"  Quantum-Enhanced-Merge: {solution_path_qem}")
    print("\nDone!")
    

if __name__ == "__main__":
    main() 
