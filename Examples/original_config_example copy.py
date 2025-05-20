"""
Example script demonstrating food production optimization with the original configuration
using only the PuLP solver.
"""
import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import matplotlib
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

# Import the required modules
from src.scenarios import load_food_data
from src.data_models import OptimizationResult

# Make sure the methods directory is in the Python path
methods_dir = os.path.join(parent_dir, "src", "methods")
sys.path.insert(0, methods_dir)

# Also ensure src directory is in path for direct imports
src_dir = os.path.join(parent_dir, "src")
sys.path.insert(0, src_dir)

# Import PuLP optimizer - fix the import statement to use the correct class name
import pulp_method 
# Let's try directly importing the optimization class that actually exists
# There are a few possibilities for the actual class name:
try:
    # Option 1: Check if LinearProgrammingOptimizer exists
    from pulp_method import LinearProgrammingOptimizer as PulpOptimizer
except ImportError:
    try:
        # Option 2: Check if PuLPSolver exists (note capitalization)
        from pulp_method import PuLPSolver as PulpOptimizer
    except ImportError:
        # Option 3: If neither exists, get the first optimizer class from the module
        # This will raise an error if no optimizer class exists
        for attr_name in dir(pulp_method):
            attr = getattr(pulp_method, attr_name)
            if isinstance(attr, type) and "optim" in attr_name.lower():
                PulpOptimizer = attr
                print(f"Using '{attr_name}' from pulp_method as the optimizer class")
                break
        else:
            # Fallback: use what's documented in the module
            from pulp_method import Optimizer as PulpOptimizer
            print("Using 'Optimizer' from pulp_method")

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
    """Run optimization using only PuLP method."""
    print("\n" + "=" * 80)
    print(" FOOD PRODUCTION OPTIMIZATION - PULP SOLVER")
    print("=" * 80)
    
    # Create results directory for text files
    results_dir = os.path.join(os.path.dirname(current_dir), "Results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create plots directory for PDFs
    plots_dir = os.path.join(os.path.dirname(current_dir), "Plots for PPT")
    os.makedirs(plots_dir, exist_ok=True)

    # Initialize solution path for PuLP
    solution_path_pulp = os.path.join(plots_dir, "pulp_solution.png")

    # Load food data from load_food_data function
    print("\nLoading food data...")
    farms, foods, food_groups, config = load_food_data(complexity_level='simple')
    
    # Create the PulpOptimizer - add error handling
    print("\nCreating PuLP optimizer...")
    try:
        optimizer = PulpOptimizer(farms=farms, foods=foods, food_groups=food_groups, config=config)
    except Exception as e:
        print(f"Error creating optimizer: {e}")
        print("Available classes in pulp_method:")
        for attr_name in dir(pulp_method):
            attr = getattr(pulp_method, attr_name)
            if isinstance(attr, type):
                print(f" - {attr_name}")
        raise
    
    # Dictionary to store results
    all_results = {}
    
    # Solve with PuLP optimization method - add more error handling
    print("\n" + "=" * 80)
    print(" PULP SOLVER")
    print("=" * 80)
    
    # Check what solver methods are available
    solver_method = getattr(optimizer, 'solve', None)
    if solver_method is None:
        # Try alternative method names
        potential_methods = [attr for attr in dir(optimizer) if 'solve' in attr.lower() and callable(getattr(optimizer, attr))]
        if potential_methods:
            print(f"'solve' method not found. Using '{potential_methods[0]}' instead.")
            solver_method = getattr(optimizer, potential_methods[0])
        else:
            raise AttributeError("No solve method found in the optimizer")
    
    # Now use the solver method we found
    start_time_pulp = time.time()
    pulp_result = solver_method()  # Use the method we found
    runtime_pulp = time.time() - start_time_pulp
    
    pulp_result.runtime = runtime_pulp
    
    # Print detailed information about the solution
    print(f"Status: {pulp_result.status}")
    print(f"Objective Value: {pulp_result.objective_value:.4f}")
    print(f"Runtime: {runtime_pulp:.2f} seconds")
    
    # Print solution details
    if pulp_result.solution:
        print("\nSolution Details:")
        for (farm, food), area in pulp_result.solution.items():
            print(f"Farm: {farm}, Food: {food}, Area: {area:.2f} hectares")
        
        print("\nFarm Utilization:")
        for farm in optimizer.farms:
            farm_area = sum(area for (f, _), area in pulp_result.solution.items() if f == farm)
            farm_utilization = farm_area / optimizer.parameters['land_availability'][farm] * 100
            print(f"{farm}: {farm_area:.2f} hectares ({farm_utilization:.2f}%)")
        
        if pulp_result.metrics:
            print("\nMetrics:")
            for metric, value in pulp_result.metrics.items():
                if metric not in [f'utilization_{farm}' for farm in optimizer.farms]:
                    print(f"{metric}: {value:.4f}")
    else:
        print("\nWARNING: No valid solution found.")
    
    # Save PuLP results to text file
    log_file_pulp = os.path.join(plots_dir, "pulp_results.txt")
    log_optimization_results(pulp_result, optimizer.farms, optimizer.parameters, 
                           'pulp', 'PuLP Solver', log_file_pulp)
    print(f"\nDetailed results logged to: {log_file_pulp}")
    
    # Plot PuLP solution
    plot_solution(pulp_result, optimizer.farms, optimizer.foods, optimizer.parameters, save_path=solution_path_pulp)
    print(f"Solution visualization saved to: {solution_path_pulp}")
    
    print("\nPuLP optimization completed.")
    

if __name__ == "__main__":
    main()
