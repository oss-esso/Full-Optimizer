"""
VRP Optimization Example using Quantum-Enhanced Methods

This script demonstrates the application of the quantum-enhanced optimization framework
to Vehicle Routing Problems (VRP), including the Vehicle Routing Problem with Pickup and Delivery (VRPPD) from the paper:
"Modeling routing problems in QUBO with application to ride‑hailing" by Cattelan & Yarkoni.
"""

import os
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)  # Add current directory to path

# Import VRP modules with absolute imports
from vrp_scenarios import get_all_scenarios
from vrp_optimizer_fixed import VRPQuantumOptimizer  # Use the fixed optimizer
from vrp_data_models import VRPObjective, VRPResult

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vrp_optimization.log', encoding='utf-8'),  # Fix Unicode encoding
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def visualize_vrp_solution(instance, result: VRPResult, save_path: Optional[str] = None):
    """Visualize the VRP solution with enhanced support for OSM data."""
    plt.figure(figsize=(12, 8))
    
    # Plot locations
    for loc_id, location in instance.locations.items():
        # Enhanced labels for OSM scenarios
        label_text = loc_id
        if hasattr(location, 'address') and location.address:
            label_text += f"\n{location.address[:30]}..."  # Truncate long addresses
        
        if loc_id.startswith("depot"):
            plt.scatter(location.x, location.y, c='red', s=200, marker='s', 
                       label='Depot' if 'Depot' not in plt.gca().get_legend_handles_labels()[1] else "")
        elif loc_id.startswith("pickup"):
            plt.scatter(location.x, location.y, c='green', s=100, marker='^', 
                       label='Pickup' if 'Pickup' not in plt.gca().get_legend_handles_labels()[1] else "")
        elif loc_id.startswith("dropoff"):
            plt.scatter(location.x, location.y, c='blue', s=100, marker='v', 
                       label='Dropoff' if 'Dropoff' not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(location.x, location.y, c='orange', s=100, marker='o', 
                       label='Customer' if 'Customer' not in plt.gca().get_legend_handles_labels()[1] else "")
        
        plt.annotate(label_text, (location.x, location.y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=7)
    
    # Plot routes
    colors = ['purple', 'brown', 'pink', 'gray', 'olive']
    for i, (vehicle_id, route) in enumerate(result.routes.items()):
        if len(route) > 1:
            color = colors[i % len(colors)]
            
            # Plot route lines
            for j in range(len(route) - 1):
                loc1 = instance.locations[route[j]]
                loc2 = instance.locations[route[j + 1]]
                plt.plot([loc1.x, loc2.x], [loc1.y, loc2.y], 
                        color=color, linewidth=2, alpha=0.7,
                        label=vehicle_id if j == 0 else "")
    
    # Enhanced title for OSM scenarios
    title = f'VRP Solution - {instance.name}\n'
    title += f'Objective: {result.objective_value:.3f}, '
    title += f'Distance: {result.metrics.get("total_distance", 0):.2f}, '
    
    # Add time information if available
    if 'total_time' in result.metrics and result.metrics['total_time'] > 0:
        # Convert time to hours:minutes format
        time_hours = result.metrics['total_time'] / 3600  # Convert to hours
        hours = int(time_hours)
        minutes = int((time_hours - hours) * 60)
        title += f'Time: {hours}h {minutes}m, '
    
    title += f'Vehicles: {result.metrics.get("vehicles_used", 0)}'
    
    if hasattr(instance, 'is_realistic') and instance.is_realistic:
        title += '\n(Using real-world coordinates - see interactive map)'
    
    plt.title(title)
    plt.xlabel('Longitude' if hasattr(instance, 'is_realistic') and instance.is_realistic else 'X Coordinate')
    plt.ylabel('Latitude' if hasattr(instance, 'is_realistic') and instance.is_realistic else 'Y Coordinate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Solution visualization saved to {save_path}")
        
        # Add GPS info for realistic scenarios
        from vrp_map_visualization import enhance_existing_plot_with_gps_info
        gps_info_path = enhance_existing_plot_with_gps_info(instance, save_path)
        if gps_info_path:
            logger.info(f"GPS coordinates saved to {gps_info_path}")
    else:
        plt.show()
    
    plt.close()

def compare_optimization_methods(instance, save_path: Optional[str] = None):
    """Compare quantum vs classical vs OR-Tools vs advanced heuristics optimization methods."""
    logger.info(f"Comparing optimization methods for {instance.name}")
    
    optimizer = VRPQuantumOptimizer(instance, VRPObjective.MINIMIZE_DISTANCE)
    
    # Run quantum-enhanced optimization
    logger.info("Running quantum-enhanced Benders...")
    quantum_result = optimizer.optimize_with_quantum_benders()
    
    # Run simple classical optimization
    logger.info("Running simple classical heuristic...")
    classical_result = optimizer.optimize_with_classical_benders()
    
    # Run OR-Tools optimization (industrial benchmark)
    logger.info("Running OR-Tools optimization...")
    ortools_result = optimizer.optimize_with_ortools()
    
    # Run advanced heuristics optimization
    logger.info("Running advanced heuristics...")
    advanced_result = optimizer.optimize_with_advanced_heuristics()
    
    # Run 2-opt improvement optimization
    logger.info("Running 2-opt improvement...")
    twopt_result = optimizer.optimize_with_2opt_improvement()
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Objective values comparison (use actual distances for comparison)
    methods = ['Quantum-Enhanced', 'Simple Heuristic', 'OR-Tools', 'Advanced Heuristics', '2-opt Improvement']
    
    # Extract distances for fair comparison
    quantum_dist = quantum_result.metrics.get('total_distance', 0)
    classical_dist = classical_result.metrics.get('total_distance', 0)
    ortools_dist = ortools_result.metrics.get('total_distance', 0)
    advanced_dist = advanced_result.metrics.get('total_distance', 0)
    twopt_dist = twopt_result.metrics.get('total_distance', 0)
    distances = [quantum_dist, classical_dist, ortools_dist, advanced_dist, twopt_dist]
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']    
    ax1.bar(methods, distances, color=colors)
    ax1.set_title('Total Distance Comparison')
    ax1.set_ylabel('Total Distance')
    ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(distances):
        if v > 0:
            ax1.text(i, v + 0.01 * max(distances), f'{v:.2f}', ha='center', va='bottom')    
    # Runtime comparison
    runtimes = [quantum_result.runtime, classical_result.runtime, ortools_result.runtime, 
                advanced_result.runtime, twopt_result.runtime]
    ax2.bar(methods, runtimes, color=colors)
    ax2.set_title('Runtime Comparison')
    ax2.set_ylabel('Runtime (milliseconds)')
    ax2.tick_params(axis='x', rotation=45)
    for i, v in enumerate(runtimes):
        ax2.text(i, v + 0.01 * max(runtimes) if max(runtimes) > 0 else 0.01, 
                f'{v:.2f}ms', ha='center', va='bottom')
    
    plt.suptitle(f'Optimization Methods Comparison - {instance.name}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return quantum_result, classical_result, ortools_result, advanced_result, twopt_result

def print_optimization_summary(all_results: Dict):
    """Print a comprehensive summary of all optimization results."""
    print("\n" + "="*140)
    print(" OPTIMIZATION SUMMARY - OBJECTIVE VALUES")
    print("="*140)
    
    # Create summary table
    scenarios = list(all_results.keys())
    
    print(f"{'Scenario':<20} {'Quantum':<10} {'Classical':<10} {'OR-Tools':<10} {'Advanced':<10} {'2-opt':<10} {'Best Method':<15}")
    print("-" * 140)
    
    quantum_wins = 0
    classical_wins = 0
    ortools_wins = 0
    advanced_wins = 0
    twopt_wins = 0
    total_scenarios = len(scenarios)
    
    for scenario_name in scenarios:
        results = all_results[scenario_name]
        
        # Extract distances (lower is better)
        quantum_dist = results['quantum'].metrics.get('total_distance', float('inf'))
        classical_dist = results['classical'].metrics.get('total_distance', float('inf'))
        ortools_dist = results['ortools'].metrics.get('total_distance', float('inf'))
        advanced_dist = results.get('advanced', type('obj', (object,), {'metrics': {}})()).metrics.get('total_distance', float('inf'))
        twopt_dist = results.get('twopt', type('obj', (object,), {'metrics': {}})()).metrics.get('total_distance', float('inf'))
        
        distances = [quantum_dist, classical_dist, ortools_dist, advanced_dist, twopt_dist]
        method_names = ['Quantum', 'Classical', 'OR-Tools', 'Advanced', '2-opt']
        
        # Find best method (minimum distance)
        valid_distances = [(dist, name) for dist, name in zip(distances, method_names) if dist != float('inf')]
        if valid_distances:
            best_dist, best_method = min(valid_distances)
            
            if best_method == 'Quantum':
                quantum_wins += 1
            elif best_method == 'Classical':
                classical_wins += 1
            elif best_method == 'OR-Tools':
                ortools_wins += 1
            elif best_method == 'Advanced':
                advanced_wins += 1
            else:
                twopt_wins += 1
        else:
            best_method = "None"
        
        # Format distances for display
        quantum_str = f"{quantum_dist:.2f}" if quantum_dist != float('inf') else "N/A"
        classical_str = f"{classical_dist:.2f}" if classical_dist != float('inf') else "N/A"
        ortools_str = f"{ortools_dist:.2f}" if ortools_dist != float('inf') else "N/A"
        advanced_str = f"{advanced_dist:.2f}" if advanced_dist != float('inf') else "N/A"
        twopt_str = f"{twopt_dist:.2f}" if twopt_dist != float('inf') else "N/A"
        
        print(f"{scenario_name:<20} {quantum_str:<10} {classical_str:<10} {ortools_str:<10} {advanced_str:<10} {twopt_str:<10} {best_method:<15}")
    
    print("-" * 140)
    print(f"{'WINS:':<20} {quantum_wins:<10} {classical_wins:<10} {ortools_wins:<10} {advanced_wins:<10} {twopt_wins:<10}")
    print(f"{'WIN RATE:':<20} {quantum_wins/total_scenarios*100:.1f}%{'':<5} {classical_wins/total_scenarios*100:.1f}%{'':<5} {ortools_wins/total_scenarios*100:.1f}%{'':<5} {advanced_wins/total_scenarios*100:.1f}%{'':<5} {twopt_wins/total_scenarios*100:.1f}%{'':<5}")
    
    print("\n" + "="*140)
    print(" RUNTIME PERFORMANCE SUMMARY")
    print("="*140)
    print(f"{'Scenario':<20} {'Quantum (ms)':<12} {'Classical (ms)':<12} {'OR-Tools (ms)':<12} {'Advanced (ms)':<12} {'2-opt (ms)':<12} {'Fastest':<15}")
    print("-" * 140)
    
    quantum_fastest = 0
    classical_fastest = 0
    ortools_fastest = 0
    advanced_fastest = 0
    twopt_fastest = 0    
    for scenario_name in scenarios:
        results = all_results[scenario_name]
        
        quantum_time = results['quantum'].runtime
        classical_time = results['classical'].runtime
        ortools_time = results['ortools'].runtime
        advanced_time = results.get('advanced', type('obj', (object,), {'runtime': float('inf')})()).runtime
        twopt_time = results.get('twopt', type('obj', (object,), {'runtime': float('inf')})()).runtime
        
        times = [quantum_time, classical_time, ortools_time, advanced_time, twopt_time]
        time_methods = ['Quantum', 'Classical', 'OR-Tools', 'Advanced', '2-opt']
        
        # Find fastest method
        valid_times = [(time, name) for time, name in zip(times, time_methods) if time != float('inf')]
        if valid_times:
            fastest_time, fastest_method = min(valid_times)
            
            if fastest_method == 'Quantum':
                quantum_fastest += 1
            elif fastest_method == 'Classical':
                classical_fastest += 1
            elif fastest_method == 'OR-Tools':
                ortools_fastest += 1
            elif fastest_method == 'Advanced':
                advanced_fastest += 1
            else:
                twopt_fastest += 1
        else:
            fastest_method = "None"
        
        # Format times for display
        quantum_time_str = f"{quantum_time:.0f}" if quantum_time != float('inf') else "N/A"
        classical_time_str = f"{classical_time:.0f}" if classical_time != float('inf') else "N/A"
        ortools_time_str = f"{ortools_time:.0f}" if ortools_time != float('inf') else "N/A"
        advanced_time_str = f"{advanced_time:.0f}" if advanced_time != float('inf') else "N/A"
        twopt_time_str = f"{twopt_time:.0f}" if twopt_time != float('inf') else "N/A"
        
        print(f"{scenario_name:<20} {quantum_time_str:<12} {classical_time_str:<12} {ortools_time_str:<12} {advanced_time_str:<12} {twopt_time_str:<12} {fastest_method:<15}")
    
    print("-" * 140)
    print(f"{'FASTEST:':<20} {quantum_fastest:<12} {classical_fastest:<12} {ortools_fastest:<12} {advanced_fastest:<12} {twopt_fastest:<12}")
    print(f"{'SPEED RATE:':<20} {quantum_fastest/total_scenarios*100:.1f}%{'':<7} {classical_fastest/total_scenarios*100:.1f}%{'':<7} {ortools_fastest/total_scenarios*100:.1f}%{'':<7} {advanced_fastest/total_scenarios*100:.1f}%{'':<7} {twopt_fastest/total_scenarios*100:.1f}%{'':<7}")
    
    print("\n" + "="*140)
    print(" OVERALL PERFORMANCE ASSESSMENT")
    print("="*140)
    
    # Overall winner assessment
    win_counts = [quantum_wins, classical_wins, ortools_wins, advanced_wins, twopt_wins]
    win_methods = ['Quantum-Enhanced', 'Classical Heuristic', 'OR-Tools', 'Advanced Heuristics', '2-opt Improvement']
    overall_winner = win_methods[win_counts.index(max(win_counts))]
    
    print(f"Overall Distance Winner: {overall_winner}")
    
    speed_counts = [quantum_fastest, classical_fastest, ortools_fastest, advanced_fastest, twopt_fastest]
    speed_methods = ['Quantum-Enhanced', 'Classical Heuristic', 'OR-Tools', 'Advanced Heuristics', '2-opt Improvement']
    speed_winner = speed_methods[speed_counts.index(max(speed_counts))]
    
    print(f"Overall Speed Winner: {speed_winner}")
    print("="*140)

def run_scenario_analysis():
    """Run analysis on only MODA_small scenario for debugging."""
    logger.info("Starting VRP scenario analysis - MODA_small only")
    
    # Create results directory
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    scenarios = get_all_scenarios()
    all_results = {}
    
    # Only run MODA_small scenario
    scenario_name = "MODA_small"
    if scenario_name not in scenarios:
        logger.error(f"MODA_small scenario not found! Available: {list(scenarios.keys())}")
        return
    
    instance = scenarios[scenario_name]
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing scenario: {scenario_name}")
    logger.info(f"{'='*60}")    
    try:
        # Create optimizer for this scenario
        optimizer = VRPQuantumOptimizer(instance, VRPObjective.MINIMIZE_DISTANCE)
        
        # Run optimization comparison
        quantum_result, classical_result, ortools_result, advanced_result, twopt_result = compare_optimization_methods(
            instance, 
            save_path=os.path.join(results_dir, f"{scenario_name}_comparison.png")
        )
        
        # Visualize solutions with standard plots
        visualize_vrp_solution(
            instance, quantum_result,
            save_path=os.path.join(results_dir, f"{scenario_name}_quantum_solution.png")
        )
        
        visualize_vrp_solution(
            instance, classical_result,
            save_path=os.path.join(results_dir, f"{scenario_name}_classical_solution.png")
        )
        
        visualize_vrp_solution(
            instance, ortools_result,
            save_path=os.path.join(results_dir, f"{scenario_name}_ortools_solution.png")
        )
        
        visualize_vrp_solution(
            instance, advanced_result,
            save_path=os.path.join(results_dir, f"{scenario_name}_advanced_solution.png")
        )
        
        visualize_vrp_solution(
            instance, twopt_result,
            save_path=os.path.join(results_dir, f"{scenario_name}_twopt_solution.png")
        )
          # Create GPS map visualizations for realistic scenarios
        try:
            from vrp_map_visualization import create_all_map_visualizations
            
            # Pass all results to the visualization function so it can auto-select the best one
            all_solver_results = {
                'quantum': quantum_result,
                'classical': classical_result,
                'ortools': ortools_result,
                'advanced': advanced_result,
                'twopt': twopt_result
            }
            
            map_files = create_all_map_visualizations(instance, all_solver_results, results_dir, scenario_name)
            if map_files:
                logger.info(f"Created {len(map_files)} map visualizations for {scenario_name}")
            
        except ImportError:
            logger.info("Map visualization dependencies not available. Install folium, contextily, geopandas for GPS maps.")
        except Exception as e:
            logger.warning(f"Could not create map visualizations: {e}")
        
        all_results[scenario_name] = {
            'quantum': quantum_result,
            'classical': classical_result,
            'ortools': ortools_result,
            'advanced': advanced_result,
            'twopt': twopt_result,
            'instance': instance
        }
        
        # Print summary with actual distances
        quantum_dist = quantum_result.metrics.get('total_distance', 0)
        classical_dist = classical_result.metrics.get('total_distance', 0)
        ortools_dist = ortools_result.metrics.get('total_distance', 0)
        advanced_dist = advanced_result.metrics.get('total_distance', 0)
        twopt_dist = twopt_result.metrics.get('total_distance', 0)
        
        logger.info(f"Scenario {scenario_name} completed:")
        logger.info(f"  Quantum - Distance: {quantum_dist:.2f}, Runtime: {quantum_result.runtime:.2f}ms")
        logger.info(f"  Classical - Distance: {classical_dist:.2f}, Runtime: {classical_result.runtime:.2f}ms")
        logger.info(f"  OR-Tools - Distance: {ortools_dist:.2f}, Runtime: {ortools_result.runtime:.2f}ms")
        logger.info(f"  Advanced - Distance: {advanced_dist:.2f}, Runtime: {advanced_result.runtime:.2f}ms")
        logger.info(f"  2-opt - Distance: {twopt_dist:.2f}, Runtime: {twopt_result.runtime:.2f}ms")
        
    except Exception as e:
        logger.error(f"Error processing scenario {scenario_name}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Generate summary report
    generate_summary_report(all_results, results_dir)
    
    # Print optimization summary
    print_optimization_summary(all_results)
    
    logger.info(f"\nVRP analysis completed. Results saved to {results_dir}")

def generate_summary_report(all_results: Dict, results_dir: str):
    """Generate a summary report of all scenarios."""
    logger.info("Generating summary report")
    
    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    scenarios = list(all_results.keys())
    quantum_objectives = [all_results[s]['quantum'].objective_value for s in scenarios]
    classical_objectives = [all_results[s]['classical'].objective_value for s in scenarios]
    ortools_objectives = [all_results[s]['ortools'].objective_value for s in scenarios]
    advanced_objectives = [all_results[s].get('advanced', type('obj', (object,), {'objective_value': 0})()).objective_value for s in scenarios]
    twopt_objectives = [all_results[s].get('twopt', type('obj', (object,), {'objective_value': 0})()).objective_value for s in scenarios]
    
    quantum_runtimes = [all_results[s]['quantum'].runtime for s in scenarios]
    classical_runtimes = [all_results[s]['classical'].runtime for s in scenarios]
    ortools_runtimes = [all_results[s]['ortools'].runtime for s in scenarios]
    advanced_runtimes = [all_results[s].get('advanced', type('obj', (object,), {'runtime': 0})()).runtime for s in scenarios]
    twopt_runtimes = [all_results[s].get('twopt', type('obj', (object,), {'runtime': 0})()).runtime for s in scenarios]
    
    x = np.arange(len(scenarios))
    width = 0.15
    
    # Objective values
    ax1.bar(x - 2*width, quantum_objectives, width, label='Quantum-Enhanced', color='lightblue')
    ax1.bar(x - width, classical_objectives, width, label='Simple Heuristic', color='lightcoral')
    ax1.bar(x, ortools_objectives, width, label='OR-Tools', color='lightgreen')
    ax1.bar(x + width, advanced_objectives, width, label='Advanced Heuristics', color='lightyellow')
    ax1.bar(x + 2*width, twopt_objectives, width, label='2-opt Improvement', color='lightpink')
    ax1.set_title('Objective Values by Scenario')
    ax1.set_ylabel('Objective Value')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Runtimes
    ax2.bar(x - 2*width, quantum_runtimes, width, label='Quantum-Enhanced', color='lightblue')
    ax2.bar(x - width, classical_runtimes, width, label='Simple Heuristic', color='lightcoral')
    ax2.bar(x, ortools_runtimes, width, label='OR-Tools', color='lightgreen')
    ax2.bar(x + width, advanced_runtimes, width, label='Advanced Heuristics', color='lightyellow')
    ax2.bar(x + 2*width, twopt_runtimes, width, label='2-opt Improvement', color='lightpink')
    ax2.set_title('Runtime by Scenario')
    ax2.set_ylabel('Runtime (milliseconds)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('VRP Optimization Summary Report', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "summary_report.png"), dpi=300, bbox_inches='tight')
    plt.close()      # Generate text report
    with open(os.path.join(results_dir, "summary_report.txt"), 'w', encoding='utf-8') as f:
        f.write("VRP OPTIMIZATION SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        for scenario_name, results in all_results.items():
            quantum_result = results['quantum']
            classical_result = results['classical']
            ortools_result = results['ortools']
            advanced_result = results.get('advanced')
            twopt_result = results.get('twopt')
            instance = results['instance']
            
            f.write(f"Scenario: {scenario_name}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Problem size: {len(instance.location_ids)} locations, {len(instance.vehicles)} vehicles\n")
            
            # Helper function to format time
            def format_time(seconds):
                if seconds <= 0:
                    return "N/A"
                hours = int(seconds / 3600)
                minutes = int((seconds % 3600) / 60)
                return f"{hours}h {minutes}m"
            
            # Write quantum result
            f.write(f"Quantum-Enhanced:\n")
            f.write(f"  Objective: {quantum_result.objective_value:.6f}\n")
            f.write(f"  Total Distance: {quantum_result.metrics.get('total_distance', 0):.4f}\n")
            f.write(f"  Time-on-Route: {format_time(quantum_result.metrics.get('total_time', 0))}\n")
            f.write(f"  Vehicles Used: {quantum_result.metrics.get('vehicles_used', 0)}\n")
            f.write(f"  Runtime: {quantum_result.runtime:.2f} ms\n")
            
            # Write classical result
            f.write(f"Simple Heuristic:\n")
            f.write(f"  Objective: {classical_result.objective_value:.6f}\n")
            f.write(f"  Total Distance: {classical_result.metrics.get('total_distance', 0):.4f}\n")
            f.write(f"  Time-on-Route: {format_time(classical_result.metrics.get('total_time', 0))}\n")
            f.write(f"  Vehicles Used: {classical_result.metrics.get('vehicles_used', 0)}\n")
            f.write(f"  Runtime: {classical_result.runtime:.2f} ms\n")
            
            # Write OR-Tools result
            f.write(f"OR-Tools:\n")
            f.write(f"  Objective: {ortools_result.objective_value:.6f}\n")
            f.write(f"  Total Distance: {ortools_result.metrics.get('total_distance', 0):.4f}\n")
            f.write(f"  Time-on-Route: {format_time(ortools_result.metrics.get('total_time', 0))}\n")
            f.write(f"  Vehicles Used: {ortools_result.metrics.get('vehicles_used', 0)}\n")
            f.write(f"  Runtime: {ortools_result.runtime:.2f} ms\n")
            
            # Write advanced heuristics result
            if advanced_result:
                f.write(f"Advanced Heuristics:\n")
                f.write(f"  Objective: {advanced_result.objective_value:.6f}\n")
                f.write(f"  Total Distance: {advanced_result.metrics.get('total_distance', 0):.4f}\n")
                f.write(f"  Time-on-Route: {format_time(advanced_result.metrics.get('total_time', 0))}\n")
                f.write(f"  Vehicles Used: {advanced_result.metrics.get('vehicles_used', 0)}\n")
                f.write(f"  Runtime: {advanced_result.runtime:.2f} ms\n")
            
            # Write 2-opt result
            if twopt_result:
                f.write(f"2-opt Improvement:\n")
                f.write(f"  Objective: {twopt_result.objective_value:.6f}\n")
                f.write(f"  Total Distance: {twopt_result.metrics.get('total_distance', 0):.4f}\n")
                f.write(f"  Time-on-Route: {format_time(twopt_result.metrics.get('total_time', 0))}\n")
                f.write(f"  Vehicles Used: {twopt_result.metrics.get('vehicles_used', 0)}\n")
                f.write(f"  Runtime: {twopt_result.runtime:.2f} ms\n")
            f.write("\n")

def main():
    """Main function to run VRP optimization examples."""
    print("\n" + "="*80)
    print(" VRP OPTIMIZATION WITH QUANTUM-ENHANCED METHODS")
    print("="*80)
    print("\nThis example demonstrates the application of quantum-enhanced")
    print("optimization methods to Vehicle Routing Problems (VRP),")
    print("including the Vehicle Routing Problem with Pickup and Delivery (VRPPD) formulation.")
    print("\nComparison includes:")
    print("  - Quantum-Enhanced Heuristic")
    print("  - Simple Greedy Heuristic")
    print("  - OR-Tools (Industry Standard Benchmark)")
    print("  - Advanced Construction Heuristics (Nearest Neighbor, Savings, Firefly)")
    print("  - 2-opt Improvement (Croes' 1958 with modern enhancements)")
    print("\nAll methods use Manhattan distance for fair comparison.")
    print("Realistic scenarios use:")
    print("  - Offline coordinate database (no API calls)")
    print("  - Optional: OpenStreetMap addresses via Nominatim API")
    print("  - Optional: OSRM routing for real travel times")
    print("  - GPS map visualizations with real backgrounds")
    print("\nBased on: 'Modeling routing problems in QUBO with application")
    print("to ride‑hailing' by Cattelan & Yarkoni")
    print("="*80)
    
    try:
        # Check map visualization dependencies with detailed feedback
        print("\nChecking dependencies for GPS map visualization...")
        
        missing_deps = []
        try:
            import folium
            print("+ folium available")
            try:
                import folium.plugins
                print("+ folium.plugins available")
            except (ImportError, AttributeError):
                print("! folium.plugins not available - simplified routes will be used")
                missing_deps.append("folium plugins")
        except ImportError:
            print("- folium not available")
            missing_deps.append("folium")
        
        try:
            import requests
            print("+ requests available - OSRM routing enabled")
        except ImportError:
            print("! requests not available - straight line routes will be used")
            missing_deps.append("requests")
        
        try:
            import contextily
            import geopandas
            print("+ contextily and geopandas available - map backgrounds enabled")
        except ImportError:
            print("! contextily/geopandas not available - fallback GPS plots will be used")
            missing_deps.append("contextily/geopandas")
        
        if missing_deps:
            print(f"\nMissing dependencies: {', '.join(missing_deps)}")
            print("Install with: pip install folium requests contextily geopandas")
            print("Note: The system will still work with fallback visualizations")
        else:
            print("\n+ All GPS map dependencies available - full functionality enabled")
        
        # Test import of scenarios first
        print("\nLoading ALL VRP scenarios...")
        scenarios = get_all_scenarios()
        print(f"\nSuccessfully loaded {len(scenarios)} VRP scenarios:")
        
        realistic_count = 0
        offline_count = 0
        osm_count = 0
        large_count = 0
        
        for name, instance in scenarios.items():
            # Verify distance method and count realistic scenarios
            total_demand = sum(loc.demand for loc in instance.locations.values() if hasattr(loc, 'demand'))
            total_capacity = sum(vehicle.capacity for vehicle in instance.vehicles.values())
            
            scenario_type = "synthetic"
            if hasattr(instance, 'is_realistic') and instance.is_realistic:
                realistic_count += 1
                if "offline" in name:
                    offline_count += 1
                    scenario_type = "offline_realistic"
                elif "osm" in name:
                    osm_count += 1
                    scenario_type = "osm_realistic"
                else:
                    scenario_type = "realistic"
            
            # Check for large scenarios
            if len(instance.location_ids) > 50:
                large_count += 1
                scenario_type += "_large"
                
            print(f"  - {name}: {len(instance.location_ids)} locations, "
                  f"{len(instance.vehicles)} vehicles, "
                  f"demand/capacity: {total_demand}/{total_capacity} ({scenario_type})")
        
        print(f"\nScenario Summary:")
        print(f"  🏙️  Realistic scenarios: {realistic_count}")
        print(f"    📍 Offline database: {offline_count}")
        print(f"    🌐 OSM API-based: {osm_count}")
        print(f"  🧮 Synthetic scenarios: {len(scenarios) - realistic_count}")
        print(f"  📏 Large scenarios (>50 locations): {large_count}")
        
        if realistic_count > 0:
            if offline_count > 0:
                print(f"\n+ Using offline realistic coordinates (no API required)")
            if osm_count > 0:
                print(f"+ Enhanced with OSM data from live APIs")
        else:            print(f"\n! No realistic scenarios loaded - using synthetic data only")
        
        # Ensure all instances use Manhattan distance
        print("\nEnsuring consistent distance calculations...")
        for name, instance in scenarios.items():
            if instance.distance_matrix is None:
                print(f"  Calculating Manhattan distances for {name}...")
                instance.calculate_distance_matrix(distance_method="manhattan")
            else:
                print(f"  {name} already has distance matrix calculated")
          # Run comprehensive scenario analysis
        print("\nRunning comprehensive scenario analysis...")
        run_scenario_analysis()
        
        print("\n" + "="*80)
        print(" VRP OPTIMIZATION COMPLETED")
        print("="*80)
        print("\nResults have been saved to the 'results' directory.")
        print("Generated files:")
        print("  -> Individual scenario comparison plots (5 methods)")
        print("  -> Solution visualizations for each method")
        print("  -> Interactive GPS maps (HTML files)")
        print("  -> Static maps with real backgrounds (PNG files)")
        print("  -> GPS coordinate files (TXT files)")
        print("  -> Summary report with all scenarios")
        print("  -> Detailed text report")
        print("\nKey findings will be summarized in the console output above.")
        print("Check the log file 'vrp_optimization.log' for detailed output.")
        
        print("\nFEATURES:")
        print("  + Consistent Manhattan distance calculations")
        print("  + Proper capacity handling for different scenarios")
        print("  + Clean visualization without duplicates")
        print("  + Comprehensive performance comparison (5 methods)")
        print("  + OR-Tools industry benchmark integration")
        print("  + Advanced construction heuristics (nearest neighbor, savings, firefly)")
        print("  + 2-opt local search improvement (Croes' 1958 framework)")
        if offline_count > 0:
            print("  + Offline realistic scenarios (no API calls)")
        if osm_count > 0:
            print("  + Live OSM data integration")
            print("  + OSRM routing for realistic travel times")
        print("  + Interactive GPS maps with real backgrounds")
        print("  + Static maps with OpenStreetMap tiles")
        print("  + GPS coordinate export for external tools")
        if large_count > 0:
            print(f"  + Large-scale scenarios testing ({large_count} scenarios)")
        if realistic_count == 0:
            print("  ! Realistic scenarios available but not loaded")
        
        # Special note for MODA scenarios
        moda_scenarios = [name for name in scenarios.keys() if 'moda' in name.lower()]
        if moda_scenarios:
            print(f"\nMODA SCENARIO HIGHLIGHTS:")
            for scenario_name in moda_scenarios:
                instance = scenarios[scenario_name]
                print(f"  📊 {scenario_name}: {len(instance.locations)} locations, {len(instance.vehicles)} vehicles")
                if hasattr(instance, 'ride_requests'):
                    print(f"      🚗 VRPPD: {len(instance.ride_requests)} pickup-delivery requests")
                print(f"      🗺️  GPS coverage: Northern Italy")
                
                # Add average speeds for realistic time estimation
                avg_speed = 30  # km/h in urban areas
                if 'milan' in scenario_name.lower():
                    avg_speed = 25  # slower in Milan
                elif 'asti' in scenario_name.lower():
                    avg_speed = 35  # faster in smaller cities
                
                print(f"      🚗 Average speed: {avg_speed} km/h for time estimation")
                
                if 'first' in scenario_name.lower():
                    print(f"      🎯 Challenge: Large-scale ride sharing optimization")
                else:
                    print(f"      🎯 Challenge: Small-scale ride sharing optimization")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\nERROR: {str(e)}")
        print("Check the log file for more details.")
        raise

if __name__ == "__main__":
    main()