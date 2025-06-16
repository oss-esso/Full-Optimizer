"""
VRP Optimization Example using Quantum-Enhanced Methods

This script demonstrates the application of the quantum-enhanced optimization framework
to Vehicle Routing Problems (VRP), including the Ride Pooling Problem (RPP) from the paper:
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
    """Compare quantum vs classical optimization methods including pyVRP."""
    logger.info(f"Comparing optimization methods for {instance.name}")
    
    optimizer = VRPQuantumOptimizer(instance, VRPObjective.MINIMIZE_DISTANCE)
    
    # Run quantum-enhanced optimization
    logger.info("Running quantum-enhanced Benders...")
    quantum_result = optimizer.optimize_with_quantum_benders()
    
    # Run pyVRP classical optimization
    logger.info("Running pyVRP classical solver...")
    pyvrp_result = optimizer.optimize_with_pyvrp_classical()
    
    # Run simple classical optimization
    logger.info("Running simple classical heuristic...")
    classical_result = optimizer.optimize_with_classical_benders()
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Objective values comparison (use actual distances for comparison)
    methods = ['Quantum-Enhanced', 'pyVRP Classical', 'Simple Heuristic']
    
    # Extract distances for fair comparison
    quantum_dist = quantum_result.metrics.get('total_distance', 0)
    pyvrp_dist = pyvrp_result.metrics.get('pyvrp_distance', pyvrp_result.metrics.get('total_distance', 0))
    classical_dist = classical_result.metrics.get('total_distance', 0)
    distances = [quantum_dist, pyvrp_dist, classical_dist]
    
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    ax1.bar(methods, distances, color=colors)
    ax1.set_title('Total Distance Comparison')
    ax1.set_ylabel('Total Distance')
    ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(distances):
        if v > 0:
            ax1.text(i, v + 0.01 * max(distances), f'{v:.2f}', ha='center', va='bottom')
    
    # Runtime comparison
    runtimes = [quantum_result.runtime, pyvrp_result.runtime, classical_result.runtime]
    ax2.bar(methods, runtimes, color=colors)
    ax2.set_title('Runtime Comparison')
    ax2.set_ylabel('Runtime (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    for i, v in enumerate(runtimes):
        ax2.text(i, v + 0.01 * max(runtimes) if max(runtimes) > 0 else 0.01, 
                f'{v:.2f}s', ha='center', va='bottom')
    
    # Cost efficiency (distance per unit time)
    efficiencies = []
    for dist, runtime in zip(distances, runtimes):
        if runtime > 0 and dist > 0:
            efficiencies.append(dist / runtime)  # Lower is better
        else:
            efficiencies.append(float('inf'))
    
    ax3.bar(methods, [1/e if e != float('inf') else 0 for e in efficiencies], color=colors)
    ax3.set_title('Solution Efficiency (Higher = Better)')
    ax3.set_ylabel('Inverse Distance/Time')
    ax3.tick_params(axis='x', rotation=45)
    
    # Vehicles used comparison
    quantum_vehicles = quantum_result.metrics.get('vehicles_used', 0)
    pyvrp_vehicles = pyvrp_result.metrics.get('pyvrp_num_routes', pyvrp_result.metrics.get('vehicles_used', 0))
    classical_vehicles = classical_result.metrics.get('vehicles_used', 0)
    vehicles = [quantum_vehicles, pyvrp_vehicles, classical_vehicles]
    
    ax4.bar(methods, vehicles, color=colors)
    ax4.set_title('Vehicles Used Comparison')
    ax4.set_ylabel('Number of Vehicles')
    ax4.tick_params(axis='x', rotation=45)
    for i, v in enumerate(vehicles):
        ax4.text(i, v + 0.01 * max(vehicles) if max(vehicles) > 0 else 0.01, 
                f'{int(v)}', ha='center', va='bottom')
    
    plt.suptitle(f'Optimization Methods Comparison - {instance.name}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return quantum_result, pyvrp_result, classical_result

def print_optimization_summary(all_results: Dict):
    """Print a comprehensive summary of all optimization results."""
    print("\n" + "="*80)
    print(" OPTIMIZATION SUMMARY - OBJECTIVE VALUES")
    print("="*80)
    
    # Create summary table
    scenarios = list(all_results.keys())
    
    print(f"{'Scenario':<20} {'Quantum':<12} {'pyVRP':<12} {'Classical':<12} {'Best Method':<15}")
    print("-" * 80)
    
    quantum_wins = 0
    pyvrp_wins = 0
    classical_wins = 0
    total_scenarios = len(scenarios)
    
    for scenario_name in scenarios:
        results = all_results[scenario_name]
        
        # Extract distances (lower is better)
        quantum_dist = results['quantum'].metrics.get('total_distance', float('inf'))
        pyvrp_dist = results['pyvrp'].metrics.get('pyvrp_distance', 
                     results['pyvrp'].metrics.get('total_distance', float('inf')))
        classical_dist = results['classical'].metrics.get('total_distance', float('inf'))
        
        distances = [quantum_dist, pyvrp_dist, classical_dist]
        method_names = ['Quantum', 'pyVRP', 'Classical']
        
        # Find best method (minimum distance)
        valid_distances = [(dist, name) for dist, name in zip(distances, method_names) if dist != float('inf')]
        if valid_distances:
            best_dist, best_method = min(valid_distances)
            
            if best_method == 'Quantum':
                quantum_wins += 1
            elif best_method == 'pyVRP':
                pyvrp_wins += 1
            else:
                classical_wins += 1
        else:
            best_method = "None"
        
        # Format distances for display
        quantum_str = f"{quantum_dist:.2f}" if quantum_dist != float('inf') else "N/A"
        pyvrp_str = f"{pyvrp_dist:.2f}" if pyvrp_dist != float('inf') else "N/A"
        classical_str = f"{classical_dist:.2f}" if classical_dist != float('inf') else "N/A"
        
        print(f"{scenario_name:<20} {quantum_str:<12} {pyvrp_str:<12} {classical_str:<12} {best_method:<15}")
    
    print("-" * 80)
    print(f"{'WINS:':<20} {quantum_wins:<12} {pyvrp_wins:<12} {classical_wins:<12}")
    print(f"{'WIN RATE:':<20} {quantum_wins/total_scenarios*100:.1f}%{'':<7} {pyvrp_wins/total_scenarios*100:.1f}%{'':<7} {classical_wins/total_scenarios*100:.1f}%{'':<7}")
    
    print("\n" + "="*80)
    print(" RUNTIME PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"{'Scenario':<20} {'Quantum (s)':<12} {'pyVRP (s)':<12} {'Classical (s)':<12} {'Fastest':<15}")
    print("-" * 80)
    
    quantum_fastest = 0
    pyvrp_fastest = 0
    classical_fastest = 0
    
    for scenario_name in scenarios:
        results = all_results[scenario_name]
        
        quantum_time = results['quantum'].runtime
        pyvrp_time = results['pyvrp'].runtime
        classical_time = results['classical'].runtime
        
        times = [quantum_time, pyvrp_time, classical_time]
        time_methods = ['Quantum', 'pyVRP', 'Classical']
        
        fastest_time, fastest_method = min(zip(times, time_methods))
        
        if fastest_method == 'Quantum':
            quantum_fastest += 1
        elif fastest_method == 'pyVRP':
            pyvrp_fastest += 1
        else:
            classical_fastest += 1
        
        print(f"{scenario_name:<20} {quantum_time:<12.2f} {pyvrp_time:<12.2f} {classical_time:<12.2f} {fastest_method:<15}")
    
    print("-" * 80)
    print(f"{'FASTEST:':<20} {quantum_fastest:<12} {pyvrp_fastest:<12} {classical_fastest:<12}")
    print(f"{'SPEED RATE:':<20} {quantum_fastest/total_scenarios*100:.1f}%{'':<7} {pyvrp_fastest/total_scenarios*100:.1f}%{'':<7} {classical_fastest/total_scenarios*100:.1f}%{'':<7}")
    
    print("\n" + "="*80)
    print(" TIME-ON-ROUTE PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"{'Scenario':<20} {'Quantum (h:m)':<15} {'pyVRP (h:m)':<15} {'Classical (h:m)':<15} {'Fastest':<15}")
    print("-" * 80)
    
    quantum_shortest_time = 0
    pyvrp_shortest_time = 0
    classical_shortest_time = 0
    
    for scenario_name in scenarios:
        results = all_results[scenario_name]
        
        # Extract times and format as hours:minutes
        def format_time(time_seconds):
            if time_seconds <= 0:
                return "N/A"
            hours = int(time_seconds / 3600)
            minutes = int((time_seconds % 3600) / 60)
            return f"{hours}:{minutes:02d}"
        
        quantum_time = results['quantum'].metrics.get('total_time', 0)
        pyvrp_time = results['pyvrp'].metrics.get('total_time', 0)
        classical_time = results['classical'].metrics.get('total_time', 0)
        
        times = [quantum_time, pyvrp_time, classical_time]
        time_methods = ['Quantum', 'pyVRP', 'Classical']
        
        # Find shortest route time (ignoring zeros)
        valid_times = [(t, m) for t, m in zip(times, time_methods) if t > 0]
        
        if valid_times:
            shortest_time, shortest_method = min(valid_times)
            
            if shortest_method == 'Quantum':
                quantum_shortest_time += 1
            elif shortest_method == 'pyVRP':
                pyvrp_shortest_time += 1
            else:
                classical_shortest_time += 1
        else:
            shortest_method = "N/A"
        
        print(f"{scenario_name:<20} {format_time(quantum_time):<15} {format_time(pyvrp_time):<15} {format_time(classical_time):<15} {shortest_method:<15}")
    
    print("-" * 80)
    print(f"{'SHORTEST TIME:':<20} {quantum_shortest_time:<15} {pyvrp_shortest_time:<15} {classical_shortest_time:<15}")
    
    if total_scenarios > 0:
        print(f"{'EFFICIENCY RATE:':<20} {quantum_shortest_time/total_scenarios*100:.1f}%{'':<10} {pyvrp_shortest_time/total_scenarios*100:.1f}%{'':<10} {classical_shortest_time/total_scenarios*100:.1f}%{'':<10}")
    
    print("\n" + "="*80)
    print(" OVERALL PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Calculate average improvements
    quantum_vs_pyvrp = []
    quantum_vs_classical = []
    pyvrp_vs_classical = []
    
    for scenario_name in scenarios:
        results = all_results[scenario_name]
        
        quantum_dist = results['quantum'].metrics.get('total_distance', 0)
        pyvrp_dist = results['pyvrp'].metrics.get('pyvrp_distance', 
                     results['pyvrp'].metrics.get('total_distance', 0))
        classical_dist = results['classical'].metrics.get('total_distance', 0)
        
        if pyvrp_dist > 0:
            quantum_vs_pyvrp.append((pyvrp_dist - quantum_dist) / pyvrp_dist * 100)
        if classical_dist > 0:
            quantum_vs_classical.append((classical_dist - quantum_dist) / classical_dist * 100)
        if classical_dist > 0:
            pyvrp_vs_classical.append((classical_dist - pyvrp_dist) / classical_dist * 100)
    
    print("Average Distance Improvements (positive = better):")
    if quantum_vs_pyvrp:
        print(f"  Quantum vs pyVRP:     {np.mean(quantum_vs_pyvrp):+6.2f}%")
    if quantum_vs_classical:
        print(f"  Quantum vs Classical:  {np.mean(quantum_vs_classical):+6.2f}%")
    if pyvrp_vs_classical:
        print(f"  pyVRP vs Classical:    {np.mean(pyvrp_vs_classical):+6.2f}%")
    
    # Calculate average runtimes
    avg_quantum_time = np.mean([results['quantum'].runtime for results in all_results.values()])
    avg_pyvrp_time = np.mean([results['pyvrp'].runtime for results in all_results.values()])
    avg_classical_time = np.mean([results['classical'].runtime for results in all_results.values()])
    
    print(f"\nAverage Runtimes:")
    print(f"  Quantum:    {avg_quantum_time:.2f} seconds")
    print(f"  pyVRP:      {avg_pyvrp_time:.2f} seconds")
    print(f"  Classical:  {avg_classical_time:.2f} seconds")
    
    # Overall recommendation
    print(f"\n{'RECOMMENDATION:':<20}")
    if quantum_wins >= pyvrp_wins and quantum_wins >= classical_wins:
        print("  → Quantum-Enhanced method shows best solution quality")
    elif pyvrp_wins >= quantum_wins and pyvrp_wins >= classical_wins:
        print("  → pyVRP shows best solution quality (industry standard)")
    else:
        print("  → Classical heuristic shows best solution quality (simple approach)")
    
    if quantum_fastest >= pyvrp_fastest and quantum_fastest >= classical_fastest:
        print("  → Quantum-Enhanced method is fastest")
    elif pyvrp_fastest >= quantum_fastest and pyvrp_fastest >= classical_fastest:
        print("  → pyVRP is fastest")
    else:
        print("  → Classical heuristic is fastest")
    
    print("="*80)

def run_scenario_analysis():
    """Run analysis on all VRP scenarios."""
    logger.info("Starting VRP scenario analysis")
    
    # Create results directory
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    scenarios = get_all_scenarios()
    all_results = {}
    
    for scenario_name, instance in scenarios.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing scenario: {scenario_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Create optimizer for this scenario
            optimizer = VRPQuantumOptimizer(instance, VRPObjective.MINIMIZE_DISTANCE)
            
            # Run optimization comparison
            quantum_result, pyvrp_result, classical_result = compare_optimization_methods(
                instance, 
                save_path=os.path.join(results_dir, f"{scenario_name}_comparison.png")
            )
            
            # Visualize solutions with standard plots
            visualize_vrp_solution(
                instance, quantum_result,
                save_path=os.path.join(results_dir, f"{scenario_name}_quantum_solution.png")
            )
            
            visualize_vrp_solution(
                instance, pyvrp_result,
                save_path=os.path.join(results_dir, f"{scenario_name}_pyvrp_solution.png")
            )
            
            visualize_vrp_solution(
                instance, classical_result,
                save_path=os.path.join(results_dir, f"{scenario_name}_classical_solution.png")
            )
              # Create GPS map visualizations for realistic scenarios
            try:
                from vrp_map_visualization import create_all_map_visualizations
                
                # For ride pooling scenarios, prefer classical solution for maps since pyVRP
                # doesn't properly enforce ride pooling constraints
                is_ride_pooling = bool(instance.ride_requests)
                if is_ride_pooling:
                    best_result = classical_result
                    logger.info(f"Using classical solution for {scenario_name} map (ride pooling constraints)")
                else:
                    # For non-ride pooling scenarios, use the solution with lowest distance
                    best_result = min([quantum_result, pyvrp_result, classical_result], 
                                    key=lambda r: r.metrics.get('total_distance', float('inf')))
                    logger.info(f"Using best distance solution for {scenario_name} map")
                
                map_files = create_all_map_visualizations(instance, best_result, results_dir, scenario_name)
                if map_files:
                    logger.info(f"Created {len(map_files)} map visualizations for {scenario_name}")
                
                
            except ImportError:
                logger.info("Map visualization dependencies not available. Install folium, contextily, geopandas for GPS maps.")
            except Exception as e:
                logger.warning(f"Could not create map visualizations: {e}")
            
            all_results[scenario_name] = {
                'quantum': quantum_result,
                'pyvrp': pyvrp_result,
                'classical': classical_result,
                'instance': instance
            }
              # Print summary with actual distances
            quantum_dist = quantum_result.metrics.get('total_distance', 0)
            pyvrp_dist = pyvrp_result.metrics.get('pyvrp_distance', pyvrp_result.metrics.get('total_distance', 0))
            classical_dist = classical_result.metrics.get('total_distance', 0)
            
            logger.info(f"Scenario {scenario_name} completed:")
            logger.info(f"  Quantum - Distance: {quantum_dist:.2f}, Runtime: {quantum_result.runtime:.2f}s")
            
            # Add warning for pyVRP if constraints aren't enforced
            pyvrp_warning = pyvrp_result.metrics.get('pyvrp_warning', '')
            if pyvrp_warning:
                logger.info(f"  pyVRP - Distance: {pyvrp_dist:.2f}, Runtime: {pyvrp_result.runtime:.2f}s, Cost: {pyvrp_result.metrics.get('pyvrp_cost', 0)} [WARNING: {pyvrp_warning}]")
            else:
                logger.info(f"  pyVRP - Distance: {pyvrp_dist:.2f}, Runtime: {pyvrp_result.runtime:.2f}s, Cost: {pyvrp_result.metrics.get('pyvrp_cost', 0)}")
            
            logger.info(f"  Classical - Distance: {classical_dist:.2f}, Runtime: {classical_result.runtime:.2f}s")
            
            
        except Exception as e:
            logger.error(f"Error processing scenario {scenario_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # Generate summary report
    generate_summary_report(all_results, results_dir)
    
    # Print optimization summary
    print_optimization_summary(all_results)
    
    logger.info(f"\nVRP analysis completed. Results saved to {results_dir}")

def generate_summary_report(all_results: Dict, results_dir: str):
    """Generate a summary report of all scenarios."""
    logger.info("Generating summary report")
    
    # Create summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    scenarios = list(all_results.keys())
    quantum_objectives = [all_results[s]['quantum'].objective_value for s in scenarios]
    pyvrp_objectives = [all_results[s]['pyvrp'].objective_value for s in scenarios]
    classical_objectives = [all_results[s]['classical'].objective_value for s in scenarios]
    quantum_runtimes = [all_results[s]['quantum'].runtime for s in scenarios]
    pyvrp_runtimes = [all_results[s]['pyvrp'].runtime for s in scenarios]
    classical_runtimes = [all_results[s]['classical'].runtime for s in scenarios]
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    # Objective values
    ax1.bar(x - width, quantum_objectives, width, label='Quantum-Enhanced', color='lightblue')
    ax1.bar(x, pyvrp_objectives, width, label='pyVRP Classical', color='lightgreen')
    ax1.bar(x + width, classical_objectives, width, label='Simple Heuristic', color='lightcoral')
    ax1.set_title('Objective Values by Scenario')
    ax1.set_ylabel('Objective Value')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Runtimes
    ax2.bar(x - width, quantum_runtimes, width, label='Quantum-Enhanced', color='lightblue')
    ax2.bar(x, pyvrp_runtimes, width, label='pyVRP Classical', color='lightgreen')
    ax2.bar(x + width, classical_runtimes, width, label='Simple Heuristic', color='lightcoral')
    ax2.set_title('Runtime by Scenario')
    ax2.set_ylabel('Runtime (seconds)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Improvement ratios (Quantum vs pyVRP)
    improvement_ratios = []
    for s in scenarios:
        quantum_obj = all_results[s]['quantum'].objective_value
        pyvrp_obj = all_results[s]['pyvrp'].objective_value
        if pyvrp_obj != 0:
            ratio = (quantum_obj - pyvrp_obj) / abs(pyvrp_obj) * 100
        else:
            ratio = 0
        improvement_ratios.append(ratio)
    
    colors = ['green' if r > 0 else 'red' for r in improvement_ratios]
    ax3.bar(scenarios, improvement_ratios, color=colors, alpha=0.7)
    ax3.set_title('Quantum vs pyVRP Improvement (%)')
    ax3.set_ylabel('Improvement (%)')
    ax3.set_xticklabels(scenarios, rotation=45)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    
    # Problem characteristics
    problem_sizes = []
    for s in scenarios:
        instance = all_results[s]['instance']
        problem_sizes.append(len(instance.location_ids))
    
    ax4.scatter(problem_sizes, quantum_objectives, label='Quantum-Enhanced', color='blue', alpha=0.7)
    ax4.scatter(problem_sizes, pyvrp_objectives, label='pyVRP Classical', color='green', alpha=0.7)
    ax4.scatter(problem_sizes, classical_objectives, label='Simple Heuristic', color='red', alpha=0.7)
    ax4.set_title('Objective vs Problem Size')
    ax4.set_xlabel('Number of Locations')
    ax4.set_ylabel('Objective Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('VRP Optimization Summary Report', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "summary_report.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate text report
    with open(os.path.join(results_dir, "summary_report.txt"), 'w', encoding='utf-8') as f:
        f.write("VRP OPTIMIZATION SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        for scenario_name, results in all_results.items():
            quantum_result = results['quantum']
            pyvrp_result = results['pyvrp']
            classical_result = results['classical']
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
            f.write(f"  Runtime: {quantum_result.runtime:.4f} seconds\n")
            
            # Write pyVRP result
            f.write(f"pyVRP Classical:\n")
            f.write(f"  Objective: {pyvrp_result.objective_value:.6f}\n")
            f.write(f"  Total Distance: {pyvrp_result.metrics.get('total_distance', 0):.4f}\n")
            f.write(f"  Time-on-Route: {format_time(pyvrp_result.metrics.get('total_time', 0))}\n")
            f.write(f"  Vehicles Used: {pyvrp_result.metrics.get('vehicles_used', 0)}\n")
            f.write(f"  Runtime: {pyvrp_result.runtime:.4f} seconds\n")
            
            # Write classical result
            f.write(f"Simple Heuristic:\n")
            f.write(f"  Objective: {classical_result.objective_value:.6f}\n")
            f.write(f"  Total Distance: {classical_result.metrics.get('total_distance', 0):.4f}\n")
            f.write(f"  Time-on-Route: {format_time(classical_result.metrics.get('total_time', 0))}\n")
            f.write(f"  Vehicles Used: {classical_result.metrics.get('vehicles_used', 0)}\n")
            f.write(f"  Runtime: {classical_result.runtime:.4f} seconds\n")
            f.write("\n")

def main():
    """Main function to run VRP optimization examples."""
    print("\n" + "="*80)
    print(" VRP OPTIMIZATION WITH QUANTUM-ENHANCED METHODS")
    print("="*80)
    print("\nThis example demonstrates the application of quantum-enhanced")
    print("optimization methods to Vehicle Routing Problems (VRP),")
    print("including the Ride Pooling Problem (RPP) formulation.")
    print("\nComparison includes:")
    print("  - Quantum-Enhanced Heuristic")
    print("  - pyVRP Classical Solver (industry standard)")
    print("  - Simple Greedy Heuristic")
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
        else:
            print(f"\n! No realistic scenarios loaded - using synthetic data only")
        
        # Check pyVRP availability
        from vrp_optimizer_fixed import PYVRP_AVAILABLE
        if PYVRP_AVAILABLE:
            print("\n+ pyVRP classical solver available")
        else:
            print("\n! pyVRP not available - install with: pip install pyvrp")
            print("  Some comparisons may be limited")
        
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
        print("  -> Individual scenario comparison plots (3 methods)")
        print("  -> Solution visualizations for each method")
        print("  -> Interactive GPS maps (HTML files)")
        print("  -> Static maps with real backgrounds (PNG files)")
        print("  -> GPS coordinate files (TXT files)")
        print("  -> Summary report with all scenarios")
        print("  -> Detailed text report")
        print("\nKey findings will be summarized in the console output above.")
        print("Check the log file 'vrp_optimization.log' for detailed output.")
        
        print("\nFEATURES:")
        print("  + Using direct Model API approach for pyVRP")
        print("  + Consistent Manhattan distance calculations")
        print("  + Proper capacity handling for different scenarios")
        print("  + Proper route conversion from pyVRP solutions")
        print("  + Clean visualization without duplicates")
        print("  + Comprehensive performance comparison")
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
                    print(f"      🚗 Ride pooling: {len(instance.ride_requests)} ride requests")
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