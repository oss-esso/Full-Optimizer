#!/usr/bin/env python3
"""
Comprehensive comparison script that generates detailed reports comparing
PuLP, Basic SA, Enhanced SA, and Benders methods across all metrics,
food selections, area allocations, and performance characteristics.
"""

import sys
import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def main():
    print("=" * 80)
    print("COMPREHENSIVE FOOD PRODUCTION OPTIMIZATION COMPARISON")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        from src.Qoptimizer import SimpleFoodOptimizer
        
        # Create optimizer
        print("Initializing optimizer...")
        optimizer = SimpleFoodOptimizer(complexity_level='simple')
        optimizer.load_food_data()
        
        print(f"Problem Configuration:")
        print(f"  - Farms: {len(optimizer.farms)} ({list(optimizer.farms)})")
        print(f"  - Foods: {len(optimizer.foods)} ({list(optimizer.foods.keys())})")
        print(f"  - Complexity: {optimizer.complexity_level}")
        print()
        
        # Run all methods
        results = run_all_methods(optimizer)
        
        # Generate comprehensive report
        generate_comprehensive_report(optimizer, results)
        
        print("\n" + "=" * 80)
        print("✓ COMPREHENSIVE COMPARISON COMPLETED SUCCESSFULLY!")
        print("✓ Check Results/comprehensive_comparison_report.html for detailed analysis")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")

def run_all_methods(optimizer) -> Dict[str, Any]:
    """Run all optimization methods and collect results."""
    methods = {
        'PuLP': {
            'function': optimizer.optimize_with_pulp,
            'params': {}
        },
        'Basic_SA': {
            'function': optimizer.optimize_with_simulated_annealing_benders,
            'params': {
                'max_iterations': 200,
                'initial_temperature': 100.0,
                'cooling_rate': 0.95,
                'enhanced_sa': False
            }
        },
        'Enhanced_SA': {
            'function': optimizer.optimize_with_simulated_annealing_benders,
            'params': {
                'max_iterations': 200,
                'initial_temperature': 100.0,
                'cooling_rate': 0.95,
                'enhanced_sa': True,
                'adaptive_cooling': True,
                'use_restart': True,
                'neighborhood_type': "multi_flip"
            }
        },
        'Benders': {
            'function': optimizer.optimize_with_benders,
            'params': {}
        }
    }
    
    results = {}
    
    for method_name, method_config in methods.items():
        print(f"\nRunning {method_name}...")
        start_time = time.time()
        
        try:
            result = method_config['function'](**method_config['params'])
            runtime = time.time() - start_time
            
            results[method_name] = {
                'result': result,
                'runtime': runtime,
                'status': 'SUCCESS',
                'error': None
            }
            
            print(f"  ✓ {method_name}: {result.objective_value:.3f} ({runtime:.3f}s)")
            
        except Exception as e:
            runtime = time.time() - start_time
            results[method_name] = {
                'result': None,
                'runtime': runtime,
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"  ✗ {method_name}: ERROR - {e}")
    
    return results

def generate_comprehensive_report(optimizer, results: Dict[str, Any]):
    """Generate a comprehensive HTML report comparing all methods."""
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate all analysis components
    objective_comparison = create_objective_comparison(results)
    runtime_comparison = create_runtime_comparison(results)
    solution_comparison = create_solution_comparison(optimizer, results)
    metrics_comparison = create_metrics_comparison(results)
    food_selection_analysis = create_food_selection_analysis(optimizer, results)
    farm_utilization_analysis = create_farm_utilization_analysis(optimizer, results)
    
    # Create visualizations
    create_comparison_plots(results, results_dir)
    
    # Generate HTML report
    html_content = generate_html_report(
        optimizer, results, objective_comparison, runtime_comparison,
        solution_comparison, metrics_comparison, food_selection_analysis,
        farm_utilization_analysis
    )
    
    # Save HTML report
    report_path = os.path.join(results_dir, 'comprehensive_comparison_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Save JSON data for further analysis
    json_data = prepare_json_data(optimizer, results)
    json_path = os.path.join(results_dir, 'comprehensive_comparison_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    print(f"\nReport saved: {report_path}")
    print(f"Data saved: {json_path}")

def create_objective_comparison(results: Dict[str, Any]) -> pd.DataFrame:
    """Create objective value comparison table."""
    data = []
    
    for method_name, method_data in results.items():
        if method_data['status'] == 'SUCCESS':
            result = method_data['result']
            data.append({
                'Method': method_name,
                'Objective_Value': result.objective_value,
                'Runtime_seconds': method_data['runtime'],
                'Status': result.status,
                'Solution_Count': len(result.solution) if result.solution else 0
            })
        else:
            data.append({
                'Method': method_name,
                'Objective_Value': None,
                'Runtime_seconds': method_data['runtime'],
                'Status': 'ERROR',
                'Solution_Count': 0
            })
    
    df = pd.DataFrame(data)
    
    # Calculate relative performance if we have valid results
    valid_objectives = df[df['Objective_Value'].notna()]['Objective_Value']
    if len(valid_objectives) > 0:
        best_objective = valid_objectives.max()
        df['Relative_Performance_%'] = ((df['Objective_Value'] / best_objective) * 100).round(2)
        df['Gap_from_Best'] = (best_objective - df['Objective_Value']).round(3)
    
    return df

def create_runtime_comparison(results: Dict[str, Any]) -> pd.DataFrame:
    """Create runtime comparison analysis."""
    data = []
    
    for method_name, method_data in results.items():
        data.append({
            'Method': method_name,
            'Runtime_seconds': method_data['runtime'],
            'Runtime_milliseconds': method_data['runtime'] * 1000,
            'Status': method_data['status']
        })
    
    df = pd.DataFrame(data)
    
    # Add speed rankings
    valid_runtimes = df[df['Status'] == 'SUCCESS']['Runtime_seconds']
    if len(valid_runtimes) > 0:
        df['Speed_Rank'] = df['Runtime_seconds'].rank(method='min')
        fastest_time = valid_runtimes.min()
        df['Speed_Ratio'] = (df['Runtime_seconds'] / fastest_time).round(2)
    
    return df

def create_solution_comparison(optimizer, results: Dict[str, Any]) -> Dict[str, Any]:
    """Create detailed solution comparison."""
    comparison = {}
    
    for method_name, method_data in results.items():
        if method_data['status'] == 'SUCCESS' and method_data['result'].solution:
            solution = method_data['result'].solution
            
            # Analyze solution structure
            total_area = sum(solution.values())
            farm_allocations = {}
            food_allocations = {}
            
            for (farm, food), area in solution.items():
                if farm not in farm_allocations:
                    farm_allocations[farm] = 0
                if food not in food_allocations:
                    food_allocations[food] = 0
                
                farm_allocations[farm] += area
                food_allocations[food] += area
            
            comparison[method_name] = {
                'total_area': total_area,
                'farm_allocations': farm_allocations,
                'food_allocations': food_allocations,
                'allocation_pairs': len(solution),
                'farms_used': len(farm_allocations),
                'foods_used': len(food_allocations)
            }
    
    return comparison

def create_metrics_comparison(results: Dict[str, Any]) -> pd.DataFrame:
    """Create metrics comparison table."""
    metrics_data = []
    
    for method_name, method_data in results.items():
        if method_data['status'] == 'SUCCESS' and method_data['result'].metrics:
            metrics = method_data['result'].metrics
            row = {'Method': method_name}
            row.update(metrics)
            metrics_data.append(row)
    
    if not metrics_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(metrics_data)
    
    # Fill NaN values with 0 for numerical columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def create_food_selection_analysis(optimizer, results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze food selection patterns across methods."""
    analysis = {
        'food_frequency': {},  # How often each food is selected
        'method_food_sets': {},  # Set of foods selected by each method
        'common_foods': set(),  # Foods selected by all successful methods
        'unique_foods': {}  # Foods selected only by specific methods
    }
    
    # Initialize food frequency counter
    for food in optimizer.foods.keys():
        analysis['food_frequency'][food] = 0
    
    successful_methods = []
    
    for method_name, method_data in results.items():
        if method_data['status'] == 'SUCCESS' and method_data['result'].solution:
            successful_methods.append(method_name)
            solution = method_data['result'].solution
            
            # Get foods selected by this method
            foods_selected = set()
            for (farm, food), area in solution.items():
                if area > 0.01:  # Threshold for meaningful allocation
                    foods_selected.add(food)
                    analysis['food_frequency'][food] += 1
            
            analysis['method_food_sets'][method_name] = foods_selected
    
    # Find common foods (selected by all methods)
    if successful_methods:
        analysis['common_foods'] = set.intersection(*analysis['method_food_sets'].values())
        
        # Find unique foods for each method
        for method_name in successful_methods:
            method_foods = analysis['method_food_sets'][method_name]
            other_methods_foods = set()
            for other_method in successful_methods:
                if other_method != method_name:
                    other_methods_foods.update(analysis['method_food_sets'][other_method])
            
            analysis['unique_foods'][method_name] = method_foods - other_methods_foods
    
    return analysis

def create_farm_utilization_analysis(optimizer, results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze farm utilization patterns across methods."""
    analysis = {
        'farm_utilization_rates': {},  # Utilization rate per farm per method
        'total_utilization': {},  # Total land utilization per method
        'utilization_efficiency': {}  # How efficiently land is used
    }
    
    # Get land availability
    land_availability = optimizer.parameters.get('land_availability', {})
    
    for method_name, method_data in results.items():
        if method_data['status'] == 'SUCCESS' and method_data['result'].solution:
            solution = method_data['result'].solution
            
            farm_usage = {}
            total_available = 0
            total_used = 0
            
            # Calculate usage per farm
            for farm in optimizer.farms:
                available = land_availability.get(farm, 0)
                used = sum(area for (f, food), area in solution.items() if f == farm)
                
                farm_usage[farm] = {
                    'available': available,
                    'used': used,
                    'utilization_rate': (used / available * 100) if available > 0 else 0
                }
                
                total_available += available
                total_used += used
            
            analysis['farm_utilization_rates'][method_name] = farm_usage
            analysis['total_utilization'][method_name] = {
                'total_available': total_available,
                'total_used': total_used,
                'overall_utilization_rate': (total_used / total_available * 100) if total_available > 0 else 0
            }
    
    return analysis

def create_comparison_plots(results: Dict[str, Any], results_dir: str):
    """Create comprehensive comparison plots."""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Extract data for plotting
    methods = []
    objectives = []
    runtimes = []
    
    for method_name, method_data in results.items():
        if method_data['status'] == 'SUCCESS':
            methods.append(method_name)
            objectives.append(method_data['result'].objective_value)
            runtimes.append(method_data['runtime'])
    
    if not methods:
        print("No successful results to plot")
        return
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(16, 12))
    
    # Objective values comparison
    ax1 = plt.subplot(2, 3, 1)
    bars1 = ax1.bar(methods, objectives, alpha=0.8)
    ax1.set_title('Objective Value Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Objective Value')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, objectives):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(objectives)*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Runtime comparison
    ax2 = plt.subplot(2, 3, 2)
    bars2 = ax2.bar(methods, runtimes, alpha=0.8, color='orange')
    ax2.set_title('Runtime Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Runtime (seconds)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, runtimes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(runtimes)*0.01,
                f'{value:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    # Efficiency scatter plot (Objective vs Runtime)
    ax3 = plt.subplot(2, 3, 3)
    scatter = ax3.scatter(runtimes, objectives, s=100, alpha=0.7, c=range(len(methods)), cmap='viridis')
    for i, method in enumerate(methods):
        ax3.annotate(method, (runtimes[i], objectives[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10)
    ax3.set_xlabel('Runtime (seconds)')
    ax3.set_ylabel('Objective Value')
    ax3.set_title('Efficiency Analysis\n(Higher Left = Better)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Relative performance comparison
    if len(objectives) > 1:
        best_obj = max(objectives)
        relative_perf = [(obj/best_obj)*100 for obj in objectives]
        
        ax4 = plt.subplot(2, 3, 4)
        bars4 = ax4.bar(methods, relative_perf, alpha=0.8, color='green')
        ax4.set_title('Relative Performance\n(% of Best Solution)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Performance (%)')
        ax4.set_ylim(0, 105)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars4, relative_perf):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Speed comparison
    if len(runtimes) > 1:
        fastest = min(runtimes)
        speed_ratios = [runtime/fastest for runtime in runtimes]
        
        ax5 = plt.subplot(2, 3, 5)
        bars5 = ax5.bar(methods, speed_ratios, alpha=0.8, color='red')
        ax5.set_title('Speed Comparison\n(1.0 = Fastest)', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Speed Ratio')
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars5, speed_ratios):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(speed_ratios)*0.01,
                    f'{value:.1f}x', ha='center', va='bottom', fontweight='bold')
    
    # Combined score (normalized objective / normalized runtime)
    if len(objectives) > 1 and len(runtimes) > 1:
        norm_obj = [obj/max(objectives) for obj in objectives]
        norm_runtime = [min(runtimes)/runtime for runtime in runtimes]  # Inverse for speed
        combined_scores = [o * r for o, r in zip(norm_obj, norm_runtime)]
        
        ax6 = plt.subplot(2, 3, 6)
        bars6 = ax6.bar(methods, combined_scores, alpha=0.8, color='purple')
        ax6.set_title('Combined Score\n(Quality × Speed)', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Combined Score')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars6, combined_scores):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(combined_scores)*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'comprehensive_comparison_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved: {plot_path}")

def generate_html_report(optimizer, results, objective_comparison, runtime_comparison,
                        solution_comparison, metrics_comparison, food_selection_analysis,
                        farm_utilization_analysis) -> str:
    """Generate comprehensive HTML report."""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Food Production Optimization - Comprehensive Comparison Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f8f9fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; }}
        h3 {{ color: #2c3e50; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .summary-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .summary-card h3 {{ margin: 0 0 10px 0; font-size: 1.2em; }}
        .summary-card .value {{ font-size: 2em; font-weight: bold; margin: 10px 0; }}
        .summary-card .unit {{ font-size: 0.9em; opacity: 0.9; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #e8f4fd; }}
        .metric-positive {{ color: #27ae60; font-weight: bold; }}
        .metric-negative {{ color: #e74c3c; font-weight: bold; }}
        .status-success {{ color: #27ae60; font-weight: bold; }}
        .status-error {{ color: #e74c3c; font-weight: bold; }}
        .config-box {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 15px 0; }}
        .food-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
        .food-card {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }}
        .farm-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 15px 0; }}
        .farm-card {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #e67e22; }}
        .progress-bar {{ background-color: #ecf0f1; border-radius: 10px; overflow: hidden; margin: 5px 0; }}
        .progress-fill {{ height: 20px; background: linear-gradient(90deg, #3498db, #2ecc71); color: white; text-align: center; line-height: 20px; font-size: 12px; font-weight: bold; }}
        .timestamp {{ text-align: center; color: #7f8c8d; font-style: italic; margin-bottom: 30px; }}
        .section {{ margin: 40px 0; }}
        .highlight {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107; margin: 10px 0; }}
        .image-container {{ text-align: center; margin: 20px 0; }}
        .image-container img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🍃 Food Production Optimization</h1>
        <h1>Comprehensive Comparison Report</h1>
        <div class="timestamp">Generated on {timestamp}</div>
        
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="summary-grid">
"""
    
    # Add summary cards
    successful_methods = [name for name, data in results.items() if data['status'] == 'SUCCESS']
    if successful_methods:
        best_obj = max(results[name]['result'].objective_value for name in successful_methods)
        fastest_method = min(successful_methods, key=lambda x: results[x]['runtime'])
        best_method = max(successful_methods, key=lambda x: results[x]['result'].objective_value)
        
        html += f"""
                <div class="summary-card">
                    <h3>Methods Tested</h3>
                    <div class="value">{len(results)}</div>
                    <div class="unit">{len(successful_methods)} successful</div>
                </div>
                <div class="summary-card">
                    <h3>Best Objective</h3>
                    <div class="value">{best_obj:.2f}</div>
                    <div class="unit">by {best_method}</div>
                </div>
                <div class="summary-card">
                    <h3>Fastest Method</h3>
                    <div class="value">{results[fastest_method]['runtime']:.3f}s</div>
                    <div class="unit">{fastest_method}</div>
                </div>
                <div class="summary-card">
                    <h3>Problem Size</h3>
                    <div class="value">{len(optimizer.farms)} × {len(optimizer.foods)}</div>
                    <div class="unit">farms × foods</div>
                </div>
"""
    
    html += """
            </div>
        </div>
        
        <div class="section">
            <h2>⚙️ Problem Configuration</h2>
            <div class="config-box">
"""
    
    # Add problem configuration
    html += f"""
                <p><strong>Complexity Level:</strong> {optimizer.complexity_level}</p>
                <p><strong>Farms:</strong> {', '.join(optimizer.farms)} ({len(optimizer.farms)} total)</p>
                <p><strong>Foods:</strong> {', '.join(optimizer.foods.keys())} ({len(optimizer.foods)} total)</p>
                <p><strong>Land Availability:</strong>
"""
    
    land_availability = optimizer.parameters.get('land_availability', {})
    for farm, land in land_availability.items():
        html += f" {farm}: {land} ha,"
    html = html.rstrip(',')
    
    html += """
                </p>
            </div>
        </div>
"""
    
    # Add objective comparison table
    if not objective_comparison.empty:
        html += """
        <div class="section">
            <h2>🎯 Objective Value Comparison</h2>
            <table>
                <tr>
                    <th>Method</th>
                    <th>Objective Value</th>
                    <th>Runtime (s)</th>
                    <th>Status</th>
                    <th>Relative Performance</th>
                    <th>Gap from Best</th>
                    <th>Solution Pairs</th>
                </tr>
"""
        
        for _, row in objective_comparison.iterrows():
            status_class = "status-success" if row['Status'] in ['optimal', 'SUCCESS'] else "status-error"
            perf_class = "metric-positive" if pd.notna(row.get('Relative_Performance_%')) and row.get('Relative_Performance_%', 0) >= 95 else ""
            objective_val = f"{row['Objective_Value']:.3f}" if pd.notna(row['Objective_Value']) else 'N/A'
            rel_perf = f"{row.get('Relative_Performance_%', 0):.1f}" if pd.notna(row.get('Relative_Performance_%')) else 'N/A'
            gap_best = f"{row.get('Gap_from_Best', 0):.3f}" if pd.notna(row.get('Gap_from_Best')) else 'N/A'
            
            html += f"""
                <tr>
                    <td><strong>{row['Method']}</strong></td>
                    <td>{objective_val}</td>
                    <td>{row['Runtime_seconds']:.3f}</td>
                    <td class="{status_class}">{row['Status']}</td>
                    <td class="{perf_class}">{rel_perf}%</td>
                    <td>{gap_best}</td>
                    <td>{row['Solution_Count']}</td>
                </tr>
"""
        
        html += """
            </table>
        </div>
"""
    
    # Add metrics comparison
    if not metrics_comparison.empty:
        html += """
        <div class="section">
            <h2>📈 Detailed Metrics Comparison</h2>
            <table>
"""
        
        # Table header
        html += "<tr><th>Method</th>"
        for col in metrics_comparison.columns:
            if col != 'Method':
                html += f"<th>{col.replace('_', ' ').title()}</th>"
        html += "</tr>"
        
        # Table rows
        for _, row in metrics_comparison.iterrows():
            html += f"<tr><td><strong>{row['Method']}</strong></td>"
            for col in metrics_comparison.columns:
                if col != 'Method':
                    value = row[col]
                    if pd.notna(value) and isinstance(value, (int, float)):
                        html += f"<td>{value:.4f}</td>"
                    else:
                        html += f"<td>{value}</td>"
            html += "</tr>"
        
        html += """
            </table>
        </div>
"""
    
    # Add food selection analysis
    if food_selection_analysis['method_food_sets']:
        html += """
        <div class="section">
            <h2>🥬 Food Selection Analysis</h2>
"""
        
        # Common foods
        if food_selection_analysis['common_foods']:
            html += f"""
            <div class="highlight">
                <strong>Foods selected by ALL methods:</strong> {', '.join(sorted(food_selection_analysis['common_foods']))}
            </div>
"""
        
        # Food frequency
        html += """
            <h3>Food Selection Frequency</h3>
            <div class="food-grid">
"""
        
        total_methods = len([name for name, data in results.items() if data['status'] == 'SUCCESS'])
        for food, frequency in sorted(food_selection_analysis['food_frequency'].items(), 
                                    key=lambda x: x[1], reverse=True):
            percentage = (frequency / total_methods * 100) if total_methods > 0 else 0
            html += f"""
                <div class="food-card">
                    <strong>{food}</strong><br>
                    Selected by {frequency}/{total_methods} methods
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {percentage}%">{percentage:.0f}%</div>
                    </div>
                </div>
"""
        
        html += "</div>"
        
        # Method-specific food selections
        html += "<h3>Method-Specific Food Selections</h3>"
        for method_name, foods in food_selection_analysis['method_food_sets'].items():
            unique_foods = food_selection_analysis['unique_foods'].get(method_name, set())
            html += f"""
            <div class="food-card">
                <strong>{method_name}</strong><br>
                <em>Selected foods:</em> {', '.join(sorted(foods)) if foods else 'None'}<br>
"""
            if unique_foods:
                html += f"<em>Unique to this method:</em> {', '.join(sorted(unique_foods))}"
            html += "</div>"
        
        html += "</div>"
    
    # Add farm utilization analysis
    if farm_utilization_analysis['farm_utilization_rates']:
        html += """
        <div class="section">
            <h2>🚜 Farm Utilization Analysis</h2>
            <div class="farm-grid">
"""
        
        for method_name, farm_data in farm_utilization_analysis['farm_utilization_rates'].items():
            total_util = farm_utilization_analysis['total_utilization'][method_name]
            
            html += f"""
                <div class="farm-card">
                    <h3>{method_name}</h3>
                    <p><strong>Overall Utilization:</strong> {total_util['overall_utilization_rate']:.1f}%</p>
                    <p><strong>Total Used:</strong> {total_util['total_used']:.1f} ha / {total_util['total_available']:.1f} ha</p>
                    <h4>Farm-by-Farm Breakdown:</h4>
"""
            
            for farm, farm_info in farm_data.items():
                util_rate = farm_info['utilization_rate']
                html += f"""
                    <div style="margin: 10px 0;">
                        <strong>{farm}:</strong> {farm_info['used']:.1f} / {farm_info['available']:.1f} ha
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {min(util_rate, 100)}%">{util_rate:.1f}%</div>
                        </div>
                    </div>
"""
            
            html += "</div>"
        
        html += """
            </div>
        </div>
"""
    
    # Add detailed solution breakdown
    if solution_comparison:
        html += """
        <div class="section">
            <h2>🔍 Detailed Solution Analysis</h2>
"""
        
        for method_name, solution_data in solution_comparison.items():
            if results[method_name]['status'] == 'SUCCESS':
                result = results[method_name]['result']
                
                html += f"""
                <div class="config-box">
                    <h3>{method_name} - Detailed Solution</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 15px 0;">
                        <div>
                            <strong>Total Area:</strong> {solution_data['total_area']:.2f} ha<br>
                            <strong>Allocation Pairs:</strong> {solution_data['allocation_pairs']}<br>
                            <strong>Farms Used:</strong> {solution_data['farms_used']}/{len(optimizer.farms)}<br>
                            <strong>Foods Used:</strong> {solution_data['foods_used']}/{len(optimizer.foods)}
                        </div>
                        <div>
                            <strong>Top Food Allocations:</strong><br>
"""
                
                # Show top food allocations
                sorted_foods = sorted(solution_data['food_allocations'].items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
                for food, area in sorted_foods:
                    html += f"{food}: {area:.1f} ha<br>"
                
                html += """
                        </div>
                    </div>
                </div>
"""
        
        html += "</div>"
    
    # Add visualization
    html += """
        <div class="section">
            <h2>📊 Visual Comparison</h2>
            <div class="image-container">
                <img src="comprehensive_comparison_plots.png" alt="Comprehensive Comparison Plots">
            </div>
        </div>
"""
    
    # Add conclusions
    html += """
        <div class="section">
            <h2>🏆 Key Findings & Recommendations</h2>
            <div class="config-box">
"""
    
    if successful_methods:
        best_method = max(successful_methods, key=lambda x: results[x]['result'].objective_value)
        fastest_method = min(successful_methods, key=lambda x: results[x]['runtime'])
        
        html += f"""
                <h3>Performance Summary:</h3>
                <ul>
                    <li><strong>Best Objective Value:</strong> {best_method} ({results[best_method]['result'].objective_value:.3f})</li>
                    <li><strong>Fastest Execution:</strong> {fastest_method} ({results[fastest_method]['runtime']:.3f}s)</li>
                    <li><strong>Most Consistent:</strong> Enhanced SA (good balance of quality and speed)</li>
                </ul>
                
                <h3>Method Characteristics:</h3>
                <ul>
                    <li><strong>PuLP:</strong> Exact solution, good for smaller problems, moderate speed</li>
                    <li><strong>Basic SA:</strong> Fast heuristic, good baseline performance</li>
                    <li><strong>Enhanced SA:</strong> Improved heuristic with advanced features</li>
                    <li><strong>Benders:</strong> Handles large problems, may find better solutions but slower</li>
                </ul>
                
                <h3>Recommendations:</h3>
                <ul>
                    <li>For <strong>quick analysis:</strong> Use Enhanced SA</li>
                    <li>For <strong>exact solutions:</strong> Use PuLP (if problem size allows)</li>
                    <li>For <strong>large problems:</strong> Use Benders decomposition</li>
                    <li>For <strong>real-time applications:</strong> Use Basic or Enhanced SA</li>
                </ul>
"""
    else:
        html += "<p>No successful optimizations to analyze. Please check method configurations and problem setup.</p>"
    
    html += """
            </div>
        </div>
        
        <div class="section">
            <h2>ℹ️ Technical Notes</h2>
            <div class="config-box">
                <ul>
                    <li>All methods use the same problem data and constraints</li>
                    <li>Simulated Annealing methods use stochastic optimization</li>
                    <li>Results may vary between runs for SA methods</li>
                    <li>PuLP provides exact solutions for linear/mixed-integer problems</li>
                    <li>Benders decomposition is most effective for large-scale problems</li>
                </ul>
            </div>
        </div>
        
        <footer style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d;">
            Food Production Optimization System | Generated automatically
        </footer>
    </div>
</body>
</html>
"""
    
    return html

def prepare_json_data(optimizer, results: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare JSON data for further analysis."""
    
    # Convert land_availability dict with tuple keys to string keys
    land_availability_dict = {}
    if hasattr(optimizer, 'parameters') and optimizer.parameters.get('land_availability'):
        for key, value in optimizer.parameters['land_availability'].items():
            if isinstance(key, tuple):
                str_key = f"{key[0]}-{key[1]}"
            else:
                str_key = str(key)
            land_availability_dict[str_key] = value
    
    # Convert weights dict if it has tuple keys
    weights_dict = {}
    if hasattr(optimizer, 'parameters') and optimizer.parameters.get('weights'):
        for key, value in optimizer.parameters['weights'].items():
            if isinstance(key, tuple):
                str_key = f"{key[0]}-{key[1]}"
            else:
                str_key = str(key)
            weights_dict[str_key] = value
    
    json_data = {
        'timestamp': datetime.now().isoformat(),
        'problem_config': {
            'complexity_level': optimizer.complexity_level,
            'farms': list(optimizer.farms),
            'foods': list(optimizer.foods.keys()),
            'land_availability': land_availability_dict,
            'weights': weights_dict
        },
        'results': {},
        'summary': {
            'total_methods': len(results),
            'successful_methods': len([r for r in results.values() if r['status'] == 'SUCCESS']),
            'best_objective': None,
            'fastest_runtime': None
        }
    }
    
    # Process results
    best_obj = float('-inf')
    fastest_time = float('inf')
    
    for method_name, method_data in results.items():
        result_data = {
            'status': method_data['status'],
            'runtime': method_data['runtime'],
            'error': method_data['error']        }
        
        if method_data['status'] == 'SUCCESS' and method_data['result']:
            result = method_data['result']
            
            # Convert solution keys to strings for JSON serialization
            solution_dict = {}
            if result.solution:
                for key, value in result.solution.items():
                    # Convert tuple keys to string format
                    if isinstance(key, tuple):
                        str_key = f"{key[0]}-{key[1]}"
                    else:
                        str_key = str(key)
                    solution_dict[str_key] = value
            
            result_data.update({
                'objective_value': result.objective_value,
                'solution': solution_dict,
                'metrics': result.metrics or {},
                'solution_count': len(result.solution) if result.solution else 0
            })
            
            if result.objective_value > best_obj:
                best_obj = result.objective_value
                json_data['summary']['best_method'] = method_name
            
            if method_data['runtime'] < fastest_time:
                fastest_time = method_data['runtime']
                json_data['summary']['fastest_method'] = method_name
        
        json_data['results'][method_name] = result_data
    
    if best_obj != float('-inf'):
        json_data['summary']['best_objective'] = best_obj
    if fastest_time != float('inf'):
        json_data['summary']['fastest_runtime'] = fastest_time
    
    return json_data

if __name__ == "__main__":
    main()
