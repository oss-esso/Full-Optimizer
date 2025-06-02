#!/usr/bin/env python3
"""
Working comparison script for PuLP, Basic SA, and Enhanced SA methods.
Based on the working debug_simple_run.py approach.
"""

import sys
import os
import time
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def main():
    print("Starting multi-method optimization comparison...")
    print("Loading SimpleFoodOptimizer...")
    
    try:
        from src.Qoptimizer import SimpleFoodOptimizer
        print("SimpleFoodOptimizer imported successfully")
        
        # Create optimizer
        print("Creating optimizer...")
        optimizer = SimpleFoodOptimizer(complexity_level='simple')
        print("Optimizer created")
        
        # Load data
        print("Loading food data...")
        optimizer.load_food_data()
        print("Food data loaded")
        
        print(f"Problem: {len(optimizer.farms)} farms, {len(optimizer.foods)} foods")
        print(f"Farms: {list(optimizer.farms)}")
        print(f"Foods: {list(optimizer.foods.keys())}")
        
        # Run PuLP
        print("\nRunning PuLP...")
        start_time = time.time()
        pulp_result = optimizer.optimize_with_pulp()
        pulp_runtime = time.time() - start_time
        print(f"PuLP result: {pulp_result.objective_value}")
        
        # Run Basic SA
        print("\nRunning Basic SA...")
        start_time = time.time()
        basic_result = optimizer.optimize_with_simulated_annealing_benders(
            max_iterations=100,
            initial_temperature=100.0,
            cooling_rate=0.95,
            enhanced_sa=False
        )
        basic_runtime = time.time() - start_time
        print(f"Basic SA result: {basic_result.objective_value}")
        
        # Run Enhanced SA
        print("\nRunning Enhanced SA...")
        start_time = time.time()
        enhanced_result = optimizer.optimize_with_simulated_annealing_benders(
            max_iterations=100,
            initial_temperature=100.0,
            cooling_rate=0.95,
            enhanced_sa=True,
            adaptive_cooling=True,
            use_restart=True,
            neighborhood_type="multi_flip"
        )
        enhanced_runtime = time.time() - start_time
        print(f"Enhanced SA result: {enhanced_result.objective_value}")
        
        # Run Benders method
        print("\nRunning Benders...")
        start_time = time.time()
        benders_result = optimizer.optimize_with_benders()
        benders_runtime = time.time() - start_time
        print(f"Benders result: {benders_result.objective_value}")
        
        print("\nAll methods completed successfully!")
        
        # Display comparison
        print("\n" + "="*60)
        print("RESULTS COMPARISON")
        print("="*60)
        print(f"PuLP LP:        {pulp_result.objective_value:10.3f} ({pulp_runtime:.3f}s)")
        print(f"Basic SA:       {basic_result.objective_value:10.3f} ({basic_runtime:.3f}s)")
        print(f"Enhanced SA:    {enhanced_result.objective_value:10.3f} ({enhanced_runtime:.3f}s)")
        print(f"Benders:        {benders_result.objective_value:10.3f} ({benders_runtime:.3f}s)")
        
        # Calculate improvements
        sa_improvement = ((basic_result.objective_value - enhanced_result.objective_value) / 
                         abs(basic_result.objective_value)) * 100 if basic_result.objective_value != 0 else 0
        
        basic_vs_pulp = ((pulp_result.objective_value - basic_result.objective_value) / 
                        abs(pulp_result.objective_value)) * 100 if pulp_result.objective_value != 0 else 0
        enhanced_vs_pulp = ((pulp_result.objective_value - enhanced_result.objective_value) / 
                           abs(pulp_result.objective_value)) * 100 if pulp_result.objective_value != 0 else 0
        benders_vs_pulp = ((pulp_result.objective_value - benders_result.objective_value) / 
                          abs(pulp_result.objective_value)) * 100 if pulp_result.objective_value != 0 else 0
        
        print(f"\nPERFORMANCE ANALYSIS:")
        print(f"Enhanced SA vs Basic SA: {sa_improvement:+.2f}% improvement")
        print(f"Basic SA vs PuLP:        {basic_vs_pulp:+.2f}%")
        print(f"Enhanced SA vs PuLP:     {enhanced_vs_pulp:+.2f}%")
        print(f"Benders vs PuLP:         {benders_vs_pulp:+.2f}%")
          # Generate simple plot
        print("\nGenerating comparison plot...")
        create_simple_plot(pulp_result, basic_result, enhanced_result, benders_result,
                          pulp_runtime, basic_runtime, enhanced_runtime, benders_runtime)
        
        print("\n" + "="*60)
        print("✓ COMPARISON COMPLETED SUCCESSFULLY!")
        print("✓ Check Results/working_comparison.png for plot")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")

def create_simple_plot(pulp_result, basic_result, enhanced_result, benders_result,
                      pulp_runtime, basic_runtime, enhanced_runtime, benders_runtime):
    """Create a simple comparison plot."""
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare data
    methods = ['PuLP LP', 'Basic SA', 'Enhanced SA', 'Benders']
    objectives = [pulp_result.objective_value, basic_result.objective_value, enhanced_result.objective_value, benders_result.objective_value]
    runtimes = [pulp_runtime, basic_runtime, enhanced_runtime, benders_runtime]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
      # Objective values
    colors = ['gold', 'lightblue', 'lightgreen', 'lightcoral']
    bars1 = ax1.bar(methods, objectives, color=colors)
    ax1.set_title('Objective Value Comparison')
    ax1.set_ylabel('Objective Value')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars1, objectives):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(objectives)*0.01,
                f'{value:.1f}', ha='center', va='bottom')
    
    # Runtime comparison
    bars2 = ax2.bar(methods, runtimes, color=colors)
    ax2.set_title('Runtime Comparison')
    ax2.set_ylabel('Runtime (seconds)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars2, runtimes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(runtimes)*0.01,
                f'{value:.3f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(results_dir, 'working_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: {plot_path}")

if __name__ == "__main__":
    main()
