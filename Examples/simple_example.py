"""
Simple example showing how to use the food production optimizer with all four solver methods.
"""
import os
import sys
import matplotlib.pyplot as plt
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s: %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

# Ensure paths are correctly set up
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Full Optimizer directory
sys.path.insert(0, parent_dir)  # Add parent directory to path

# Import optimizer
try:
    from src.optimizer import FoodProductionOptimizer
    from src.data_models import OptimizationObjective
except ImportError as e:
    print(f"Error importing optimizer components: {e}")
    print("Please ensure 'src' directory is in the Python path and contains the necessary files.")
    sys.exit(1)

def main():
    """
    Run a simple demonstration of the food production optimizer with all four solution methods.
    """
    print("=" * 60)
    print(" FOOD PRODUCTION OPTIMIZER - SIMPLE EXAMPLE")
    print("=" * 60)
    
    # Define example farms, foods, and food groups
    farms = ['Farm1', 'Farm2']
    
    # Define foods with their attributes
    foods = {
        'Wheat': {
            'nutritional_value': 0.7,
            'nutrient_density': 0.6,
            'environmental_impact': 0.3,
            'affordability': 0.8,
            'sustainability': 0.7
        },
        'Corn': {
            'nutritional_value': 0.6,
            'nutrient_density': 0.5,
            'environmental_impact': 0.4,
            'affordability': 0.9,
            'sustainability': 0.6
        },
        'Soybeans': {
            'nutritional_value': 0.9,
            'nutrient_density': 0.8,
            'environmental_impact': 0.2,
            'affordability': 0.6,
            'sustainability': 0.8
        },
        'Potatoes': {
            'nutritional_value': 0.5,
            'nutrient_density': 0.4,
            'environmental_impact': 0.3,
            'affordability': 0.9,
            'sustainability': 0.7
        }
    }
    
    # Define food groups
    food_groups = {
        'Grains': ['Wheat', 'Corn'],
        'Legumes': ['Soybeans'],
        'Vegetables': ['Potatoes']
    }
    
    # Define optimization configuration
    config = {
        'parameters': {
            # Objective weights
            'objective_weights': {
                'nutritional_value': 0.3,
                'nutrient_density': 0.2,
                'environmental_impact': 0.2,
                'affordability': 0.1,
                'sustainability': 0.2
            },
            # Land availability per farm (hectares)
            'land_availability': {
                'Farm1': 100,
                'Farm2': 150
            },
            # Minimum planting area per crop
            'minimum_planting_area': {
                'Wheat': 10,
                'Corn': 10,
                'Soybeans': 5,
                'Potatoes': 5
            },
            # Food group constraints
            'food_group_constraints': {
                'Grains': {'min_foods': 1, 'max_foods': 2},
                'Legumes': {'min_foods': 0, 'max_foods': 1},
                'Vegetables': {'min_foods': 0, 'max_foods': 1}
            }
        },
        # Algorithm parameters
        'benders_tolerance': 0.01,
        'benders_max_iterations': 50,
        'benders_multi_cut': True,
        'benders_use_norm_cuts': True
    }
    
    # Create the optimizer
    print("Creating food production optimizer...")
    optimizer = FoodProductionOptimizer(farms, foods, food_groups, config)
    
    # Dictionary to store results from all methods
    results = {}
    
    # 1. Solve with Classical Benders Decomposition
    print("\n" + "=" * 60)
    print(" METHOD 1: CLASSICAL BENDERS DECOMPOSITION")
    print("=" * 60)
    start_time = time.time()
    benders_result = optimizer.solve_with_benders()
    benders_runtime = time.time() - start_time
    
    print(f"Status: {benders_result.status}")
    print(f"Objective Value: {benders_result.objective_value:.4f}")
    print(f"Runtime: {benders_runtime:.2f} seconds")
    
    # Print solution summary
    print("\nLand Allocation:")
    for (farm, food), area in benders_result.solution.items():
        print(f"{farm}: {food} - {area:.2f} hectares")
    
    # Store results
    results['Benders'] = {
        'objective_value': benders_result.objective_value,
        'runtime': benders_runtime
    }
    
    # 2. Solve with PuLP
    print("\n" + "=" * 60)
    print(" METHOD 2: DIRECT PULP SOLVER")
    print("=" * 60)
    start_time = time.time()
    pulp_result = optimizer.solve_with_pulp()
    pulp_runtime = time.time() - start_time
    
    print(f"Status: {pulp_result.status}")
    print(f"Objective Value: {pulp_result.objective_value:.4f}")
    print(f"Runtime: {pulp_runtime:.2f} seconds")
    
    # Print solution summary
    print("\nLand Allocation:")
    for (farm, food), area in pulp_result.solution.items():
        print(f"{farm}: {food} - {area:.2f} hectares")
    
    # Store results
    results['PuLP'] = {
        'objective_value': pulp_result.objective_value,
        'runtime': pulp_runtime
    }
    
    # 3. Solve with Quantum-Inspired Benders
    print("\n" + "=" * 60)
    print(" METHOD 3: QUANTUM-INSPIRED BENDERS")
    print("=" * 60)
    start_time = time.time()
    quantum_inspired_result = optimizer.solve_with_quantum_inspired()
    quantum_inspired_runtime = time.time() - start_time
    
    print(f"Status: {quantum_inspired_result.status}")
    print(f"Objective Value: {quantum_inspired_result.objective_value:.4f}")
    print(f"Runtime: {quantum_inspired_runtime:.2f} seconds")
    
    # Print solution summary
    print("\nLand Allocation:")
    for (farm, food), area in quantum_inspired_result.solution.items():
        print(f"{farm}: {food} - {area:.2f} hectares")
    
    # Store results
    results['Quantum-Inspired'] = {
        'objective_value': quantum_inspired_result.objective_value,
        'runtime': quantum_inspired_runtime
    }
    
    # 4. Solve with Quantum-Enhanced Benders
    print("\n" + "=" * 60)
    print(" METHOD 4: QUANTUM-ENHANCED BENDERS")
    print("=" * 60)
    start_time = time.time()
    quantum_enhanced_result = optimizer.solve_with_quantum_enhanced()
    quantum_enhanced_runtime = time.time() - start_time
    
    print(f"Status: {quantum_enhanced_result.status}")
    print(f"Objective Value: {quantum_enhanced_result.objective_value:.4f}")
    print(f"Runtime: {quantum_enhanced_runtime:.2f} seconds")
    
    # Print solution summary
    print("\nLand Allocation:")
    for (farm, food), area in quantum_enhanced_result.solution.items():
        print(f"{farm}: {food} - {area:.2f} hectares")
    
    # Store results
    results['Quantum-Enhanced'] = {
        'objective_value': quantum_enhanced_result.objective_value,
        'runtime': quantum_enhanced_runtime
    }
    
    # Print summary comparing all methods
    print("\n" + "=" * 60)
    print(" COMPARISON OF ALL METHODS")
    print("=" * 60)
    
    # Print objective values
    print("Objective values:")
    print(f"  Classical Benders:      {results['Benders']['objective_value']:.4f}")
    print(f"  PuLP:                   {results['PuLP']['objective_value']:.4f}")
    print(f"  Quantum-Inspired:       {results['Quantum-Inspired']['objective_value']:.4f}")
    print(f"  Quantum-Enhanced:       {results['Quantum-Enhanced']['objective_value']:.4f}")
    
    # Print runtimes
    print("\nRuntimes:")
    print(f"  Classical Benders:      {results['Benders']['runtime']:.2f} seconds")
    print(f"  PuLP:                   {results['PuLP']['runtime']:.2f} seconds")
    print(f"  Quantum-Inspired:       {results['Quantum-Inspired']['runtime']:.2f} seconds")
    print(f"  Quantum-Enhanced:       {results['Quantum-Enhanced']['runtime']:.2f} seconds")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 