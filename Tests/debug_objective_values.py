#!/usr/bin/env python3
"""
Debug script to investigate the objective value calculation and complexity level usage.
"""

import os
import sys
import logging

# Add project directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Qoptimizer import SimpleFoodOptimizer
from src.scenarios import load_food_data

def debug_scenario_loading():
    """Debug the scenario loading for different complexity levels."""
    print("=" * 60)
    print("DEBUGGING SCENARIO LOADING")
    print("=" * 60)
    
    for complexity in ['simple', 'intermediate', 'full']:
        print(f"\n--- Testing complexity level: {complexity} ---")
        
        try:
            farms, foods, food_groups, config = load_food_data(complexity)
            
            print(f"Farms ({len(farms)}): {farms}")
            print(f"Foods ({len(foods)}): {list(foods.keys())}")
            print(f"Food groups ({len(food_groups)}): {list(food_groups.keys())}")
            
            # Check weights in config
            if 'parameters' in config and 'weights' in config['parameters']:
                weights = config['parameters']['weights']
                print(f"Weights: {weights}")
                
                # Calculate max possible positive objective for a single food
                if foods:
                    first_food = list(foods.keys())[0]
                    food_data = foods[first_food]
                    print(f"Sample food '{first_food}' data: {food_data}")
                    
                    # Calculate theoretical max objective contribution
                    max_contrib = 0
                    for attr, weight in weights.items():
                        if attr in food_data:
                            value = food_data[attr]
                            # For positive objectives, higher is better
                            if attr in ['nutritional_value', 'nutrient_density', 'affordability', 'sustainability']:
                                contrib = weight * value
                            # For negative objectives (impact), lower is better, so negate
                            elif attr == 'environmental_impact':
                                contrib = -weight * value
                            else:
                                contrib = weight * value
                            max_contrib += contrib
                            print(f"  {attr}: {value} * {weight} = {contrib:.4f}")
                    
                    print(f"  Total max contribution per food: {max_contrib:.4f}")
                    
                    # For 3 farms with this food
                    total_theoretical = max_contrib * 3
                    print(f"  Theoretical max for 3 farms: {total_theoretical:.4f}")
            
        except Exception as e:
            print(f"Error loading {complexity}: {e}")
            import traceback
            traceback.print_exc()

def debug_sa_objective():
    """Debug the simulated annealing objective function."""
    print("\n" + "=" * 60)
    print("DEBUGGING SIMULATED ANNEALING OBJECTIVE")
    print("=" * 60)
    
    # Test with simple complexity
    optimizer = SimpleFoodOptimizer(complexity_level='simple')
    farms, foods, food_groups, config = optimizer.load_food_data()
    
    print(f"Loaded data: {len(farms)} farms, {len(foods)} foods")
    print(f"Weights: {config['parameters']['weights']}")
    
    # Test objective function calculation manually
    F = len(farms)
    C = len(foods)
    
    # Create a test solution: select first food for each farm
    test_solution = [0.0] * (F * C)
    for fi in range(F):
        test_solution[fi * C] = 1.0  # Select first food for each farm
    
    print(f"\nTest solution (selecting first food for each farm):")
    selection_matrix = [[test_solution[fi * C + ci] for ci in range(C)] for fi in range(F)]
    
    for fi, farm in enumerate(farms):
        selected_foods = [list(foods.keys())[ci] for ci in range(C) if selection_matrix[fi][ci] > 0.5]
        print(f"  {farm}: {selected_foods}")
    
    # Manual objective calculation
    total_objective = 0.0
    weights = config['parameters']['weights']
    
    print(f"\nManual objective calculation:")
    for fi, farm in enumerate(farms):
        for ci, food in enumerate(foods.keys()):
            if selection_matrix[fi][ci] > 0.5:
                food_data = foods[food]
                print(f"  {farm} - {food}:")
                
                # Calculate contribution for each objective
                nutritional = food_data.get('nutritional_value', 0.0)
                environmental = food_data.get('environmental_impact', 0.0)
                affordability = food_data.get('affordability', 0.0)
                sustainability = food_data.get('sustainability', 0.0)
                
                # This matches the SA objective function
                contrib = (
                    -weights.get('nutritional_value', 0.3) * nutritional +
                    weights.get('environmental_impact', 0.2) * environmental +
                    -weights.get('affordability', 0.2) * affordability +
                    -weights.get('sustainability', 0.0) * sustainability
                )
                
                print(f"    nutritional_value: {nutritional} * {-weights.get('nutritional_value', 0.3)} = {-weights.get('nutritional_value', 0.3) * nutritional:.4f}")
                print(f"    environmental_impact: {environmental} * {weights.get('environmental_impact', 0.2)} = {weights.get('environmental_impact', 0.2) * environmental:.4f}")
                print(f"    affordability: {affordability} * {-weights.get('affordability', 0.2)} = {-weights.get('affordability', 0.2) * affordability:.4f}")
                print(f"    sustainability: {sustainability} * {-weights.get('sustainability', 0.0)} = {-weights.get('sustainability', 0.0) * sustainability:.4f}")
                print(f"    Total contribution: {contrib:.4f}")
                
                total_objective += contrib
    
    print(f"\nTotal objective value: {total_objective:.6f}")
    
    # Compare with expected range (60-70)
    print(f"Expected range: 60-70")
    print(f"Actual value: {total_objective:.6f}")
    print(f"Difference: {abs(total_objective - 65):.6f} from midpoint")

def test_different_weights():
    """Test what weights would give values in the 60-70 range."""
    print("\n" + "=" * 60)
    print("TESTING ALTERNATIVE WEIGHT CONFIGURATIONS")
    print("=" * 60)
    
    optimizer = SimpleFoodOptimizer(complexity_level='simple')
    farms, foods, food_groups, config = optimizer.load_food_data()
    
    # Get food data for calculation
    F = len(farms)
    C = len(foods)
    
    # Test solution: select first food for each farm
    selection_matrix = [[1.0 if ci == 0 else 0.0 for ci in range(C)] for fi in range(F)]
    
    # Test different weight configurations
    test_configs = [
        {
            'name': 'Current SA weights',
            'weights': {
                'nutritional_value': 0.3,
                'environmental_impact': 0.2,
                'affordability': 0.2,
                'sustainability': 0.0
            }
        },
        {
            'name': 'Scenario weights',
            'weights': config['parameters']['weights']
        },
        {
            'name': 'Maximization weights (all positive)',
            'weights': {
                'nutritional_value': 25.0,
                'environmental_impact': -25.0,  # Negative to minimize
                'affordability': 20.0,
                'sustainability': 20.0
            }
        },
        {
            'name': 'Large positive weights',
            'weights': {
                'nutritional_value': 100.0,
                'environmental_impact': -50.0,
                'affordability': 50.0,
                'sustainability': 50.0
            }
        }
    ]
    
    for config_test in test_configs:
        print(f"\n--- {config_test['name']} ---")
        weights = config_test['weights']
        print(f"Weights: {weights}")
        
        total_objective = 0.0
        
        for fi, farm in enumerate(farms):
            for ci, food in enumerate(foods.keys()):
                if selection_matrix[fi][ci] > 0.5:
                    food_data = foods[food]
                    
                    # Calculate contribution
                    contrib = (
                        weights.get('nutritional_value', 0) * food_data.get('nutritional_value', 0.0) +
                        weights.get('environmental_impact', 0) * food_data.get('environmental_impact', 0.0) +
                        weights.get('affordability', 0) * food_data.get('affordability', 0.0) +
                        weights.get('sustainability', 0) * food_data.get('sustainability', 0.0)
                    )
                    
                    total_objective += contrib
        
        print(f"Total objective: {total_objective:.6f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    debug_scenario_loading()
    debug_sa_objective()
    test_different_weights()
