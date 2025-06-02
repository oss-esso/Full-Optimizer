#!/usr/bin/env python3
"""
Debug script to test the simple scenario step by step.
"""

import sys
import os
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Configure logging to show output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def debug_simple_test():
    """Debug test of simple scenario."""
    try:
        print("Starting debug test...")
        
        from src.Qoptimizer import SimpleFoodOptimizer
        print("SimpleFoodOptimizer imported successfully")
        
        # Create optimizer instance
        print("Creating optimizer...")
        optimizer = SimpleFoodOptimizer(complexity_level='simple')
        print("Optimizer created")
        
        print("Loading food data...")
        optimizer.load_food_data()
        print("Food data loaded")
        
        print(f"Problem: {len(optimizer.farms)} farms, {len(optimizer.foods)} foods")
        print(f"Farms: {list(optimizer.farms)}")
        print(f"Foods: {list(optimizer.foods.keys())}")
        
        # Test PuLP method
        print("Running PuLP...")
        pulp_result = optimizer.optimize_with_pulp()
        print(f"PuLP result: {pulp_result.objective_value}")
        
        # Test Basic SA
        print("Running Basic SA...")
        basic_result = optimizer.optimize_with_simulated_annealing_benders(
            max_iterations=100,  # Reduced for faster testing
            initial_temperature=100.0,
            cooling_rate=0.95,
            enhanced_sa=False
        )
        print(f"Basic SA result: {basic_result.objective_value}")
        
        # Test Enhanced SA
        print("Running Enhanced SA...")
        enhanced_result = optimizer.optimize_with_simulated_annealing_benders(
            max_iterations=100,  # Reduced for faster testing
            initial_temperature=100.0,
            cooling_rate=0.95,
            enhanced_sa=True
        )
        print(f"Enhanced SA result: {enhanced_result.objective_value}")
        
        print("All methods completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_simple_test()
    sys.exit(0 if success else 1)
