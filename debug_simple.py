#!/usr/bin/env python3
"""
Debug version to see what's happening.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

print("Starting test...")

try:
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
    
    # Try PuLP first
    print("Running PuLP...")
    pulp_result = optimizer.optimize_with_pulp()
    print(f"PuLP result: {pulp_result.objective_value}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
