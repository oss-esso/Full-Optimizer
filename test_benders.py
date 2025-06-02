#!/usr/bin/env python3
"""
Quick test of Benders method
"""

import sys
import os
import time

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def main():
    try:
        from src.Qoptimizer import SimpleFoodOptimizer
        
        optimizer = SimpleFoodOptimizer(complexity_level='simple')
        optimizer.load_food_data()
        
        print('Running Benders method...')
        start_time = time.time()
        benders_result = optimizer.optimize_with_benders()
        benders_runtime = time.time() - start_time
        
        print(f'Benders result: {benders_result.objective_value} ({benders_runtime:.3f}s)')
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
