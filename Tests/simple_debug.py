#!/usr/bin/env python3
print("Starting simple debug test...")

try:
    import sys
    import os
    print("Basic imports successful")
    
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    print(f"Added project root to path: {project_root}")
    
    from src.Qoptimizer import SimpleFoodOptimizer
    print("SimpleFoodOptimizer imported successfully")
    
    optimizer = SimpleFoodOptimizer(complexity_level='simple')
    print("Optimizer created")
    
    optimizer.load_food_data()
    print("Food data loaded")
    
    print(f"Problem: {len(optimizer.farms)} farms, {len(optimizer.foods)} foods")
    print(f"Farms: {list(optimizer.farms)}")
    print(f"Foods: {list(optimizer.foods.keys())}")
    
    print("Running PuLP...")
    pulp_result = optimizer.optimize_with_pulp()
    print(f"PuLP result: {pulp_result.objective_value}")
    
    print("Running Basic SA...")
    basic_result = optimizer.optimize_with_simulated_annealing_benders(
        max_iterations=100,
        initial_temperature=100.0,
        cooling_rate=0.95,
        enhanced_sa=False
    )
    print(f"Basic SA result: {basic_result.objective_value}")
    
    print("Running Enhanced SA...")
    enhanced_result = optimizer.optimize_with_simulated_annealing_benders(
        max_iterations=100,
        initial_temperature=100.0,
        cooling_rate=0.95,
        enhanced_sa=True
    )
    print(f"Enhanced SA result: {enhanced_result.objective_value}")
    
    print("All methods completed successfully!")

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()

input("Press Enter to exit...")
