#!/usr/bin/env python3
"""
Debug script to understand PuLP variable extraction issue.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def main():
    print("="*60)
    print("DEBUGGING PULP VARIABLE EXTRACTION")
    print("="*60)
    
    try:
        from src.Qoptimizer import SimpleFoodOptimizer
        
        # Create optimizer
        optimizer = SimpleFoodOptimizer(complexity_level='simple')
        optimizer.load_food_data()
        
        print(f"Problem: {len(optimizer.farms)} farms, {len(optimizer.foods)} foods")
        print(f"Farms: {list(optimizer.farms)}")
        print(f"Foods: {list(optimizer.foods.keys())}")
        
        # Temporarily patch the PuLP method to add debugging
        original_method = optimizer.optimize_with_pulp
        
        def debug_optimize_with_pulp():
            import pulp
            from src.data_models import OptimizationResult
            import time
            
            # Create the optimization model
            model = pulp.LpProblem("Food_Production_Optimization", pulp.LpMaximize)
            
            # Decision variables
            x = {}
            y = {}
            
            # Initialize variables
            for farm in optimizer.farms:
                for food in optimizer.foods:
                    x[farm, food] = pulp.LpVariable(f"x_{farm}_{food}", lowBound=0)
                    y[farm, food] = pulp.LpVariable(f"y_{farm}_{food}", cat='Binary')
            
            print(f"\nCreated {len(x)} x variables and {len(y)} y variables")
            
            # Get weight parameters
            weights = optimizer.parameters['weights']
            
            # Objective function
            objective_terms = []
            for farm in optimizer.farms:
                for food in optimizer.foods:
                    for obj_key, weight in weights.items():
                        food_value = optimizer.foods[food].get(obj_key, 0)
                        if obj_key == 'environmental_impact':
                            objective_terms.append(-weight * food_value * x[farm, food])
                        else:
                            objective_terms.append(weight * food_value * x[farm, food])
            
            model += pulp.lpSum(objective_terms)
            print(f"Added objective with {len(objective_terms)} terms")
            
            # Add constraints (simplified for debugging)
            # Land availability constraints
            for farm in optimizer.farms:
                land_constraint = pulp.lpSum([x[farm, food] for food in optimizer.foods]) <= optimizer.parameters['land_availability'][farm]
                model += land_constraint
            
            # Linking constraints
            for farm in optimizer.farms:
                for food in optimizer.foods:
                    model += x[farm, food] <= optimizer.parameters['land_availability'][farm] * y[farm, food]
            
            print(f"Added constraints")
            
            # Solve
            print("\nSolving...")
            start_time = time.time()
            solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=120, gapRel=0.05)
            model.solve(solver)
            runtime = time.time() - start_time
            
            # Check status
            status_str = pulp.LpStatus[model.status]
            print(f"\nOptimization status: {status_str}")
            obj_value = pulp.value(model.objective)
            print(f"Objective value: {obj_value}")
            
            # Debug variable extraction
            solution = {}
            if model.status == pulp.LpStatusOptimal:
                print("\nDebugging variable values:")
                for farm in optimizer.farms:
                    print(f"\nFarm: {farm}")
                    farm_total = 0
                    for food in optimizer.foods:
                        x_var = x[farm, food]
                        y_var = y[farm, food]
                        
                        print(f"  Variable x[{farm}, {food}]: {x_var}")
                        print(f"    Name: {x_var.name}")
                        print(f"    Value: {x_var.value()}")
                        print(f"  Variable y[{farm}, {food}]: {y_var}")
                        print(f"    Name: {y_var.name}")
                        print(f"    Value: {y_var.value()}")
                        
                        x_val = x_var.value()
                        if x_val is not None and x_val > 0.01:
                            solution[(farm, food)] = x_val
                            farm_total += x_val
                            print(f"    Added to solution: {x_val}")
                    
                    print(f"  Total for {farm}: {farm_total}")
            
            print(f"\nFinal solution dictionary: {solution}")
            print(f"Number of entries: {len(solution)}")
            
            # Calculate manual objective
            manual_obj = 0.0
            for (farm, food), area in solution.items():
                for obj_key, weight in weights.items():
                    food_value = optimizer.foods[food].get(obj_key, 0)
                    if obj_key == 'environmental_impact':
                        manual_obj += -weight * food_value * area
                    else:
                        manual_obj += weight * food_value * area
            
            print(f"Manual objective calculation: {manual_obj}")
            print(f"PuLP objective: {obj_value}")
            print(f"Difference: {abs(obj_value - manual_obj) if obj_value else 'N/A'}")
            
            return OptimizationResult(
                status="optimal" if solution else "infeasible",
                objective_value=obj_value if obj_value is not None else 0.0,
                solution=solution,
                metrics={},
                runtime=runtime
            )
        
        # Run the debug version
        result = debug_optimize_with_pulp()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
