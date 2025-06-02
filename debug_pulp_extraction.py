#!/usr/bin/env python3
"""
Debug script to investigate PuLP solution extraction issue.
This will help identify why the solution dictionary is empty when PuLP reports a positive objective.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.Qoptimizer import SimpleFoodOptimizer

def debug_pulp_solution():
    """Debug the PuLP solution extraction process."""
    
    # Simple configuration for testing
    config = {
        'farms': ['Farm1', 'Farm2'],
        'foods': {
            'Wheat': {
                'nutritional_value': 0.8,
                'nutrient_density': 0.7,
                'environmental_impact': 0.3,
                'affordability': 0.6,
                'sustainability': 0.7
            },
            'Corn': {
                'nutritional_value': 0.7,
                'nutrient_density': 0.6,
                'environmental_impact': 0.4,
                'affordability': 0.8,
                'sustainability': 0.6
            }
        },
        'parameters': {
            'land_availability': {
                'Farm1': 10.0,
                'Farm2': 15.0
            },
            'weights': {
                'nutritional_value': 0.3,
                'nutrient_density': 0.2,
                'environmental_impact': 0.2,
                'affordability': 0.15,
                'sustainability': 0.15
            },
            'minimum_planting_area': 0.5,  # Minimum area per food
            'farm_utilization_min': 0.8,   # Minimum farm utilization
            'farm_utilization_max': 1.0,   # Maximum farm utilization
            'max_foods_per_farm': 4,       # Maximum variety per farm
            'min_foods_per_farm': 1        # Minimum variety per farm
        }
    }
    
    # Create optimizer
    optimizer = SimpleFoodOptimizer(config)
    
    print("="*60)
    print("DEBUGGING PULP SOLUTION EXTRACTION")
    print("="*60)
    
    # Run PuLP optimization
    result = optimizer.optimize_with_pulp()
    
    print(f"\nResult status: {result.status}")
    print(f"Objective value: {result.objective_value}")
    print(f"Solution dictionary: {result.solution}")
    print(f"Number of solution entries: {len(result.solution)}")
    
    # Print metrics
    print(f"\nMetrics:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Check for the discrepancy
    calculated_obj = result.metrics.get('calculated_objective', 0.0)
    if abs(result.objective_value - calculated_obj) > 0.01:
        print(f"\n⚠️  DISCREPANCY DETECTED!")
        print(f"   PuLP objective: {result.objective_value:.4f}")
        print(f"   Calculated objective: {calculated_obj:.4f}")
        print(f"   Difference: {abs(result.objective_value - calculated_obj):.4f}")
    else:
        print(f"\n✅ No discrepancy detected")

if __name__ == "__main__":
    debug_pulp_solution()
