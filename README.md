
# Full Food Production Optimizer

This directory contains a complete implementation of the multi-objective food production optimizer, featuring different optimization methods:

1.**Benders Decomposition**: A powerful mathematical decomposition approach for large-scale mixed-integer problems.

2.**PuLP Direct Solver**: A straightforward implementation using the PuLP library's direct solver.

3.**Quantum-Enhanced Benders**: An innovative approach using quantum-inspired techniques to solve the master problem.

## Structure

-`src/`: Contains the source code for the optimizer.

  -`__init__.py`: Makes `src` a Python package.

  -`data_models.py`: Defines data structures like `OptimizationObjective` and `OptimizationResult`.

  -`optimizer.py`: Contains the main `FoodProductionOptimizer` class that integrates the different optimization methods.

  -`methods/`: Contains specialized implementations for each optimization approach.

    -`benders_method.py`: Implementation of the Benders decomposition approach.

    -`pulp_method.py`: Implementation of the PuLP direct optimization.

    -`quantum_methods.py`: Implementation of the quantum-enhanced optimization methods.

-`run_comparison.py`: Script to run and compare all optimization methods.

-`requirements.txt`: Lists the Python dependencies.

-`README.md`: This file.

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- pulp
- Additional quantum libraries for quantum-enhanced methods (optional)

## Dependencies

Make sure the `my_functions` directory (expected to be in the parent directory relative to `Codes`) is available in your Python path. This contains core functionality like:

-`my_functions/benders.py`: Benders decomposition implementation

-`my_functions/plotter.py`: Plotting utilities

-`my_functions/qubo_converter.py`: QUBO conversion utilities for quantum methods

-`my_functions/mean_field_base.py`: Mean-field approximation utilities

## Usage

### Running Comparison

To run a comparison of all methods:

```bash

cd"Codes/Full Optimizer"

pythonrun_comparison.py

```

This will:

1. Run optimization using all three methods
2. Save results to the `Results/` directory
3. Generate solution plots and convergence graphs
4. Print a comparison report to the console

### Using the Optimizer in Custom Code

```python

from src.optimizer import FoodProductionOptimizer

from src.data_models import OptimizationObjective


# Define your data

farms = ['Farm1', 'Farm2']

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

    }

}

food_groups = {'Grains': ['Wheat', 'Corn']}

config = {

    'parameters': {

        'objective_weights': {

            'nutritional_value': 0.25,

            'nutrient_density': 0.25,

            'environmental_impact': 0.5,

            'affordability': 0,

            'sustainability': 0

        },

        'land_availability': {'Farm1': 75, 'Farm2': 100}

    }

}


# Create optimizer

optimizer =FoodProductionOptimizer(farms, foods, food_groups, config)


# Solve using Benders decomposition (default method)

result = optimizer.solve_optimization_problem()


# Access solution

print(f"Objective value: {result.objective_value}")

for (farm, food), area in result.solution.items():

    print(f"Farm {farm}, Food {food}: {area:.2f} hectares")


# Alternatively, use another method

pulp_result = optimizer.solve_with_pulp()

quantum_result = optimizer.solve_with_quantum(use_quantum_inspired=True)

```

### Using with Real Data

To use with custom data, create appropriate dictionaries for:

-`farms`: List of farm names

-`foods`: Dictionary of foods with their attribute scores

-`food_groups`: Groups of foods for constraints

-`config`: Configuration parameters, including objective weights, land availability, etc.

## Results

Results are saved in the `Results/` directory and include:

- Text files with detailed results for each method
- Visualizations of the solutions
- Convergence plots for the Benders-based methods
- Comparison plots between classical and quantum-enhanced approaches

## Notes on the Quantum-Enhanced Method

The quantum-enhanced method in this package is a quantum-inspired approach using mean-field approximation to solve the master problem in the Benders decomposition. It represents the problem as a QUBO (Quadratic Unconstrained Binary Optimization) problem, which is well-suited for quantum computation.

This implementation does not require actual quantum hardware and can run on classical machines, providing a good benchmark for comparing with future real quantum implementations.
