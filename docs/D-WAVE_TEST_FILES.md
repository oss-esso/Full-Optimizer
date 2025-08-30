# D-Wave Test Scenario - Essential Files

This document lists the essential files required to run the D-Wave test scenario and understand its implementation.

## 1. Test Script

This is the main script to execute.

- **File:** `Tests/test_dwave_cost_estimation.py`
- **Purpose:** Runs the food optimization problem across different complexity levels ('simple', 'intermediate', 'full') using the D-Wave adapter. It demonstrates how to load a scenario, create a QUBO, and solve it using the adapter.

## 2. Core Logic & Adapters

These modules contain the primary logic for the D-Wave integration and problem definition.

- **File:** `my_functions/dwave_qpu_adapter.py`
- **Purpose:** This is the core D-Wave adapter. It handles the conversion of a QUBO matrix into a `BinaryQuadraticModel` (BQM) object and interfaces with the D-Wave samplers (in this case, a local simulator).

- **File:** `src/scenarios.py`
- **Purpose:** Loads the problem data (farms, foods, constraints, weights) for different complexity levels. The test script calls this module to get the problem to solve.

- **File:** `my_functions/qubo_converter.py`
- **Purpose:** Provides the necessary functions to convert the problem defined in the scenarios into the QUBO (Quadratic Unconstrained Binary Optimization) format, which is a prerequisite for the D-Wave adapter.

## 3. Data Models & Input Files

These files provide the necessary data structures and raw input data.

- **File:** `src/data_models.py`
- **Purpose:** Defines the common data structures used across the project, such as `OptimizationResult`, ensuring consistent data handling.

- **File:** `Inputs/Combined_Food_Data.xlsx`
- **Purpose:** The raw Excel data file that is read by `src/scenarios.py` to build the 'full' complexity problem. This file is essential for running that specific scenario.
