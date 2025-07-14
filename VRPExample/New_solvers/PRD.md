# TODO: Enhance `test_overnight.py` with multiple solvers

The goal is to extend the `test_overnight.py` script to benchmark multiple solvers against the same scenario. This will help us compare their performance and solution quality.

## Requirements:

1.  **Integrate Multiple Solvers:**
    *   Modify `test_overnight.py` to run the scenario through three different solvers:
        1.  The existing `vrp_multiday_sequential.py` solver (baseline).
        2.  The `tsp_multiple_days1.py` solver.
        3.  The `tsp_multiple_days2.py` solver.

2.  **Scenario Compatibility:**
    *   **Analyze Solver Inputs:** For `tsp_multiple_days1.py` and `tsp_multiple_days2.py`, the coding agent must first read their source code to determine the exact input data structure they expect for a scenario.
    *   **Print Requirements:** The agent must print a clear list of all required and optional parameters that the scenario data must contain for each new solver. This is for clarification before implementation.
    *   **Data Transformation:** Create data transformation layers or functions within `test_overnight.py`. These will adapt the base `furgoni_scenario` to the specific format required by `tsp_multiple_days1.py` and `tsp_multiple_days2.py`.

3.  **Standardized Output:**
    *   **Reports:** For each solver run, generate a printed report in the console. This report must follow the exact same format as the one currently produced by `test_overnight.py`.
    *   **Plots:** Each solver run must generate and save a plot visualizing the solution. The plot's style and format should be identical to the existing one.

4.  **Solution Validation:**
    *   **Objective Value Check:** Implement a validation step. After all three solvers have run, compare the objective value (e.g., total distance, cost) from `tsp_multiple_days1.py` and `tsp_multiple_days2.py` against the objective value from the baseline solver.
    *   **Interval Check:** The objective values from the new solvers must fall within a reasonable interval (e.g., +/- 15%) of the baseline solver's objective value. The agent should define and justify this interval.

## Implementation Premises:

### `tsp_multiple_days1.py` Data Extraction:
*   **Time and Distance Matrix:** Extract time details from the scenario using the same logic as `vrp_multiday_sequential.py`. This includes:
    *   Creating time and distance matrices using CachedOSRMDistanceCalculator
    *   Including service time details for each location
    *   Applying truck speed ratios based on vehicle type
*   **Parameter Flexibility:** Keep solver parameters (days, start/end times, etc.) flexible but ensure they respect the maximum hours per vehicle set in the scenario (typically 8 hours = 480 minutes per day).
*   **Missing Dependencies:** Create stub implementation for missing `time_details.py` module or extract required logic directly into the integration.

### `tsp_multiple_days2.py` JSON Generation:
*   **Scenario-to-JSON:** Generate the required JSON file dynamically from the furgoni scenario, including:
    *   `Locations`: Convert scenario locations with coordinates and metadata
    *   `Matrix`: Extract time/distance matrix using the same OSRM logic
    *   `Windows`: Convert location time windows and working hours
    *   `ServiceCosts`: Extract service times from scenario locations
    *   `Penalties`: Set reasonable penalties for dropping locations
    *   `NumberOfDays`, `NumberOfEvents`, `Duration`: Extract from scenario constraints
*   **Time Compliance:** Ensure all time parameters respect the scenario's maximum drive time per vehicle and daily working hour limits.

### Constraint Preservation:
*   **Vehicle Limits:** Both solvers must respect the scenario's vehicle capacity, time, and operational constraints.
*   **Working Hours:** Maintain consistency with the baseline solver's daily time limits and working hour windows.
*   **Route Feasibility:** Ensure generated solutions are feasible within the same constraint framework as the baseline.

## Agent Instructions:

*   **Step 1: Analyze Solvers:** ✅ Complete - Solvers analyzed and input requirements identified.
*   **Step 2: Implementation Premises:** ✅ Complete - Detailed data extraction and JSON generation approaches defined above.
*   **Step 3: Implement and Test:** Proceed with the implementation of the integration, data transformation, standardized output, and validation checks using the premises outlined above.
