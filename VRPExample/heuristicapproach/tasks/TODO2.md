# TODO List for Advanced EPDT Implementation

This document outlines the necessary steps to integrate the advanced, realistic constraints described in Chapter 3 of the thesis into the EPDT algorithm, and details the MILP formulation from Chapter 6 and the data-driven approaches from Chapter 7.

## 6. Advanced Data-Driven Approaches for Dynamic/Stochastic VRP (from Chapter 7)

**Objective:** To improve solution quality in dynamic and stochastic environments by proactively anticipating future demand using historical data.

### 6.1. Approach 1: Representative Orders
- [ ] **Goal:** Guide the heuristic to position vehicles in strategic space-time locations by introducing artificial "representative" orders into the problem instance.
- [ ] **Implementation Steps:**
    - [ ] **Historical Data Clustering:** Implement a module that uses a K-Means-based clustering technique to analyze historical order data (pickup/delivery coordinates, time of request).
    - [ ] **Generate Representative Orders:** Create new `Order` objects from the centroids of the resulting clusters. These orders are artificial and represent demand hotspots.
    - [ ] **Augment VRP Instance:** Add a mechanism to inject these representative orders into the problem instance that is fed to the heuristic.
    - [ ] **Calculate Reliability Factor:** For each cluster (and its representative order), implement the calculation of a `reliability_factor` (`R(C) = a * F(C) - b * S(C)`), which balances the cluster's frequency (`F(C)`) and sparsity (`S(C)`).
    - [ ] **Implement Anticipatory Strategies:**
        - **Relocation:** After the main optimization, use the `reliability_factor` to decide whether to keep a representative order in the solution, effectively relocating an idle vehicle to its location.
        - **Waiting:** Modulate the vehicle waiting time before it departs for a representative order based on the order's `reliability_factor`. Lower reliability should result in a longer waiting time.
    - [ ] **Action:** Add a configuration step or a separate utility for tuning the `a` and `b` parameters of the reliability factor. Their values will significantly impact the behavior of the anticipatory strategies.

### 6.2. Approach 2: Accessibility-Based Scoring
- [ ] **Goal:** Enhance the route evaluation function to favor routes that are better positioned to intercept future requests.
- [ ] **Implementation Steps:**
    - [ ] **Pre-calculate Accessibility Map:** Implement a module to pre-compute an accessibility measure `Φi(t)` for all locations `i` and discretized time points `t`. This value, based on the logit model from urban logistics, represents how easily a vehicle at `(i, t)` can reach potential future orders based on historical data.
    - [ ] **Modify Second-Level Score Function:** Enhance `calculate_z2_score` by adding a new term that sums the accessibility values `Φi(t)` for each task in the route. The modified score will be `Ž2(r) = Z2(r) + φ(r)`, where `φ(r)` is the cumulative accessibility. This will make routes that travel through high-accessibility areas more attractive.
    - [ ] **Implementation Note:** The accessibility map `Φi(t)` should be pre-computed and stored, as its calculation is too computationally intensive to be performed in real-time during the main heuristic run. This should be an offline process that runs on the historical dataset.

## 13. Fix `l1_heuristic` Usage in Test Runner

**Objective:** Instruct a large language model (LLM) to fix the usage of the `l1_heuristic` in `tests/run_scenario_test.py`, as the current implementation incorrectly reports that the heuristic is not implemented.

- [ ] **Problem:** The test runner script (`tests/run_scenario_test.py`) currently uses a mock solution because it fails to correctly import and use the `l1_heuristic` from `algo/first_level.py`. The output shows the warning: "`l1_heuristic` not yet implemented".
- [ ] **Goal:** Modify the script to correctly call the `l1_heuristic` and process its results, removing the mock solution fallback.
- [ ] **Instructions for LLM:**
    1.  **Analyze `tests/run_scenario_test.py`:** Identify the section where the `l1_heuristic` is called.
    2.  **Remove Mock Solution:** Delete the `if l1_heuristic is None:` block and the call to `_create_mock_solution`.
    3.  **Ensure Correct Call:** The script should now directly call `solution = l1_heuristic(orders, vehicles, params)`.
    4.  **Parameter Configuration:** The `configure_algorithm_parameters` function should be used to set up the `params` object that is passed to the heuristic. Ensure that the parameters are appropriate for a test run (e.g., a reasonable number of iterations).
    5.  **Run and Validate:** After the changes, the script should be executed. The expected outcome is that the real `l1_heuristic` runs, and the `print_solution_summary` function displays the actual results from the solver, not from a mock object.

## 14. Solver Performance and Assignment Analysis

**Objective:** Investigate why not all tasks are being assigned by the heuristic solver, improve the assignment rate, and reduce the overall solve time.

- [ ] **Problem:** The current heuristic solver has a high runtime (over 200 seconds) and fails to assign all tasks, whereas the mock fallback solution successfully assigns all tasks. This indicates potential issues with the heuristic's feasibility checks or search strategy.
- [ ] **Goal:** Achieve a total assignment of all tasks, similar to the mock solution, and significantly reduce the solver's execution time.

- [ ] **Investigation and Improvement Steps:**
    1.  **Analyze Unassigned Tasks:**
        - **Action:** In the `print_solution_summary` function, add a detailed analysis of the unassigned orders. For each unassigned order, print its properties (weight, volume, priority, etc.) and the potential reasons why it might have been rejected (e.g., "too large for any vehicle," "no feasible insertion point found").
    2.  **Compare with Mock Solution:**
        - **Action:** Run the `_create_mock_solution` function from the test runner to generate a baseline "perfect" assignment. Compare the routes and vehicle loads from the mock solution with the heuristic's solution to identify key differences in how tasks are distributed.
    3.  **Review Heuristic Feasibility Checks:**
        - **Action:** Carefully review the feasibility checks in `algo/second_level.py`, particularly the `is_feasible` function and the HoS simulation (`_simulate_hos_advanced`). It's possible that overly strict or incorrect constraints are preventing valid task insertions.
    4.  **Profile and Optimize:**
        - **Action:** Use a profiler (e.g., `cProfile`) to identify the most time-consuming parts of the `l1_heuristic`. The high runtime suggests that there may be bottlenecks in the neighborhood generation, scoring functions, or local search.
        - **Action:** Based on the profiling results, explore optimization strategies such as:
            - Using more efficient data structures.
            - Reducing the size of the search neighborhoods.
            - Implementing faster, approximate scoring functions for candidate evaluation.

## 15. Interactive Map Visualization

**Objective:** Integrate the interactive map visualization functionality into the main solver and test runner to generate an HTML map of the solution.

### 15.1. Create a new module `algo/solution_visualizer.py`

- [ ] **Action:** Create a new Python script `algo/solution_visualizer.py` that will contain the adapted map visualization logic.
- [ ] **Details:**
    1.  This module will import the necessary libraries (`folium`, `matplotlib`, etc.) and the EPDT data structures (`Solution`, `Route`, `Task`, `Vehicle`).
    2.  It will contain a new class, `EPDTMapVisualizer`, adapted from the `VRPMapVisualizer` in `src/vrp_map_visualization.py`.

### 15.2. Adapt `VRPMapVisualizer` to `EPDTMapVisualizer`

- [ ] **Action:** Modify the `create_interactive_map` function to work with the EPDT `Solution` object.
- [ ] **Logic:**
    1.  The function signature will be `create_interactive_map(solution: Solution, save_path: str)`.
    2.  The function will iterate through the `solution.routes` dictionary to get the routes for each vehicle.
    3.  The map will be centered based on the coordinates of the tasks in the solution.
    4.  **Markers:** For each task in each route, a marker will be placed on the map. The marker's popup will display:
        -   Task ID, Order ID, and Location ID.
        -   Load (demand and volume).
        -   Expected arrival time (from the task's time window).
        -   Actual arrival time (calculated from the HoS simulation).
    5.  **Routes:** The routes will be color-coded for each vehicle.
    6.  **Legend:** An interactive legend will be created with clickable items for each vehicle. Clicking on a vehicle in the legend will toggle the visibility of its corresponding route and markers on the map.

### 15.3. Integration with Test Runner (`tests/run_scenario_test.py`)

- [ ] **Action:** Modify the `run_scenario_test.py` script to generate and save the interactive map.
- [ ] **Logic:**
    1.  After the `l1_heuristic` returns a solution, call the `create_interactive_map` function from the new `solution_visualizer` module.
    2.  The map will be saved to the `results` directory with a filename that includes the scenario name and a timestamp (e.g., `furgoni_solution_map_1678886400.html`).
    3.  The path to the generated map file will be printed to the console at the end of the test run.

## 16. Comprehensive Integration Test

**Objective:** Create a new test file that runs a single, comprehensive, full-featured scenario to validate the integration of all advanced features.

- [ ] **Create `tests/comprehensive_integration_test.py`:**
    - **Action:** Develop a new Python script that mimics the structure of `tests/run_scenario_test.py` but is designed for a single, all-encompassing test run rather than multiple small, isolated tests.

- [ ] **Phase 1: Heuristic Solver Test**
    1.  **Load Scenario:** Use `create_scenario_from_excel` to load vehicle and order data from `src/furgoni.xlsx`.
    2.  **Run Heuristic:** Execute the `l1_heuristic` to generate a `Solution` object containing optimized vehicle routes (drivers are not assigned at this stage).
    3.  **Validate Routes:** Print a summary of the routes, including the number of tasks, total distance, and feasibility, to ensure the heuristic solver is working correctly.

- [ ] **Phase 2: Driver Assignment Integration**
    1.  **Load Drivers:** Use `load_drivers_from_excel_enhanced` to load the list of `EnhancedDriver` objects.
    2.  **Enhance Qualifications:** Call the `enhance_drivers_with_vehicle_capabilities` utility to ensure drivers are qualified for the vehicles in the generated routes.
    3.  **Run Assignment:** Pass the routes from the heuristic solution and the list of enhanced drivers to the `assign_drivers_to_routes_enhanced` function.
    4.  **Print Final Summary:** Call `print_assignment_summary` to display the final, complete solution with drivers assigned to routes.

## 19. Debug and Resolve Assignment Failures

**Objective:** Systematically debug the `l1_heuristic` to understand why it fails to assign all orders and implement fixes to achieve a 100% assignment rate.

- [ ] **Problem:** The `comprehensive_integration_test.py` output reveals that a significant number of orders are left unassigned by the heuristic. This indicates that the solver's constraints or search operators are too restrictive, preventing valid insertions.
- [ ] **Goal:** Identify the exact reasons for assignment failures and modify the heuristic to assign all orders, mirroring the behavior of a baseline or mock solution.

- [ ] **Debugging and Resolution Steps:**

    1.  **Instrument the Heuristic for Detailed Logging:**
        - **Action:** Modify the `l1_heuristic` and its sub-components (especially `best_insertion_initializer` and neighborhood search functions) to produce detailed logs.
        - **Logging Details:**
            - When an order is considered for insertion, log the order ID and the vehicle being tested.
            - If an insertion fails, log the *exact* reason (e.g., "Capacity violation: weight exceeds limit," "Time window violation: arrival at task X is too late," "HOS violation: driver runs out of driving time").
            - To do this, the `is_feasible` function in `algo/second_level.py` should be modified to return not just `True` or `False`, but a tuple `(bool, str)`, where the string contains the reason for failure.

    2.  **Analyze Unassigned Orders Systematically:**
        - **Action:** Enhance the `print_route_validation_summary` function in `tests/comprehensive_integration_test.py`.
        - **Enhancement:**
            - Create a set of all order IDs at the beginning of the function.
            - As you iterate through the solution's routes, remove the assigned order IDs from this set.
            - After the loop, the set will contain only the unassigned order IDs.
            - Iterate through the unassigned orders and print a detailed profile for each one, including:
                - Order ID, weight, volume, pickup/delivery locations.
                - Time windows for both pickup and delivery tasks.
                - Any other relevant properties (e.g., priority).

    3.  **Create a Focused Debugging Test:**
        - **Action:** Create a new test script, `tests/debug_order_assignment.py`, that focuses on a *single* unassigned order.
        - **Test Logic:**
            1.  Load the full scenario (`furgoni.xlsx`).
            2.  Identify one of the unassigned orders from the comprehensive test output.
            3.  Run a simplified version of the heuristic that attempts to insert only this specific order into every available vehicle's route.
            4.  Use the detailed logging from step 1 to trace exactly why the insertion fails for each vehicle. This will isolate the constraint or logic that is causing the issue.

    4.  **Review and Refine Constraints:**
        - **Action:** Based on the debugging output, critically review the constraints in `algo/second_level.py`.
        - **Common Issues to Look For:**
            - **Overly Strict Time Windows:** Are the time windows too narrow, making it impossible to arrive on time?
            - **Incorrect Capacity Calculations:** Is the vehicle's remaining capacity being calculated correctly after each task?
            - **Flawed HOS Simulation:** Is the Hours of Service simulation prematurely marking routes as infeasible? Check for off-by-one errors or incorrect time accumulation.
            - **Depot Return Time:** Is the heuristic correctly calculating the time required to return to the depot at the end of the route?

    5.  **Iterate and Validate:**
        - **Action:** After applying a fix, re-run the `comprehensive_integration_test.py` to see if the assignment rate improves.
        - **Goal:** Continue this cycle of debugging, fixing, and validating until all orders are successfully assigned.

## 17. Enforce Pickup-Before-Delivery Precedence Constraint

**Objective:** Fix a critical bug in the route validation logic that allows the solver to create infeasible routes where delivery tasks are performed before all pickup tasks have been completed. This violates the fundamental EPDT problem definition from Chapter 3 of the thesis.

- [ ] **Problem:** The current implementation in `algo/second_level.py` does not enforce the rule that all pickups on a route must precede all deliveries. This leads to invalid solutions.
- [ ] **Goal:** Modify the route feasibility check to correctly identify and reject routes that interleave pickups and deliveries.

- [ ] **Implementation Steps:**
    1.  **Locate the Simulation Loop:**
        - **File:** `algo/second_level.py`
        - **Function:** Find the primary function responsible for checking route feasibility (likely `is_feasible` or a similar function that iterates through a route's tasks chronologically).
    2.  **Implement Precedence Check Logic:**
        - **Action:** Inside the task iteration loop, introduce a new boolean state variable, e.g., `delivery_phase_started = False`.
        - **Logic:**
            - Before the loop begins, initialize `delivery_phase_started = False`.
            - For each `task` in the route's sequence:
                - If the `task.type` is a delivery (e.g., `task.is_delivery()` returns `True`), set `delivery_phase_started = True`.
                - If the `task.type` is a pickup AND `delivery_phase_started` is already `True`, this is a constraint violation. The function must immediately return `False` for the route's feasibility.
        - **Success Condition:** If the loop completes without ever triggering the violation, the route respects the precedence constraint.
    3.  **Create a Validation Test:**
        - **File:** `tests/test_route_validation.py` (or create a new dedicated test file).
        - **Action:** Add a new test case `test_pickup_before_delivery_precedence`.
        - **Logic:**
            - Manually construct a `Route` object.
            - Add tasks to it in an interleaved sequence (e.g., Pickup A -> Pickup B -> Delivery A -> Pickup C).
            - Call the `is_feasible` function on this route.
            - Assert that the function correctly returns `False`.
            - Construct a second, valid route (e.g., Pickup A -> Pickup B -> Pickup C -> Delivery A) and assert that `is_feasible` returns `True`.

## 18. Optimize OSRM Usage and First-Run Performance

**Objective:** Drastically reduce the number of OSRM HTTP calls during the initial run of the heuristic solver to ensure a reasonable runtime, even with an empty cache.

**Phase 1: Pre-computation and Caching**

1.  **Implement a Standalone Pre-computation Script:**
    *   **Action:** Create a new script, `utils/precompute_routes.py`.
    *   **Goal:** This script will be run *before* the main heuristic. It will identify all unique pairs of locations in a given scenario and pre-populate the `moda_routes.db` cache with their OSRM data.
    *   **Implementation:**
        1.  The script will take a scenario file (e.g., `src/furgoni.xlsx`) as input.
        2.  It will extract all unique location coordinates from the orders and depots.
        3.  It will generate all possible pairs of these unique locations.
        4.  For each pair, it will call the `_query_osrm_and_cache` function (which should be moved from `algo/route_provider.py` to a more accessible utility module) to fetch the OSRM data and save it to the database.
    *   **Benefit:** This turns the expensive, online OSRM calls into a one-time, offline pre-computation step.

**Phase 2: Intelligent Neighborhood Search**

1.  **Introduce a "Delta Evaluation" in Neighborhood Functions:**
    *   **Action:** Modify the neighborhood generation functions in `algo/first_level.py` (e.g., `single_order_relocation_neighborhood`).
    *   **Goal:** Instead of re-calculating the entire route cost from scratch for every neighbor, calculate only the *change* in cost (the "delta").
    *   **Implementation:**
        1.  When an order is moved from route A to route B, the change in cost is primarily related to the edges that are removed and added.
        2.  The cost of removing the order from route A can be calculated by subtracting the travel time of the two edges connected to its tasks.
        3.  The cost of inserting the order into route B can be calculated by adding the travel time of the two new edges.
        4.  This requires only a few lookups from the now pre-populated cache, instead of re-simulating the entire route.
    *   **Benefit:** This will reduce the number of calls to the `calculate_z2_score` function, which is a major source of the redundant calculations.

2.  **Limit the Scope of Neighborhood Searches:**
    *   **Action:** Add parameters to the `l1_heuristic` to control the size and scope of the neighborhood searches.
    *   **Goal:** Prevent the solver from exploring an excessive number of neighbors, especially in the early stages of the search.
    *   **Implementation:**
        1.  Introduce a `max_neighbors_to_evaluate` parameter. The neighborhood functions will stop generating new neighbors after this limit is reached.
        2.  Implement a "best `k` insertions" strategy for the `best_insertion_initializer`. Instead of trying to insert an order into every possible position, only evaluate the `k` most promising positions (e.g., based on Euclidean distance as a cheap proxy).
