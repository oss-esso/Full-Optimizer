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
tions" strategy for the `best_insertion_initializer`. Instead of trying to insert an order into every possible position, only evaluate the `k` most promising positions (e.g., based on Euclidean distance as a cheap proxy).

## 20. Advanced Order Insertion Strategies for Large Orders

**Objective:** Address the failure to assign large or constrained orders by implementing more advanced insertion heuristics that can intelligently re-organize the solution, as inspired by academic literature on VRPs (e.g., Gastaldon, 2018).

- [ ] **Problem:** The current `best_insertion` heuristic is a greedy, myopic approach. It assigns the "easiest" orders first, which can fill up vehicles in a suboptimal way. When a large, difficult-to-place order is considered later, there may be no single vehicle that can accommodate it, even if a combination of smaller orders could be rearranged across the fleet to make space.
- [ ] **Goal:** Implement and test two advanced strategies—Regret-k Insertion and Destroy and Repair—to improve the assignment rate for large orders and find higher-quality solutions.

### 20.1. Implement Regret-k Insertion Heuristic

- [ ] **Goal:** Prioritize orders that have the fewest good placement options (i.e., a high "regret" if they are not placed in their best possible route). This prevents the solver from using up the best slots on cheap, easy-to-place orders.
- [ ] **Action:** Create a new initialization function `regret_k_initializer` in `algo/first_level.py`.
- [ ] **Logic:**
    1.  For each unassigned order, calculate the cost of inserting it into every feasible position in every vehicle's route. Store all these potential insertion costs.
    2.  For each order, find its best insertion cost (`cost_1`) and its `k`-th best insertion cost (`cost_k`). A `k` value of 2 or 3 is common.
    3.  Calculate the **regret value** for each order: `regret = cost_k - cost_1`. A high regret value means the order has one very good position and the next best options are much worse.
    4.  In a loop, select the order with the **highest regret value** and place it in its best position (`cost_1`).
    5.  Update the routes and repeat until all orders are assigned.
- [ ] **Integration:** Modify `l1_heuristic` to allow selecting `regret_k_initializer` as the `initialization_method` via the `params` dictionary.

### 20.2. Implement Destroy and Repair Operator

- [ ] **Goal:** When a large order cannot be assigned, intelligently "destroy" a part of the existing solution to "repair" it by inserting the difficult order.
- [ ] **Action:** Create a new module `algo/destroy_and_repair.py` and integrate it into the `l1_heuristic` loop.
- [ ] **Logic:**
    1.  **Trigger:** This operator is called from `l1_heuristic` if, after the initialization phase, there are still unassigned orders.
    2.  **Select Order:** Pick the largest (or highest priority) unassigned order, let's call it `Order_X`.
    3.  **Destroy Phase:**
        -   Identify a target vehicle `V` that is the "closest" to being able to handle `Order_X` (e.g., it has the most compatible capacity type but is slightly overloaded).
        -   From vehicle `V`'s route, remove a set of smaller, already-assigned orders. The orders to be removed can be selected based on:
            -   **Proximity:** Orders whose pickup/delivery locations are near `Order_X`'s locations.
            -   **Cost:** Orders that are the "cheapest" to re-insert elsewhere.
    4.  **Repair Phase:**
        -   Attempt to insert `Order_X` into the now-emptier route of vehicle `V`.
        -   If successful, take the orders that were removed during the "destroy" phase and attempt to re-insert them into the solution using the standard `best_insertion` or `regret_k_insertion` logic.
- [ ] **Integration:** The `l1_heuristic` will call `destroy_and_repair(solution, unassigned_orders)` if necessary. The function will modify the `solution` object in place.

### 20.3. Update Configuration and Testing

- [ ] **Action:** Update `configure_algorithm_parameters` in `tests/comprehensive_integration_test.py`.
- [ ] **Details:**
    -   Add a new parameter `initialization_method` that can be set to `'best_insertion'` or `'regret_k'`.
    -   Add a boolean parameter `enable_destroy_and_repair` to control whether the new operator is used.
- [ ] **Action:** Create a new test file `tests/test_large_order_assignment.py` that specifically loads a scenario with known difficult orders and asserts that the new heuristics can successfully assign them where the old one failed.tions" strategy for the `best_insertion_initializer`. Instead of trying to insert an order into every possible position, only evaluate the `k` most promising positions (e.g., based on Euclidean distance as a cheap proxy).

## 21. Validate Travel Time Calculations and Depot Operations

**Objective:** Ensure the accuracy of travel time calculations by cross-validating OSRM and Haversine methods, and verify that all routes correctly start and end at a depot or depot bay.

- [ ] **Problem:** There are discrepancies in travel time calculations, and it's unclear if the Haversine fallback is being used correctly. Additionally, route validation needs to be stricter about depot starts and ends.
- [ ] **Goal:** Implement a robust validation system to catch inconsistencies in travel times and ensure all routes adhere to depot constraints.

- [ ] **Implementation Steps:**

    1.  **Cross-Validate Travel Times:**
        -   **Action:** In `algo/route_provider.py`, when `USE_OSRM` is `True`, after fetching a route from OSRM, also calculate the Haversine distance for the same pair of coordinates.
        -   **Logging:** Log both the OSRM time and the Haversine time, along with a percentage difference. If the difference exceeds a certain threshold (e.g., 50%), log a warning.
        -   **Action:** In `tests/run_scenario_test.py`, enhance the `print_solution_summary` function. When printing the route details, for each leg of the journey, display both the OSRM-based travel time and the Haversine-based travel time. This will make it easy to spot significant discrepancies during testing.

    2.  **Strengthen Depot Start/End Validation:**
        -   **Action:** In `algo/second_level.py`, within the `is_feasible` function, add an explicit check at the beginning and end of the task sequence.
        -   **Logic:**
            -   The first task in a route's task list *must* be a depot start task (e.g., `task.is_depot_start()` is `True`).
            -   The last task in a route's task list *must* be a depot return task (e.g., `task.is_depot_return()` is `True`).
            -   If either of these conditions is not met, the route should be marked as infeasible.
        -   **Action:** Create a dedicated test in `tests/test_route_validation.py` to ensure this check works correctly. Create a route that does not start with a depot task and another that does not end with one, and assert that `is_feasible` returns `False` for both.

## 22. Replace Photon with Nominatim for Geocoding

**Objective:** Standardize the geocoding service to Nominatim to ensure consistency and reliability, removing the deprecated Photon service.

- [ ] **Problem:** The current geocoding implementation in `utils/scenario_creator.py` uses a mix of services and fallbacks, including the less reliable Photon service. The `test_photon_geocoding.py` script highlights that Nominatim is a more robust alternative.
- [ ] **Goal:** Refactor the `get_coordinates` function in `utils/scenario_creator.py` to exclusively use Nominatim, adopting the best practices identified in the `test_nominatim_geocoding` function.

- [ ] **Implementation Steps:**

    1.  **Refactor `get_coordinates` in `utils/scenario_creator.py`:**
        -   **Action:** Replace the entire body of the `get_coordinates` function with a new implementation based on the `test_nominatim_geocoding` function from `tests/test_photon_geocoding.py`.
        -   **Key Logic to Port:**
            -   Use the Nominatim API endpoint: `https://nominatim.openstreetmap.org/search`.
            -   Set a proper `User-Agent` header in the request to comply with Nominatim's usage policy (e.g., `'EPDT-Scenario-Creator/1.0'`).
            -   Construct the request parameters as a dictionary (`'q': address, 'format': 'json', etc.`).
            -   Implement a simple retry mechanism with a delay (`time.sleep`) to handle `GeocoderTimedOut` or `GeocoderServiceError` exceptions gracefully.
            -   After a successful API call, parse the JSON response to extract the latitude and longitude from the first result.
            -   Ensure the function continues to use the existing caching mechanism: check the cache before the API call and save the result to the cache after a successful call.

    2.  **Create a New Geocoding Validation Test:**
        -   **Action:** Create a new test file, `tests/test_scenario_geocoding.py`.
        -   **Goal:** This test will validate that the refactored `scenario_creator.py` correctly geocodes addresses from an Excel file.
        -   **Test Logic:**
            1.  The test should use a small, sample Excel file (`tests/sample_scenario_for_geocoding.xlsx`) with a few addresses.
            2.  It should call the `create_scenario_from_excel` function from `utils/scenario_creator.py`.
            3.  **Assertions:**
                -   Assert that the function returns a list of `Order` objects.
                -   For each `Order`, iterate through its `Task` objects.
                -   Assert that each `Task` has valid latitude and longitude attributes (i.e., they are not `None` and are within the valid range for coordinates).
                -   Check that the `geocode_cache.json` file is created or updated.
            4.  **Cache Testing:**
                -   Run the `create_scenario_from_excel` function a second time.
                -   Use a mock to verify that the geocoding API (`requests.get`) is *not* called on the second run, proving that the results were successfully retrieved from the cache.


## 23 Guide for Interpreting the Excel Scenario File (`src/furgoni2.xlsx`)

This guide provides detailed instructions for a coding agent on how to parse the three sheets of the Excel file (`CONSEGNE`, `AUTISTI`, `VEICOLI`) into the required Python data structures (`Order`, `Task`, `Driver`, `Vehicle`).

### 1. The `CONSEGNE` Sheet

**Purpose:** This sheet defines all individual tasks (pickups or deliveries). Each row is a single task. Tasks are grouped into `Order` objects based on the `ORDER_ID` column.

**Column-by-Column Interpretation:**

-   **`ORDER`**: **(Primary Key for Grouping)** Read this string value. Use `pandas.groupby('ORDER_ID')` to iterate through all rows belonging to the same order. For each group, create one `Order` object.
-   **`COMPANY`**: Read as a string. This is descriptive information for the task.
-   **`STREET`, `HOUSE NUMBER`, `CITY`, `PROVINCE`, `POSTAL CODE`, `COUNTRY`**: Read these as strings. Concatenate them into a single, comma-separated address string (e.g., "Via Roma 10, 10121, Torino, TO,  Italy") to be passed to the geocoding service.
-   **`EARLIEST DAY`, `LATEST DAY`**: Parse as integers. These define the multi-day time window for the task. Store them in the `Task` object.
-   **`TIME WINDOW START`, `TIME WINDOW END`**: Parse these time strings (format `HH:MM:SS`). Convert them into total minutes from the start of the day (e.g., `09:00:00` becomes `540`). Store these integer values in the `Task` object.
-   **`SERVICE TIME`**: Parse as a float. This is the time in minutes required to complete the task at the location.
-   **`TASK`**: Read the string. If it is `"DELIVERY"`, map it to `TaskType.DELIVERY`. If it is `"PICKUP"`, map it to `TaskType.PICKUP`.
-   **`LOAD KG`, `LOAD VOLUME M^3`, `PALLETS`**: Parse these as numeric values. **Crucially**, for `DELIVERY` tasks, these values must be made **negative** (e.g., `-150.5`) to signify that capacity is being freed up. For `PICKUP` tasks, they remain **positive**.
-   `"LOW_TEMP, LOADER"` , '"HANGERS"` store them in the `Task` object. This will be used for matching with vehicle and driver capabilities.

### 2. The `AUTISTI` Sheet

**Purpose:** This sheet defines the available drivers. Each row corresponds to a single `Driver` object.

**Column-by-Column Interpretation:**

-   **`LICENSE PLATE`**: Read as a string. This is the number plate of the driver's **preferred vehicle**. Store this to create an initial, default pairing between a driver and a vehicle.
-   **`DRIVER`**: Read as a string. This is the unique identifier for the `Driver` object.
-   **`LICENSE`**: Read as a string (e.g., `"B"`, `"C"`, `"CE"`). This is a **hard constraint**. The algorithm must ensure that a driver is only ever assigned to a vehicle that their license permits them to drive.
-   **`COST PER HOUR`**: Parse as a float. This is the driver's hourly wage and is a key component of the total route cost.

### 3. The `VEICOLI` Sheet

**Purpose:** This sheet defines the vehicle fleet. Each row corresponds to a single `Vehicle` object.

**Column-by-Column Interpretation:**

-   **`NUMBER PLATE`**: Read as a string. This is the unique identifier for the `Vehicle` object.
-   **`TYPE OF VEHICLE`**: Read as a string (e.g., `"Van"`, `"Truck"`). This will be used to check compatibility against a driver's `LICENSE`.
-   **`MAX LOAD KG`, `MAX LOAD VOLUME M^3`, `PALLET`**: Parse as numeric types. These are the capacity limits of the vehicle.
-   **`COST PER KM`, `FIXED COST`**: Parse as floats. These are the vehicle-specific operational costs.
-   `"LOW_TEMP, LOADER"` , '"HANGERS"` store them in the `Vehicle` object. This will be used for matching with tasks.

-   **`REGULATIONS`**: Parse as a boolean (`YES`/`NO`). This determines which set of Hours of Service (HOS) rules apply to the vehicle's route.

- **`LAST IN FIRST OUT`**: used for the Vehicle object to define the LIFO loading

## 24. Enhance Driver Summary with Daily Work/Drive Time

**Objective:** Add a detailed, day-by-day breakdown of work and drive time to the final driver assignment summary to provide a clearer overview of each driver's workload.

- [ ] **Problem:** The current driver assignment summary only shows which driver is assigned to which vehicle. While the main route validation summary contains detailed HoS (Hours of Service) data, it's not presented in the final, driver-centric summary, making it difficult to quickly assess if a driver's schedule is balanced and compliant over the multi-day planning horizon.
- [ ] **Goal:** Modify the `print_assignment_summary` function to include a daily breakdown of work time and drive time for each assigned driver, similar to the data shown in the HoS violation reports.

- [ ] **Instructions for LLM:**

    1.  **Locate the Data Source:**
        -   **File:** `algo/second_level.py`
        -   **Function:** `_simulate_hos_advanced`
        -   **Analysis:** Examine this function to understand how it calculates and stores the daily work and drive times. It likely returns a data structure (e.g., a dictionary or a list of objects) containing the simulation results for each day of the route. This data is then attached to the `Route` object, likely in a field like `route.hos_daily_summary`.

    2.  **Find the Summary Generation Function:**
        -   **File:** `tests/comprehensive_integration_test.py` (or a similar test runner script where driver assignments are printed).
        -   **Function:** `print_assignment_summary` (or a similar named function).
        -   **Analysis:** This function currently iterates through the assigned routes and prints the driver-vehicle pairing.

    3.  **Modify the Summary Output:**
        -   **Action:** Inside the `print_assignment_summary` function, for each `route` in the solution, access the stored HoS simulation results.
        -   **Logic:**
            -   After printing the driver and vehicle, iterate through the daily summary data stored on the `route` object.
            -   For each day, print the `Day`, `Work Time`, and `Drive Time` in a clear, indented format.
        -   **Example Output:**
            ```
            Vehicle FX194HX    → Driver Martinas 1      (License: CE, Type: heavy)
                - Day 1: Work: 480m, Drive: 240m
                - Day 2: Work: 520m, Drive: 300m
                - Day 3: Work: 700.8m, Drive: 640.8m (HOS VIOLATION)
            ```
    4.  **Handle Feasibility Status:**
        -   **Action:** In the daily breakdown, if a day's work or drive time exceeds the legal limits, highlight it as a "HOS VIOLATION" as shown in the example. The feasibility status and reason should be available from the `is_feasible` check which is also stored on the route object.

## 25. Implement Soft Constraint for HOS Violations

**Objective:** Convert the hard Hours of Service (HOS) constraint into a soft constraint with a configurable penalty, allowing the solver to create routes with minor HOS violations if it leads to a better overall solution.

- [ ] **Problem:** Currently, any HOS violation makes a route completely infeasible. This is too rigid and can prevent the solver from finding good solutions, especially when a small amount of overtime for one driver could prevent the need for an entirely new vehicle.
- [ ] **Goal:** Implement a penalty-based system where HOS violations add to the route's total cost instead of making it invalid. This provides a "tweaking knob" to control how strictly HOS rules are enforced.

- [ ] **Instructions for LLM:**

    1.  **Add New Configuration Parameter:**
        -   **File:** `tests/comprehensive_integration_test.py`
        -   **Function:** `configure_algorithm_parameters`
        -   **Action:** Add a new parameter `hos_violation_penalty_per_minute` and set it to a reasonable default value (e.g., `100.0`). This will be the cost for every minute a driver goes over the allowed work or drive time.

    2.  **Modify HOS Simulation to Return Violation Minutes:**
        -   **File:** `algo/second_level.py`
        -   **Function:** `_simulate_hos_advanced`
        -   **Action:** Modify the function to return not just a boolean indicating feasibility, but the total number of minutes the route is in violation.
        -   **Logic:**
            -   Inside the simulation, if the daily drive or work time exceeds the limits, calculate the difference (the number of minutes in violation).
            -   The function should return a tuple: `(is_feasible, violation_minutes, reason)`. If there is no violation, `violation_minutes` should be `0`.

    3.  **Update Feasibility Check to Calculate Penalty:**
        -   **File:** `algo/second_level.py`
        -   **Function:** `is_feasible`
        -   **Action:** Modify this function to use the new return value from `_simulate_hos_advanced`.
        -   **Logic:**
            -   Call `_simulate_hos_advanced` and get the `violation_minutes`.
            -   Instead of returning `False` immediately on an HOS violation, the function should now always return `True` for HOS checks, but it should also return the calculated `violation_minutes`. The function signature might change to return a dictionary of penalties, e.g., `{'hos_penalty': total_hos_penalty}`.

    4.  **Integrate Penalty into the Main Score Function:**
        -   **File:** `algo/second_level.py`
        -   **Function:** `calculate_z2_score`
        -   **Action:** Modify the score calculation to include the new HOS penalty.
        -   **Logic:**
            -   The `calculate_z2_score` function will now need to get the HOS violation penalty. It might need to call a modified `is_feasible` or another helper function that runs the HOS simulation.
            -   Add the HOS penalty to the total score: `total_cost += hos_violation_minutes * params['hos_violation_penalty_per_minute']`. This makes routes with HOS violations more "expensive" but not impossible.

## 26. Support Date-Based Time Windows in Scenario Creator

**Objective:** Modify the scenario creator to accept date strings (e.g., "2025-07-30") for time windows, making it easier to use with calendar-based planning.

- [ ] **Problem:** The `EARLIEST DAY` and `LATEST DAY` columns in the Excel file currently require integer day indices (1, 2, 3,...). This is not intuitive and requires manual conversion from a calendar.
- [ ] **Goal:** Enhance the `create_scenario_from_excel` function in `utils/scenario_creator.py` to automatically handle date strings, converting them into the required zero-based integer day indices for the solver.

- [ ] **Instructions for LLM:**

    1.  **Pre-scan for a Global Start Date:**
        -   **File:** `utils/scenario_creator.py`
        -   **Function:** `create_scenario_from_excel`
        -   **Action:** Before processing the rows, iterate through the `EARLIEST DAY` column of the `CONSEGNE` sheet to find the overall earliest date in the entire dataset. This date will become the reference point for the entire planning horizon (Day 0).
        -   **Logic:**
            -   Use `pd.to_datetime` to parse the values in the column, ignoring errors for now.
            -   Find the minimum (earliest) date among all valid dates found. This is your `global_start_date`.

    2.  **Modify Task Creation to Convert Dates to Day Indices:**
        -   **File:** `utils/scenario_creator.py`
        -   **Function:** `create_task_from_row` (or the main loop in `create_scenario_from_excel` that calls it).
        -   **Action:** When parsing the `EARLIEST DAY` and `LATEST DAY` for each task, convert the date string into a relative, zero-based integer day index.
        -   **Logic:**
            -   For each task's `EARLIEST DAY`, parse it into a datetime object.
            -   Calculate the difference between the task's date and the `global_start_date`. The result will be a `timedelta`.
            -   The day index is `timedelta.days`. This will be `0` for the first day, `1` for the second, and so on.
            -   Repeat for `LATEST DAY`.

    3.  **Update Absolute Time Calculation:**
        -   **File:** `utils/scenario_creator.py`
        -   **Function:** `create_task_from_row`
        -   **Action:** Adjust the formula that calculates the absolute time in minutes.
        -   **Logic:**
            -   The old formula was `(day - 1) * 1440 + time_minutes`.
            -   The new formula should be `day_index * 1440 + time_minutes`, where `day_index` is the zero-based integer calculated in the previous step.

    4.  **Ensure Backward Compatibility:**
        -   **Action:** The code should still handle integer values in the `EARLIEST DAY` and `LATEST DAY` columns gracefully.
        -   **Logic:**
            -   When parsing, check the type of the value. If it's an integer, use it directly (you may need to adjust it to be zero-based, e.g., `day_index = int_value - 1`). If it's a string, attempt to parse it as a date.
