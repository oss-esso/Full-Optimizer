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

## 13. Interactive Map Visualization

**Objective:** Integrate the interactive map visualization functionality into the main solver and test runner to generate an HTML map of the solution.

### 13.1. Create a new module `algo/solution_visualizer.py`

- [ ] **Action:** Create a new Python script `algo/solution_visualizer.py` that will contain the adapted map visualization logic.
- [ ] **Details:**
    1.  This module will import the necessary libraries (`folium`, `matplotlib`, etc.) and the EPDT data structures (`Solution`, `Route`, `Task`, `Vehicle`).
    2.  It will contain a new class, `EPDTMapVisualizer`, adapted from the `VRPMapVisualizer` in `src/vrp_map_visualization.py`.

### 13.2. Adapt `VRPMapVisualizer` to `EPDTMapVisualizer`

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

### 13.3. Integration with Test Runner (`tests/run_scenario_test.py`)

- [ ] **Action:** Modify the `run_scenario_test.py` script to generate and save the interactive map.
- [ ] **Logic:**
    1.  After the `l1_heuristic` returns a solution, call the `create_interactive_map` function from the new `solution_visualizer` module.
    2.  The map will be saved to the `results` directory with a filename that includes the scenario name and a timestamp (e.g., `furgoni_solution_map_1678886400.html`).
    3.  The path to the generated map file will be printed to the console at the end of the test run.

## 14. Comprehensive Integration Test

**Objective:** Create a new test file that runs a single, comprehensive, full-featured scenario to validate the integration of all advanced features, including the enhanced driver assignment and scenario creation from the `furgoni.xlsx` file.

- [ ] **Create `tests/comprehensive_integration_test.py`:**
    - **Action:** Develop a new Python script that mimics the structure of `tests/run_scenario_test.py` but is designed for a single, all-encompassing test run rather than multiple small, isolated tests.
    - **Key Differences from `run_scenario_test.py`:**
        - No command-line arguments for different tests.
        - The script will execute a single, hardcoded, full-featured run.

- [ ] **Test Execution Flow:**
    1.  **Load Scenario:**
        - Use the `create_scenario_from_excel` function from `utils/scenario_creator.py` to load the vehicle, order, and driver data from `src/furgoni.xlsx`.
    2.  **Load Drivers:**
        - Use the `load_drivers_from_excel_enhanced` function from `algo/driver_assignment_enhanced.py` to load the list of `EnhancedDriver` objects.
    3.  **Run First-Level Heuristic:**
        - Execute the `l1_heuristic` to generate the vehicle routes (without assigned drivers).
    4.  **Run Driver Assignment:**
        - Pass the generated routes and the list of `EnhancedDriver` objects to the `assign_drivers_to_routes_enhanced` function from `algo/driver_assignment_enhanced.py`.
    5.  **Print Summary:**
        - Call the `print_assignment_summary` function from `algo/driver_assignment_enhanced.py` to display the final driver assignments and a summary of the solution.
