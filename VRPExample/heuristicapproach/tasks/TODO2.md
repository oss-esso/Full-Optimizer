# TODO List for Advanced EPDT Implementation

This document outlines the necessary steps to integrate the advanced, realistic constraints described in Chapter 3 of the thesis into the EPDT algorithm, and details the MILP formulation from Chapter 6 and the data-driven approaches from Chapter 7.

## 1. Data Model Enhancements

**Objective:** Update the core data structures to support multi-day planning and advanced constraints.

- [ ] **`Task` Data Structure:**
    - **Action:** Add a `day` attribute to the `Task` class (e.g., in `epdt_data_structures.py`). This will be an integer representing the day relative to the planning horizon (-1 for yesterday, 0 for today, 1 for tomorrow).
    - **Reason:** This is essential for the route simulation logic to correctly sequence tasks chronologically.

- [ ] **`Vehicle` Data Structure:**
    - **Action:** Add a `lifo_required` boolean attribute to the `Vehicle` class.
    - **Action:** Add an `initial_state` attribute to the `Vehicle` class. This should store the vehicle's position and any pending tasks from the previous day at the start of the simulation.
    - **Reason:** To identify vehicles that do not have side doors and must follow a Last-In, First-Out loading policy.
    - **Reason:** To handle open routes that continue from the previous day's state.

- [ ] **`Order` Data Structure:**
    - **Action:** Add a `priority` attribute (e.g., 'mandatory', 'urgent', 'normal').
    - **Reason:** To allow the scoring functions to apply the correct penalties for unassigned orders.

## 2. Second-Level Heuristic (Intra-Route) Modifications (`algo/second_level.py`)

**Objective:** Enhance the route feasibility and cost calculation functions to handle the advanced constraints.

- [ ] **Implement Multi-Day Route Simulation:**
    - **Action:** Modify the chronological simulation logic within `is_feasible` and `calculate_z2_score`.
    - **Details:**
        1.  The simulation must start from the vehicle's `initial_state`.
        2.  It must process tasks in strict chronological order: yesterday's tasks, then today's, then tomorrow's.
        3.  The Hours-of-Service (`_check_hos`) simulation must be initialized with the driver's state from the end of the previous day.

- [ ] **Implement LIFO Loading Constraint:**
    - **Action:** Add a new check within the `is_feasible` function.
    - **Logic:**
        1.  If `route.vehicle.lifo_required` is `True`, simulate the loading process.
        2.  Use a stack (`cargo_stack`) to track the `order_id` of loaded items.
        3.  When a pickup task is encountered, push its `order_id` onto the stack.
        4.  When a delivery task is encountered, check if its `order_id` matches the top of the stack. If not, the route is infeasible. If it matches, pop from the stack.
        5.  The route is only feasible if the stack is empty at the end.

- [ ] **Implement Prospective Cost Calculation:**
    - **Action:** Enhance the `calculate_z2_score` function.
    - **Logic:** The `A(r)` (prospective cost) component should be calculated for routes that have "tomorrow" tasks. This can be estimated as the travel time/distance from the location of the last "today" task to the locations of all "tomorrow" tasks, respecting their internal sequence.

- [ ] **Implement Soft Time Window Penalties:**
    - **Action:** Enhance the `calculate_z2_score` function.
    - **Logic:** For tasks with soft time windows, calculate the delay beyond the window and apply the corresponding penalty `L_i` to the `W(r)` component of the score.

- [ ] **Implement Detailed Hours of Service (HoS) Rules:**
    - **Action:** Ensure the `_check_hos` function correctly implements the European Hours of Service Regulations.
    - **Details:**
        - After 4.5 hours of driving, a 45-minute break is mandatory (can be split into 15 + 30 mins).
        - Maximum 9 hours of driving per day (extendable to 10 hours twice a week).
        - Maximum 13 hours of work per day (extendable to 14 hours twice a week).
        - Minimum 11 hours of daily rest (can be reduced to 9 hours under certain conditions).

## 3. First-Level Heuristic (Inter-Route) Modifications (`algo/first_level.py`)

**Objective:** Ensure the main search algorithm correctly uses the enhanced second-level logic.

- [ ] **Update `calculate_z1_score`:**
    - **Action:** Modify the main scoring function to correctly penalize unassigned orders based on their new `priority` attribute.
    - **Logic:**
        - Apply a very high penalty for unassigned "mandatory" orders.
        - Apply the `Lo` penalty for unassigned "urgent" orders.
        - Apply no penalty for unassigned "normal" orders.

- [ ] **Handle Open Routes in Initializers:**
    - **Action:** Modify `best_insertion_initializer` and `round_robin_insertion_with_priority_initializer`.
    - **Logic:** The initial solution construction must respect the initial state of vehicles. Any pending "yesterday" tasks must be considered fixed and part of the initial routes before any new orders are inserted.

## 4. Test Runner (`run_scenario_test.py`)

**Objective:** Create a test script to validate the implementation of the advanced features.

- [x] **Create `run_scenario_test.py`:**
    - **Action:** Develop a new Python script dedicated to running a full scenario test.
    - **Steps:**
        1.  **Load Data:** Use `create_furgoni_scenario` from `src/moda_scenarios.py` to generate the `VRPInstance`.
        2.  **Adapt Data:** Use the `data_adapter.py` module to convert the `VRPInstance` into the `Order` and `Vehicle` lists required by the EPDT algorithm.
        3.  **Configure:** Set up the `params` dictionary to control the heuristic (e.g., iterations, tabu tenure, enabling/disabling enhancements).
        4.  **Execute:** Call `l1_heuristic` with the prepared data and parameters.
        5.  **Report:** Create a `print_solution_summary` function to display the results, including the final score, vehicle routes, and a list of unassigned orders.

## 5. MILP Formulation for Optimality Bounds (from Chapter 6)

**Objective:** To assess the performance of the EPDT heuristic, Chapter 6 of the thesis proposes using a Column Generation algorithm to find optimality bounds for the EPDT problem. This involves formulating the problem as a Mixed-Integer Linear Program (MILP).

### 5.1. Overview
- [ ] The approach is based on a **set covering formulation** of the EPDT problem. The continuous relaxation of this model is solved using Column Generation, which decomposes the problem into a Master Problem and a Pricing Problem.

### 5.2. Master Problem (MP)
- [ ] **Formulation:** The Master Problem is a **set covering model**. It is defined on a restricted set of routes initially and is augmented with new routes (columns) generated by the Pricing Problem.
- [ ] **Objective:** The objective function of the MP is to minimize the total cost, which includes route costs and penalties for soft constraint violations.
- [ ] **Variables:**
    - `yr`: A binary variable that is 1 if route `r` is selected, 0 otherwise.
    - `xo`: A binary variable that is 1 if order `o` is not served, 0 otherwise.
- [ ] **Constraints:**
    - **Set Covering:** Ensures that each mandatory task is served by at least one route. For non-mandatory tasks, the constraints allow them to be unassigned (if `xo` = 1).
    - **Fleet Size:** The total number of selected routes cannot exceed the number of available vehicles.

### 5.3. Pricing Problem (PP)
- [ ] **Formulation:** The Pricing Problem is an **Elementary Shortest Path Problem with Resource Constraints (ESPPRC)**. Its goal is to find a new route (a column) with a negative reduced cost to add to the Master Problem.
- [ ] **Solution Method:** The ESPPRC is solved using a **label correcting algorithm**. This algorithm is specifically adapted to handle the unique attributes of EPDT, which is a key contribution of the thesis.
- [ ] **Label Definition:** A label `L` for a partial path represents its state and includes:
    - The last node visited.
    - Cumulative working and driving time.
    - Current vehicle load (volume and weight).
    - The set of "open" orders (orders for which some, but not all, pickups have been made).
    - The set of "unreachable" nodes (e.g., delivery tasks for which not all pickups have been completed).
- [ ] **Dominance Rules:** Custom dominance rules are defined to efficiently prune the search space. These rules are crucial for handling the multi-pickup, multi-delivery precedence constraints, as classic dominance rules are insufficient. For an order to be considered "dominated," not only must the cost and resources be better, but the set of unreachable nodes must also be a subset, ensuring that the dominating path has at least as many feasible completions.
- [ ] **Implementation Note:** The adaptation of the label-correcting algorithm is the most challenging part of this task. The custom label structure and dominance rules described in Chapter 6 are critical for correctness and performance and must be implemented precisely.

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

