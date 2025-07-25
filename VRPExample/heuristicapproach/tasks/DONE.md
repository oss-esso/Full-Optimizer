## 7. QUBO Formulation for Quantum Annealing ✅ **COMPLETED**

**Objective:** To explore solving the EPDT problem on quantum annealing hardware, the MILP formulation must be converted into a Quadratic Unconstrained Binary Optimization (QUBO) model.

### 7.1. Overview ✅
- [x] ✅ **IMPLEMENTED:** The conversion process involves taking the constraints from the MILP and representing them as quadratic penalty terms in the objective function. This transforms the problem into a single polynomial to be minimized.
- [x] ✅ **IMPLEMENTED:** The recommended starting point is the **set partitioning** version of the problem, where each task must be served by **exactly one** route.

### 7.2. QUBO Objective Function ✅
- [x] ✅ **IMPLEMENTED:** The total QUBO objective function, or Hamiltonian (`H`), is the sum of the original MILP objective and the new penalty terms.
- [x] ✅ **IMPLEMENTED:** **`H = H_objective + H_penalty_tasks + H_penalty_fleet`**

### 7.3. Components of the QUBO Formulation ✅
- [x] ✅ **IMPLEMENTED:** **Objective Term (`H_objective`):**
    - **Action:** Directly translate the MILP objective function.
    - **Formula:** `H_objective = Σ_v Σ_r (cr * yr) + Σ_o (qo * xo)`
    - **Note:** `yr` and `xo` are the binary variables representing route selection and order rejection, respectively.
    - **File:** `algo/qubo_formulation.py` - `_formulate_objective_term()`

- [x] ✅ **IMPLEMENTED:** **Task Servicing Penalty (`H_penalty_tasks`):**
    - **Action:** For each task `i`, add a quadratic penalty that is minimized (equals zero) only when the task is serviced by exactly one route or is rejected.
    - **Formula:** `H_penalty_tasks = Σ_i P_task * ( (Σ_r air * yr + xo(i)) - 1 )^2`
    - **`P_task`:** A large positive coefficient to penalize violations of the task-servicing constraint.
    - **File:** `algo/qubo_formulation.py` - `_formulate_task_penalties()`

- [x] ✅ **IMPLEMENTED:** **Fleet Size Penalty (`H_penalty_fleet`):**
    - **Action:** Convert the fleet size inequality (`Σ_r yr <= |V|`) into an equality using binary slack variables, and then penalize its violation.
    - **Step 1: Introduce Slack Variables.** Define an integer slack variable `s` such that `Σ_v Σ_r yr + s = |V|`.
    - **Step 2: Binary Expansion.** Represent `s` using a set of binary variables `sk`: `s = Σ_k (2^k * sk)`. The number of slack variables needed depends on the maximum possible fleet size.
    - **Step 3: Formulate Penalty.** Create a quadratic penalty term.
    - **Formula:** `H_penalty_fleet = P_fleet * ( (Σ_v Σ_r yr + Σ_k (2^k * sk)) - |V| )^2`
    - **`P_fleet`:** A large positive coefficient to penalize violations of the fleet size constraint.
    - **File:** `algo/qubo_formulation.py` - `_formulate_fleet_penalties()`

### 7.4. Implementation Steps ✅
- [x] ✅ **IMPLEMENTED:** **Expand and Simplify:**
    - **Action:** Algebraically expand all the squared terms in the penalty functions. This will produce linear (diagonal) and quadratic (off-diagonal) terms for the QUBO matrix.
    - **Example:** `(a + b - 1)^2 = a^2 + b^2 + 1 + 2ab - 2a - 2b`. Since `a` and `b` are binary, `a^2 = a` and `b^2 = b`. The expression simplifies to `2ab - a - b + 1`.
    - **File:** `algo/qubo_formulation.py` - `_formulate_task_penalties()` and `_formulate_fleet_penalties()`

- [x] ✅ **IMPLEMENTED:** **Determine Penalty Coefficients:**
    - **Action:** Choose appropriate values for `P_task` and `P_fleet`.
    - **Guideline:** The penalty coefficients must be larger than any possible change in the original objective function to ensure that feasible solutions are always preferred over infeasible ones. A common rule of thumb is to set the penalty to be greater than the maximum possible value of the objective function.
    - **File:** `algo/qubo_formulation.py` - `_calculate_penalty_coefficients()`

- [x] ✅ **IMPLEMENTED:** **Construct QUBO Matrix:**
    - **Action:** Create the final QUBO matrix `Q`, where `Q_ii` will contain the linear coefficients of the binary variables and `Q_ij` will contain the quadratic coefficients.
    - **Goal:** The final model will be in the form `min(x'Qx)`, where `x` is the vector of all binary variables (`yr`, `xo`, `sk`). This matrix can then be submitted to a quantum annealer or a classical QUBO solver.
    - **File:** `algo/qubo_formulation.py` - `_construct_qubo_matrix()`

### 7.5. Implementation Files Created ✅
- [x] ✅ **`algo/qubo_formulation.py`** - Core QUBO formulation and solver integration
- [x] ✅ **`algo/qubo_integration.py`** - Hybrid Column Generation + QUBO workflow
- [x] ✅ **`algo/qubo_test.py`** - Comprehensive testing suite
- [x] ✅ **Integration with test runner** - Command line options for QUBO testing

### 7.6. Quantum Solver Support ✅
- [x] ✅ **Neal Simulated Annealing** - Classical QUBO solver (always available)
- [x] ✅ **D-Wave Quantum Annealer** - Quantum hardware support (when available)
- [x] ✅ **Exact Solver** - Small problem exact solutions
- [x] ✅ **Hybrid CG+QUBO** - Best of both classical and quantum approaches

### 7.7. Usage Examples ✅
```powershell
# Test QUBO formulation
python "testsun_scenario_test.py" --test-qubo

# Benchmark quantum vs classical
python "testsun_scenario_test.py" --test-quantum-benchmark

# Hybrid Column Generation + QUBO
python "testsun_scenario_test.py" --test-enhanced
```
## 1. Data Model Enhancements

**Objective:** Update the core data structures to support multi-day planning and advanced constraints.

- [x] **`Task` Data Structure:**
    - **Action:** Add a `day` attribute to the `Task` class (e.g., in `epdt_data_structures.py`). This will be an integer representing the day relative to the planning horizon (-1 for yesterday, 0 for today, 1 for tomorrow).
    - **Reason:** This is essential for the route simulation logic to correctly sequence tasks chronologically.

- [x] **`Vehicle` Data Structure:**
    - **Action:** Add a `lifo_required` boolean attribute to the `Vehicle` class.
    - **Action:** Add an `initial_state` attribute to the `Vehicle` class. This should store the vehicle's position and any pending tasks from the previous day at the start of the simulation.
    - **Reason:** To identify vehicles that do not have side doors and must follow a Last-In, First-Out loading policy.
    - **Reason:** To handle open routes that continue from the previous day's state.

- [x] **`Order` Data Structure:**
    - **Action:** Add a `priority` attribute (e.g., 'mandatory', 'urgent', 'normal').
    - **Reason:** To allow the scoring functions to apply the correct penalties for unassigned orders.

## 2. Second-Level Heuristic (Intra-Route) Modifications (`algo/second_level.py`)

**Objective:** Enhance the route feasibility and cost calculation functions to handle the advanced constraints.

- [x] **Add Pallet Capacity as a Hard Constraint:**
    - **Action:** Modify the `is_feasible` function to check for pallet capacity.
    - **Logic:**
        1.  Simulate the pallet load throughout the route.
        2.  If the pallet load at any point exceeds the vehicle's `pallet_capacity`, the route is infeasible.
    - **Status:** ✅ **COMPLETED** - Implemented in `algo/second_level.py`

- [x] **Move Weight to a Soft Capacity Constraint:**
    - **Action:** Modify the `is_feasible` and `calculate_z2_score` functions.
    - **Logic:**
        1.  In `is_feasible`, remove the hard check for weight capacity.
        2.  In `calculate_z2_score`, add a penalty to the score for any weight capacity violations. The penalty should be proportional to the amount of excess weight.
    - **Status:** ✅ **COMPLETED** - Implemented in `algo/second_level.py`

- [x] **Implement Multi-Day Route Simulation:**
    - **Action:** Modify the chronological simulation logic within `is_feasible` and `calculate_z2_score`.
    - **Details:**
        1.  The simulation must start from the vehicle's `initial_state`.
        2.  It must process tasks in strict chronological order: yesterday's tasks, then today's, then tomorrow's.
        3.  The Hours-of-Service (`_check_hos`) simulation must be initialized with the driver's state from the end of the previous day.

- [x] **Implement LIFO Loading Constraint:**
    - **Action:** Add a new check within the `is_feasible` function.
    - **Logic:**
        1.  If `route.vehicle.lifo_required` is `True`, simulate the loading process.
        2.  Use a stack (`cargo_stack`) to track the `order_id` of loaded items.
        3.  When a pickup task is encountered, push its `order_id` onto the stack.
        4.  When a delivery task is encountered, check if its `order_id` matches the top of the stack. If not, the route is infeasible. If it matches, pop from the stack.
        5.  The route is only feasible if the stack is empty at the end.

- [x] **Implement Prospective Cost Calculation:**
    - **Action:** Enhance the `calculate_z2_score` function.
    - **Logic:** The `A(r)` (prospective cost) component should be calculated for routes that have "tomorrow" tasks. This can be estimated as the travel time/distance from the location of the last "today" task to the locations of all "tomorrow" tasks, respecting their internal sequence.

- [x] **Implement Soft Time Window Penalties:**
    - **Action:** Enhance the `calculate_z2_score` function.
    - **Logic:** For tasks with soft time windows, calculate the delay beyond the window and apply the corresponding penalty `L_i` to the `W(r)` component of the score.

- [x] **Implement Detailed Hours of Service (HoS) Rules:**
    - **Action:** Ensure the `_check_hos` function correctly implements the European Hours of Service Regulations.
    - **Details:**
        - After 4.5 hours of driving, a 45-minute break is mandatory (can be split into 15 + 30 mins).
        - Maximum 9 hours of driving per day (extendable to 10 hours twice a week).
        - Maximum 13 hours of work per day (extendable to 14 hours twice a week).
        - Minimum 11 hours of daily rest (can be reduced to 9 hours under certain conditions).

## 3. First-Level Heuristic (Inter-Route) Modifications (`algo/first_level.py`)

**Objective:** Ensure the main search algorithm correctly uses the enhanced second-level logic.

- [x] **Update `calculate_z1_score`:**
    - **Action:** Modify the main scoring function to correctly penalize unassigned orders based on their new `priority` attribute.
    - **Logic:**
        - Apply a very high penalty for unassigned "mandatory" orders.
        - Apply the `Lo` penalty for unassigned "urgent" orders.
        - Apply no penalty for unassigned "normal" orders.

- [x] **Handle Open Routes in Initializers:**
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
- [/] The approach is based on a **set covering formulation** of the EPDT problem. The continuous relaxation of this model is solved using Column Generation, which decomposes the problem into a Master Problem and a Pricing Problem.

### 5.2. Master Problem (MP)
- [x] **Formulation:** The Master Problem is a **set covering model**. It is defined on a restricted set of routes initially and is augmented with new routes (columns) generated by the Pricing Problem.
- [x] **Objective:** The objective function of the MP is to minimize the total cost, which includes route costs and penalties for soft constraint violations.
- [x] **Variables:**
    - `yr`: A binary variable that is 1 if route `r` is selected, 0 otherwise.
    - `xo`: A binary variable that is 1 if order `o` is not served, 0 otherwise.
- [x] **Constraints:**
    - **Set Covering:** Ensures that each mandatory task is served by at least one route. For non-mandatory tasks, the constraints allow them to be unassigned (if `xo` = 1).
    - **Fleet Size:** The total number of selected routes cannot exceed the number of available vehicles.

### 5.3. Pricing Problem (PP)
- [x] **Formulation:** The Pricing Problem is an **Elementary Shortest Path Problem with Resource Constraints (ESPPRC)**. Its goal is to find a new route (a column) with a negative reduced cost to add to the Master Problem.
- [x] **Solution Method:** The ESPPRC is solved using a **label correcting algorithm**. This algorithm is specifically adapted to handle the unique attributes of EPDT, which is a key contribution of the thesis.
- [x] **Label Definition:** A label `L` for a partial path represents its state and includes:
    - The last node visited.
    - Cumulative working and driving time.
    - Current vehicle load (volume and weight).
    - The set of "open" orders (orders for which some, but not all, pickups have been made).
    - The set of "unreachable" nodes (e.g., delivery tasks for which not all pickups have been completed).
- [x] **Dominance Rules:** Custom dominance rules are defined to efficiently prune the search space. These rules are crucial for handling the multi-pickup, multi-delivery precedence constraints, as classic dominance rules are insufficient. For an order to be considered "dominated," not only must the cost and resources be better, but the set of unreachable nodes must also be a subset, ensuring that the dominating path has at least as many feasible completions.
- [x] **Implementation Note:** The adaptation of the label-correcting algorithm is the most challenging part of this task. The custom label structure and dominance rules described in Chapter 6 are critical for correctness and performance and must be implemented precisely.

## 8. OSRM Integration Guide

This guide provides instructions on how to query an OSRM (Open Source Routing Machine) server to obtain route information, from basic distance and duration to advanced details like road composition. These instructions are based on the usage within `New_solvers/vrp_multiday_sequential.py`.

**Prerequisites:**
- A running OSRM server. For public testing, you can use `http://router.project-osrm.org`.
- The `requests` library in Python (`pip install requests`).

---

### 8.1. Basic Route Request: Distance & Duration

- [x] The simplest OSRM call is to get the total distance and duration for a route between two points. This uses the `route` service.



### 8.2. Getting Route Geometry

- [x] For visualization on a map, you need the full geometry of the route. This is achieved by changing the `overview` and `geometries` parameters.



### 8.3. Advanced Route Information: Road Composition

- [x] To implement realistic vehicle constraints (e.g., different speeds for trucks on motorways vs. urban roads), you need to know the composition of the route. The `steps=true` parameter provides access to metadata for each segment of the route.



## 9. Advanced Travel Time Calculation

**Objective:** Replace the simplistic Euclidean distance-based travel time calculation with a realistic model using OSRM and a local cache to handle vehicle-specific speeds.

### 9.1. OSRM Caching and Integration Module

- [x] **Create a new module `algo/route_provider.py`:**
    - **Action:** This module will be responsible for fetching route data, either from a local database cache or from the OSRM API. It will encapsulate all the logic from `tests/road_composition.py`.
    - **Key Functions:**
        - `get_route_details(start_node_id, end_node_id)`: The main public function. It will orchestrate the process of getting route information.
        - `_query_local_db(start_node_id, end_node_id)`: A private function to look up a route in the database.
        - `_query_osrm_and_cache(start_coords, end_coords, start_node_id, end_node_id)`: A private function to call the OSRM API and store the results in the database.

### 9.2. Database Integration for Route Caching

- [x] **Set up a local database:**
    - **Action:** Use the `sqlite3` module (built-in to Python) to create a simple local database file named `moda_routes.db`.
    - **Schema:** Create a table to store the vehicle-agnostic route data. This keeps the database clean and flexible for changes in vehicle specifications.
        ```sql
        CREATE TABLE IF NOT EXISTS routes (
            start_node_id TEXT,
            end_node_id TEXT,
            distance_km REAL,
            base_duration_minutes REAL, -- OSRM's default duration (for a car)
            road_composition_json TEXT,
            route_geometry_json TEXT,   -- Full GPS trace of the route
            PRIMARY KEY (start_node_id, end_node_id)
        );
        ```
- [x] **Implement Caching Logic in `route_provider.py`:**
    - **Action:** When `get_route_details` is called, it should first query the `routes` table in `moda_routes.db`.
    - **If route exists:** Return the stored data (distance, duration, composition, geometry).
    - **If route does not exist:**
        1.  Call the OSRM API with `annotations=true` and `overview=full` to get the most detailed route data.
        2.  From the response, infer the road composition as shown in `tests/road_composition.py`.
        3.  Extract the total distance, baseline duration, road composition, and the full route geometry (the GPS trace).
        4.  Convert the distance to kilometers and the duration to minutes.
        5.  Store these vehicle-agnostic data points in the `routes` table.
        6.  Return the newly fetched data.

### 9.3. Solver Integration

- [x] **Modify `algo/second_level.py`:**
    - **Action:** Replace the `_calculate_travel_time_between_tasks` function.
    - **Logic:**
        1.  The function will now take `(task1, task2, vehicle)` as input.
        2.  It will call `route_provider.get_route_details(task1.node_id, task2.node_id)` to get the base route info (road composition and base duration in minutes).
        3.  Using the `vehicle.type` (e.g., 'standard', 'heavy'), it will retrieve the correct speed profiles (e.g., `TRUCK_SPEEDS['heavy']`).
        4.  It will then calculate the final, vehicle-specific travel time by applying the speed profiles to the road composition data, as demonstrated in `tests/road_composition.py`.
        5.  The function will return the final calculated travel time in minutes.
- [x] **Update `Vehicle` Data Structure:**
    - **Action:** Add a `type` attribute to the `Vehicle` class in `epdt_data_structures.py`. This will store the vehicle's profile (e.g., 'car', 'standard', 'heavy') to be used for the dynamic time calculation.
- [x] **Remove Old Calculation:**
    - **Action:** Once the new `route_provider` is integrated and tested, the old Euclidean distance calculation in `_calculate_travel_time_between_tasks` can be safely removed.

### 9.5. Handling Multi-Day and Weekly Hours of Service (HoS)

**Objective:** The current HoS check is too simplistic for multi-day, long-haul routes. It must be replaced with a detailed, iterative simulation that correctly implements the full scope of European driving and working time regulations as detailed in Regulation (EC) 561/2006 and Directive 2002/15/EC (Chapter 4 of the thesis).

- [x] **Enhance `_check_hos` (or create `_simulate_hos_advanced`) in `algo/second_level.py`:**

    - **Action:** Implement a stateful, iterative simulation for all travel and work activities.

    - **Driver State Representation:** The simulation must track a comprehensive set of state variables for each driver, initialized at the start of the planning horizon:
        - `drive_since_break`: Accumulated driving since the last 45min break (or split break part).
        - `work_since_break`: Accumulated working time since the last 30/45min break.
        - `drive_today`: Accumulated driving in the current daily period.
        - `work_today`: Accumulated working time in the current daily period.
        - `time_in_daily_period`: Time elapsed since the end of the last daily rest (must not exceed 24h for the start of the next daily rest).
        - `drive_this_week`: Accumulated driving from Monday 00:00 to current time.
        - `drive_last_week`: Total driving time from the previous week.
        - `work_this_week`: Accumulated working time from Monday 00:00.
        - `time_since_weekly_rest`: Time elapsed since the last weekly rest ended (must not exceed 144 hours).
        - `daily_drive_extensions_used`: Count of 10h driving days used this week (max 2).
        - `daily_rest_reductions_used`: Count of 9h daily rests used between weekly rests (max 3).
        - `is_weekly_rest_reduction_taken`: Flag indicating if a reduced weekly rest was taken in the last two-week period.

    - **Iterative Simulation Logic:**
        1.  For any activity (travel or service), do not check it as a single block. Instead, enter a loop that continues as long as the activity has time remaining (e.g., `travel_remaining > 0`).
        2.  **Inside the loop, calculate `drivable_time`:** This is the maximum time the driver can continue the current activity before a mandatory stop. It is the **minimum** of all applicable limits:
            - `4.5h - drive_since_break`
            - `6h - work_since_break` (if the activity is work/driving)
            - `9h - drive_today` (or `10h` if an extension is available and strategically chosen)
            - `56h - drive_this_week`
            - `90h - (drive_last_week + drive_this_week)`
            - `60h - work_this_week`
            - Time until a daily rest becomes mandatory (e.g., `24h - time_in_daily_period`).
            - Time until a weekly rest becomes mandatory (`144h - time_since_weekly_rest`).
            - The actual time remaining for the current activity (e.g., `travel_remaining`).
        3.  **Advance Simulation Time:**
            - Advance the `current_time` by `drivable_time`.
            - Update all relevant driver state counters based on the elapsed time.
            - Decrement the activity's remaining time.
        4.  **Trigger and Simulate Rests/Breaks:**
            - If the loop terminated because a regulatory limit was hit (not because the activity finished), simulate the required rest/break.
            - **Driving Break:** If `drive_since_break` reaches 4.5h, simulate a 45-minute break (or a 30-minute if a 15-minute break was already taken). Advance `current_time` and reset `drive_since_break`, `work_since_break`.
            - **Daily Rest:** If a daily limit is reached, simulate an 11-hour rest (or a reduced 9-hour rest if permissible). Advance `current_time`, reset all daily counters, and increment `time_since_weekly_rest`.
            - **Weekly Rest:** If a weekly limit is reached (e.g., 144 hours since last one), simulate a 45-hour rest (or reduced 24-hour rest). Advance `current_time` and reset all weekly and bi-weekly counters.
        5.  **Continue Loop:** Repeat the process until the activity is complete. This ensures mandatory rests are correctly simulated *during* long travel legs, allowing routes to be feasible across multiple days and weeks.

## 10. Advanced HoS Simulation and Route Day Calculation

**Objective:** Implement a detailed, iterative Hours of Service (HoS) simulation that accurately calculates the total time and number of days required for a route, including all mandatory breaks and rests as per European regulations.

### 10.1. Implement Route Day Calculation Function

- [x] **File:** `algo/second_level.py`
- [x] **Action:** Create a new public function `calculate_route_days(route: 'Route') -> int`.
- [x] **Logic:**
    1.  This function will serve as a high-level wrapper for the advanced HoS simulation.
    2.  If the route has no tasks, it should return `1` (takes one day).
    3.  Initialize a new `DriverState` object with default values.
    4.  Sort the route's tasks chronologically using the existing `_sort_tasks_chronologically` helper function.
    5.  Call the `_simulate_hos_advanced` function with the route, the new driver state, and the sorted tasks.
    6.  The `_simulate_hos_advanced` function will return `(is_feasible, total_time_minutes)`.
    7.  If `is_feasible` is `False`, the route is impossible under HoS rules. Return `float('inf')` to represent an infinitely long (infeasible) route.
    8.  If the route is feasible, calculate the number of days by dividing `total_time_minutes` by the number of minutes in a day (`24 * 60`) and taking the ceiling of the result (e.g., `(total_time // (24 * 60)) + 1`).
    9.  Return the calculated number of days.

### 10.2. Refactor Advanced HoS Simulation

- [x] **File:** `algo/second_level.py`
- [x] **Action:** Replace the entire body of the `_simulate_hos_advanced` function with a new, cleaner implementation. The new implementation should not handle day transitions internally but should focus purely on simulating the sequence of tasks.
- [x] **New `_simulate_hos_advanced` Logic:**
    1.  **Signature:** `_simulate_hos_advanced(route: 'Route', driver_state: 'DriverState', sorted_tasks: List) -> tuple[bool, float]`
    2.  **Initialization:**
        - If `sorted_tasks` is empty, return `(True, 0.0)`.
        - Create a `deepcopy` of the incoming `driver_state` to ensure the simulation is non-destructive.
        - Initialize `current_time = 0.0`.
    3.  **Main Loop:**
        - Iterate through the `sorted_tasks` list. For each `current_task` at index `i`:
            a. **Simulate Service Time:**
               - Get the `service_time` for the `current_task`.
               - If `service_time > 0`, call a new helper `_simulate_activity(state, service_time, 'service', current_time)`.
               - This helper will return `(feasible, elapsed_time)`. If not `feasible`, the main function should immediately return `(False, current_time)`.
               - Add the `elapsed_time` (which includes breaks) to `current_time`.
            b. **Simulate Travel Time:**
               - If it's not the last task (`i < len(sorted_tasks) - 1`):
                 - Get the `next_task`.
                 - Calculate `travel_time` between `current_task` and `next_task` using `_calculate_travel_time_between_tasks`.
                 - If `travel_time > 0`, call `_simulate_activity(state, travel_time, 'driving', current_time)`.
                 - Again, check for feasibility and update `current_time` with the `elapsed_time`.
            c. **Check Time Windows:**
               - After each activity, call the existing `_check_time_windows(current_task, current_time)` helper. If it returns `False`, the main function should return `(False, current_time)`.
    4.  **Return Value:** If the loop completes successfully, return `(True, current_time)`.

### 10.3. Update Tests and Scenarios

- [x] **File:** `tests/simple_multi_day_hos_test.py`
- [x] **Action:** Modify the test to use the new `calculate_route_days` function.
- [x] **Logic:**
    1.  Create a long route that is expected to span multiple days.
    2.  Call `calculate_route_days` on this route.
    3.  Assert that the returned number of days is correct (e.g., `self.assertEqual(days, 3)`).
    4.  Create another route that is intentionally infeasible (e.g., a single travel leg that is longer than the bi-weekly driving limit).
    5.  Assert that `calculate_route_days` returns `float('inf')` for the infeasible route.

- [x] **File:** `src/moda_scenarios.py` (or relevant test runner)
- [x] **Action:** Integrate the `calculate_route_days` check into the final solution evaluation for the "furgoni" scenario.
- [x] **Logic:**
    1.  After the main heuristic (`l1_heuristic`) has produced a final `Solution`.
    2.  Iterate through each `route` in `solution.routes.values()`.
    3.  For each route, call `calculate_route_days(route)`.
    4.  Print the number of days required for each vehicle's route as part of the final solution summary. This will provide a clear, realistic assessment of the operational plan.

### 10.4. Critical Implementation Notes for HoS Simulation

**Objective:** Address a critical bug in the current HoS simulation logic that causes incorrect feasibility checks for long multi-day routes.

- [x] **Identify and Fix the Root Cause of Infeasibility Loop:**
    - **Problem:** The `_simulate_mandatory_rest` function has a critical bug. It only triggers a weekly rest based on the 144-hour (6-day) rule. It completely ignores violations of the **56-hour weekly driving limit** and the **90-hour bi-weekly driving limit**.
    - **Symptom:** When a route exceeds the 56-hour or 90-hour driving limit, the simulation enters an infinite loop. `_calculate_max_continuous_time` correctly returns `0` (can't drive), but `_simulate_mandatory_rest` fails to trigger the necessary *weekly rest*. Instead, it simulates a *daily rest*, after which the weekly driving limits are still violated. The loop repeats, causing the feasibility check to fail incorrectly.
    - **Action:** Modify the logic that determines if a weekly rest is needed.
    - **File:** `algo/second_level.py`
    - **Function:** `_simulate_mandatory_rest`
    - **Logic Change:** The `needs_weekly_rest` condition must be updated to check for all three weekly rest triggers, not just the time-based one.

    - **Current (Buggy) Logic:**
      ```python
      needs_weekly_rest = state.time_since_weekly_rest >= 144 * 60
      ```

    - **Correct (Fixed) Logic:**
      ```python
      # Constants should be defined in DriverState for clarity
      # MAX_TIME_BEFORE_WEEKLY_REST = 144 * 60
      needs_weekly_rest = (
          state.time_since_weekly_rest >= state.MAX_TIME_BEFORE_WEEKLY_REST or
          state.drive_this_week >= state.MAX_DRIVE_PER_WEEK or
          (state.drive_this_week + state.drive_last_week) >= state.MAX_DRIVE_TWO_WEEKS
      )
      ```
    - **Reason:** This ensures that the simulation correctly identifies and enforces a weekly rest whenever *any* of the relevant HoS limits are reached, preventing the infinite loop and allowing for correct feasibility assessment of long-haul routes.

## 11. Multi-Day Time Window Calendarization

**Objective:** Implement support for multi-day time windows to allow tasks to be scheduled on specific days within the planning horizon, rather than just specific times.

### 11.1. Time Window Representation

- [x] **Scenario Definition:**
    - **Action:** Update the scenario creation logic (e.g., in `src/moda_scenarios.py`) to support a new time window format.
    - **Format:** Time windows will be represented as a single integer value calculated as: `(day_index * 1440) + time_in_minutes`.
        - `day_index`: 0 for the first day, 1 for the second, and so on.
        - `time_in_minutes`: The time of day, from 0 (midnight) to 1439.
    - **Example:** A time window of `(1, 480)` (Day 1, 8:00 AM) would be stored as `(1 * 1440) + 480 = 1920`. A window of `(1, 1020)` (Day 1, 5:00 PM) would be `(1 * 1440) + 1020 = 2460`. The task's time window would be `(1920, 2460)`.

### 11.2. Second-Level Heuristic Modifications (`algo/second_level.py`)

- [x] **Update `_check_time_windows`:**
    - **Action:** Modify the time window checking logic to correctly interpret the new multi-day format.
    - **Logic:**
        1.  The `current_time` in the simulation represents the total elapsed minutes from the start of the planning horizon.
        2.  The check `task.tw_start <= current_time <= task.tw_end` will now naturally handle multi-day windows without requiring explicit day-by-day comparisons.
        3.  If `current_time` is less than `task.tw_start`, the vehicle must wait. The waiting time is simply `task.tw_start - current_time`. This logic remains the same but now works across days.

- [x] **Update `_sort_tasks_chronologically`:**
    - **Action:** Ensure the sorting function correctly handles the new time window format.
    - **Logic:** The function should sort tasks based on their `tw_start` value. Since `tw_start` now encodes the day, this will automatically result in a chronologically correct sequence across multiple days. No major change is needed if it's already sorting by `tw_start`.

### 11.3. Test Implementation

- [x] **Create a new test file `tests/test_multiday_time_windows.py`:**
    - **Action:** Develop a dedicated test to validate the calendarization logic.
    - **Steps:**
        1.  Create a scenario with tasks that have time windows spanning multiple days.
            - **Task 1:** Day 0, 8:00-10:00 (tw_start=480, tw_end=600)
            - **Task 2:** Day 1, 9:00-11:00 (tw_start=1440+540=1980, tw_end=1440+660=2100)
            - **Task 3:** Day 0, 14:00-16:00 (tw_start=840, tw_end=960)
        2.  Construct a route that serves these tasks in a non-chronological order (e.g., Task 1 -> Task 2 -> Task 3).
        3.  Use `_sort_tasks_chronologically` to verify that the tasks are correctly re-ordered to (Task 1 -> Task 3 -> Task 2).
        4.  Simulate the route execution using the advanced HoS simulation.
        5.  Verify that the `current_time` at each step correctly reflects the waiting times and travel times, and that the time window checks pass. For example, after Task 3, there should be a significant waiting period (overnight) before the simulation proceeds to Task 2.

- [x] **Integrate Time Window Tests into Test Runner:**
    - **Action:** Enhance `tests/run_scenario_test.py` to include a test case for multi-day time windows.
    - **Details:** The test should load a scenario with calendarized time windows and verify that the final solution is valid and that all time constraints are respected by the feasibility check.

## 12. Return to Depot

**Objective:** Ensure all vehicles return to the depot at the end of their routes.

- [x] **Modify Route Generation:**
    - **Action:** After a route is finalized, add a final task that directs the vehicle back to the depot.
    - **Details:**
        1.  This should be the last task in the sequence for every vehicle.
        2.  The travel time and distance for this final leg must be included in the route's total cost and duration.
        3.  This must be accounted for in the HoS simulation to ensure the return trip doesn't violate any regulations.
