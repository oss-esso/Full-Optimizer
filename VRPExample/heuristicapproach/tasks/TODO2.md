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
## 8. OSRM Integration Guide

This guide provides instructions on how to query an OSRM (Open Source Routing Machine) server to obtain route information, from basic distance and duration to advanced details like road composition. These instructions are based on the usage within `New_solvers/vrp_multiday_sequential.py`.

**Prerequisites:**
- A running OSRM server. For public testing, you can use `http://router.project-osrm.org`.
- The `requests` library in Python (`pip install requests`).

---

### 8.1. Basic Route Request: Distance & Duration

The simplest OSRM call is to get the total distance and duration for a route between two points. This uses the `route` service.

**OSRM Request:**
-   **Service:** `route`
-   **Coordinates:** Provided as `{lon},{lat}` pairs separated by a semicolon.
-   **URL Format:** `http://<osrm-server>/route/v1/driving/{lon1},{lat1};{lon2},{lat2}`
-   **Parameters:** `overview=false` to keep the response minimal.

**Python Code Snippet:**

```python
import requests

def get_basic_route_info(start_coords, end_coords):
    '''
    Gets the distance and duration for a route from OSRM.

    :param start_coords: Tuple of (longitude, latitude) for the start point.
    :param end_coords: Tuple of (longitude, latitude) for the end point.
    :return: A tuple of (distance_km, duration_minutes), or (None, None) on error.
    '''
    osrm_url = "http://router.project-osrm.org"
    url = f"{osrm_url}/route/v1/driving/{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
    params = {'overview': 'false'}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data['code'] == 'Ok' and data['routes']:
            route = data['routes'][0]
            distance_meters = route['distance']
            duration_seconds = route['duration']
            
            distance_km = distance_meters / 1000.0
            duration_minutes = duration_seconds / 60.0
            
            print(f"Route found: {distance_km:.2f} km, {duration_minutes:.2f} minutes")
            return distance_km, duration_minutes
        else:
            print(f"OSRM could not find a route. Code: {data.get('code')}")
            return None, None

    except requests.exceptions.RequestException as e:
        print(f"Error calling OSRM: {e}")
        return None, None

# Example usage:
start = (9.18951, 45.46427) # Milan
end = (12.49637, 41.90278) # Rome
get_basic_route_info(start, end)
```

**Example JSON Response (relevant parts):**
```json
{
    "code": "Ok",
    "routes": [
        {
            "distance": 573343.7,
            "duration": 20015.1,
            "legs": [...]
        }
    ],
    "waypoints": [...]
}
```

---

### 8.2. Getting Route Geometry

For visualization on a map, you need the full geometry of the route. This is achieved by changing the `overview` and `geometries` parameters.

**OSRM Request:**
-   **Service:** `route`
-   **Parameters:**
    -   `overview=full`: Requests the most detailed geometry.
    -   `geometries=geojson`: Returns the geometry in GeoJSON format, which is easy to parse.

**Python Code Snippet:**

```python
import requests

def get_route_geometry(start_coords, end_coords):
    '''
    Gets the full route geometry from OSRM.

    :param start_coords: Tuple of (longitude, latitude) for the start point.
    :param end_coords: Tuple of (longitude, latitude) for the end point.
    :return: A list of [lon, lat] coordinates, or None on error.
    '''
    osrm_url = "http://router.project-osrm.org"
    url = f"{osrm_url}/route/v1/driving/{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
    params = {
        'overview': 'full',
        'geometries': 'geojson'
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data['code'] == 'Ok' and data['routes']:
            geometry = data['routes'][0]['geometry']['coordinates']
            print(f"Route geometry found with {len(geometry)} points.")
            return geometry
        else:
            print(f"OSRM could not find a route. Code: {data.get('code')}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error calling OSRM: {e}")
        return None

# Example usage:
start = (9.18951, 45.46427) # Milan
end = (8.22751, 45.07034) # Turin
geometry = get_route_geometry(start, end)
if geometry:
    print(f"First 5 points: {geometry[:5]}")
```

**Example JSON Response (relevant parts):**
```json
{
    "code": "Ok",
    "routes": [
        {
            "geometry": {
                "coordinates": [
                    [9.18951, 45.46427],
                    [9.18931, 45.46421],
                    ...
                ],
                "type": "LineString"
            },
            "legs": [...],
            "distance": 142520.3,
            "duration": 5667.9
        }
    ],
    "waypoints": [...]
}
```

---

### 8.3. Advanced Route Information: Road Composition

To implement realistic vehicle constraints (e.g., different speeds for trucks on motorways vs. urban roads), you need to know the composition of the route. The `steps=true` parameter provides access to metadata for each segment of the route.

**OSRM Request:**
-   **Service:** `route`
-   **Parameters:**
    -   `steps=true`: This is the key parameter to get step-by-step instructions which include road metadata.

**Python Code Snippet:**

```python
import requests
from collections import defaultdict

def get_route_road_composition(start_coords, end_coords):
    '''
    Gets detailed route information from OSRM and calculates road composition.
    This requires steps=true to access road metadata.

    :param start_coords: Tuple of (longitude, latitude) for the start point.
    :param end_coords: Tuple of (longitude, latitude) for the end point.
    :return: A dictionary with road types and their percentage of the total distance.
    '''
    osrm_url = "http://router.project-osrm.org"
    url = f"{osrm_url}/route/v1/driving/{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
    params = {
        'steps': 'true'
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if data['code'] != 'Ok' or not data['routes']:
            print(f"OSRM could not find a route. Code: {data.get('code')}")
            return None

        road_distances = defaultdict(float)
        total_distance = data['routes'][0]['distance']

        if total_distance == 0:
            return {}

        # Iterate through legs and steps to get road classes
        for leg in data['routes'][0]['legs']:
            for step in leg['steps']:
                distance = step['distance']
                road_class = 'unknown'
                
                # The 'intersections' array often contains road classification
                if 'intersections' in step and len(step['intersections']) > 0:
                    # The 'classes' array can contain hints like 'motorway'
                    if 'classes' in step['intersections'][0]:
                        road_class = step['intersections'][0]['classes'][0]

                road_distances[road_class] += distance

        # Calculate percentages
        composition = {
            road_type: (dist / total_distance) * 100
            for road_type, dist in road_distances.items()
        }
        
        print("Road Composition:")
        for road_type, percentage in composition.items():
            print(f"- {road_type}: {percentage:.2f}%")
            
        return composition

    except requests.exceptions.RequestException as e:
        print(f"Error calling OSRM: {e}")
        return None

# Example usage:
start = (9.18951, 45.46427) # Milan
end = (8.22751, 45.07034) # Turin
get_route_road_composition(start, end)
```

## 9. Advanced Travel Time Calculation

**Objective:** Replace the simplistic Euclidean distance-based travel time calculation with a realistic model using OSRM and a local cache to handle vehicle-specific speeds.

### 9.1. OSRM Caching and Integration Module

- [ ] **Create a new module `algo/route_provider.py`:**
    - **Action:** This module will be responsible for fetching route data, either from a local database cache or from the OSRM API. It will encapsulate all the logic from `tests/road_composition.py`.
    - **Key Functions:**
        - `get_route_details(start_node_id, end_node_id)`: The main public function. It will orchestrate the process of getting route information.
        - `_query_local_db(start_node_id, end_node_id)`: A private function to look up a route in the database.
        - `_query_osrm_and_cache(start_coords, end_coords, start_node_id, end_node_id)`: A private function to call the OSRM API and store the results in the database.

### 9.2. Database Integration for Route Caching

- [ ] **Set up a local database:**
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
- [ ] **Implement Caching Logic in `route_provider.py`:**
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

- [ ] **Modify `algo/second_level.py`:**
    - **Action:** Replace the `_calculate_travel_time_between_tasks` function.
    - **Logic:**
        1.  The function will now take `(task1, task2, vehicle)` as input.
        2.  It will call `route_provider.get_route_details(task1.node_id, task2.node_id)` to get the base route info (road composition and base duration in minutes).
        3.  Using the `vehicle.type` (e.g., 'standard', 'heavy'), it will retrieve the correct speed profiles (e.g., `TRUCK_SPEEDS['heavy']`).
        4.  It will then calculate the final, vehicle-specific travel time by applying the speed profiles to the road composition data, as demonstrated in `tests/road_composition.py`.
        5.  The function will return the final calculated travel time in minutes.
- [ ] **Update `Vehicle` Data Structure:**
    - **Action:** Add a `type` attribute to the `Vehicle` class in `epdt_data_structures.py`. This will store the vehicle's profile (e.g., 'car', 'standard', 'heavy') to be used for the dynamic time calculation.
- [ ] **Remove Old Calculation:**
    - **Action:** Once the new `route_provider` is integrated and tested, the old Euclidean distance calculation in `_calculate_travel_time_between_tasks` can be safely removed.

### 9.5. Handling Multi-Day Travel in HoS

- [ ] **Enhance `_check_hos_multiday` in `algo/second_level.py`:**
    - **Action:** The current HoS check is too rigid for long-haul routes, as it checks if the entire travel leg can be completed within the remaining daily hours. This needs to be replaced with a more realistic simulation.
    - **Logic:**
        1.  Instead of checking the entire `travel_time` at once, simulate the drive in segments.
        2.  Create a loop that continues as long as `travel_remaining > 0`.
        3.  Inside the loop, determine the `drivable_time` before a mandatory break or daily rest is required.
        4.  Subtract this `drivable_time` from `travel_remaining`.
        5.  If a break or rest is triggered, simulate it by advancing the `current_time` and resetting the relevant driver state counters (e.g., `drive_since_break`, `drive_today`).
        6.  This ensures that mandatory rests are correctly simulated *during* a long travel leg, allowing the route to be feasible across multiple days.