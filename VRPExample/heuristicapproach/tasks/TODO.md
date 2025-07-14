
# Task List for EPDT Heuristic Algorithm Implementation

## 1. Second-Level Heuristic (Intra-route Optimization)
- [x] **Implement a comprehensive `is_feasible(route: Route) -> bool` function.**
    - **Details:** This function is crucial and will be called frequently. It should take a `Route` object and return `True` if it respects all hard constraints, `False` otherwise. It needs to verify:
        - **H1: Vehicle Capacity:** Iterate through the route's tasks chronologically. Maintain a running `current_load` (both weight and volume). After each pickup, increase the load. After each delivery, decrease it. At no point should the load exceed the `route.vehicle.capacity`.
        - **H2: Precedence Constraints:** For each order with multiple tasks in the route, find the index of the last pickup task and the first delivery task. The last pickup's index must be less than the first delivery's index.
        - **H3: Hard Time Windows:** During the time calculation for the route, if the arrival time at any task is later than the task's `hard_time_window_end`, the route is infeasible.
        - **H6: Hours-of-Service Regulations:** This is the most complex check and requires simulating the route chronologically. You will need to track several state variables for the driver, which are reset at different times (after breaks, daily rests, or weekly rests). Create a helper class or dataclass `DriverState` to manage this.
            - **State variables to track:**
                - `drive_since_break`: Accumulated driving time since the last 45-min break (or split 15+30 min breaks). Resets to 0 after a valid break. *Limit: 4.5 hours.*
                - `work_since_break`: Accumulated working time (drive + service) since the last work break. Resets to 0 after a valid break. *Limit: 6 hours.*
                - `drive_today`: Total driving time in the current 24-hour duty period. Resets after a daily rest. *Limit: 9 hours (extendable to 10).*
                - `work_today`: Total duty time (work + drive + waiting) in the current 24-hour period. Resets after a daily rest. *Limit: 13 hours (extendable to 15).*
                - `drive_this_week` / `drive_last_week`: To check the 90-hour limit over two weeks.
            - **Implementation Logic:** Create a function `_check_hos(route: Route) -> bool:`. This function iterates through the route's tasks from start to finish.
                1. Initialize `DriverState` and `current_time` at the route start.
                2. For each leg of the route (from task `i` to `i+1`):
                    a. Determine the travel time `t_travel`.
                    b. Before starting the travel, check if `t_travel` can be completed without violating the `drive_since_break` or `drive_today` limits.
                    c. If not, you must insert a break/rest *before* travel. This is a complex decision point. The simplest approach (as per the paper) is to iteratively schedule the minimum required break/rest to make the next segment feasible. For example, if `drive_since_break` is 4.0 hours and `t_travel` is 1.0 hour, you can only drive for 0.5 hours. You must insert a 45-minute break, update `current_time`, reset `drive_since_break`, and then re-calculate the remaining travel time.
                    d. Update all state variables based on the travel time.
                    e. Upon arrival at task `i+1`, update `current_time`.
                    f. Before starting service, check if the service time `s_i+1` would violate the `work_since_break` or `work_today` limits. If so, insert a break and update times and state variables accordingly.
                    g. Update state variables based on the service time.
                3. If the loop completes without returning `False`, the route is feasible regarding HOS. Return `True`.
        - **H7: Prescribed Route Ending Position:** If the route has a prescribed end position, check if the last task in the sequence matches this position.

- [x] **Implement the `L2` function (Algorithm 8) for the second-level heuristic.**
    - **Details:** Create a function `l2_heuristic(route: Route, order: Order) -> Route:`. This function will be the main entry point for the second-level heuristic. It should first call a helper to generate an initial, feasible route that includes the new order. Then, it should pass this initial route to a local search procedure to optimize the task sequence. It returns the best route found.

- [x] **Implement the initial task sequence generation (Section 5.3.1).**
    - **Details:** Create a helper function `_generate_initial_task_sequence(route: Route, order: Order) -> Route:`. This function will implement the logic from section 5.3.1. It iterates through all pairs of indices `(i, j)` in the existing route `r`. For each pair, it inserts the pickup tasks of the new order `o` after index `i` and delivery tasks after index `j`. Use a "nearest neighbor" criterion to decide the insertion order of the tasks within the pickup/delivery blocks. Each time a new valid route is formed (check with `is_feasible`), calculate its `Z2` score and keep the best one found. Return the single best initial route.

- [x] **Implement the second-level neighborhoods: `task insertion` and `task swap` (Section 5.3.2).**
    - **Details:** These should be implemented as generator functions that yield new `Route` objects.
        - `_task_insertion_neighborhood(route: Route) -> Iterator[Route]:` This generator iterates through each task `t_i` in the route (that is not fixed). For each task, it iterates through every other possible position `j` in the route. It `yield`s a new `Route` object for each valid move of task `t_i` to position `j`.
        - `_task_swap_neighborhood(route: Route) -> Iterator[Route]:` This generator iterates through all unique pairs of tasks `(t_i, t_j)` in the route. For each pair, it `yield`s a new `Route` object where the positions of `t_i` and `t_j` are swapped. Ensure you handle route feasibility (e.g., precedence constraints) after the swap.

- [x] **Implement the `Z2(r)` score function (Equation 5.3) for route evaluation.**
    - **Details:** Create a function `calculate_z2_score(route: Route) -> float:`. This function is critical and needs to be accurate and efficient. It takes a `Route` object and calculates its total cost. This involves:
        1. Iterating through the sequence of tasks in the route.
        2. Cumulatively calculating travel time, work time, and driving time, inserting mandatory driver breaks where required by Hours of Service regulations.
        3. Calculating penalties for any soft time window violations.
        4. Adding the prospective cost for "tomorrow tasks".
        5. Adding penalties related to preferred vehicle assignments and end-of-day positions.
        The `Route` object should cache its score once calculated to avoid re-computation.

- [x] **Implement the local search procedure used within `L2`.**
    - **Details:** Create a function `local_search_L2(initial_route: Route, neighborhoods: list[Callable]) -> Route:`. This function will implement a simple hill-climbing or steepest-descent algorithm.
        - Start with `current_route = best_route = initial_route`.
        - Enter a `while` loop that continues as long as improvements are found.
        - Inside the loop, iterate through the provided `neighborhoods` (e.g., `[_task_swap_neighborhood]`).
        - For each neighbor route, calculate its `Z2` score.
        - Keep track of the best neighbor found in the current iteration.
        - If the best neighbor is better than `current_route`, update `current_route` and `best_route` and continue the loop. Otherwise, break the loop.
        - Return `best_route`.

## 2. First-Level Heuristic (Order-to-Vehicle Assignment)
- [x] **Implement the `L1` function (Algorithm 9) for the first-level heuristic.**
    - **Details:** This is the main function of the algorithm, e.g., `l1_heuristic(orders: list[Order], vehicles: list[Vehicle], params: AlgoParams) -> Solution:`. It will contain the main Tabu Search VND loop.
        1. **Initialization**:
            - Create an initial `Solution` object by calling one of the construction methods (e.g., `best_insertion_initializer`).
            - Initialize `best_solution = center_solution = initial_solution`.
            - Initialize iteration counters `non_improving_iters = 0`, `total_iters = 0`.
            - Initialize the tabu list: `tabu_list = collections.deque(maxlen=params['tabu_tenure'])`. Add the initial solution's characteristics to the tabu list to prevent immediately reversing the first move.
        2. **Main Loop**:
            - Loop `while non_improving_iters < M1 and total_iters < M2`.
            - Inside the loop, implement the chosen exploration strategy (VND is a good starting point).
            - **VND Logic**:
                - Set `improvement_found_in_iteration = False`.
                - For each neighborhood function `N` in `[neighborhood_1R, neighborhood_2S, ...]`:
                    - Find the best neighbor solution from that neighborhood that is not tabu (or meets aspiration criteria).
                    - If this neighbor is better than `center_solution`:
                        - `center_solution = best_neighbor`.
                        - `improvement_found_in_iteration = True`.
                        - Update `best_solution` if `center_solution` is better.
                        - Update the tabu list with the move that led to this new center.
                        - `break` the neighborhood loop to restart the VND from the first neighborhood (`1R`).
            - **If no improvement was found** in the entire VND loop:
                - Increment `non_improving_iters`.
                - Select a non-improving move (diversification). Choose the best-scoring non-improving solution found across all neighborhoods and make it the new `center_solution`.
                - Update the tabu list with this move.
        3. **Termination**: Return `best_solution`.

- [x] **Implement the initial solution construction methods (Section 5.4):**
    - **Details:**
        - `best_insertion_initializer(orders: list[Order], vehicles: list[Vehicle]) -> Solution:`
            - Start with an empty `Solution` where all orders are in a dummy "unassigned" list.
            - Loop until the unassigned list is empty.
            - In each iteration, find the best possible move (order, vehicle) pair. This requires a nested loop: `for order in unassigned_orders: for vehicle in vehicles:`.
            - To evaluate a move, tentatively place the order in the vehicle's route and call `l2_heuristic` to get the cost of the *new* route. The "cost" of the move is the `Z1` score change (`new_score - old_score`).
            - After checking all possible moves, execute the one with the best score improvement.
            - Update the solution state and remove the order from the unassigned list.
        - `round_robin_insertion_with_priority_initializer(orders: list[Order], vehicles: list[Vehicle]) -> Solution:`
            - As described in the paper, this is a faster, more heuristic initializer.
            - Group vehicles into the three classes.
            - Sort orders by priority and distance.
            - Assign orders to vehicles in a round-robin fashion within the vehicle classes, calling `l2_heuristic` to update the route after each assignment.

- [x] **Implement the first-level Tabu Search neighborhoods (Section 5.5):**
    - **Details:** These functions take a `Solution` and a `tabu_list` and `yield` new `Solution` objects.
        - **`single_order_relocation_neighborhood` (1R):**
            - `for v_from in solution.vehicles: for order in v_from.route.orders: for v_to in solution.vehicles:`
            - If `v_from == v_to`, continue.
            - **Tabu Check**: The "move" can be represented by the order and its new vehicle, e.g., `(order.id, v_to.id)`. Check if this tuple is in the `tabu_list`. If it is, skip this move unless the aspiration criteria are met.
            - **Aspiration Criteria**: Calculate the score of the potential new solution. If `new_score > best_solution.score`, ignore the tabu status.
            - If the move is allowed, create a `deepcopy` of the solution, perform the order relocation, call `l2_heuristic` on the two affected routes (`v_from` and `v_to`), and `yield` the new solution.
        - **`two_orders_swap_neighborhood` (2S):**
            - `for v1 in solution.vehicles: for o1 in v1.route.orders: for v2 in solution.vehicles: for o2 in v2.route.orders:`
            - Check if the move is tabu (e.g., `(o1.id, v2.id)` or `(o2.id, v1.id)`).
            - If allowed, `deepcopy` the solution, swap `o1` and `o2`, run `l2_heuristic` on both routes, and `yield` the result.

- [x] **Implement the `Z1(s)` score function (Equation 5.2) for solution evaluation.**
    - **Details:** Create a function `calculate_z1_score(solution: Solution) -> float:`.
        - Initialize `total_score = 0`.
        - For each route `r` in `solution.routes`: `total_score -= calculate_z2_score(r)`.
        - Add revenue for all assigned orders: `total_score += sum(o.revenue for o in assigned_orders)`.
        - Subtract penalties for unassigned urgent orders.
        - Subtract penalties for violated solution-wide soft constraints (e.g., `ID(s)` - preferred subsets, `IE(s)` - missing vehicles at depot).

- [x] **Implement the Tabu Search exploration strategies (Section 5.6):**
    - **Details:** This logic is part of the main `L1` loop.
        - **VND (Sequential Switch):** This is the primary strategy. A `for` loop over the neighborhood list `[1R, 2S, mR, 2C]`. If an improving move is found, `break` this loop and restart it from `1R`.
        - **Cyclic Switch:** This is an alternative strategy. Maintain a state variable for the current neighborhood. If in `1R` and no improvement is found after `In` iterations, switch to `2S`. Stay in `2S` for `Is` iterations, then switch back to `1R`. This requires more complex state management in the main loop than VND.

- [x] **Implement the Tabu list, aspiration criteria, and termination criteria (Section 5.6.4).**
    - **Details:**
        - **Tabu List**: `tabu_list = collections.deque(maxlen=TABU_TENURE)`. The items stored in the deque should represent the *attributes* of a move, not the solution itself. For a relocation (1R), this is typically `(order_id, source_vehicle_id)`. This means order `order_id` is forbidden from being moved *back* to `source_vehicle_id` for `TABU_TENURE` iterations. When a move is made, add its attribute to the list.
        - **Aspiration Criteria**: The most common criterion is to override the tabu status if the move results in a solution that is better than any solution found so far (`best_solution`). So, the check is: `if is_tabu(move) and new_solution.score > best_solution.score: # Allow move`.
        - **Termination Criteria**: In the main `L1` loop, simply check `if non_improving_iters >= M1 or total_iters >= M2: break`.


## 3. Enhancements
- [x] **Implement Granular Tabu Search (Section 5.8).**
    - **Details:** Before exploring a large neighborhood like `mR`, build a proximity graph of the orders in a route as described in Definition 5.2. You can use a library like `networkx` to represent the graph. Then, when generating moves, only consider moving subsets of orders that form cliques in this graph. This requires a clique-finding algorithm.

- [x] **Implement the Destroy and Repair phase (Section 5.9).**
    - **Details:** Create a high-level function `destroy_and_repair(solution: Solution) -> Solution:`.
        1. **Check:** Iterate through routes in the solution to find infeasibilities.
        2. **Destroy:** For each infeasible route, identify the problematic orders and move them to a "dummy" vehicle (or an unassigned list).
        3. **Repair:** Run the `L1` heuristic again, but this time provide the partially destroyed solution as the starting point. The heuristic will then try to re-insert the unassigned orders.

- [x] **Implement parallelization techniques (Section 5.10).**
    - **Details:** Use the `concurrent.futures.ProcessPoolExecutor`.
        - **Parallel Evaluation (PE):** In the neighborhood exploration loop, gather all possible neighbor solutions into a list. Use `executor.map(calculate_z1_score, list_of_neighbors)` to evaluate them in parallel.
        - **Parallel Neighborhood (PN):** Submit each neighborhood exploration (e.g., `generate_1R_neighbors`, `generate_2S_neighbors`) as a separate task to the process pool. This is more complex as the tasks need to communicate if an improving solution is found to stop the other tasks early. Use a `multiprocessing.Event` or a `Manager` object for synchronization.

# Pseudocode

## Second-Level Heuristic (Algorithm 8)

```python
from typing import List, Optional

# Assume Route, Order, Task are defined classes
def l2_heuristic(route: Route, order: Order) -> Optional[Route]:
    """Finds the best way to insert an order into a route."""
    
    # 1. Generate initial task sequences (Section 5.3.1)
    # This involves trying all insertion points for pickups and deliveries
    initial_routes: List[Route] = _generate_initial_task_sequences(route, order)

    if not initial_routes:
        return None # Infeasible insertion

    # 2. Find the best initial route based on Z2 score
    best_initial_route = max(initial_routes, key=calculate_z2_score)

    # 3. Perform local search to improve the sequence
    neighborhoods_to_search = [_task_swap_neighborhood]
    if order.is_fixed:
        neighborhoods_to_search.append(_task_insertion_neighborhood)
        
    final_route = local_search_L2(best_initial_route, neighborhoods_to_search)
    
    return final_route
```

## First-Level Heuristic (Algorithm 9)

```python
import collections

def l1_heuristic(orders: list, vehicles: list, params: dict) -> Solution:
    """Main Tabu Search VND algorithm."""
    
    # 1. Create initial solution
    initial_solution = best_insertion_initializer(orders, vehicles)
    
    # 2. Initialize state
    best_solution = center_solution = initial_solution
    tabu_list = collections.deque(maxlen=params['tabu_tenure'])
    non_improving_iters = 0
    total_iters = 0

    # 3. Main loop
    while non_improving_iters < params['M1'] and total_iters < params['M2']:
        total_iters += 1
        improvement_found = False
        best_neighbors_pool = []

        # 4. VND Loop
        for neighborhood_func in [single_order_relocation_neighborhood, two_orders_swap_neighborhood]:
            
            # Explore neighborhood, find best valid neighbor
            best_neighbor_in_N = None
            for neighbor in neighborhood_func(center_solution):
                # Simplified logic: check tabu, aspiration, and update best_neighbor_in_N
                # In reality, this is more complex
                pass 
            
            # Check for improvement
            if calculate_z1_score(best_neighbor_in_N) > calculate_z1_score(center_solution):
                center_solution = best_neighbor_in_N
                non_improving_iters = 0
                improvement_found = True
                
                # Update global best if needed
                if calculate_z1_score(center_solution) > calculate_z1_score(best_solution):
                    best_solution = center_solution
                
                # Add move to tabu list
                tabu_list.append(move_representation)
                break # Go back to the first neighborhood
            else:
                # Add to pool for diversification later
                best_neighbors_pool.append(best_neighbor_in_N)

        # 5. Diversification / Non-improving move
        if not improvement_found:
            non_improving_iters += 1
            # Select a non-improving move from the pool (deterministic or random)
            center_solution = select_from_pool(best_neighbors_pool, params['exploration_strategy'])
            # Add move to tabu list
            tabu_list.append(move_representation)

    return best_solution
```
# Python Implementation Suggestions

- **Data Structures**:
    - Use dataclasses or simple Python classes to represent your main entities: `Order`, `Vehicle`, `Route`, `Task`. This will make the code more readable and maintainable.
    - A `Solution` class could hold the current state of assignments (e.g., a dictionary mapping vehicle IDs to `Route` objects) and the overall score.
    - For the Tabu list, a `collections.deque` with a fixed size (`maxlen`) is a natural fit for managing the recency-based memory.

- **Libraries**:
    - **NumPy**: Can be useful for any vectorized calculations, especially if you get into more complex scoring or constraint evaluations.
    - **Multiprocessing/Concurrent.futures**: For implementing the parallel exploration of neighborhoods (Section 5.10), Python's built-in `multiprocessing` module or the higher-level `concurrent.futures.ProcessPoolExecutor` are excellent choices. This will allow you to leverage multiple CPU cores to speed up the search.
    - **Numba**: For performance-critical parts of the code, like the route evaluation or neighborhood generation, consider using `numba`'s `@jit` decorator to just-in-time compile Python code to native machine code, which can provide C-like speed with minimal code changes.

- **Code Architecture**:
    - **Modularity**: Keep the implementations of the different neighborhoods (1R, 2S, mR, 2C) in separate functions or methods. This makes the code easier to test and debug.
    - **Separation of Concerns**: Clearly separate the first-level heuristic (order-to-vehicle assignment logic) from the second-level heuristic (intra-route optimization). The `L1` function should call the `L2` function, as shown in the pseudocode.
    - **Scoring Functions**: Implement `Z1(s)` and `Z2(r)` as separate, well-defined functions. This makes it easier to modify the objective function later if needed.
    - **State Management**: The main loop of the `L1` function will manage the state (current solution, best solution, tabu list). Be careful about object mutation; use `copy.deepcopy()` when you need to modify a solution without affecting the original (e.g., when exploring neighbors).
# Improvement Suggestions

- **Refine `calculate_z2_score` Costing:** The current `driver_break_cost` is a simple placeholder. To improve accuracy, this should be replaced with a more realistic cost derived from the actual break/rest durations calculated in the `_check_hos` simulation. A longer break means more driver downtime, which has a higher cost.

- **Use Generators for Neighborhoods:** The `_task_insertion_neighborhood` and `_task_swap_neighborhood` functions currently build and return a full list of all possible neighbor routes. For routes with many tasks, this can consume a large amount of memory. Converting them to be true Python generators (using `yield` for each new route instead of appending to a list) would be much more memory-efficient.

- **Optimize Initial Sequence Generation:** The `_generate_initial_task_sequence` function has a nested loop structure that gives it a time complexity of roughly O(n^2), where n is the number of tasks in the route. For long routes, this could become a performance bottleneck. Consider evaluating a simpler, faster heuristic for the initial insertion, such as a single pass that inserts the new tasks at the cheapest position found, rather than exhaustively checking all O(n^2) combinations.

- **Clarify Local Search Strategy:** The `local_search_l2` implementation uses a "first improvement" strategy (it takes the first better neighbor it finds and restarts the search). The academic papers often describe a "best improvement" (or "steepest descent") strategy, which would require evaluating all neighbors in the current neighborhood and then choosing the best one. The current implementation is faster but might not find as good a solution. The strategy should be chosen deliberately and the code commented to reflect the choice.

## 5. Testing the Heuristic Algorithm

This section outlines the methodology for testing the EPDT heuristic algorithm using the scenarios defined in `src/moda_scenarios.py`.

### 5.1 Data Model Transformation

The algorithm is designed to work with a specific set of data structures (`Order`, `Vehicle`, `Route`, `Task`) as defined in the EPDT paper. The test scenarios, however, produce a `VRPInstance` object with a different structure (`Location`, `Vehicle`, `RideRequest`). A data transformation layer is required to bridge this gap.

**Action:** Create a data adapter module (e.g., `data_adapter.py`) with a primary function `convert_instance_to_epdt_input(instance: VRPInstance) -> (List[Order], List[Vehicle])`.

**Transformation Logic:**
1.  **`VRPInstance.ride_requests` to `List[Order]`:**
    *   Iterate through each `RideRequest` in the instance.
    *   For each `RideRequest`, create a corresponding `Order` object.
    *   The `RideRequest.id` becomes the `Order.id`.
    *   Create two `Task` objects for the order:
        *   A **pickup task** using the `pickup_loc_id` from the `RideRequest`. The location details (coordinates, time windows, service time) are retrieved from `VRPInstance.locations`.
        *   A **delivery task** using the `dropoff_loc_id`.
    *   The `RideRequest.passengers` and `RideRequest.volume` map to the `Order`'s demand attributes.

2.  **`VRPInstance.vehicles` to `List[Vehicle]`:**
    *   This is a more direct mapping. Iterate through `VRPInstance.vehicles` and create a corresponding `Vehicle` object for each, copying attributes like `id`, `capacity`, `volume_capacity`, and `depot_id`.

### 5.2 Algorithm Configuration

The heuristic's behavior is controlled by a `params` dictionary. A test script must be created to configure these parameters for different test runs.

**Key Parameters to Configure:**
-   **Tabu Search:**
    -   `tabu_tenure`: Size of the tabu list (e.g., 10).
    -   `M1`: Maximum number of non-improving iterations (e.g., 50).
    -   `M2`: Maximum total iterations (e.g., 500).
-   **Exploration Strategy:**
    -   `exploration_strategy`: Can be `'deterministic'` or `'random'`.
-   **Enhancements (Flags):**
    -   `enable_advanced_neighborhoods`: `True` or `False`.
    -   `enable_granular_search`: `True` or `False`.
    -   `enable_parallelization`: `True` or `False`.
    -   `parallel_strategy`: `'PE'` (Parallel Evaluation) or `'PN'` (Parallel Neighborhood).
-   **Scoring Penalties:**
    -   `vehicle_penalty_per_vehicle`: (e.g., 100)
    -   `unassigned_order_base_penalty`: (e.g., 1000)
    -   And other penalties defined in `calculate_z1_score`.

### 5.3 Test Execution Flow

A main test runner script (e.g., `run_scenario_test.py`) should orchestrate the test.

**Execution Steps:**
1.  **Import necessary modules:** `create_furgoni_scenario`, `l1_heuristic`, `convert_instance_to_epdt_input`, etc.
2.  **Load Scenario:** `instance = create_furgoni_scenario()`.
3.  **Transform Data:** `orders, vehicles = convert_instance_to_epdt_input(instance)`.
4.  **Configure Parameters:** Define the `params` dictionary for the specific test case.
5.  **Run Heuristic:** `solution = l1_heuristic(orders, vehicles, params)`.
6.  **Analyze and Print Results:** Call a helper function `print_solution_summary(solution)` to display the output in a readable format.

### 5.4 Results Analysis

The output of the test should be analyzed to verify both the correctness and the quality of the solution.

**`print_solution_summary` Function:**
This function should take the final `Solution` object and print:
-   **Overall Score:** The final `Z1` score.
-   **Vehicle Usage:** The total number of vehicles used.
-   **Route Details (for each vehicle):**
    -   Vehicle ID.
    -   Sequence of tasks/orders.
    -   Total route cost (`Z2` score).
    -   Total route distance and duration.
-   **Unassigned Orders:** A list of any orders that were not assigned to a vehicle.

This structured output will allow for easy verification of solution feasibility (e.g., checking if all mandatory orders are assigned) and a clear assessment of the solution's quality based on its score and other metrics.


