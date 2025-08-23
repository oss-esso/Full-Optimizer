# Implementation Plan: Advanced Order Splitting and Sequencing

## 1. Objective

To implement a robust, multi-stage strategy for assigning large or complex orders that cannot be placed in a single vehicle using standard insertion methods. The system will hierarchically attempt different task sequencing and splitting strategies, from simple single-vehicle re-sequencing to complex multi-vehicle splits, until a feasible solution is found.

## 2. High-Level Logic Flow

When a standard order insertion fails for an `order`, the following master function in `algo/first_level.py` should be called:

```python
# In algo/first_level.py, inside the initializer (e.g., regret_k_initializer)
# when an order fails to be assigned:
if not assignment_successful:
    was_force_assigned = handle_unassigned_order(order, vehicles, solution)
    if not was_force_assigned:
        solution.unassigned_orders.add(order.id)
```

The new `handle_unassigned_order` function will orchestrate the process:

```python
# In algo/first_level.py
def handle_unassigned_order(order, vehicles, solution):
    """
    Orchestrates a series of advanced strategies to assign a difficult order.
    """
    print(f"Executing advanced assignment for order {order.id}...")

    # Strategy 1: Try different single-vehicle sequencing
    if try_single_vehicle_strategies(order, vehicles, solution):
        print(f"SUCCESS: Assigned order {order.id} using a single-vehicle strategy.")
        return True

    # Strategy 2: Try splitting the order across multiple vehicles
    if try_multi_vehicle_splitting(order, vehicles, solution):
        print(f"SUCCESS: Assigned order {order.id} by splitting across multiple vehicles.")
        return True

    print(f"FAILURE: All advanced strategies failed for order {order.id}.")
    return False
```

---

## 3. Phase 1: Single-Vehicle Strategies

This phase focuses on finding a way to fit all tasks of an order into one vehicle by trying different sequencing patterns.

### 3.1. Modify `algo/second_level.py`

1.  **Enhance `l2_heuristic`:**
    *   Add a new parameter: `sequencing_strategy: str = 'clustered'`. This parameter can be `'clustered'` or `'interleaved'`.

2.  **Enhance `_generate_initial_task_sequence`:**
    *   This function should accept the `sequencing_strategy` parameter from `l2_heuristic`.
    *   **If `sequencing_strategy == 'clustered'`:** Use the existing logic that groups all pickups together, followed by all deliveries (`P-P-D-D`). This is the default, most efficient pattern.
    *   **If `sequencing_strategy == 'interleaved'`:** Implement a new logic that attempts to sequence tasks chronologically based on their time windows, while respecting pickup-before-delivery precedence for each pair. This will naturally create `P-D-P-D` patterns.
        *   Get all tasks (`pickups` + `deliveries`).
        *   Sort them based on their `earliest_time`.
        *   Iterate through the sorted list, attempting to insert them one by one into a route, always checking that a delivery's corresponding pickup has already been inserted.

### 3.2. Implement `try_single_vehicle_strategies` in `algo/first_level.py`

```python
# In algo/first_level.py
def try_single_vehicle_strategies(order, vehicles, solution):
    """
    Tries to assign all tasks of an order to a single vehicle using different sequencing.
    """
    # Iterate through all available vehicles
    for vehicle in vehicles:
        # Get an empty route for the vehicle
        base_route = _create_base_route(vehicle)

        # Strategy 1.1: Try clustered sequencing (P-P-D-D)
        clustered_route = l2_heuristic(base_route, order, sequencing_strategy='clustered')
        if clustered_route:
            solution.routes[vehicle.id] = clustered_route
            return True

        # Strategy 1.2: Try interleaved sequencing (P-D-P-D)
        interleaved_route = l2_heuristic(base_route, order, sequencing_strategy='interleaved')
        if interleaved_route:
            solution.routes[vehicle.id] = interleaved_route
            return True
            
    return False
```

---

## 4. Phase 2: Multi-Vehicle Splitting

If single-vehicle strategies fail, the system will attempt to split the order's tasks across two or more vehicles.

### 4.1. Implement `try_multi_vehicle_splitting` in `algo/first_level.py`

This function will be complex and may require helper functions for clarity.

```python
# In algo/first_level.py
from itertools import combinations

def try_multi_vehicle_splitting(order, vehicles, solution):
    """
    Tries to assign an order by splitting its tasks across multiple vehicles.
    """
    tasks = order.get_all_tasks()
    num_tasks = len(tasks)

    if num_tasks < 2:
        return False

    # Iterate from a 50/50 split down to a 1 vs. N-1 split
    for i in range(num_tasks // 2, 0, -1):
        # Generate all combinations of `i` tasks for the first group
        for task_group_1_indices in combinations(range(num_tasks), i):
            task_group_1 = [tasks[j] for j in task_group_1_indices]
            task_group_2 = [tasks[j] for j in range(num_tasks) if j not in task_group_1_indices]

            # Create temporary orders for each task group
            order_split_1 = create_temp_order_from_tasks(task_group_1, order.id + "_split1")
            order_split_2 = create_temp_order_from_tasks(task_group_2, order.id + "_split2")

            # Find the best assignment for this specific split
            best_assignment = find_best_vehicle_pair_for_split(
                order_split_1, order_split_2, vehicles, solution
            )

            if best_assignment:
                # If a valid assignment is found, apply it to the main solution
                v1_id, route1, v2_id, route2 = best_assignment
                solution.routes[v1_id] = route1
                solution.routes[v2_id] = route2
                return True
    return False

def find_best_vehicle_pair_for_split(order1, order2, vehicles, solution):
    """
    Finds the best pair of vehicles to serve a two-way order split.
    """
    best_cost = float('inf')
    best_assignment = None

    # Iterate through all unique pairs of vehicles
    for v1, v2 in combinations(vehicles, 2):
        # Try assigning order1 to v1 and order2 to v2
        route1 = l2_heuristic(_create_base_route(v1), order1)
        if route1:
            route2 = l2_heuristic(_create_base_route(v2), order2)
            if route2:
                cost = calculate_z2_score(route1) + calculate_z2_score(route2)
                if cost < best_cost:
                    best_cost = cost
                    best_assignment = (v1.id, route1, v2.id, route2)

        # Try assigning order1 to v2 and order2 to v1
        route1_rev = l2_heuristic(_create_base_route(v2), order1)
        if route1_rev:
            route2_rev = l2_heuristic(_create_base_route(v1), order2)
            if route2_rev:
                cost = calculate_z2_score(route1_rev) + calculate_z2_score(route2_rev)
                if cost < best_cost:
                    best_cost = cost
                    best_assignment = (v2.id, route1_rev, v1.id, route2_rev)

    return best_assignment

def create_temp_order_from_tasks(tasks, new_order_id):
    """
    Helper to create a temporary Order object from a list of tasks.
    This is needed to pass to the l2_heuristic.
    """
    # This function needs to be implemented. It will create a new Order object,
    # assign it the new_order_id, and populate its task list.
    # Note: The demand/volume of this new order must be calculated from the tasks.
    pass
```

## 5. Logging

Add detailed `print` statements throughout this new logic to ensure the decision-making process is transparent. Log which strategy is being attempted, which splits are being evaluated, and the reason for any failures. This will be critical for debugging and future enhancements.

## 6. LLM Agent Implementation and Testing Loop

To effectively implement the features outlined above, the agent should follow an iterative development and testing cycle for each sub-task.

1.  **Select a Task:** Choose a small, specific task from the implementation plan (e.g., "Add `sequencing_strategy` parameter to `l2_heuristic`").

2.  **Implement the Change:** Modify the code in `algo/first_level.py` or `algo/second_level.py` to implement the selected task.

3.  **Run Comprehensive Test & Capture Baseline:**
    *   Before making a change, run the test to get a baseline output.
    *   Execute the command: `python tests/comprehensive_integration_test.py > baseline_output.txt`

4.  **Run Comprehensive Test & Capture New Output:**
    *   After implementing the change, run the test again.
    *   Execute the command: `python tests/comprehensive_integration_test.py > modification_output.txt`

5.  **Analyze and Compare:**
    *   Read the content of both `baseline_output.txt` and `modification_output.txt`.
    *   Compare the "Order Assignment Summary" and the "UNASSIGNED ORDERS ANALYSIS" sections.
    *   **Success Criteria:** The change is successful if the number of unassigned orders decreases or if the new logic correctly handles a previously failing case, without causing regressions.
    *   **Failure Analysis:** If the test fails or the results are worse, analyze the error messages and logs in `modification_output.txt` to debug the implementation.

6.  **Iterate:**
    *   If the change is successful, commit the change (conceptually) and move to the next task.
    *   If it fails, revert or fix the change and repeat the cycle.

This loop ensures that each modification is validated against the full system, preventing regressions and providing clear feedback on the effectiveness of the implemented logic.

## 7. Code Review and Analysis (Agent Notes)

### 7.1. Implementation Status

As of the last review, the advanced splitting and sequencing logic outlined in sections 2, 3, and 4 of this document **has not been implemented**.

*   **`algo/first_level.py`:** Still contains the old `intelligent_order_splitting_during_optimization` which splits order *demand* (weight/volume) instead of splitting the *tasks* across multiple vehicles. The new functions (`handle_unassigned_order`, `try_single_vehicle_strategies`, `try_multi_vehicle_splitting`) are absent.
*   **`algo/second_level.py`:** The `l2_heuristic` and `_generate_initial_task_sequence` functions have not been modified to accept a `sequencing_strategy` parameter.

**Action:** The implementation should proceed as outlined in this document. The agent should start by implementing the changes in Phase 1.

### 7.2. Analysis of Time Window Failures

The user correctly identified that time window violations are a primary cause of assignment failures. The investigation points to the following core issue:

*   **Problem:** The heuristic constructs routes based on cost and distance first, and only validates them against Hours of Service (HoS) and time windows afterward. When the HoS simulation adds mandatory rests (e.g., 45-minute breaks, 11-hour daily rests), these rests push subsequent tasks forward in time, causing them to violate their delivery time windows.

*   **Root Cause:** This is not a bug in the time window *check* itself, but a flaw in the *heuristic strategy*. The route construction logic is not "HoS-aware." It doesn't proactively account for the impact of mandatory rests when it builds the initial routes.

*   **Recommendation:**
    1.  **The advanced splitting logic in this plan is the correct solution.** By splitting a long, multi-day order into smaller chunks that can be handled by different vehicles in a single day, it avoids triggering the long mandatory rests that cause the time window violations.
    2.  **The implementation should proceed as planned.** The agent should focus on implementing the task-based splitting and sequencing strategies, as this will directly address the root cause of the time window failures.