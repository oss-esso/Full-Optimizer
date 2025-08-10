# Production Readiness Task List

**REFERENCE OUTPUT (UNDESIRED BEHAVIOR):** `results/MCK500000 copy.TXT`

This document provides a detailed breakdown of the tasks required to make the vehicle routing optimization engine production-ready. Each issue includes a root cause analysis, a clear set of instructions for an external LLM agent, and a precise testing plan to verify the fix.

**IMPORTANT NOTE ON TESTING:** For each issue, you must create a **new, targeted test script** in the `tests/` directory. This script should create a minimal scenario that specifically triggers the issue you are fixing. The testing workflow is as follows:

1.  **Implement the fix** as described in the task breakdown.
2.  **Create and run your new targeted test.**
3.  **Read the terminal output** from your test run.
4.  **Compare the new terminal output** with the corresponding error in the reference file (`results/MCK500000 copy.TXT`) to confirm the specific bug is resolved.
5.  **Run the `tests/comprehensive_integration_test.py` script** to ensure your changes have not introduced any regressions.

---

## Issue 1: Fix Hours of Service (HoS) Violations

**Root Cause:** The current HoS feasibility check in `algo/second_level.py` is a placeholder (`_check_hos_multiday`) that always returns `True`. This means the optimizer is not constrained by legal driving limits, leading to routes that are only identified as illegal in a post-processing step.

### Task Breakdown for LLM Agent:

1.  **Navigate to the Correct File:**
    *   Open the file `algo/second_level.py`.

2.  **Locate the Defective Function:**
    *   Find the function `_check_hos_multiday(route: 'Route', driver_state: 'DriverState', sorted_tasks: List) -> bool`.

3.  **Implement a Robust HoS Simulation:**
    *   The function currently contains placeholder logic. You must replace it with a detailed simulation that accurately models European HoS regulations.
    *   **Driver State Tracking:** The simulation must iterate through the `sorted_tasks` list chronologically. For each step (travel between tasks and service at a task), you must update the `driver_state` object, which tracks:
        *   `drive_since_break`: Cumulative driving time (in minutes) since the last qualifying break.
        *   `drive_today`: Total driving time (in minutes) for the current calendar day.
        *   `work_today`: Total on-duty time (in minutes), which includes driving, service time, and any waiting time.
    *   **Enforce Legal Limits (EU Regulation 561/2006):**
        *   **4.5-Hour Driving Limit:** Before starting any travel segment, check if `drive_since_break` is approaching 4.5 hours (270 minutes). If the upcoming `travel_time` would exceed this limit, the driver must take a **45-minute break** first. This break time must be added to `work_today`, and `drive_since_break` must be reset to 0.
        *   **9-Hour Daily Driving Limit:** The total `drive_today` must not exceed 9 hours (540 minutes). If at any point this limit is surpassed, the route is **infeasible**.
        *   **13-Hour Daily Work Limit:** The total `work_today` must not exceed 13 hours (780 minutes). If this limit is surpassed, the route is **infeasible**.
    *   **Integration with `is_feasible`:** This function is called by the main `is_feasible` check. It must be strictly enforced, meaning it should not be bypassed during any phase of the optimization, including the initial solution generation.

### Testing the Fix:

1.  **Create a Targeted Test:**
    *   Create a new test file: `tests/test_hos_violations.py`.
    *   In this file, create a scenario with a single vehicle and a series of long-distance tasks that will force a violation of the 9-hour driving limit if the HoS check is not working.
2.  **Run the Test and Verify:**
    *   Execute your new test script.
    *   Read the terminal output. It should show that the route is correctly identified as infeasible.
    *   Compare this with the behavior in `results/MCK500000 copy.TXT`, where similar routes were marked `Feasible: False` but only after the fact. Your fix should prevent these routes from being considered feasible at all.
3.  **Run Comprehensive Test:**
    *   Execute the `tests/comprehensive_integration_test.py` script to ensure no regressions were introduced.

---

## Issue 2: Fix Time Window Violations

**Root Cause:** The time window validation in `is_feasible` is incomplete. It fails to simulate the route's timing, meaning it doesn't calculate the actual arrival time at each task to check against its `[earliest_time, latest_time]` window.

### Task Breakdown for LLM Agent:

1.  **Navigate to the Correct File:**
    *   Open the file `algo/second_level.py`.

2.  **Locate the Defective Logic:**
    *   Find the `is_feasible` function. The current time window check is a simplistic placeholder.

3.  **Implement Arrival Time Simulation:**
    *   Inside the `is_feasible` function, you must implement a loop that simulates the route's chronological progression.
    *   **Maintain `current_time`:** Initialize a `current_time` variable (in minutes from the start of the planning horizon).
    *   **Iterate Through Tasks:** For each task `i` in the route's task list:
        1.  **Calculate Travel Time:** Use the `calculate_travel_time_between_tasks` function to get the travel time from the previous task (`i-1`) to the current task (`i`).
        2.  **Calculate Arrival Time:** `arrival_time = current_time + travel_time`.
        3.  **Check for Lateness:** If `arrival_time` is greater than the task's `latest_time`, the route is **infeasible**. The function must return `False`.
        4.  **Handle Waiting Time:** If `arrival_time` is less than the task's `earliest_time`, the vehicle must wait. Update `current_time` to be equal to the `earliest_time`.
        5.  **Update `current_time`:** After arriving and potentially waiting, add the task's `service_time` to `current_time`.
    *   This logic must be integrated into the main feasibility check to prevent the optimizer from creating routes that violate time windows.

### Testing the Fix:

1.  **Create a Targeted Test:**
    *   Create a new test file: `tests/test_time_window_violations.py`.
    *   In this file, create a scenario with an order that has a very tight time window that is impossible to meet given the travel times.
2.  **Run the Test and Verify:**
    *   Execute your new test script.
    *   Read the terminal output. The test should show that the order is either unassigned or the route is correctly marked as infeasible.
    *   Compare this with the behavior in `results/MCK500000 copy.TXT`, where late arrivals are present. Your fix should eliminate them from feasible routes.
3.  **Run Comprehensive Test:**
    *   Execute the `tests/comprehensive_integration_test.py` script.

---

## Issue 5: Ensure Depot Tasks are Added to All Final Routes

**Root Cause:** The current method of adding depot tasks as a final clean-up step in `l1_heuristic` is not consistently applied across all code paths. Routes modified by various heuristics (neighborhood operators, destroy and repair) may not be correctly finalized, leading to invalid route structures.

### Task Breakdown for LLM Agent:

1.  **Locate the Depot Task Logic:**
    *   Navigate to `algo/first_level.py` and find the `l1_heuristic` function.
    *   Towards the end of this function, there is a loop that iterates through `best_solution.routes.items()` and is responsible for adding `depot_start_order` and `depot_return_order` tasks.

2.  **Verify and Harden the Logic:**
    *   **Robustness Check:** The existing logic must be analyzed to ensure it correctly handles all possible scenarios:
        *   Routes that are already well-formed (i.e., already have depot tasks).
        *   Routes that are empty.
        *   Routes that contain only a single task.
    *   **Defensive Programming:** The code should be made more robust. For example, it should check `if not route.tasks[0].is_depot_start():` before inserting a new start task, and `if not route.tasks[-1].is_depot_return():` before appending a return task.

### Testing the Fix:

1.  **Create a Targeted Test:**
    *   Create a new test file: `tests/test_depot_task_creation.py`.
    *   This test will run the optimizer on a small scenario with just a few orders.
    *   After the optimizer runs, the test will programmatically iterate through the routes in the final solution and assert that `route.tasks[0].is_depot_start()` is `True` and `route.tasks[-1].is_depot_return()` is `True` for every route.
2.  **Run the Test and Verify:**
    *   Execute your new test script. The test should pass.
    *   Read the terminal output to confirm that the assertions are being met.
3.  **Run Comprehensive Test:**
    *   Execute the `tests/comprehensive_integration_test.py` script.

---

## Issue 6: Fix Incorrect Order Insertion

**Root Cause:** The L2 insertion heuristic (`_generate_initial_task_sequence` in `algo/second_level.py`) has been refactored to use a cluster-based approach. The original problem of inserting tasks outside depot boundaries is no longer relevant. The new focus is to ensure that the routes generated by this heuristic are valid and can be correctly finalized with depot tasks by the `l1_heuristic`.

### Task Breakdown for LLM Agent:

1.  **Analyze the New Insertion Logic:**
    *   Navigate to `algo/second_level.py` and find the `_generate_initial_task_sequence` function.
    *   The new logic first groups all pickup tasks and then all delivery tasks. This function does not, and should not, handle depot tasks. Its only responsibility is to create a valid sequence of customer-related tasks.

2.  **Verify Final Route Structure:**
    *   The responsibility for adding depot tasks now lies entirely with the `l1_heuristic` in `algo/first_level.py` (as detailed in Issue 5).
    *   The test for this issue is to confirm that the routes produced by `_generate_initial_task_sequence` are correctly processed and finalized by `l1_heuristic`.

### Testing the Fix:

*   The test for this issue is covered by the targeted test in **Issue 5** (`tests/test_depot_task_creation.py`). If that test passes, it implicitly confirms that the task sequences generated by L2 are valid and can be correctly finalized.
*   Run the main optimization script and examine the terminal output. Compare it with `results/MCK500000 copy.TXT` to ensure all customer orders are between `DEPOT_START` and `DEPOT_RETURN` tasks.
*   Run the `tests/comprehensive_integration_test.py` script for overall validation.

---

## Issue 7: Remove Empty Depot-Only Routes from Final Solution

**Root Cause:** The final solution may contain routes that were initialized but never assigned any orders. These "empty" routes, which might only contain `DEPOT_START` and `DEPOT_RETURN` tasks, should be filtered out. The current filtering logic in `_validate_and_filter_solution` is not sufficient.

### Task Breakdown for LLM Agent:

1.  **Locate the Filtering Function:**
    *   Navigate to `algo/first_level.py` and find the `_validate_and_filter_solution` function.

2.  **Update Emptiness Check:**
    *   The current check `if hasattr(route, 'tasks') and route.tasks:` is too generic.
    *   You must modify this condition to be more specific. A route should be considered non-empty and valid only if it contains more than just the two depot tasks.
    *   The check should be updated to filter out routes where `len(route.tasks) <= 2`.

3.  **Ensure Correct Removal:**
    *   Make sure that any route failing this new, stricter check is not included in the `validated_routes` dictionary that is returned.

### Testing the Fix:

1.  **Create a Targeted Test:**
    *   Create a new test file: `tests/test_empty_route_filtering.py`.
    *   In this test, create a scenario with more vehicles than orders, ensuring at least one vehicle will be assigned an empty route.
2.  **Run the Test and Verify:**
    *   Execute your new test script. The test should run the `l1_heuristic` and then assert that no route in the final solution has only 2 tasks.
    *   Read the terminal output to confirm the assertion passes.
3.  **Run Comprehensive Test:**
    *   Execute the `tests/comprehensive_integration_test.py` script.

---

## Issue 8: Ensure Depot Tasks in Destroy and Repair

**Root Cause:** The `destroy_and_repair` heuristic in `algo/destroy_and_repair.py` modifies routes by removing orders. It is critical that this process does not inadvertently remove the essential `DEPOT_START` and `DEPOT_RETURN` tasks.

### Task Breakdown for LLM Agent:

1.  **Locate the Route Modification Logic:**
    *   Navigate to `algo/destroy_and_repair.py`.
    *   Focus on the `_destroy_orders_from_vehicle` function, which is responsible for removing tasks from a route.

2.  **Verify Preservation of Depot Tasks:**
    *   Inside `_destroy_orders_from_vehicle`, there is a check: `if hasattr(task, 'order_id') and task.order_id and 'depot' not in str(task.order_id).lower():`.
    *   This line is intended to prevent the removal of any task whose `order_id` contains the word "depot". You must verify that this is a reliable and sufficient way to identify and preserve the depot tasks.

3.  **Confirm Final Route Structure:**
    *   Because depot tasks are now added as a final step in `l1_heuristic`, the primary concern is ensuring that the routes coming out of the destroy and repair process are structurally sound. They should not be malformed in a way that prevents the final depot-adding logic from working correctly.

### Testing the Fix:

1.  **Create a Targeted Test:**
    *   Create a new test file: `tests/test_destroy_repair_depot_tasks.py`.
    *   Create a scenario that is likely to trigger the `destroy_and_repair` mechanic (e.g., a few large, conflicting orders).
2.  **Run the Test and Verify:**
    *   Execute your new test script with `destroy_and_repair` enabled.
    *   The test should assert that all routes in the final solution have the correct depot start and end tasks.
    *   Read the terminal output to confirm the assertion passes.
3.  **Run Comprehensive Test:**
    *   Execute the `tests/comprehensive_integration_test.py` script with `destroy_and_repair` enabled.




Previòus feedback:

I have analyzed the codebase. Here's a summary of the progress on the tasks from
  REFACTOR_ROUTE_CREATION.md and TODO_SOLVE_PROD_ISSUES.md.

  REFACTOR_ROUTE_CREATION.md

   * Step 1: `_create_base_route` Factory: DONE. The function _create_base_route exists in
     algo/first_level.py.
   * Step 2: Update Initializer Functions: DONE. The initializer functions regret_k_initializer,
     best_insertion_initializer, and cluster_aware_initializer in algo/first_level.py all use
     _create_base_route.
   * Step 3: Update Neighborhood Operators: DONE. The neighborhood operators
     unassigned_order_insertion_neighborhood and single_order_relocation_neighborhood in
     algo/first_level.py use _create_base_route.
   * Step 4: Modify L2 Insertion Heuristic: PARTIALLY DONE. The logic in _generate_initial_task_sequence in        
     algo/second_level.py has been updated to insert tasks between depot tasks, but the progress notes
     indicate it was INCORRECTLY IMPLEMENTED. The loops are not correctly constrained.
   * Step 5: Remove Redundant Logic: DONE. The redundant depot-adding logic has been removed from
     l1_heuristic in algo/first_level.py.
   * Issue 9: Filter Empty Depot-Only Routes: NOT DONE. The _validate_and_filter_solution function in
     algo/first_level.py does not filter out routes that only contain depot tasks.
   * Issue 10: Ensure Depot Tasks in Destroy and Repair: NOT DONE. The _insert_order_into_vehicle function
     in algo/destroy_and_repair.py does not use the _create_base_route factory.

  TODO_SOLVE_PROD_ISSUES.md

   * Issue 1: Fix Hours of Service (HoS) Violations: IN PROGRESS. The _check_hos_multiday function in
     algo/second_level.py contains a detailed simulation, but it is not yet fully integrated and validated.        
   * Issue 2: Fix Time Window Violations: IN PROGRESS. The is_feasible function in algo/second_level.py has        
     an implementation for arrival time simulation, but it needs to be verified.
   * Issue 5: Ensure Depot Tasks are Added to All Final Routes: This is the same as the refactoring task
     and is mostly DONE, but the destroy_and_repair part is missing.
   * Issue 6: Fix Incorrect Order Insertion: This is the same as Step 4 of the refactoring and is
     INCORRECTLY IMPLEMENTED.
   * Issue 7: Remove Empty Depot-Only Routes from Final Solution: This is the same as Issue 9 of the
     refactoring and is NOT DONE.
   * Issue 8: Ensure Depot Tasks in Destroy and Repair: This is the same as Issue 10 of the refactoring and        
     is NOT DONE.

  In summary, the main refactoring of route creation is largely complete, but there are still critical
  issues to address, especially in the L2 insertion logic, empty route filtering, and the
  destroy_and_repair functionality. The production issues are being worked on but are not yet fully
  resolved.can y
---

## Issue 9: Fix Depot Task Finalization and Duplication

**Root Cause:** There are two related issues causing problems with depot task assignment.
1.  **Missing Finalization Function:** The `l1_heuristic` function in `algo/first_level.py` calls a helper function `_add_depot_tasks_to_route` to ensure all routes are correctly finalized with start and end depot tasks. However, this helper function is not defined anywhere, causing the finalization step to fail with a `NameError`. This is the primary reason why many routes in the output are marked as infeasible for missing depot tasks.
2.  **Duplicate Depot Task Creation:** The `cluster_aware_initializer` creates a valid base route with depot tasks using `_create_base_route`. It then passes this valid route to the `build_clustered_route` function, which proceeds to add *another* set of depot start and return tasks, resulting in routes with duplicated and incorrectly placed depot tasks.

### Task Breakdown for LLM Agent:

#### Part 1: Implement the Missing Finalization Function

1.  **Navigate to the Correct File:**
    *   Open the file `algo/first_level.py`.

2.  **Add the Missing Function:**
    *   Locate the line that defines the `l1_heuristic` function: `def l1_heuristic(orders: List['Order'], vehicles: List['Vehicle'], params: dict) -> 'Solution':`.
    *   **Immediately before this line**, insert the following complete function definition for `_add_depot_tasks_to_route`:

    ```python
    def _add_depot_tasks_to_route(route: 'Route'):
        """
        Ensures a route has depot start and return tasks, adding them if missing.
        This is a defensive function to fix routes that were created incorrectly.
        """
        if not hasattr(route, 'tasks') or not route.tasks:
            # Don't add depot tasks to a completely empty route.
            # It will be filtered out later anyway.
            return

        # Defensive check: if route already has them, do nothing
        if route.tasks[0].is_depot_start() and route.tasks[-1].is_depot_return():
            return

        try:
            from .epdt_data_structures import Task, TaskType
        except ImportError:
            from epdt_data_structures import Task, TaskType

        depot_location_id = "DEPOT-ASTI"
        depot_lat, depot_lon = 44.9009, 8.2057

        # Check for and add start task
        if not route.tasks[0].is_depot_start():
            start_task = Task(
                id=f"depot_start_order_{route.vehicle.id}",
                location_id=depot_location_id,
                task_type=TaskType.DEPOT_START,
                order_id=f"depot_start_order_{route.vehicle.id}",
                lat=depot_lat,
                lon=depot_lon,
                service_time=0.0,
                demand=0.0,
                volume=0.0
            )
            route.tasks.insert(0, start_task)

        # Check for and add return task
        if not route.tasks[-1].is_depot_return():
            return_task = Task(
                id=f"depot_return_order_{route.vehicle.id}",
                location_id=depot_location_id,
                task_type=TaskType.DEPOT_RETURN,
                order_id=f"depot_return_order_{route.vehicle.id}",
                lat=depot_lat,
                lon=depot_lon,
                service_time=0.0,
                demand=0.0,
                volume=0.0
            )
            route.tasks.append(return_task)
    ```

#### Part 2: Remove Duplicate Depot Task Creation

1.  **Navigate to the Correct File:**
    *   Open the file `algo/first_level.py`.

2.  **Locate the Defective Function:**
    *   Find the function `build_clustered_route(route: 'Route', orders: List, debug_assignment: bool = False) -> Optional['Route']:`.

3.  **Remove Redundant Code:**
    *   Inside this function, delete the two blocks of code that create and insert the `depot_start_task` and `depot_return_task`. The route passed to this function already contains the correct depot tasks.
    *   **Block to remove (at the beginning of the function):**
        ```python
        # Add depot start task if we have any actual cargo tasks
        if all_pickups or all_deliveries:
            # Get depot information from vehicle
            depot_location_id = getattr(route.vehicle, 'depot_id', 'main_depot')
            depot_lat = getattr(route.vehicle, 'depot_lat', 44.9009)  # Default Asti coordinates (Via del Lavoro 38)
            depot_lon = getattr(route.vehicle, 'depot_lon', 8.2057)
            
            # Import Task class for creating depot tasks
            from epdt_data_structures import Task
            
            # Create and add depot start task at the beginning
            depot_start_task = Task.create_depot_start_task(
                vehicle_id=route.vehicle.id,
                depot_location_id=depot_location_id,
                depot_lat=depot_lat,
                depot_lon=depot_lon
            )
            current_route.insert_task_without_reordering(0, depot_start_task)
        ```
    *   **Block to remove (at the end of the function):**
        ```python
        # Add depot return task at the end if we have any actual cargo tasks
        if all_pickups or all_deliveries:
            # Get depot information from vehicle (same as start)
            depot_location_id = getattr(route.vehicle, 'depot_id', 'main_depot')
            depot_lat = getattr(route.vehicle, 'depot_lat', 44.9009)  # Default Asti coordinates (Via del Lavoro 38)
            depot_lon = getattr(route.vehicle, 'depot_lon', 8.2057)
            
            # Create and add depot return task at the end
            depot_return_task = Task.create_depot_return_task(
                vehicle_id=route.vehicle.id,
                depot_location_id=depot_location_id,
                depot_lat=depot_lat,
                depot_lon=depot_lon
            )
            current_route.insert_task_without_reordering(len(current_route.tasks), depot_return_task)
        ```

### Testing the Fix:

1.  **Run the full test script** that generated the problematic output (`tests/refactor_output.txt`).
2.  **Examine the new output.**
    *   Verify that the number of routes marked `Feasible: False` due to missing depot tasks has significantly decreased or been eliminated.
    *   Inspect the routes that are marked `Feasible: True`. Confirm that they no longer contain duplicated or misplaced depot tasks. All routes should have exactly one `DEPOT_START` as the first task and one `DEPOT_RETURN` as the last task.
3.  **Run the `tests/comprehensive_integration_test.py` script** to ensure no regressions were introduced.
