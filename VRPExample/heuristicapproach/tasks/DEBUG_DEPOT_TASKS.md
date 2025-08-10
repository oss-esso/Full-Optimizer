# Test Suite: Debugging Depot Task Issues

**Objective:** To systematically identify the component responsible for removing or failing to add `DEPOT_START` and `DEPOT_RETURN` tasks to routes.

This document outlines a series of tests, starting from the most basic components and progressively adding complexity. For each step, you will create a new, dedicated test script in the `/tests` directory, execute it, and analyze the output to verify its success before proceeding to the next.

---

## Issue: Force Assignment Creates Malformed Routes

**Root Cause:** The `force_assign_order_to_vehicle` function within `tests/comprehensive_integration_test.py` creates new routes for unassigned orders without the required `DEPOT_START` and `DEPOT_RETURN` tasks. It manually creates a simple route object instead of using the `_create_base_route` factory, leading to routes that are guaranteed to fail validation.

### Task Breakdown:

1.  **Navigate to the Correct File:**
    *   Open the file `tests/comprehensive_integration_test.py`.

2.  **Locate the Defective Function:**
    *   Find the function `force_assign_order_to_vehicle(solution, order, vehicle)`.

3.  **Replace the Implementation:**
    *   Replace the entire body of the function with the following corrected code. This new version uses the proper `_create_base_route` factory for new routes and the `l2_heuristic` to ensure tasks are inserted correctly between the depot tasks.

    ```python
    def force_assign_order_to_vehicle(solution, order, vehicle):
        """Force assign an order to a specific vehicle by adding its tasks to the vehicle's route."""
        try:
            from algo.first_level import _create_base_route
            from algo.second_level import l2_heuristic
        except ImportError:
            print("ERROR: Could not import helper functions for force_assign_order_to_vehicle.")
            return False

        # Get or create the route for this vehicle
        if vehicle.id not in solution.routes or solution.routes[vehicle.id] is None or not solution.routes[vehicle.id].tasks:
            # If route doesn't exist or is empty, create a proper base route
            route = _create_base_route(vehicle)
            solution.routes[vehicle.id] = route
        else:
            route = solution.routes[vehicle.id]

        # Use the existing l2_heuristic to insert the order's tasks correctly
        # This is much safer than manually appending tasks.
        new_route = l2_heuristic(route, order)

        if new_route:
            solution.routes[vehicle.id] = new_route
            print(f"✅ Force assigned order {order.id} to vehicle {vehicle.id}")
            return True
        else:
            print(f"❌ Force assignment of {order.id} to {vehicle.id} failed feasibility check during L2 insertion.")
            return False
    ```

### Testing the Fix:

1.  **Execute the comprehensive test:** Run `python tests/comprehensive_integration_test.py`.
2.  **Analyze the output:** Observe the logs for the "SMART FORCE ASSIGNMENT" phase. Confirm that force-assigned orders are now placed in valid routes that pass the feasibility checks and that the number of `Feasible: False` routes in the final summary is significantly reduced.

---

## Test 1: Core Component Validation (`_create_base_route`)

**Goal:** Confirm that the most fundamental route-creation function works as expected and that the feasibility check correctly validates a simple, valid route.

### Task Breakdown:

1.  **Create a new test script:** `tests/debug_test_step1_base_route.py`.
2.  **Write the test code:** The script should perform the following actions:
    *   Import `Route`, `Vehicle`, `Task`, and `TaskType` from `epdt_data_structures`.
    *   Import `_create_base_route` from `algo.first_level`.
    *   Import `is_feasible` from `algo.second_level`.
    *   Create a dummy `Vehicle` object.
    *   Call `route = _create_base_route(vehicle)`.
    *   Assert that `len(route.tasks)` is 2.
    *   Assert that `route.tasks[0].is_depot_start()` is `True`.
    *   Assert that `route.tasks[-1].is_depot_return()` is `True`.
    *   Call `feasible, reason = is_feasible(route, return_reason=True)`.
    *   Assert that `feasible` is `True`.
    *   Print a success message if all assertions pass.
3.  **Execute the test:** Run `python tests/debug_test_step1_base_route.py`.
4.  **Analyze the output:** The test must pass without any assertion errors. If it fails, the problem lies in the most basic data structures or the feasibility check itself.

---

## Test 2: L2 Heuristic Validation (`l2_heuristic`)

**Goal:** Verify that the L2 heuristic (intra-route optimization) correctly inserts a single, simple order into a base route while preserving the depot task structure.

### Task Breakdown:

1.  **Create a new test script:** `tests/debug_test_step2_l2_insertion.py`.
2.  **Write the test code:** The script should:
    *   Set up a simple `Order` and `Vehicle`.
    *   Create a valid `base_route` using `_create_base_route`.
    *   Import and call `l2_heuristic(base_route, order)`.
    *   Check the `new_route` returned by the heuristic.
    *   Assert that the `new_route` is not `None`.
    *   Assert that `new_route.tasks[0].is_depot_start()` is `True`.
    *   Assert that `new_route.tasks[-1].is_depot_return()` is `True`.
    *   Call `is_feasible(new_route)` and assert it returns `True`.
    *   Print a success message.
3.  **Execute and Analyze:** The test must pass. A failure here points to a bug in `l2_heuristic` or its sub-functions like `_generate_initial_task_sequence`.

---

## Test 3: Initializer Validation (`cluster_aware_initializer`)

**Goal:** Check if a complete initialization function generates a solution where all routes are valid and correctly structured.

### Task Breakdown:

1.  **Create a new test script:** `tests/debug_test_step3_initializer.py`.
2.  **Write the test code:** The script should:
    *   Set up a simple scenario with 2-3 orders and a few vehicles.
    *   Import and call `cluster_aware_initializer(orders, vehicles, params)`.
    *   Iterate through every `route` in the returned `solution.routes.values()`.
    *   For each route, assert that it has tasks, that the first task is a depot start, and the last is a depot return.
    *   Print a success message, including the number of routes validated.
3.  **Execute and Analyze:** A failure here indicates a problem in the initialization logic, likely in how `build_clustered_route` handles the routes it receives.

---

## Test 4: L1 Heuristic Finalization Validation

**Goal:** Verify that the main `l1_heuristic` loop, specifically its finalization and validation steps, correctly processes a solution, even one with malformed routes.

### Task Breakdown:

1.  **Create a new test script:** `tests/debug_test_step4_l1_finalization.py`.
2.  **Write the test code:** The script should:
    *   Create a simple scenario (a few orders and vehicles).
    *   Manually create a `Solution` object containing a `Route` that is intentionally missing its depot tasks (e.g., just a pickup and delivery task).
    *   Import `l1_heuristic`.
    *   **Crucially,** call `l1_heuristic` with parameters that **disable all neighborhood operators** to isolate the main loop's logic. The goal is not to optimize, but to see if the function can fix the broken input route.
    *   Check the `final_solution` returned by the heuristic.
    *   Assert that the route for the vehicle has been corrected and now has the proper depot start and end tasks.
3.  **Execute and Analyze:** This directly tests the `_add_depot_tasks_to_route` function and the final validation logic. If it fails, the finalization step itself is flawed.

---

## Test 5: Comprehensive Test Execution

**Goal:** Run the full `comprehensive_integration_test.py` and analyze its output in light of the previous findings.

### Task Breakdown:

1.  **Execute the full test:** `python tests/comprehensive_integration_test.py`.
2.  **Analyze the output:**
    *   If tests 1-4 passed, but this fails, the issue is likely an emergent bug from the interaction of neighborhood operators or the complexity of the full dataset.
    *   Examine the logs to see which routes are failing. Cross-reference the vehicle and order IDs with the logs from the simpler tests to see if there are any patterns.
    *   This final step should pinpoint any remaining issues that only appear under full system load.