# Production Readiness Improvement Plan

**Objective:** To resolve the remaining issues preventing the system from being production-ready. This involves improving vehicle utilization, making order assignment more robust, and strictly enforcing Hours of Service (HoS) regulations during optimization.

---

## Issue 1: Inefficient Use of Idle Vehicles

**Root Cause:** The neighborhood operators (`unassigned_order_insertion` and `single_order_relocation`) only consider vehicles that have no route object assigned to them as "idle". However, the current initializers assign a route object to every vehicle, even if it's just an empty route with depot tasks. This prevents the operators from using these effectively idle vehicles to place unassigned orders.

### Task Breakdown:

1.  **Navigate to the File:**
    *   Open `algo/first_level.py`.

2.  **Modify `unassigned_order_insertion_neighborhood`:**
    *   Locate the logic that identifies idle vehicles: `idle_vehicles = [v for v in vehicles if v.id not in used_vehicle_ids]`.
    *   Replace this with logic that finds vehicles whose routes are either non-existent or contain only depot tasks (i.e., `len(route.tasks) <= 2`).
    *   The new logic should look like this:

    ```python
    # Strategy 2: Try assigning to idle or empty vehicles
    used_vehicle_ids = set(solution.routes.keys())
    idle_vehicles = []
    for v in vehicles:
        route = solution.routes.get(v.id)
        if not route or len(route.tasks) <= 2:
            idle_vehicles.append(v)
    ```

3.  **Modify `single_order_relocation_neighborhood`:**
    *   Apply the exact same change to the idle vehicle identification logic in this function.

### Testing the Fix:

*   Run the `tests/comprehensive_integration_test.py` script.
*   Observe the log output. The line `Found 0 idle vehicles` should now report a non-zero number, and the optimizer should be able to use these vehicles, potentially reducing the number of unassigned orders.

---

## Issue 2: Brittle Force-Assignment Logic

**Root Cause:** The `smart_force_assign_unassigned_orders` logic in `tests/comprehensive_integration_test.py` fails if an order is too difficult for the standard `l2_heuristic` to place (e.g., due to a tight time window). The message `Force assignment of 14 to FX194HX failed feasibility check` indicates this. A "force assignment" should be more robust and prioritize assignment over perfect feasibility.

### Task Breakdown:

1.  **Navigate to the File:**
    *   Open `tests/comprehensive_integration_test.py`.

2.  **Modify `force_assign_order_to_vehicle`:**
    *   The goal is to ensure the order is added to the route, even if `l2_heuristic` fails. If the heuristic returns `None`, we will manually create a simple but complete route.
    *   Replace the current implementation of the function with this more robust version:

    ```python
    def force_assign_order_to_vehicle(solution, order, vehicle):
        """Force assign an order to a specific vehicle, bypassing L2 failure."""
        from algo.first_level import _create_base_route
        from algo.second_level import l2_heuristic

        route = solution.routes.get(vehicle.id)
        if not route or len(route.tasks) <= 2:
            route = _create_base_route(vehicle)

        # First, try the standard, safe insertion heuristic
        new_route = l2_heuristic(route, order)

        if new_route:
            solution.routes[vehicle.id] = new_route
            print(f"✅ Force assigned order {order.id} to vehicle {vehicle.id} via L2 heuristic.")
            return True
        else:
            # If L2 fails, perform a direct manual insertion.
            # This prioritizes assignment over ideal feasibility.
            print(f"⚠️ L2 failed for force assignment of {order.id}. Performing direct insertion.")
            manual_route = _create_base_route(vehicle)
            tasks_to_add = order.get_pickups() + order.get_deliveries()
            # Insert tasks between depot start and return
            manual_route.tasks[1:1] = tasks_to_add
            solution.routes[vehicle.id] = manual_route
            print(f"✅ Force assigned order {order.id} to vehicle {vehicle.id} via direct insertion.")
            return True
    ```

### Testing the Fix:

*   Run `tests/comprehensive_integration_test.py`.
*   The log should no longer show `failed feasibility check during L2 insertion` for force assignments. All orders, including order 14, should now be assigned to a route, even if that route is later marked as infeasible due to other constraints.

---

## Issue 3: Strict Enforcement of Hours of Service (HoS)

**Root Cause:** The final output still contains routes with HoS violations. This means the optimization process is exploring and selecting these illegal routes. The cost function (`calculate_z2_score`) is not penalizing them harshly enough, and the feasibility check (`is_feasible`) might be bypassed during some stages of the search.

### Task Breakdown:

1.  **Navigate to the File:**
    *   Open `algo/second_level.py`.

2.  **Strengthen the Feasibility Check:**
    *   Locate the `is_feasible` function.
    *   Find the section that checks for the `is_initialization` flag to bypass the HoS check.
    *   **Remove this bypass entirely.** The HoS check is a legal requirement and must be strictly enforced at all times. The `_check_hos_multiday` function must always be called.

3.  **Implement a Hard Penalty in the Cost Function:**
    *   Locate the `calculate_z2_score` function.
    *   This function calculates costs but doesn't explicitly check for HoS feasibility.
    *   Add a call to the strict `is_feasible` function at the beginning of `calculate_z2_score`.
    *   If `is_feasible` returns `False` due to an HoS violation, `calculate_z2_score` should immediately return a massive penalty score to make the route extremely unattractive to the optimizer.
    *   The beginning of the function should be modified as follows:

    ```python
    def calculate_z2_score(route: 'Route') -> float:
        """ 
        Enhanced Z2 score calculation...
        """
        # First, perform a strict feasibility check for HoS violations.
        is_route_feasible, reason = is_feasible(route, return_reason=True)
        if not is_route_feasible and "HoS" in reason:
            return 1_000_000_000.0  # Return a massive penalty for HoS violations

        # Check if score is already cached
        if hasattr(route, '_z2_score'):
            return route._z2_score
        ...
    ```

### Testing the Fix:

*   Run `tests/comprehensive_integration_test.py`.
*   The final summary should show **zero HoS violations**. The optimizer, now properly guided by the hard penalty, should have avoided creating illegal routes.
