# Production Readiness Task List

This document outlines the necessary fixes to address critical issues related to Hours of Service (HoS) violations, time window violations, and suboptimal vehicle utilization. The goal is to make the optimization engine production-ready.

## Issue 1: Fix Hours of Service (HoS) Violations

**Root Cause:** The core feasibility check for HoS is currently a placeholder and does not enforce any regulations. The optimizer creates routes that violate legal driving limits, which are only caught by a post-process simulation.

### Task Breakdown:

1.  **Locate the Defective Function:**
    *   Navigate to `algo/second_level.py`.
    *   Find the function `_check_hos_multiday`. You will see it is a stub that always returns `True`.

2.  **Implement a Robust HoS Check:**
    *   Replace the placeholder logic in `_check_hos_multiday` with a proper simulation that tracks driver hours.
    *   The simulation must iterate through the route's tasks chronologically and, for each step, track:
        *   `drive_since_break`: Cumulative driving time since the last 45-minute break.
        *   `drive_today`: Total driving time for the current day.
        *   `work_today`: Total on-duty time for the current day (driving + service time + waiting).
    *   Enforce the following **legal limits** (these are commented in the code but not implemented):
        *   If `drive_since_break` exceeds **4.5 hours**, a **45-minute break** must be taken. This adds to `work_today` but resets `drive_since_break`.
        *   If `drive_today` exceeds **9 hours**, the route is **infeasible**.
        *   If `work_today` exceeds **13 hours**, the route is **infeasible**.
    *   This function should be called from the main `is_feasible` function and must be strictly enforced (i.e., not bypassed during initialization).

3.  **Testing the Fix:**
    *   Run the `tests/comprehensive_integration_test.py` script.
    *   **Expected Outcome:** The "HoS violation" messages in the final summary should be eliminated for all routes marked as "Feasible: True".
    *   Verify that the optimizer either creates compliant routes or correctly identifies routes with unavoidable HoS violations as infeasible, preventing them from being part of the final solution.

---

## Issue 2: Fix Time Window Violations

**Root Cause:** The time window validation in `is_feasible` is incomplete. It doesn't simulate the route's timing to check if arrival at a task location occurs within its specified `[earliest_time, latest_time]` window.

### Task Breakdown:

1.  **Locate the Defective Logic:**
    *   Navigate to `algo/second_level.py`.
    *   Examine the `is_feasible` function. Notice that the time window check is simplistic and does not calculate the actual arrival time at each task.

2.  **Implement Arrival Time Simulation:**
    *   Within the `is_feasible` function, implement a loop that simulates the route's progression from the first task to the last.
    *   Inside the loop, you must maintain a `current_time` variable. For each task `i` in the route:
        1.  Calculate `travel_time` from task `i-1` to task `i` using `calculate_travel_time_between_tasks`.
        2.  Calculate `arrival_time` = `current_time` + `travel_time`.
        3.  Check for lateness: If `arrival_time` > `task.latest_time`, the route is **infeasible**. Return `False`.
        4.  Check for waiting: If `arrival_time` < `task.earliest_time`, the vehicle must wait. Update `current_time` to `task.earliest_time`.
        5.  Update `current_time` by adding the task's `service_time`.
    *   This logic should be integrated directly into the main feasibility check to ensure the optimizer cannot create routes with time window violations.

3.  **Testing the Fix:**
    *   Run the `tests/comprehensive_integration_test.py` script.
    *   **Expected Outcome:** The "(LATE by ...)" messages in the final summary should be eliminated for all routes marked as "Feasible: True".
    *   Confirm that the optimizer now produces routes that respect all hard time window constraints.

---

## Issue 3: Investigate and Fix Constant Vehicle Usage

**Root Cause:** The optimizer consistently uses the same number of vehicles, even when parameters are set to encourage using more vehicles (e.g., `vehicle_penalty_per_vehicle = 0`). This indicates a flaw in the neighborhood operators, which are not exploring moves that would activate idle vehicles.

### Task Breakdown:

1.  **Analyze the Cost Function:**
    *   Go to `algo/first_level.py` and review the `calculate_z1_score` function.
    *   Confirm that the `vehicle_penalty_per_vehicle` parameter correctly penalizes the solution based on the number of active routes. While this is likely correct, it's a necessary first check.

2.  **Investigate Neighborhood Operators:**
    *   The primary focus is on the neighborhood generation functions in `algo/first_level.py`.
    *   **`single_order_relocation_neighborhood`**: Check if this function considers moving an order from an existing route to an **idle vehicle**. It currently appears to only move orders between active routes.
    *   **`unassigned_order_insertion_neighborhood`**: Check if this function attempts to place an unassigned order into an **idle vehicle**, thereby creating a new route.
    *   The investigation will likely reveal that idle vehicles are being ignored.

3.  **Implement the Fix:**
    *   Modify the neighborhood operators (especially `single_order_relocation_neighborhood`) to include idle vehicles as valid destinations for orders.
    *   When an order is moved to an idle vehicle, a new route should be created for that vehicle, containing the relocated order.
    *   This will allow the optimizer to explore solutions that use more vehicles, which is critical for resolving over-packed or infeasible routes.

4.  **Testing the Fix:**
    *   Modify the parameters in `tests/comprehensive_integration_test.py` by setting `'vehicle_penalty_per_vehicle': 0.0`.
    *   Run the test.
    *   **Expected Outcome:** The number of vehicles used in the solution should **increase** compared to previous runs with higher penalties. Observe if this increase helps resolve previously infeasible routes by spreading the workload across more vehicles.

---

## Issue 4: Correct the Flawed HoS Implementation

**Root Cause:** The implementation of the HoS check in `_check_hos_multiday` is incorrect and does not accurately model European regulations.

### Task Breakdown:

1.  **Correct the Break Logic:**
    *   Navigate to `algo/second_level.py` and find the `_check_hos_multiday` function.
    *   The current logic incorrectly takes a full break *before* a travel segment if the segment would cause a violation.
    *   Modify this to correctly simulate driving, taking breaks, and resuming driving.
        *   For each travel segment, calculate `remaining_drive_before_break` = `4.5_hours` - `drive_since_break`.
        *   If `travel_time` > `remaining_drive_before_break`:
            1.  Drive for `remaining_drive_before_break`. Update `drive_today`, `work_today`, and `drive_since_break`.
            2.  Take a **45-minute break**. Add 45 minutes to `work_today` and reset `drive_since_break` to 0.
            3.  Drive for the rest of the travel segment (`travel_time` - `remaining_drive_before_break`). Update driver state again.
        *   If `travel_time` <= `remaining_drive_before_break`, simply add the full `travel_time` to the driver state variables.

2.  **Use Consistent Travel Time Calculation:**
    *   The function currently uses `calculate_travel_time_haversine`. This is inconsistent with the rest of the application, which uses OSRM-based calculations.
    *   Replace all calls to `calculate_travel_time_haversine` inside `_check_hos_multiday` with `calculate_travel_time_between_tasks`. This ensures the HoS check uses the same realistic, road-network-based travel times as the route planning and final simulation.

3.  **Update `work_today` Correctly:**
    *   Ensure that when a 45-minute break is taken, the `driver_state.work_today` is incremented by 45 minutes. Breaks are considered on-duty time.

4.  **Testing the Fix:**
    *   Run the `tests/comprehensive_integration_test.py` script.
    *   **Expected Outcome:** The HoS violation messages should be completely eliminated from all feasible routes. The optimizer should now be able to create complex, multi-day routes that are fully compliant with European HoS regulations.




solution.routes returns list not dict