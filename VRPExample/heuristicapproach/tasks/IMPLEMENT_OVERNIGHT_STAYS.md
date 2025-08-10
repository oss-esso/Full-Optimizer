# Implementation Plan: Realistic and Comprehensive HoS Simulation (Agent-Friendly Refactoring)

**Objective:** To implement a comprehensive, stateful Hours of Service (HoS) simulation that correctly models all required rest periods. 

**Guiding Principle:** This plan is designed for an LLM agent. It focuses on **modifying and enhancing the existing code in small, precise steps** rather than replacing entire functions. 

---

## Task Breakdown for LLM Agent

### Part 1: Step-by-Step Refactoring of `_check_hos_multiday`

**Goal:** Incrementally upgrade the existing HoS check to handle complex, multi-stop rest logic and to calculate the associated costs.

1.  **Navigate to the File:**
    *   Open `algo/second_level.py`.

2.  **Locate the Target Function:**
    *   Find the function `_check_hos_multiday(route: 'Route', driver_state: 'DriverState', sorted_tasks: List) -> bool`.

3.  **Step 1: Change Function Signature and Initialize State**
    *   **Action:** Modify the function definition line. Change the return type annotation from `-> bool` to `-> tuple[bool, float]`.
    *   **Action:** Immediately after the function definition line, insert the following code to handle empty routes and initialize cost-tracking variables:
    ```python
    if not sorted_tasks or len(sorted_tasks) < 2:
        return True, 0.0

    total_rest_cost = 0.0
    driver_cost_per_minute = (route.vehicle.cost_per_hour / 60.0) if route.vehicle else (25.0 / 60.0)
    sim_state = driver_state.copy()
    ```

4.  **Step 2: Add Service Time Simulation**
    *   **Action:** Inside the main `for i in range(len(sorted_tasks) - 1):` loop, as the very first lines, insert this block to simulate the service time at the current task:
    ```python
    # Placed at the start of the for loop
    start_task = sorted_tasks[i]
    end_task = sorted_tasks[i+1]

    # 1. Simulate Service Time at the start_task
    service_time = start_task.service_time
    if service_time > 0:
        if sim_state.work_today + service_time > sim_state.MAX_WORK_PER_DAY:
            return False, total_rest_cost # Infeasible
        sim_state.work_today += service_time
        sim_state.total_work_this_week += service_time
    ```

5.  **Step 3: Introduce Interruptible Travel Loop**
    *   **Action:** Find the line inside the `for` loop where `travel_time` is calculated. Immediately after it, insert these two lines to start the `while` loop:
    ```python
    travel_time_remaining = travel_time
    while travel_time_remaining > 0:
    ```
    *   **Action:** Indent all of the code that comes *after* the `travel_time` calculation (the logic that simulates driving and updates the driver state) so that it is now *inside* the `while` loop. Add a closing brace `}` if your editor requires it.

6.  **Step 4: Implement Detailed Rest Logic**
    *   **Action:** Inside the `while` loop, replace the simple driver state updates with the more detailed logic below. This new code calculates the maximum time the driver can legally drive and then simulates what happens if a rest is required.
    ```python
    # This block replaces the old state update logic inside the new while loop
    max_drive_before_break = sim_state.MAX_DRIVE_WITHOUT_BREAK - sim_state.drive_since_break
    max_drive_before_daily_limit = sim_state.MAX_DRIVE_PER_DAY - sim_state.drive_today
    max_work_before_daily_limit = sim_state.MAX_WORK_PER_DAY - sim_state.work_today

    drivable_time = min(max_drive_before_break, max_drive_before_daily_limit, max_work_before_daily_limit, travel_time_remaining)

    # Simulate driving for the calculated time
    sim_state.drive_since_break += drivable_time
    sim_state.drive_today += drivable_time
    sim_state.work_today += drivable_time
    sim_state.drive_this_week += drivable_time
    sim_state.total_work_this_week += drivable_time
    travel_time_remaining -= drivable_time

    # If travel is not complete, a rest is required
    if travel_time_remaining > 0:
        # A) 4.5-hour driving break
        if sim_state.drive_since_break >= sim_state.MAX_DRIVE_WITHOUT_BREAK:
            rest_duration = 45
            total_rest_cost += rest_duration * driver_cost_per_minute
            sim_state.work_today += rest_duration
            sim_state.total_work_this_week += rest_duration
            sim_state.drive_since_break = 0
            continue # Continue the while loop for the remaining travel time

        # B) Daily or Weekly limit reached
        if sim_state.drive_today >= sim_state.MAX_DRIVE_PER_DAY or sim_state.work_today >= sim_state.MAX_WORK_PER_DAY:
            # Check for weekly rest first
            if sim_state.total_work_this_week >= sim_state.MAX_WORK_PER_WEEK:
                rest_duration = 45 * 60 # 45-hour regular weekly rest
                total_rest_cost += rest_duration * driver_cost_per_minute
                sim_state.reset_weekly() # Assumes a method that resets all counters
            else: # Otherwise, take a daily rest
                rest_duration = 11 * 60 # 11-hour daily rest
                total_rest_cost += rest_duration * driver_cost_per_minute
                sim_state.total_work_this_week += rest_duration # Time passes for weekly work limit
                sim_state.reset_daily() # Assumes method resets daily counters
    ```

7.  **Step 5: Update All Return Statements**
    *   **Action:** Search for every `return` statement in the function and ensure it returns a tuple consistent with the new signature. For example:
        *   `return False` becomes `return False, total_rest_cost`
        *   `return True` becomes `return True, total_rest_cost`

### Part 2: Integrate the Refactored HoS Function

(These instructions remain the same as they correctly integrate the newly refactored function)

1.  **Navigate to `algo/second_level.py`**.
2.  **Update `is_feasible`** to correctly unpack the tuple from `_check_hos_multiday` and use the boolean result.
3.  **Update `calculate_z2_score`** to call `_check_hos_multiday`, use the boolean for a hard penalty, and add the `hos_cost` to the route's total cost.

### Testing the Fix

(The testing plan remains the same)

1.  Create a targeted test `tests/test_full_hos_simulation.py` for a multi-day route.
2.  Run the main `tests/comprehensive_integration_test.py` and verify zero HoS violations.
