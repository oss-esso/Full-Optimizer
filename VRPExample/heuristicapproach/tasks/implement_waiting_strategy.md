# Implementing a Vehicle Waiting Strategy

## 1. Problem Overview

The current solver is producing infeasible routes because it does not account for waiting time. Vehicles are arriving at customer locations before their time windows open, leading to `Feasible: False` statuses. Additionally, some routes do not start at the depot, which is a fundamental requirement.

To solve this, we need to implement a waiting strategy that allows vehicles to wait, either at the depot before starting a route or at a customer location, to ensure that all time window constraints are met.

## 2. Core Concepts

*   **Waiting at Depot:** A vehicle assigned a route where the first task has a late time window (e.g., on Day 2) should not depart on Day 1. It should wait at the depot and begin its route at an appropriate time to meet the first task's window.

*   **Waiting at Task Location:** If a vehicle arrives at a customer location before the `earliest_time` of the task, it must wait until the window opens before starting its service.

*   **Feasibility:** The `is_feasible` check must be updated to account for this waiting time. A route is only feasible if, after accounting for all travel, service, and waiting times, no task is serviced *after* its `latest_time`.

## 3. Remediation Plan

### Task 1: Implement Waiting Logic in Feasibility Check

*   **File to modify:** `algo/second_level.py`
*   **Goal:** Update the `is_feasible` function to correctly calculate route timing, including waiting time, and validate against time windows.

**Instructions:**

Replace the entire `is_feasible` function with the following corrected and enhanced version. This version introduces a `current_time` variable to track the vehicle's progress and correctly calculates waiting time.

```python
def is_feasible(route: Route, debug_feasibility: bool = False, return_reason: bool = False) -> Union[bool, Tuple[bool, str]]:
    """
    Check if a route is feasible considering all constraints, including multi-day operations and HoS.
    This is the definitive check for a route's validity.
    """
    if not route or not route.tasks:
        return (True, "Empty route") if return_reason else True

    if debug_feasibility:
        print(f"DEBUG FEASIBILITY: Checking route feasibility for vehicle {route.vehicle.id}")
        print(f"DEBUG FEASIBILITY: Route has {len(route.tasks)} tasks")

    # 1. Depot Start/End Validation
    if not route.tasks[0].is_depot_start() or not route.tasks[-1].is_depot_return():
        reason = f"Route validation failed: First task ({route.tasks[0].id}) is not a depot start task or last task ({route.tasks[-1].id}) is not a depot return task"
        if debug_feasibility:
            print(f"DEBUG FEASIBILITY: {reason}")
        return (False, reason) if return_reason else False
    if debug_feasibility:
        print("DEBUG FEASIBILITY: ✅ Depot validation passed - route starts and ends at depot")

    # 2. Capacity Validation
    current_weight = 0
    current_volume = 0
    current_pallets = 0
    for task in route.tasks:
        current_weight += getattr(task, 'demand', 0)
        current_volume += getattr(task, 'volume', 0)
        current_pallets += getattr(task, 'pallets', 0)
        if (current_weight > route.vehicle.weight_capacity or 
            current_volume > route.vehicle.volume_capacity or 
            current_pallets > route.vehicle.pallet_capacity):
            reason = f"Capacity violation at task {task.id}: W:{current_weight}/{route.vehicle.weight_capacity}, V:{current_volume}/{route.vehicle.volume_capacity}, P:{current_pallets}/{route.vehicle.pallet_capacity}"
            if debug_feasibility:
                print(f"DEBUG FEASIBILITY: {reason}")
            return (False, reason) if return_reason else False
    if debug_feasibility:
        print("DEBUG FEASIBILITY: ✅ Capacity validation passed")

    # 3. Time Window and HoS Validation
    try:
        from route_provider import calculate_travel_time_between_tasks
    except ImportError:
        return (False, "Route provider not available for time calculations") if return_reason else False

    # Initialize time tracking. Start at 0, but the first task's earliest_time will dictate the actual departure.
    current_time = 0
    daily_driving_time = 0
    daily_work_time = 0
    current_day = 1

    for i in range(len(route.tasks)):
        task = route.tasks[i]
        travel_time = 0
        if i > 0:
            travel_time = calculate_travel_time_between_tasks(route.tasks[i-1], task, route.vehicle)

        # Update time with travel
        arrival_time = current_time + travel_time

        # --- WAITING LOGIC ---
        # Check if we need to wait for the time window to open
        wait_time = 0
        if task.earliest_time and arrival_time < task.earliest_time:
            wait_time = task.earliest_time - arrival_time
        
        service_start_time = arrival_time + wait_time

        # Check for lateness
        if task.latest_time and service_start_time > task.latest_time:
            reason = f"Time window violation at task {task.id}: Service start {format_absolute_minutes(service_start_time)} is after latest {format_absolute_minutes(task.latest_time)}"
            if debug_feasibility:
                print(f"DEBUG FEASIBILITY: {reason}")
            return (False, reason) if return_reason else False

        service_time = getattr(task, 'service_time', 0)
        current_time = service_start_time + service_time

        # HoS (Hours of Service) checks
        # This is a simplified check. A full implementation would be more complex.
        new_day = int(current_time // 1440) + 1
        if new_day > current_day:
            daily_driving_time = 0
            daily_work_time = 0
            current_day = new_day

        daily_driving_time += travel_time
        daily_work_time += travel_time + service_time + wait_time

        if daily_driving_time > route.vehicle.max_driving_time or daily_work_time > route.vehicle.max_work_time:
            reason = f"HoS constraint violated (LEGAL LIMITS) on day {current_day}: Drive={daily_driving_time:.1f}m, Work={daily_work_time:.1f}m"
            if debug_feasibility:
                print(f"DEBUG FEASIBILITY: {reason}")
            return (False, reason) if return_reason else False

    if debug_feasibility:
        print("DEBUG FEASIBILITY: ✅ Time window and HoS validation passed")

    return (True, "Route is feasible") if return_reason else True
```

### Task 2: Fix Depot Start in Heuristic

*   **File to modify:** `algo/first_level.py`
*   **Goal:** Ensure that all routes generated by the `l1_heuristic` start and end with a depot task.

**Instructions:**

In the `l1_heuristic` function, after the main optimization loop finishes, add a final check to ensure all routes that have tasks are properly framed by depot start/return tasks.

**At the end of the `l1_heuristic` function, right before the `return solution` statement, insert this block of code:**

```python
    # Final check: Ensure all active routes have depot start/end tasks
    depot_location = solution.depot_location # Assuming solution has this
    for vehicle_id, route in solution.routes.items():
        if route.tasks: # Only for routes that are not empty
            # Check for depot start
            if not route.tasks[0].is_depot_start():
                start_task = Task(id=f"depot_start_order_{vehicle_id}", location_id=depot_location.id, task_type=TaskType.DEPOT_START, order_id=f"depot_start_order_{vehicle_id}", lat=depot_location.lat, lon=depot_location.lon)
                route.tasks.insert(0, start_task)
            
            # Check for depot return
            if not route.tasks[-1].is_depot_return():
                return_task = Task(id=f"depot_return_order_{vehicle_id}", location_id=depot_location.id, task_type=TaskType.DEPOT_RETURN, order_id=f"depot_return_order_{vehicle_id}", lat=depot_location.lat, lon=depot_location.lon)
                route.tasks.append(return_task)

    return solution
```

By implementing these changes, the solver will be able to create feasible, multi-day routes that respect time windows by intelligently waiting, and the test output will accurately reflect all timing calculations.
