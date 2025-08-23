
# Guide: Implementing Detailed Optimizer Logging

This document outlines the necessary code modifications to enable a highly verbose, step-by-step logging output from the `comprehensive_integration_test.py` script. This will provide deep insight into the decision-making process of the EPDT heuristic solver.

## Objective

To trace the entire execution pipeline, from data loading to final route generation, with detailed printouts at each significant step. This includes:
1.  **Scenario Loading**: Full details of all loaded orders, tasks, vehicles, and drivers.
2.  **Heuristic Optimization (L1/L2)**: A play-by-play of the Tabu Search, including neighborhood exploration and solution evaluation.
3.  **Special Assignments**: Detailed logs for order splitting and force-assignment strategies.
4.  **Driver Assignment**: A clear view of the cost calculations and matching process.
5.  **Final Routes**: The existing detailed final route summary.

---

## Step 1: Add a Global Logging Flag

To easily control the verbosity, we will introduce a logging flag in the parameters.

**File to modify**: `tests/comprehensive_integration_test.py`

1.  **Location**: Inside the `configure_algorithm_parameters` function.
2.  **Action**: Add a new parameter to the returned dictionary.

```python
# In configure_algorithm_parameters()
return {
    'verbose_logging': True,  # <--- ADD THIS LINE
    'tabu_tenure': 50,
    # ... other parameters
}
```

---

## Step 2: Log Loaded Scenario Details

Modify the integration test script to print all data immediately after loading.

**File to modify**: `tests/comprehensive_integration_test.py`

1.  **Location**: Inside `run_phase1_heuristic_test`, after the `create_scenario_from_excel` call.
2.  **Action**: Add the following code to iterate and print details.

```python
# After: epdt_vehicles = vehicles
print("\n" + "="*80)
print("PHASE 1.1: SCENARIO DATA LOADED")
print("="*80)

print(f"\n--- {len(epdt_vehicles)} Vehicles Loaded ---")
for v in epdt_vehicles:
    print(f"  - Vehicle ID: {v.id}, Type: {v.vehicle_type}, Capacity: {v.weight_capacity}kg, {v.volume_capacity}m³, {v.pallet_capacity} pallets")

print(f"\n--- {len(epdt_orders)} Orders Loaded ---")
for o in epdt_orders:
    print(f"  - Order ID: {o.id} (Priority: {o.priority}, Mandatory: {o.is_mandatory})")
    for task in o.get_all_tasks():
        tw = f"[{task.earliest_time}, {task.latest_time}]" if task.earliest_time is not None else "Any"
        print(f"    - Task: {task.id}, Type: {task.task_type.value}, Loc: {task.location_id}, Demand: {task.demand}kg, Volume: {task.volume}m³, Pallets: {task.pallets}, TW: {tw}")

# In run_phase2_driver_assignment, after load_drivers_from_excel_enhanced
print("\n" + "="*80)
print("PHASE 2.1: DRIVER DATA LOADED")
print("="*80)
print(f"\n--- {len(drivers)} Drivers Loaded ---")
for d in drivers:
    print(f"  - Driver: {d.name}, ID: {d.id}, License: {d.license}, Default Vehicle: {d.default_vehicle_id}, Cost/hr: {d.cost_per_hour}, Qualifications: {d.qualifications}")

```

---

## Step 3: Log L1 & L2 Heuristic Loops

Modify the core heuristic functions to explain their thought process.

**File to modify**: `algo/first_level.py`

1.  **Location**: Inside the `l1_heuristic` function's main `while` loop.
2.  **Action**: Add detailed print statements.

```python
# In l1_heuristic, inside the while loop
verbose = params.get('verbose_logging', False)
if verbose:
    print(f"\n--- L1 Heuristic: Iteration {total_iters} ---")
    print(f"  - Center Solution Score (Z1): {calculate_z1_score(center_solution, params, orders):.2f}")
    print(f"  - Non-improving iterations: {non_improving_iters}/{params['M1']}")

# Inside the VND loop (for neighborhood_func in neighborhoods)
if verbose:
    print(f"\n  -> Exploring Neighborhood: {neighborhood_func.__name__}")

# After iterating through a neighborhood's neighbors
if verbose and best_neighbor_in_N:
    print(f"    - Best neighbor in this neighborhood has score: {best_neighbor_score:.2f}")

# When an improving move is found and made
if improvement_found:
    if verbose:
        print(f"  => Improvement found! New center score: {calculate_z1_score(center_solution, params, orders):.2f}. Restarting VND.")
# In the diversification step
if not improvement_found and best_neighbors_pool:
    if verbose:
        print(f"  => No improvement. Making diversification move. New center score: {calculate_z1_score(center_solution, params, orders):.2f}")

```

**File to modify**: `algo/second_level.py`

1.  **Location**: Inside the `l2_heuristic` function.
2.  **Action**: Add logging to show the insertion process.

```python
# At the start of l2_heuristic
verbose = debug_assignment or enhanced_diagnostics
if verbose:
    print(f"    -- L2 Heuristic: Inserting Order {order.id} into Vehicle {route.vehicle.id} --")

# In _generate_initial_task_sequence, inside the loops for inserting tasks
if debug_assignment:
    # When trying a position
    print(f"      - Trying to insert {pickup.id} at position {pos}...")
    # After checking feasibility
    print(f"        Feasible: {is_feasible_for_insertion(test_route)}, New Route Score (Z2): {calculate_z2_score(test_route):.2f}")

# At the end of l2_heuristic, before returning
if verbose:
    if final_route:
        print(f"    -- L2 Heuristic Complete: Found best insertion for Order {order.id}. Final Route Score (Z2): {calculate_z2_score(final_route):.2f} --")
    else:
        print(f"    -- L2 Heuristic Complete: No feasible insertion found for Order {order.id}. --")
```

---

## Step 4: Log Special Assignment and Driver Assignment

**File to modify**: `tests/comprehensive_integration_test.py`

1.  **Location**: Inside `smart_force_assign_unassigned_orders`.
2.  **Action**: Add detailed logging.

```python
# At the start of smart_force_assign_unassigned_orders
print("\n--- Smart Force Assignment ---")
print(f"Attempting to force-assign {len(unassigned_orders)} orders.")

# Inside the loop for each order
print(f"\n- Processing difficult order: {order.id} (Weight: {order_weight:.1f}kg, Volume: {order_volume:.1f}m³)")

# Inside the loop for each vehicle
# After can_handle, penalty_score = ...
print(f"  - Checking Vehicle {vehicle.id}: Can handle: {can_handle}, Penalty: {penalty_score:.2f}")

# After finding the best vehicle
print(f"  => Best option: Vehicle {best_vehicle.id} with combined score {combined_score:.2f}.")
```

**File to modify**: `algo/driver_assignment_enhanced.py`

1.  **Location**: Inside `assign_drivers_to_routes_enhanced`.
2.  **Action**: Add logging for the greedy assignment logic.

```python
# At the start of assign_drivers_to_routes_enhanced
verbose = config.get('verbose_logging', False)
if verbose:
    print("\n--- Driver Assignment Process ---")

# In Phase 1 (CE drivers)
if verbose:
    print(f"\n- Phase 1: Assigning {len(ce_drivers)} CE drivers to {len(heavy_routes)} heavy routes.")

# Inside the loop for each heavy route
if verbose:
    print(f"  - Evaluating Route: {route.vehicle.id}")
# Inside the loop for each driver
if verbose:
    # After calculating score
    print(f"    - Driver {driver.name}: Score = {score:.2f} (Depot match: {driver.home_depot_id == route.vehicle.depot_id}, Default vehicle: {driver.default_vehicle_id == route.vehicle.id})")
# After finding the best driver
if verbose and best_driver:
    print(f"  => Assignment: Route {route.vehicle.id} -> Driver {best_driver.name} (Best Score: {best_score:.2f})")

# Repeat similar logging for Phase 2 (remaining drivers)
```

By making these changes, running the `comprehensive_integration_test.py` script will produce a highly detailed, human-readable log that traces the optimizer's logic from start to finish.

---

## Step 5: Controlling Order Splitting

This section describes how to add a parameter to control the advanced assignment strategies for oversized or unassigned orders, such as order splitting.

### 1. Add the Control Parameter

**File to modify**: `tests/comprehensive_integration_test.py`

**Location**: Inside the `configure_algorithm_parameters` function.

**Action**: Add the `separate_orders` parameter and set it to `False` by default to disable it.

```python
# In configure_algorithm_parameters()
return {
    'verbose_logging': True,
    'separate_orders': False, # <-- ADD THIS LINE
    'tabu_tenure': 50,
    # ... other parameters
}
```

### 2. Make the Splitting Logic Conditional

The `comprehensive_integration_test.py` script contains functions like `attempt_order_splitting_assignment` and `smart_force_assign_unassigned_orders` to handle difficult orders. We will make their execution conditional based on the new flag.

**File to modify**: `tests/comprehensive_integration_test.py`

**Location**: Inside the `run_phase1_heuristic_test` function, find the block for "Step 3.5".

**Action**: Replace the existing block with the following logic, which separates the control for order splitting and force assignment.

```python
        # Step 3.5: Apply special assignment strategies for unassigned orders
        if params.get('separate_orders', False):
            print("\n" + "-"*40)
            print("PHASE 1.5a: APPLYING ORDER SPLITTING")
            print("-"*40)
            # This function will attempt to split oversized orders and assign the smaller parts.
            attempt_order_splitting_assignment(solution, epdt_orders, epdt_vehicles)

        if params.get('enable_force_assignment', False):
            print("\n" + "-"*40)
            print("PHASE 1.5b: APPLYING SMART FORCE ASSIGNMENT")
            print("-"*40)
            # This function will attempt to force-assign remaining unassigned orders.
            smart_force_assign_unassigned_orders(solution, epdt_orders, epdt_vehicles)
```

### 3. Add Logging to Splitting Functions

To ensure the new logs are detailed, add print statements inside the splitting and force-assignment functions.

**File to modify**: `tests/comprehensive_integration_test.py`

**Location**: Inside `attempt_order_splitting_assignment` and `split_large_order`.

**Action**: Add the following logging statements.

```python
# In attempt_order_splitting_assignment, inside the loop
if exceeds_limits:
    print(f"\n- Found oversized order: {order.id}. Attempting to split.")

# In split_large_order
print(f"  - Splitting order {order.id} into {num_splits} sub-orders based on {constraining_factor}.")
# In the loop creating sub-orders
print(f"    - Created sub-order {split_order_id} with requirements: W:{split_weight:.1f}, V:{split_volume:.2f}, P:{split_pallets:.0f}")

# In attempt_order_splitting_assignment, after trying to assign a split order
if split_assigned > 0:
    print(f"  - SUCCESS: Split order {split_order.id} was assigned.")
else:
    print(f"  - FAILURE: Split order {split_order.id} could not be assigned.")
```

---

## Step 6: Reformatting the Final Report

This step replaces the existing `print_detailed_route_breakdown` function with a new version that produces a chronological, journey-style report as specified in the example.

**File to modify**: `tests/comprehensive_integration_test.py`

**Action**: Replace the entire `print_detailed_route_breakdown` function with the new code below. Also, add the new helper function `_format_time_hhmm`.

```python

def _format_time_hhmm(minutes: float) -> str:
    """Formats minutes into a hh:mm string."""
    if minutes < 0:
        return "00:00"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"

def print_detailed_route_breakdown(vehicle_id: str, route, vehicle=None):
    """
    Prints a detailed, chronological journey log for a route, interleaving
    tasks with travel and rest events from the HoS timeline.
    """
    if not route or not route.tasks:
        return

    # --- 1. Print Header ---
    vehicle_capacity_str = f"(Max: {getattr(vehicle, 'volume_capacity', 'N/A')}m³, {getattr(vehicle, 'weight_capacity', 'N/A')}kg, {getattr(vehicle, 'pallet_capacity', 'N/A')}pal)"
    driver_str = f"Driver {getattr(route.driver, 'name', 'Unassigned')} (License: {getattr(route.driver, 'license', 'N/A')}, Type: {getattr(vehicle, 'vehicle_type', 'N/A')})"
    print(f"\nVehicle: {vehicle_id} {vehicle_capacity_str} -> {driver_str}")

    hos_timeline = getattr(route, 'hos_timeline', [])
    if not hos_timeline:
        print("    HoS Timeline: Not available.")
        return

    # --- 2. Print HoS Debug Info ---
    total_duration_h = hos_timeline[-1].end_time / 60.0 if hos_timeline else 0
    drive_time_h = sum(e.duration for e in hos_timeline if e.event_type == 'DRIVE') / 60.0
    work_time_h = sum(e.duration for e in hos_timeline if e.event_type in ['DRIVE', 'WORK']) / 60.0
    print(f"DEBUG: FIRST PATH - Total elapsed timeline: {total_duration_h:.2f} hours, Drive: {drive_time_h:.2f} hours, Work: {work_time_h:.2f} hours")

    # --- 3. Process and Print Chronological Log ---
    tasks_in_route = [t for t in route.tasks if not (t.is_depot_start() or t.is_depot_return())]
    current_task_idx = 0
    current_day = -1
    daily_drive = 0
    daily_breaks = 0
    daily_salary = 0
    driver_cost_per_min = getattr(route.driver, 'cost_per_hour', 25) / 60.0

    # Print initial Depot Start task
    print(f"\n          1. DEPOT-ASTI")
    print(f"             Arrival at 00:00:00 [No window -> No window] - Status: On time")
    print(f"             Load: +0.0kg, +0.0m³ -> Total: 0.0kg, 0.0m³")

    # Group timeline events by journeys between tasks
    journey_events = []
    for event in hos_timeline:
        if event.event_type != 'WORK':
            journey_events.append(event)
        else:
            # WORK event signifies arrival at a task. First, print the journey leading here.
            if journey_events:
                next_task_location = getattr(tasks_in_route[current_task_idx], 'location_id', 'Unknown')
                total_journey_time = sum(e.duration for e in journey_events)
                print(f"\n            DRIVE to {next_task_location} - {_format_time_hhmm(total_journey_time)}")
                for e in journey_events:
                    print(f"               - {e.event_type}: {_format_time_hhmm(e.duration)}")
                journey_events = []

            # Now print the WORK task details
            task = tasks_in_route[current_task_idx]
            day_of_task = int(event.start_time // 1440)

            if day_of_task != current_day:
                if current_day != -1:
                    # Print summary for the previous day
                    print(f"\n      - Day {current_day + 1}: Drive: {_format_time_hhmm(daily_drive)}, Breaks: {_format_time_hhmm(daily_breaks)}, Salary: EUR{daily_salary:.2f}")
                current_day = day_of_task
                daily_drive, daily_breaks, daily_salary = 0, 0, 0
                print(f"\n      - Day {current_day + 1} (Date: ...):")

            # Update daily stats
            daily_drive += sum(e.duration for e in journey_events if e.event_type == 'DRIVE')
            daily_breaks += sum(e.duration for e in journey_events if e.event_type == 'REST')
            daily_salary += (sum(e.duration for e in journey_events) + event.duration) * driver_cost_per_min

            arrival_str = format_absolute_minutes(event.start_time)
            tw_str = get_time_window_info(task)
            status_str = "On time" # Simplified, full logic can be added
            weight_change, volume_change = get_load_change(task)
            # You need to track total load correctly here if needed

            print(f"\n          {current_task_idx + 2}. {task.location_id} (Order: {task.order_id})")
            print(f"             Arrival at: {arrival_str} [{tw_str}] - Status: {status_str}")
            print(f"             WORK {_format_time_hhmm(event.duration)} - Load: {weight_change:+.1f}kg, {volume_change:+.2f}m³ (add pallets) -> Total: ...")
            
            current_task_idx += 1

    # Print final journey back to depot
    if journey_events:
        total_journey_time = sum(e.duration for e in journey_events)
        print(f"\n            DRIVE to DEPOT-ASTI - {_format_time_hhmm(total_journey_time)}")
        for e in journey_events:
            print(f"               - {e.event_type}: {_format_time_hhmm(e.duration)}")

    # Print final day summary
    if current_day != -1:
        print(f"\n      - Day {current_day + 1}: Drive: {_format_time_hhmm(daily_drive)}, Breaks: {_format_time_hhmm(daily_breaks)}, Salary: EUR{daily_salary:.2f}")

```

---

## Step 5: Controlling Order Splitting

This section describes how to add a parameter to control the advanced assignment strategies for oversized or unassigned orders, such as order splitting.

### 1. Add the Control Parameter

**File to modify**: `tests/comprehensive_integration_test.py`

**Location**: Inside the `configure_algorithm_parameters` function.

**Action**: Add the `separate_orders` parameter and set it to `False` by default to disable it.

```python
# In configure_algorithm_parameters()
return {
    'verbose_logging': True,
    'separate_orders': False, # <-- ADD THIS LINE
    'tabu_tenure': 50,
    # ... other parameters
}
```

### 2. Make the Splitting Logic Conditional

The `comprehensive_integration_test.py` script contains functions like `attempt_order_splitting_assignment` and `smart_force_assign_unassigned_orders` to handle difficult orders. We will make their execution conditional based on the new flag.

**File to modify**: `tests/comprehensive_integration_test.py`

**Location**: Inside the `run_phase1_heuristic_test` function, find the block for "Step 3.5".

**Action**: Replace the existing block with the following logic, which separates the control for order splitting and force assignment.

```python
        # Step 3.5: Apply special assignment strategies for unassigned orders
        if params.get('separate_orders', False):
            print("\n" + "-"*40)
            print("PHASE 1.5a: APPLYING ORDER SPLITTING")
            print("-"*40)
            # This function will attempt to split oversized orders and assign the smaller parts.
            attempt_order_splitting_assignment(solution, epdt_orders, epdt_vehicles)

        if params.get('enable_force_assignment', False):
            print("\n" + "-"*40)
            print("PHASE 1.5b: APPLYING SMART FORCE ASSIGNMENT")
            print("-"*40)
            # This function will attempt to force-assign remaining unassigned orders.
            smart_force_assign_unassigned_orders(solution, epdt_orders, epdt_vehicles)
```

### 3. Add Logging to Splitting Functions

To ensure the new logs are detailed, add print statements inside the splitting and force-assignment functions.

**File to modify**: `tests/comprehensive_integration_test.py`

**Location**: Inside `attempt_order_splitting_assignment` and `split_large_order`.

**Action**: Add the following logging statements.

```python
# In attempt_order_splitting_assignment, inside the loop
if exceeds_limits:
    print(f"\n- Found oversized order: {order.id}. Attempting to split.")

# In split_large_order
print(f"  - Splitting order {order.id} into {num_splits} sub-orders based on {constraining_factor}.")
# In the loop creating sub-orders
print(f"    - Created sub-order {split_order_id} with requirements: W:{split_weight:.1f}, V:{split_volume:.2f}, P:{split_pallets:.0f}")

# In attempt_order_splitting_assignment, after trying to assign a split order
if split_assigned > 0:
    print(f"  - SUCCESS: Split order {split_order.id} was assigned.")
else:
    print(f"  - FAILURE: Split order {split_order.id} could not be assigned.")
```

```