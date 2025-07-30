# Plan: Implement Explicit Waiting Time Simulation at Departure

## 1. Objective

The goal is to enhance the solver to explicitly calculate and track vehicle waiting times, modeling the wait as occurring *before* departure to a task. This ensures that a vehicle waits at its current location (e.g., the depot) to arrive at the next task just in time for its time window. This change will provide more realistic routes and accurate Hours of Service (HoS) calculations.

The final output should reflect this waiting time being associated with the departure from a location. For example, a long wait to serve the first customer should be shown as waiting at the depot.

A route that currently looks like this:
`VEHICLE: XA345KW: ... 2. VIA LEGNANO, MESERO (MI), ITALY (Order: 5) - ... Status: (Arrived early, would wait 1d 21h 15m)`

Should be transformed to reflect that the waiting happens at the previous stop (e.g., DEPOT-ASTI). The cumulative time at each step should include any waiting that occurred before departing to that step.

## 2. Implementation Plan

This will be achieved in three steps:

### Task 1: Enhance the Route Data Structure

To track waiting time, we first need a place to store it.

*   **File to Modify:** `algo/epdt_data_structures.py`
*   **Action:** Add a `total_wait_time` attribute to the `Route` dataclass. This will hold the accumulated waiting time for the entire route.

**Instructions:**

In the `Route` dataclass definition, add the `total_wait_time` field as shown below.

```python
# In algo/epdt_data_structures.py
@dataclass
class Route:
    # ... existing fields ...
    _cached_time: Optional[float] = field(default=None, init=False)
    _is_feasible_cached: Optional[bool] = field(default=None, init=False)
    total_wait_time: float = 0.0 # <-- ADD THIS LINE
    
    # ... rest of the class ...
```

### Task 2: Calculate Waiting Time at Departure

The core logic for calculating route feasibility must be updated to model waiting at the departure location.

*   **File to Modify:** `algo/second_level.py`
*   **Action:** Update the `is_feasible` function to calculate `wait_time` based on the next task's time window and apply this wait at the current location.

**Instructions:**

Replace the existing `is_feasible` function with the following enhanced version. This version calculates wait time to ensure on-time arrival at the next task and updates timing calculations accordingly.

```python
# In algo/second_level.py

def is_feasible(route: Route, debug_feasibility: bool = False, return_reason: bool = False) -> Union[bool, Tuple[bool, str]]:
    """
    Check if a route is feasible considering all constraints, including multi-day operations, HoS, and waiting times.
    Waiting is modeled as occurring at the departure location to arrive "just-in-time" for the next task.
    """
    if not route or not route.tasks:
        return (True, "Empty route") if return_reason else True

    # ... (Depot and Capacity validation remains the same) ...

    # 3. Time Window, HoS, and Waiting Time Validation
    try:
        from route_provider import calculate_travel_time_between_tasks
    except ImportError:
        return (False, "Route provider not available for time calculations") if return_reason else False

    # Initialize time and waiting time tracking
    completion_time = 0.0  # Tracks the completion time of the last task
    route.total_wait_time = 0.0
    daily_driving_time = 0
    daily_work_time = 0
    current_day = 1

    for i in range(len(route.tasks)):
        task = route.tasks[i]
        
        departure_time = completion_time
        travel_time = 0
        wait_time = 0

        if i > 0:
            prev_task = route.tasks[i-1]
            travel_time = calculate_travel_time_between_tasks(prev_task, task, route.vehicle)
            
            # If the next task has an earliest start time, we might need to wait at the previous location.
            if task.earliest_time and task.earliest_time > 0:
                # Required departure time from previous location to arrive exactly at earliest_time
                required_departure_time = task.earliest_time - travel_time
                
                # If we are ready to depart earlier than required, we wait.
                if completion_time < required_departure_time:
                    wait_time = required_departure_time - completion_time
                    route.total_wait_time += wait_time
        
        # Departure time from previous task location includes waiting
        departure_time = completion_time + wait_time
        
        # Arrival at current task
        arrival_time = departure_time + travel_time
        service_start_time = arrival_time

        # Check for lateness
        if task.latest_time and service_start_time > task.latest_time:
            reason = f"Time window violation at task {task.id}: Service start {format_absolute_minutes(service_start_time)} is after latest {format_absolute_minutes(task.latest_time)}"
            return (False, reason) if return_reason else False

        service_time = getattr(task, 'service_time', 0)
        completion_time = service_start_time + service_time

        # ... (HoS checks remain the same, but are now based on more accurate timing) ...
        # Note: Ensure HoS logic correctly handles the 'wait_time'. Depending on regulations,
        # long waiting periods might count as rest. This implementation assumes waiting
        # is 'on-duty' time. The HoS logic should be reviewed for compliance.

    return (True, "Route is feasible") if return_reason else True
```

### Task 3: Display Waiting Time in Output

Update the visualization logic to display where waiting occurs. The waiting time before starting a leg of the journey should be displayed at the *departure* task, and the cumulative time of the arrival task should reflect this wait.

*   **File to Modify:** `algo/solution_visualizer.py`
*   **Action:** Modify the function that formats the route details to calculate and display the waiting time at each task before departure to the next. The cumulative time for subsequent tasks must include this waiting period.

**Instructions:**

In the function that formats the route summary (e.g., `get_solution_summary_string`), you must replicate the timing logic from the new `is_feasible` function. As you iterate through the tasks to print them, you must calculate the waiting time that occurs *at the current task* before departing to the *next* task.

The status message for a task should indicate any waiting time. The cumulative time for the next task should be presented to clearly show the components of travel and waiting time.

**Example of new output format:**

`1. DEPOT-ASTI ... Status: (Waiting 1d 21h 15m before departure)`
`2. VIA LEGNANO, MESERO ... Cumulative: 1d 23h 59m (+2h 39m travel, +1d 21h 15m wait) ...`

**Example Logic:**

```python
# In algo/solution_visualizer.py ...

# You will need to maintain the state of `completion_time` as you loop through tasks to print them.
#
# For each task `i` in the route:
# 1. Calculate `wait_time` that occurs at task `i` before departing for task `i+1`.
#    (This requires `completion_time` at `i`, `travel_time` to `i+1`, and `earliest_time` of `i+1`)
#
# 2. Print details for task `i`, including the `wait_time` if it's > 0.
#
# 3. For task `i+1`, when printing its cumulative time, break down the increment into
#    travel time from `i` and wait time at `i`.
#
# 4. Update `completion_time` for the next iteration.
#    completion_time_i = arrival_time_i + service_time_i
#    departure_time_i = completion_time_i + wait_time_at_i
#    arrival_time_i+1 = departure_time_i + travel_time
```