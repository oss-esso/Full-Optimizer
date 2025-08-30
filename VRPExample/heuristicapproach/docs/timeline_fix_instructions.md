# Fixing the HoS Timeline and Waiting Time Logic

This document outlines the necessary steps to fix two issues identified in the `comprehensive_integration_test.py` output:
1.  The `HosEngine` does not account for waiting time when a vehicle arrives early for a delivery, leading to incorrect timeline calculations and failure to detect late arrivals later in the route.
2.  The "HoS TIMELINE JOURNEY" output format is not as detailed as the "SIMPLIFIED CHRONOLOGICAL JOURNEY" and lacks a clear breakdown of drive, work, and rest events.

## Summary of the Problem

The root cause is that the logic for calculating waiting time exists only within the `_print_simplified_chronological_view` function in `tests/comprehensive_integration_test.py`. This is a display function and its logic is not part of the core `HosEngine` in `algo/hos_simulation.py`, which is responsible for generating the actual route timeline.

The `HosEngine` produces a timeline of events without considering the `earliest_time` of tasks. When the test script prints the "HoS TIMELINE JOURNEY", it is formatting this incomplete data, resulting in an inaccurate representation of the route.

## Plan for the Fix

The fix involves two main parts:
1.  **Integrating Waiting Time Calculation into the `HosEngine`**: Modify the `build_compliant_timeline` function in `algo/hos_simulation.py` to correctly simulate waiting periods.
2.  **Enhancing the HoS Timeline Output**: Modify the `_print_hos_timeline_view` function in `tests/comprehensive_integration_test.py` to parse the corrected timeline and display it in the desired detailed format.

---

### Part 1: Modify `algo/hos_simulation.py`

The goal is to make the `build_compliant_timeline` function the single source of truth for all timeline calculations, including waiting time.

**File to Edit:** `algo/hos_simulation.py`

**Function to Modify:** `build_compliant_timeline`

**Detailed Steps:**

1.  **Refactor the Main Loop:** The function iterates through the tasks in a route. The simulation logic within this loop needs to follow a clean `DRIVE -> (optional REST) -> (optional WAIT) -> WORK` sequence for each segment of the journey.

2.  **Implement the `DRIVE -> REST -> WAIT -> WORK` Sequence:**
    Inside the `for i in range(len(route.tasks) - 1):` loop, for each `start_task` and `end_task`, the logic should be as follows:

    a. **Simulate Travel and HoS Breaks:**
        - Calculate the `travel_time` from `start_task` to `end_task`.
        - Create a loop that consumes the `travel_time_remaining`.
        - Inside the loop, determine the maximum `drivable_time` before a mandatory 45-minute break is needed.
        - Add a `DRIVE` `SimulatedEvent` for the `drivable_time`.
        - Update `current_time`, driver state, and `travel_time_remaining`.
        - If `travel_time_remaining` is still greater than zero, it means a break is required. Add a `REST` `SimulatedEvent` for 45 minutes, update `current_time`, and reset the driver's `drive_since_break` counter.

    b. **Simulate Arrival and Waiting (Critical Fix):**
        - After the travel simulation loop is complete, `current_time` represents the vehicle's arrival time at the `end_task` location.
        - Retrieve the `earliest_time` for the `end_task`.
        - **If `current_time < earliest_time`:**
            - Calculate `wait_duration = earliest_time - current_time`.
            - Create a new `SimulatedEvent` with `event_type='WAIT'`, `start_time=current_time`, `end_time=earliest_time`, and `duration=wait_duration`. Add it to the timeline.
            - **Crucially, update `current_time = earliest_time`**. This advances the simulation clock to the end of the waiting period.

    c. **Simulate Work:**
        - Add the `WORK` `SimulatedEvent` for the `end_task`'s `service_time`, starting at the now-correct `current_time`.
        - Update `current_time` by adding the `service_time`.

**Example Snippet (Illustrative):**
```python
# Inside build_compliant_timeline, after simulating travel to end_task
# current_time now holds the arrival time at end_task's location

earliest_start = getattr(end_task, 'earliest_time', None)
if earliest_start is not None and current_time < earliest_start:
    wait_duration = earliest_start - current_time
    timeline.append(SimulatedEvent(
        event_type='WAIT',
        start_time=current_time,
        end_time=earliest_start,
        duration=wait_duration,
        description=f"Wait at {end_task.location_id}",
        location=getattr(end_task, 'location', None),
        task_id=end_task.id
    ))
    current_time = earliest_start # Advance clock past the wait

# Now, add the work event
service_time = getattr(end_task, 'service_time', 0)
if service_time > 0:
    timeline.append(SimulatedEvent(
        event_type='WORK',
        start_time=current_time,
        end_time=current_time + service_time,
        # ... other event details
    ))
    current_time += service_time
```

---

### Part 2: Modify `tests/comprehensive_integration_test.py`

The goal is to make the "HoS TIMELINE JOURNEY" output as rich and readable as the "SIMPLIFIED" one by properly interpreting the new, more accurate timeline data from the `HosEngine`.

**File to Edit:** `tests/comprehensive_integration_test.py`

**Function to Modify:** `_print_hos_timeline_view` (and potentially rename it for clarity, e.g., `print_detailed_hos_journey`)

**Detailed Steps:**

1.  **Restructure the Function:** The current function simply loops through the timeline events and prints them one by one. It needs to be restructured to group events chronologically between tasks.

2.  **Group Events by Journey Segment:**
    - Iterate through the `timeline` events from the `HosEngine`.
    - Collect all `DRIVE`, `REST`, and `WAIT` events into a temporary list (e.g., `journey_events`).
    - When a `WORK` event is encountered, it signifies the end of a travel segment and arrival at a task.
    - At this point, print the journey summary for the collected `journey_events`.

3.  **Format the Journey Segment Output:**
    - Before printing the `WORK` event details, print a header like:
      `DRIVE to {task.location_id} - {total_journey_duration}`
    - Iterate through the `journey_events` list and print each event (`DRIVE`, `REST`, `WAIT`) with its duration, mimicking the indented format of the "SIMPLIFIED" view.
    - Clear the `journey_events` list to prepare for the next segment.

4.  **Print the `WORK` Event:**
    - After printing the journey details, print the details of the `WORK` event, including arrival time, time window, status, and load changes. The status should now be correct because the arrival time from the timeline accounts for waiting.

5.  **Handle the Final Return to Depot:** After the main loop, if there are any remaining events in `journey_events`, this represents the final drive back to the depot. Print this final segment.

By following these steps, the `HosEngine` will produce a legally compliant and chronologically accurate timeline, and the test output will correctly visualize this timeline, resolving both issues.
