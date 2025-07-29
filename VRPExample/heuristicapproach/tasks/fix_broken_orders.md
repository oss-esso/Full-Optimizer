# Fixing Broken Order and Load Calculation in Comprehensive Test

## 1. Problem Overview
The `comprehensive_integration_test.py` is failing. The output in `brken_utput.txt` shows that all vehicle routes have 0kg and 0.0m³ load changes for all tasks. This indicates two core problems:
1.  Orders with multiple tasks are not being created correctly.
2.  The test's output generation is not reading task load data correctly.

## 2. Root Cause Analysis

*   **Issue 1: Incorrect Order Creation in `utils/scenario_creator.py`**
    The `create_scenario_from_excel` function currently creates a new `Order` for each row in the `CONSEGNE` sheet. This is incorrect when multiple rows belong to the same logical order (identified by `ORDER_ID`). The script needs to group rows by `ORDER_ID` and create a single `Order` object that contains all the tasks for that order.

*   **Issue 2: Flawed Load Calculation in `tests/comprehensive_integration_test.py`**
    The `print_detailed_route_breakdown` function in the test script contains a helper function `get_load_change` that is responsible for determining the weight and volume change for a task. This function incorrectly tries to read load data from a `task.order` attribute, which doesn't exist. The correct load information is stored directly on the `Task` object in the `demand` (for weight) and `volume` attributes.

*   **Issue 3: Incorrect Depot Task Generation**
    The logic in `create_scenario_from_excel` was incorrectly creating a corresponding "depot bay" task for every single customer task. This is wrong. An `Order` should only contain the tasks that happen at customer locations. The vehicle's journey starting from and returning to the depot is part of the `Route`, which is determined by the optimization algorithm, not part of the `Order` definition itself. This flaw leads to nonsensical routes (e.g., depot-to-depot trips for a customer order) and orders with double the necessary tasks.

*   **Issue 4: Incorrect Time Window Parsing**
    The logic in `create_task_from_row` does not correctly parse the multi-day time window format. It reads the time columns but ignores the `EARLIEST DAY` and `LATEST DAY` columns, failing to convert them into an absolute time in minutes from the start of the planning horizon. This results in all tasks having no effective time window.

*   **Issue 5: Infeasible Routes and Time Window Display**
    The latest output shows `Feasible: False` for many routes. This is because the core heuristic is not being penalized for violating time windows, so it creates routes that are then correctly identified as infeasible. The test output also fails to display the time windows because the formatting function is broken, and it doesn't account for the service time (delay) at each location.

## 3. Remediation Plan

Here are the detailed instructions to fix these issues.

### Task 1: Fix Route Feasibility and Time Window Handling

*   **Goal:** Correct the feasibility checks, algorithm parameters, and test output to properly handle and display time windows and service delays.

**Instructions:**

1.  **Update Algorithm Parameters in `tests/comprehensive_integration_test.py`**
    The heuristic is ignoring time windows because the penalty is too low. In the `configure_algorithm_parameters` function, increase the penalty significantly.

    **Change this line:**
    ```python
            'time_window_violation_penalty': 10.0,  # Further reduced from 25.0 - very tolerant of time violations
    ```
    **To this:**
    ```python
            'time_window_violation_penalty': 500.0, # Significantly increased to enforce time windows
    ```

2.  **Update the Route Breakdown Function in `tests/comprehensive_integration_test.py`**
    The `print_detailed_route_breakdown` function needs a major overhaul to correctly calculate cumulative time (including service delays) and display the multi-day time windows.

    **Replace the entire `print_detailed_route_breakdown` function with this corrected version:**

    ```python
    def print_detailed_route_breakdown(vehicle_id: str, route, vehicle=None):
        """
        Print detailed route breakdown with task sequence, load tracking, and timing.
        """
        if not route or not route.tasks:
            return

        def get_location_name(task):
            return getattr(getattr(task, 'location', None), 'name', getattr(task, 'location_id', "Unknown"))

        def get_order_info(task):
            return f"Order: {getattr(task, 'order_id', 'N/A')}"

        def format_absolute_minutes(minutes):
            if minutes is None:
                return ""
            day = int(minutes // 1440) + 1
            hour = int((minutes % 1440) // 60)
            minute = int(minutes % 60)
            return f"Day {day}, {hour:02d}:{minute:02d}"

        def get_time_window_info(task):
            earliest = getattr(task, 'earliest_time', None)
            latest = getattr(task, 'latest_time', None)
            if earliest is None and latest is None:
                return "[No specific time window]"
            
            start_str = format_absolute_minutes(earliest) if earliest is not None else "Start"
            end_str = format_absolute_minutes(latest) if latest is not None else "End"
            return f"[{start_str} -> {end_str}]"

        def get_load_change(task):
            weight_change = getattr(task, 'demand', 0.0)
            volume_change = getattr(task, 'volume', 0.0)
            return weight_change, volume_change

        try:
            from second_level import is_feasible
            feasible, reason = is_feasible(route, debug_feasibility=True, return_reason=True)
        except ImportError:
            feasible, reason = True, "(Feasibility check unavailable)"

        print(f"   {vehicle_id}:")
        print(f"      Feasible: {feasible}")
        if not feasible:
            print(f"      Reason: {reason}")

        total_duration_minutes = 0
        if len(route.tasks) > 1:
            try:
                from route_provider import calculate_travel_time_between_tasks
                for i in range(1, len(route.tasks)):
                    total_duration_minutes += calculate_travel_time_between_tasks(route.tasks[i-1], route.tasks[i], route.vehicle)
            except ImportError:
                total_duration_minutes = len(route.tasks) * 60

        duration_formatted = format_duration_detailed(total_duration_minutes)
        days = int(total_duration_minutes / 1440)
        hos_warning = " (would violate HoS if attempted without proper rests)" if total_duration_minutes > 11 * 60 else ""

        print(f"      DURATION: Route duration: {days} day(s) ({duration_formatted}){hos_warning}")
        print(f"      TASKS: Task sequence ({len(route.tasks)} tasks) - Real-time monitoring:")

        current_weight, current_volume, cumulative_time = 0, 0, 0

        for i, task in enumerate(route.tasks, 1):
            travel_time = 0
            if i == 1:
                # First task has no travel time from a previous task, but we set a 5 min start delay.
                cumulative_time = 5
            else:
                try:
                    from route_provider import calculate_travel_time_between_tasks
                    travel_time = calculate_travel_time_between_tasks(route.tasks[i-2], task, route.vehicle)
                except ImportError:
                    travel_time = 60 # Fallback
            
            # Arrival time is after travel
            arrival_time = cumulative_time + travel_time
            
            # Account for waiting if arriving before the time window opens
            wait_time = 0
            if task.earliest_time and arrival_time < task.earliest_time:
                wait_time = task.earliest_time - arrival_time
            
            # Service starts after travel and any waiting
            service_start_time = arrival_time + wait_time
            service_time = getattr(task, 'service_time', 0)
            
            # Update cumulative time to be when the current task is finished
            cumulative_time = service_start_time + service_time

            location = get_location_name(task)
            order_info = get_order_info(task)
            time_window = get_time_window_info(task)
            weight_change, volume_change = get_load_change(task)
            current_weight += weight_change
            current_volume += volume_change

            delta_str = f" (+{format_duration_detailed(travel_time)} travel"
            if wait_time > 0:
                delta_str += f", +{format_duration_detailed(wait_time)} wait"
            if service_time > 0:
                delta_str += f", +{format_duration_detailed(service_time)} service"
            delta_str += ")"

            print(f"          {i}. {location} ({order_info}) - Arrival: {format_duration_detailed(arrival_time)} - Cumulative: {format_duration_detailed(cumulative_time)}{delta_str} {time_window}")
            
            weight_sign = "+" if weight_change >= 0 else ""
            volume_sign = "+" if volume_change >= 0 else ""
            print(f"             Load: {weight_sign}{weight_change:.1f}kg, {volume_sign}{volume_change:.1f}m³ → Total: {current_weight:.1f}kg, {current_volume:.1f}m³")
    ```

### Task 2: Fix Time Window Parsing in `utils/scenario_creator.py`

*   **File to modify:** `utils/scenario_creator.py`
*   **Goal:** Update the time window parsing logic to correctly handle multi-day scenarios.

**Instructions:**

1.  **Replace the `parse_time_window` function** with this more robust version that can handle `datetime.time` objects directly from pandas.

    ```python
    def parse_time_window(time_input) -> Optional[float]:
        """
        Parse a time input (string, datetime.time) into minutes from start of day.
        """
        if pd.isna(time_input) or not time_input:
            return None

        if isinstance(time_input, datetime.time):
            return time_input.hour * 60 + time_input.minute

        time_str_clean = str(time_input).strip()
        
        # Try HH:MM:SS format first
        try:
            time_obj = datetime.strptime(time_str_clean, "%H:%M:%S").time()
            return time_obj.hour * 60 + time_obj.minute
        except ValueError:
            pass
        
        # Fall back to HH:MM format
        try:
            time_obj = datetime.strptime(time_str_clean, "%H:%M").time()
            return time_obj.hour * 60 + time_obj.minute
        except ValueError:
            logger.warning(f"Could not parse time: {time_input}")
            return None
    ```

2.  **In the `create_task_from_row` function, replace the time window parsing section.**

    **Find and replace this block:**
    ```python
        # Parse time windows - support new format (TIME_WINDOW_START/END) and legacy format (TIME WINDOW HOURS)
        earliest_time = None
        latest_time = None
        
        # New format: explicit start and end times
        time_start = safe_parse_value(row, 'TIME WINDOW START', None, str)
        time_end = safe_parse_value(row, 'TIME WINDOW END', None, str)
        
        if time_start:
            earliest_time = parse_time_window(time_start)
        if time_end:
            latest_time = parse_time_window(time_end)
        
        # Legacy format: time range string (e.g., "8-12" or "08:00-12:00")
        if earliest_time is None and latest_time is None:
            time_window_hours = safe_parse_value(row, 'TIME WINDOW HOURS', None, str)
            # ... (rest of legacy parsing)
    ```

    **With this new, corrected logic:**
    ```python
        # --- Time Window Parsing ---
        earliest_time = None
        latest_time = None

        # New format: Multi-day with hours
        earliest_day = safe_parse_value(row, 'EARLIEST DAY', 1, int)
        latest_day = safe_parse_value(row, 'LATEST DAY', earliest_day, int)
        time_start_input = safe_parse_value(row, 'TIME WINDOW START', '00:00', None) # Keep as object
        time_end_input = safe_parse_value(row, 'TIME WINDOW END', '23:59', None)   # Keep as object

        time_start_minutes = parse_time_window(time_start_input)
        time_end_minutes = parse_time_window(time_end_input)

        if time_start_minutes is not None:
            # Convert to absolute minutes from the beginning of the planning horizon (Day 1, 00:00)
            # Day 1 corresponds to an offset of 0 days.
            earliest_time = (earliest_day - 1) * 1440 + time_start_minutes

        if time_end_minutes is not None:
            # Day 1 corresponds to an offset of 0 days.
            latest_time = (latest_day - 1) * 1440 + time_end_minutes

        # Legacy format: time range string (e.g., "8-12" or "08:00-12:00")
        if earliest_time is None and latest_time is None:
            time_window_hours = safe_parse_value(row, 'TIME WINDOW HOURS', None, str)
            if time_window_hours and '-' in time_window_hours:
                try:
                    start_str, end_str = time_window_hours.split('-', 1)
                    if ':' in start_str:
                        earliest_time = parse_time_window(start_str.strip())
                        latest_time = parse_time_window(end_str.strip())
                    else:
                        start_hour = int(start_str.strip())
                        end_hour = int(end_str.strip())
                        earliest_time = start_hour * 60
                        latest_time = end_hour * 60
                except Exception as e:
                    logger.warning(f"Could not parse legacy time window '{time_window_hours}' for {company_name}: {e}")
    ```

### Task 3: Fix Order Creation in `utils/scenario_creator.py`

*   **File to modify:** `utils/scenario_creator.py`
*   **Goal:** Modify the `create_scenario_from_excel` function to group tasks by order ID and remove the flawed depot task generation.

**Instructions:**

In the file `utils/scenario_creator.py`, you will perform two actions:

1.  **Delete the `create_depot_bay_task` function entirely.** It is not needed and is the source of the errors.

2.  **Replace the `create_scenario_from_excel` function** with the corrected version below. This new version correctly groups tasks by order ID and no longer creates the erroneous depot tasks.

**Replace the entire `create_scenario_from_excel` function with this:**

```python
def create_scenario_from_excel(file_path: str) -> Tuple[List[Order], List[Vehicle], List[Driver]]:
    """
    Read an Excel file and create a VRP scenario for the EPDT algorithm.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        Tuple of (orders_list, vehicles_list, drivers_list)
        
    Raises:
        ScenarioCreationError: If scenario creation fails
    """
    logger.info(f"Reading scenario from {file_path}...")
    
    try:
        excel_file = pd.ExcelFile(file_path)
        orders_df = pd.read_excel(excel_file, sheet_name="CONSEGNE")
        vehicles_df = pd.read_excel(excel_file, sheet_name="VEICOLI")
        drivers_df = pd.read_excel(excel_file, sheet_name="DRIVERS") if "DRIVERS" in excel_file.sheet_names else None
    except Exception as e:
        raise ScenarioCreationError(f"Failed to read Excel file: {e}")

    geocode_cache = load_geocode_cache()
    
    vehicles = [v for v in [create_vehicle_from_row(row) for _, row in vehicles_df.iterrows()] if v]
    logger.info(f"Created {len(vehicles)} vehicles")

    if drivers_df is not None:
        drivers = [d for d in [create_driver_from_row(row) for _, row in drivers_df.iterrows()] if d]
        logger.info(f"Created {len(drivers)} drivers from DRIVERS sheet")
    else:
        drivers = [] # No default drivers

    orders = []
    logger.info("Creating orders and tasks...")

    order_id_col = 'ORDER' if 'ORDER' in orders_df.columns else 'ORDER_ID'
    if order_id_col not in orders_df.columns:
        raise ScenarioCreationError("Could not find order ID column ('ORDER' or 'ORDER_ID') in CONSEGNE sheet.")

    grouped_by_order = orders_df.groupby(order_id_col)
    
    for order_id, group in grouped_by_order:
        pickup_tasks = []
        delivery_tasks = []
        order_priority = 1
        is_urgent = False

        for _, row in group.iterrows():
            customer_task = create_task_from_row(row, geocode_cache)
            if customer_task:
                if customer_task.is_pickup():
                    pickup_tasks.append(customer_task)
                else:
                    delivery_tasks.append(customer_task)
                
                order_priority = max(order_priority, customer_task.priority)
                if customer_task.priority >= 3:
                    is_urgent = True

        if pickup_tasks or delivery_tasks:
            order = Order(
                id=str(order_id),
                pickup_tasks=pickup_tasks,
                delivery_tasks=delivery_tasks,
                priority=order_priority,
                is_urgent=is_urgent,
                is_mandatory=True
            )
            orders.append(order)

    logger.info(f"Created {len(orders)} orders from grouped tasks.")
    save_geocode_cache(geocode_cache)

    if not orders or not vehicles:
        raise ScenarioCreationError("No valid orders or vehicles were created")

    logger.info(f"Scenario creation complete: {len(orders)} orders, {len(vehicles)} vehicles, {len(drivers)} drivers")
    return orders, vehicles, drivers
```

### Task 4: Add Photon Geocoder Fallback

*   **File to modify:** `utils/scenario_creator.py`
*   **Goal:** Add a fallback to the Photon geocoder to improve address lookup success.

**Instructions:**

1.  **Add a new function `try_geocode_with_photon`** to `utils/scenario_creator.py`. Place it before the `get_coordinates_with_fallback` function.

    ```python
    def try_geocode_with_photon(address: str, cache: Dict[str, Dict[str, float]]) -> Optional[Tuple[float, float]]:
        """Try to geocode an address using the Photon API."""
        if address in cache:
            coords = cache[address]
            return (coords["lat"], coords["lon"])

        photon_url = "https://photon.komoot.io/api/"
        try:
            time.sleep(0.5) # Rate limiting
            response = requests.get(photon_url, params={'q': address, 'limit': 1}, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data and data.get('features'):
                # IMPORTANT: Photon returns [lon, lat]
                lon, lat = data['features'][0]['geometry']['coordinates']
                coords = {"lat": float(lat), "lon": float(lon)}
                cache[address] = coords
                logger.info(f"Success with Photon: {address} -> ({lat:.6f}, {lon:.6f})")
                return (lat, lon)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Photon API request failed for address '{address}': {e}")
        except (ValueError, KeyError, IndexError) as e:
            logger.warning(f"Could not parse Photon response for address '{address}': {e}")
        return None
    ```

2.  **Update the `get_coordinates_with_fallback` function** to include the Photon call.

    **Replace the existing `get_coordinates_with_fallback` function with this new version:**

    ```python
    def get_coordinates_with_fallback(address: str, cache: Dict[str, Dict[str, float]], row: pd.Series) -> Optional[Tuple[float, float]]:
        """
        Enhanced geocoding with multiple fallback strategies for maximum precision.
        """
        # Strategy 1: Try exact address with Nominatim
        coords = try_geocode_single_address(address, cache)
        if coords:
            logger.info(f"Geocoded successfully with Nominatim: {address} -> ({coords[0]:.6f}, {coords[1]:.6f})")
            return coords
        
        # Strategy 2: Try alternative address formats with Nominatim
        alternative_addresses = generate_alternative_addresses(row)
        for alt_address in alternative_addresses:
            if alt_address != address:
                coords = try_geocode_single_address(alt_address, cache)
                if coords:
                    logger.info(f"Success with improved format (Nominatim): {alt_address} -> ({coords[0]:.6f}, {coords[1]:.6f})")
                    return coords

        # Strategy 3: Try original full address with Photon as a fallback
        coords = try_geocode_with_photon(address, cache)
        if coords:
            return coords

        # Strategy 4: Conservative fallback (city-level) with Nominatim
        fallback_address = construct_fallback_address(row)
        if fallback_address and fallback_address != address:
            coords = try_geocode_single_address(fallback_address, cache)
            if coords:
                logger.warning(f"Fallback to city-level geocoding (Nominatim): {fallback_address} -> ({coords[0]:.6f}, {coords[1]:.6f})")
                return coords
        
        # Strategy 5: Last resort - postal code only with Nominatim
        postal_only = construct_postal_only_address(row)
        if postal_only:
            coords = try_geocode_single_address(postal_only, cache)
            if coords:
                logger.error(f"Last resort postal-only geocoding (Nominatim): {postal_only} -> ({coords[0]:.6f}, {coords[1]:.6f})")
                return coords
        
        return None
    ```

## 4. Debugging with Incremental Tests

To help debug the process and ensure each part works correctly before running the full comprehensive test, here are instructions for creating smaller, targeted tests.

### Task 5: Update the Test for Time Window Parsing

*   **File to modify:** `tests/test_time_windows.py`
*   **Goal:** Verify that the new time window logic correctly calculates absolute time in minutes.

**Instructions:**

1.  **Modify the test script `tests/test_time_windows.py`:**

    **Replace the entire contents of the file with this:**
    ```python
    import unittest
    import os
    import sys
    from pathlib import Path
    import pandas as pd
    from datetime import time

    # Add necessary paths
    current_dir = Path(__file__).parent
    heuristic_root = current_dir.parent
    utils_dir = heuristic_root / 'utils'
    sys.path.insert(0, str(heuristic_root))
    sys.path.insert(0, str(utils_dir))

    from scenario_creator import create_scenario_from_excel

    class TestTimeWindowParsing(unittest.TestCase):

        def setUp(self):
            """Create a sample Excel file for testing time windows."""
            self.test_excel_path = current_dir / 'sample_time_window_test.xlsx'
            
            consegne_data = {
                'ORDER_ID': ['TW_ORDER_1', 'TW_ORDER_2'],
                'COMPANY_NAME': ['Time Test 1', 'Time Test 2'],
                'STREET': ['Via Prova', 'Via Test'],
                'HOUSE NUMBER': [1, 2],
                'CITY': ['Asti', 'Asti'],
                'PROVINCE': ['AT', 'AT'],
                'POSTAL CODE': [14100, 14100],
                'COUNTRY': ['ITALY', 'ITALY'],
                'DELIVERY OR PICKUP': ['DELIVERY', 'DELIVERY'],
                'LOAD KG': [10, 10],
                'LOAD VOLUME M^3': [1, 1],
                'EARLIEST DAY': [2, 1],
                'LATEST DAY': [3, 1],
                'TIME WINDOW START': [time(9, 0), time(14, 30)],
                'TIME WINDOW END': [time(17, 0), time(18, 0)]
            }
            consegne_df = pd.DataFrame(consegne_data)

            veicoli_data = {'NUMBER PLATE': ['TW123CD']}
            veicoli_df = pd.DataFrame(veicoli_data)

            with pd.ExcelWriter(self.test_excel_path) as writer:
                consegne_df.to_excel(writer, sheet_name='CONSEGNE', index=False)
                veicoli_df.to_excel(writer, sheet_name='VEICOLI', index=False)

        def tearDown(self):
            if os.path.exists(self.test_excel_path):
                os.remove(self.test_excel_path)

        def test_multi_day_time_window_parsing(self):
            """Verify that multi-day time windows are parsed into absolute minutes."""
            orders, _, _ = create_scenario_from_excel(str(self.test_excel_path))

            self.assertEqual(len(orders), 2)

            # Test Order 1 (Day 2, 09:00 to Day 3, 17:00)
            order1 = next((o for o in orders if o.id == 'TW_ORDER_1'), None)
            self.assertIsNotNone(order1)
            task1 = order1.delivery_tasks[0]
            # Expected earliest: (2-1) * 1440 + 9*60 = 1440 + 540 = 1980
            self.assertEqual(task1.earliest_time, 1980)
            # Expected latest: (3-1) * 1440 + 17*60 = 2880 + 1020 = 3900
            self.assertEqual(task1.latest_time, 3900)

            # Test Order 2 (Day 1, 14:30 to Day 1, 18:00)
            order2 = next((o for o in orders if o.id == 'TW_ORDER_2'), None)
            self.assertIsNotNone(order2)
            task2 = order2.delivery_tasks[0]
            # Expected earliest: (1-1) * 1440 + 14*60 + 30 = 870
            self.assertEqual(task2.earliest_time, 870)
            # Expected latest: (1-1) * 1440 + 18*60 = 1080
            self.assertEqual(task2.latest_time, 1080)

    if __name__ == '__main__':
        unittest.main()
    ```

2.  **Run the test.**
    ```bash
    python tests/test_time_windows.py
    ```
    This test should pass after the changes to `utils/scenario_creator.py` are correctly implemented.

### Task 6: Display Arrival Lateness/Earliness

*   **File to modify:** `tests/comprehensive_integration_test.py`
*   **Goal:** Add details to the route breakdown to show if a task is serviced early, on time, or late relative to its time window.

**Instructions:**

In the `print_detailed_route_breakdown` function, add a new line to the output that shows the arrival status.

**Inside the `for` loop, after the `Load:` print statement, add this block:**

```python
            # Display arrival status relative to time window
            arrival_status = ""
            if task.earliest_time is not None and service_start_time < task.earliest_time:
                # This case should be handled by wait_time, but we check for logic errors
                arrival_status = f"(ERROR: Arrived {-int(task.earliest_time - service_start_time)}m early)"
            elif task.latest_time is not None and service_start_time > task.latest_time:
                arrival_status = f"(LATE by {format_duration_detailed(service_start_time - task.latest_time)})"
            elif wait_time > 0:
                arrival_status = f"(Arrived early, waited {format_duration_detailed(wait_time)})"
            else:
                arrival_status = "(On time)"
            print(f"             Status: {arrival_status}")
```