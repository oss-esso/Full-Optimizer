# Strategy for Driver-Vehicle Matching

This document outlines a strategy for implementing a driver-vehicle matching system within the EPDT algorithm framework. The goal is to accurately model real-world operational constraints where driver-specific factors, such as Hours of Service (HoS), are decoupled from the vehicles themselves.

## 1. The Core Challenge: Decoupling Driver and Vehicle

The current system associates Hours of Service (HoS) regulations directly with vehicles. In reality, these regulations apply to **drivers**. A vehicle can be used by multiple drivers, and a driver's working hours are independent of the specific vehicle they are operating. Furthermore, as specified, these regulations only apply to drivers operating **heavy trucks** (e.g., capacity > 3500kg).

This requires a more sophisticated model that can:
1.  Track the state of each driver independently.
2.  Assign drivers to vehicle routes in an optimal way.
3.  Handle team driving scenarios where two drivers operate a single vehicle.

## 2. Proposed Solution: A Two-Stage Approach

Solving the driver assignment and vehicle routing problems simultaneously is computationally very expensive. A more practical and effective approach is to decompose the problem into two sequential stages:

**Stage 1: Vehicle-Centric Route Optimization**
*   First, solve the Vehicle Routing Problem using the existing EPDT algorithm. The primary focus at this stage is on **vehicle constraints**: capacity, type, LIFO requirements, etc. The output is a set of globally efficient routes assigned to specific vehicles.

**Stage 2: Driver-Centric Assignment**
*   Second, with the optimal vehicle routes determined, assign the available drivers to these vehicle-route pairs. This is a classic **assignment problem** that can be solved efficiently.

## 3. Implementation Details

### 3.1. Data Loading from Excel

The initial list of drivers should be loaded from the `AUTISTI` sheet in the `src/furgoni.xlsx` file.

*   **Sheet Name:** `AUTISTI`
*   **Columns:**
    *   `NUMBER PLATE`: The default vehicle for the driver.
    *   `DRIVER`: The name of the driver.
    *   `LICENSE`: The driver's license type (`B` or `CE`).

The script should iterate through this sheet to create a list of `Driver` objects.

### 3.2. Data Model Enhancements (`epdt_data_structures.py`)

1.  **Create a `Driver` Class:** Introduce a new data class to represent a driver.

    ```python
    from dataclasses import dataclass, field
    from typing import Set

    @dataclass
    class DriverState:
        # All HoS related counters from the thesis
        drive_since_break: float = 0.0
        work_since_break: float = 0.0
        drive_today: float = 0.0
        # ... and so on

    @dataclass
    class Driver:
        id: str
        name: str
        license: str  # 'B' or 'CE'
        default_vehicle_id: str
        cost_per_hour: float
        home_depot_id: str
        qualifications: Set[str] = field(default_factory=set)
        hos_state: DriverState = field(default_factory=DriverState)
    ```

2.  **Update the `Route` Class:** Modify the `Route` class to include a `driver` attribute.

    ```python
    @dataclass
    class Route:
        vehicle: Vehicle
        driver: Optional[Driver] = None # Initially unassigned
        tasks: List[Task] = field(default_factory=list)
        # ... rest of the class
    ```

### 3.3. Decouple Hours of Service Logic (`second_level.py`)

The HoS simulation logic must be updated to operate on a `DriverState` object, not a `Vehicle`.

*   The `_simulate_hos_advanced` function should accept a `DriverState` object as an input.
*   When checking the feasibility of assigning a driver to a route, the simulation will use the `hos_state` from the specific `Driver` object being considered.

### 3.4. Driver Assignment Algorithm (Stage 2)

This stage can be modeled as a **minimum weight bipartite matching** problem, which can be solved using the Hungarian algorithm or a linear programming solver.

1.  **Construct a Cost Matrix:**
    *   **Rows:** Available Drivers
    *   **Columns:** Vehicle-Route pairs generated in Stage 1.

2.  **Calculate Costs:** The cost `C(d, r)` for assigning driver `d` to route `r` is calculated as follows:

    ```
    cost = 0

    # 1. HoS Feasibility and Cost (ONLY FOR HEAVY TRUCKS)
    if route.vehicle.vehicle_type == 'heavy':
        is_feasible, route_duration = _simulate_hos_advanced(route, d.hos_state)
        if not is_feasible:
            cost = float('inf') # Infeasible assignment
        else:
            cost += route_duration * d.cost_per_hour
    else:
        # For light trucks, HoS rules do not apply
        route_duration = calculate_route_duration_without_hos(route)
        cost += route_duration * d.cost_per_hour

    # 2. Other Penalties (if applicable)
    if d.home_depot_id != route.start_depot_id:
        cost += PENALTY_WRONG_DEPOT

    # 3. License Qualification Check
    if route.vehicle.vehicle_type == 'heavy' and d.license != 'CE':
        cost = float('inf') # Infeasible: CE license required for CAMION
    elif route.vehicle.vehicle_type == 'standard' and d.license != 'B':
        cost = float('inf') # Infeasible: B license required for FURGONE

    if not d.qualifications.issuperset(route.required_qualifications):
        cost = float('inf') # Infeasible assignment
    ```

3.  **Solve the Assignment Problem:** Use a solver to find the assignment that minimizes the total cost.

### 3.5. Handling Team Driving

For routes with two drivers, the `Route` class can be extended to have `driver1` and `driver2`. The `_simulate_hos_advanced` function would need to be modified to track the state of both drivers, alternating their driving and resting periods to maximize the vehicle's operational time.

## 4. Code Review and Improvements

After a thorough review of the driver-vehicle matching implementation, the following areas for improvement and potential errors have been identified:

### 4.1. `algo/driver_assignment.py`

*   **Hardcoded Values:** The `load_drivers_from_excel` function uses hardcoded values for `cost_per_hour` and `home_depot_id`. These should be read from the Excel file or a configuration file to make the system more flexible.
*   **Incomplete `Driver` Object:** The `load_drivers_from_excel` function does not populate the `qualifications` field of the `Driver` object from the Excel file. This could lead to incorrect assignments if a vehicle requires a specific qualification that is not the default for the driver's license type.
*   **Simplistic Cost Function:** The `calculate_assignment_cost` function uses a simple bonus for assigning a driver to their default vehicle. A more sophisticated cost function could consider factors like the driver's experience, performance, or the distance from their home depot to the route's start depot.
*   **Dummy Assignments:** The `assign_drivers_to_routes` function uses a high cost for dummy assignments when the number of drivers and routes are unequal. This could be improved by using a more sophisticated method for handling unbalanced assignment problems, such as the auction algorithm.

### 4.2. `algo/epdt_data_structures.py`

*   **`DriverState` Class:** The `DriverState` class is well-defined, but it could be enhanced to include more detailed tracking of HoS-related events, such as the start and end times of breaks and rests. This would be useful for auditing and reporting purposes.
*   **`Driver` Class:** The `Driver` class could be extended to include more attributes, such as the driver's preferred working hours, their performance history, or any special skills they may have.

### 4.3. `algo/second_level.py`

*   **HoS Simulation:** The `_simulate_hos_advanced` function is a complex piece of logic that could be simplified and made more modular. For example, the logic for handling breaks, daily rests, and weekly rests could be extracted into separate functions.
*   **Circular Imports:** The file has circular imports with `epdt_data_structures.py`. This should be refactored to avoid potential issues.

## 5. Academic Papers for Further Reading

Here are the titles of some relevant academic papers on the Vehicle Routing Problem with Crew Scheduling (VRCSP) and Driver Assignment:

*   *A Unified Framework for Deterministic Time Constrained Vehicle Routing and Crew Scheduling Problems*
*   *A branch-and-price algorithm for the multi-depot heterogeneous-fleet pickup and delivery problem with soft time windows*
*   *A Column Generation Algorithm for a Rich Vehicle-Routing Problem*
*   *Rich vehicle routing problems: from a taxonomy to a definition*
*   *Heuristics for the vehicle routing problem*
*   *A general heuristic for vehicle routing problems*
*   *A two-stage hybrid algorithm for pickup and delivery vehicle routing problems with time windows*

These papers provide a deep dive into the mathematical models and algorithms used to solve these complex logistical problems.

## 6. Enhanced Driver Assignment Implementation (`algo/driver_assignment_enhanced.py`)

This section details the enhanced implementation that addresses the feedback from the code review.

### 6.1. Enhanced Data Loading (`load_drivers_from_excel_enhanced`)

-   **Flexible Configuration:** A `DriverAssignmentConfig` class is introduced to manage parameters like `default_cost_per_hour`, `penalty_wrong_depot`, etc. This configuration can be loaded from a JSON file, removing hardcoded values.
-   **License Correction:** The function now reads the `VEICOLI` sheet to create a lookup of vehicle types. It uses this to correct invalid `C` licenses in the `AUTISTI` sheet:
    -   If a driver with a `C` license is assigned to a `CAMION`, their license is corrected to `CE`.
    -   If assigned to a `FURGONE`, it's corrected to `B`.
-   **Full Qualification Parsing:** The function now reads additional columns from the `AUTISTI` sheet (e.g., `QUALIFICATION_LOW_TEMP`, `QUALIFICATION_LOADER`) to populate the `qualifications` set for each driver.

### 6.2. Sophisticated Cost Function (`calculate_enhanced_assignment_cost`)

-   The cost function is now more comprehensive, considering:
    -   **HoS Feasibility:** For heavy trucks, it calls the `_simulate_hos_advanced` function to get an accurate duration and feasibility check.
    -   **License and Qualification:** It performs a strict check to ensure the driver has the required license and all necessary qualifications for the vehicle's capabilities.
    -   **Depot Penalty:** A configurable penalty is applied if the driver's home depot does not match the vehicle's depot.
    -   **Bonuses:** A configurable bonus is applied if the driver is assigned to their default vehicle.
    -   **Route Complexity:** A penalty is added for routes with a high number of tasks.

### 6.3. Improved Assignment Algorithm (`assign_drivers_to_routes_enhanced`)

-   **Unbalanced Problems:** The function now handles cases where the number of drivers and routes are unequal by using an augmented cost matrix. This is a more robust approach than simply adding a high cost for dummy assignments.
-   **Clearer Reporting:** The function provides more detailed logging, including the number of heavy/light routes, the number of available drivers by license type, and a summary of successful and failed assignments.

### 6.4. Pre-Assignment Qualification Enhancement

-   **`enhance_drivers_with_vehicle_capabilities` function:** To prevent assignment failures due to missing qualifications, a utility function was created.
-   **Logic:**
    1.  It collects all unique `capabilities` from the entire vehicle fleet.
    2.  It then iterates through the list of drivers and adds these capabilities to their `qualifications` set based on their license type (`CE` drivers get all capabilities, `B` drivers get all non-heavy vehicle capabilities).
-   **Note:** This is a pragmatic approach for the test environment to ensure a feasible assignment can always be found. In a production system, this would be replaced by loading actual driver certifications.