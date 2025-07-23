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

### 3.1. Data Model Enhancements (`epdt_data_structures.py`)

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

### 3.2. Decouple Hours of Service Logic (`second_level.py`)

The HoS simulation logic must be updated to operate on a `DriverState` object, not a `Vehicle`.

*   The `_simulate_hos_advanced` function should accept a `DriverState` object as an input.
*   When checking the feasibility of assigning a driver to a route, the simulation will use the `hos_state` from the specific `Driver` object being considered.

### 3.3. Driver Assignment Algorithm (Stage 2)

This stage can be modeled as a **minimum weight bipartite matching** problem, which can be solved using the Hungarian algorithm or a linear programming solver.

1.  **Construct a Cost Matrix:**
    *   **Rows:** Available Drivers
    *   **Columns:** Vehicle-Route pairs generated in Stage 1.

2.  **Calculate Costs:** The cost `C(d, r)` for assigning driver `d` to route `r` is calculated as follows:

    ```
    cost = 0

    # 1. HoS Feasibility and Cost (ONLY FOR HEAVY TRUCKS)
    if route.vehicle.weight_capacity > 3500:
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

    if not d.qualifications.issuperset(route.required_qualifications):
        cost = float('inf') # Infeasible assignment
    ```

3.  **Solve the Assignment Problem:** Use a solver to find the assignment that minimizes the total cost.

### 3.4. Handling Team Driving

For routes with two drivers, the `Route` class can be extended to have `driver1` and `driver2`. The `_simulate_hos_advanced` function would need to be modified to track the state of both drivers, alternating their driving and resting periods to maximize the vehicle's operational time.

## 4. Academic Papers for Further Reading

Here are the titles of some relevant academic papers on the Vehicle Routing Problem with Crew Scheduling (VRCSP) and Driver Assignment:

*   *A Unified Framework for Deterministic Time Constrained Vehicle Routing and Crew Scheduling Problems*
*   *A branch-and-price algorithm for the multi-depot heterogeneous-fleet pickup and delivery problem with soft time windows*
*   *A Column Generation Algorithm for a Rich Vehicle-Routing Problem*
*   *Rich vehicle routing problems: from a taxonomy to a definition*
*   *Heuristics for the vehicle routing problem*
*   *A general heuristic for vehicle routing problems*
*   *A two-stage hybrid algorithm for pickup and delivery vehicle routing problems with time windows*

These papers provide a deep dive into the mathematical models and algorithms used to solve these complex logistical problems.
