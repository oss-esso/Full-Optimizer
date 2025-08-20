# Technical Report: Transforming a Static VRP Solver into a Dynamic Real-Time Fleet Management System

**Author:** Gemini AI
**Date:** August 20, 2025
**Version:** 1.0

## Abstract

This report outlines a comprehensive strategy for transforming the existing static EPDT (Enhanced Pickup and Delivery with Time Windows) heuristic solver into a fully-fledged Dynamic Vehicle Routing Problem (DVRP) system. The goal is to create a real-time fleet management platform that can handle dynamic order assignments, track vehicles in real-time, and support multi-day planning, thereby significantly boosting operational efficiency and profitability for a trucking company. This document details the required architectural evolution, algorithmic adaptations, new components, and a phased implementation roadmap. We will leverage the strengths of the existing sophisticated heuristic while building the necessary infrastructure to support a dynamic operational environment. The proposed system will feature a live dashboard for operations managers, a real-time event-driven solver, and a vehicle simulation module to provide a complete, end-to-end solution.

---

## 1. Introduction

The current VRP solver is a powerful tool for static planning, capable of optimizing complex routes with various constraints, including time windows, driver Hours of Service (HoS), and vehicle capacities. It is built upon a sophisticated tabu search metaheuristic, featuring two levels of optimization (L1 for inter-route and L2 for intra-route), destroy-and-repair mechanisms, and detailed cost modeling. However, its static nature—where all orders and constraints must be known in advance—limits its applicability in a real-world logistics environment where conditions change continuously.

### 1.1. The Need for a Dynamic Approach

Modern trucking and delivery operations are inherently dynamic. New orders arrive throughout the day, customers change their requirements, traffic conditions fluctuate, and vehicles can experience unexpected delays or breakdowns. A static plan, no matter how optimal at the start of the day, quickly becomes obsolete. This leads to manual, often suboptimal, adjustments by dispatchers, resulting in decreased efficiency, higher operational costs, and reduced customer satisfaction.

The transition to a Dynamic Vehicle Routing Problem (DVRP) framework is essential to address these challenges. A DVRP system can:

*   **React to new orders in real-time**, intelligently assigning them to the most suitable vehicle on the fly.
*   **Adapt to unforeseen events** like traffic jams or vehicle issues by re-routing vehicles dynamically.
*   **Provide real-time visibility** into the entire fleet's operations.
*   **Improve asset utilization** by filling empty capacity on returning vehicles.
*   **Boost profits** by minimizing mileage, reducing idle time, and increasing the number of completed deliveries.

### 1.2. Objectives

The primary objective of this project is to evolve the current solver into a dynamic system with the following key capabilities:

1.  **Real-time Order Management:** A dashboard interface for dispatchers to input new orders as they arrive.
2.  **Dynamic Route Assignment:** An intelligent engine that assigns new orders to the best vehicle in real-time based on cost, location, and constraints.
3.  **Live Vehicle Tracking:** A simulation module to approximate real-time vehicle GPS data (position, speed) and status (driving, on break), which can be replaced by actual telematics data in the future.
4.  **Calendarization:** The ability to plan and schedule tasks for future days, providing a forward-looking operational view.

---

## 2. Conceptual Architecture of the Dynamic System

To achieve the desired dynamism, we must move from a monolithic, single-run solver to a continuously running, event-driven system. The proposed architecture is based on a set of interacting services that manage state, process events, and trigger optimizations.

![DVRP System Architecture](https://i.imgur.com/9Y5ZkE7.png)

### 2.1. Core Components

1.  **Data Ingestion Layer:** This is the entry point for all dynamic information.
    *   **Order API:** A RESTful API endpoint (e.g., `/api/orders/new`) for the dashboard to submit new customer orders.
    *   **Vehicle Telematics Feed:** A service that receives or simulates real-time GPS updates for each vehicle. Initially, this will be our **Vehicle Simulation Service**.

2.  **State Management (Database):** A robust database (e.g., PostgreSQL with PostGIS for geospatial queries) is the system's source of truth. It will store:
    *   **Static Data:** `Vehicles`, `Drivers`, `Depots`.
    *   **Dynamic Data:** `Orders` (with status: pending, assigned, completed), `Routes` (the planned sequence of tasks), `VehicleState` (real-time location, current load, HoS status, active route).

3.  **Event Bus (Message Broker):** A message broker (e.g., RabbitMQ, Kafka) will decouple the system components. Events like `NewOrderReceived`, `VehiclePositionUpdated`, or `TaskCompleted` will be published to the bus.

4.  **Dynamic Solver Engine:** This is the heart of the system. It subscribes to events and decides when and how to alter the current plans. It contains:
    *   **Insertion Heuristic Processor:** A fast processor for inserting new orders into existing routes.
    *   **Re-optimization Trigger:** A component that monitors solution quality and triggers a full re-optimization when necessary.
    *   **The Adapted EPDT Heuristic:** The existing L1/L2 heuristics, modified to work on a single route or a subset of routes rather than the entire problem from scratch.

5.  **Vehicle Simulation Service:** This service simulates the fleet's real-time behavior. For each vehicle with an active route, it will:
    *   Periodically (e.g., every 30 seconds) calculate the vehicle's expected position along the OSRM-defined route path.
    *   Update the vehicle's HoS status based on simulated driving time.
    *   Publish `VehiclePositionUpdated` events.
    *   Publish `TaskCompleted` events when a vehicle arrives at a customer location.

6.  **Frontend (Dashboard):** A web-based user interface for the operations team.
    *   **Technology:** React with a mapping library like Leaflet or Mapbox.
    *   **Communication:** WebSockets for receiving real-time updates from the backend.

---

## 3. Deep Dive: Algorithmic and Component-Level Changes

This section details the necessary modifications to the existing algorithms and the design of new components.

### 3.1. The Dynamic Solver Engine: From Static to Real-Time

The core challenge is adapting the batch-oriented EPDT heuristic into an engine that can react to events. The engine will operate in two primary modes: **fast insertion** and **full re-optimization**.

#### 3.1.1. Fast Insertion Heuristic

When a `NewOrderReceived` event occurs, the system must quickly decide which vehicle should serve it. This is not a full re-planning problem but an insertion problem.

**Algorithm:**

1.  **Receive New Order:** The engine is triggered with the details of a new order.
2.  **Identify Candidate Vehicles:** Identify a subset of vehicles that could potentially serve this order. Candidates are selected based on:
    *   Proximity of the vehicle's current location or upcoming tasks to the new order's pickup location.
    *   Sufficient remaining capacity (weight, volume, pallets).
    *   Feasible remaining HoS for the driver.
3.  **Evaluate Insertion Cost for Each Candidate:** For each candidate vehicle, we must find the best position to insert the new order's pickup and delivery tasks into its current route. This is where the existing **L2 Heuristic (`second_level.py`)** becomes invaluable.
    *   The `l2_heuristic(route, order)` function is perfectly suited for this. It already finds the best way to insert an order into a single route.
    *   We will iterate through each candidate vehicle's current route and call the L2 heuristic. The function `calculate_z2_score` will return the cost of the *new* proposed route.
    *   The insertion cost is the difference: `Z2_new - Z2_original`.
4.  **Select Best Insertion:** The insertion with the lowest non-negative cost increase is chosen. This is a "cheapest insertion" strategy.
5.  **Regret-k Insertion (Advanced):** To avoid myopic decisions, a "regret-k" strategy can be implemented. For each new order, we calculate the insertion costs for the *k* best vehicles. The "regret" is the cost difference between the second-best insertion and the best insertion (`cost_2nd_best - cost_best`). The system then prioritizes inserting the order with the highest regret, as this order has the most to lose if it doesn't get its best option.
6.  **Update Route:** Once the best vehicle is chosen, its route plan is updated in the database, and the new plan is sent to the driver's interface via the Dispatch Service.

#### 3.1.2. Re-optimization Strategies

Continuous simple insertions can lead to a gradual degradation of the overall solution quality. The system must periodically trigger a more comprehensive re-optimization.

**Triggers for Re-optimization:**

*   **Time-based:** Run a full optimization on all pending and active orders every N minutes (e.g., 30-60 minutes).
*   **Event-based:** A major disruption, such as a vehicle breakdown, might trigger a re-optimization for all affected orders.
*   **Quality-based:** If the average cost per delivery increases by a certain percentage (e.g., 15%) compared to the initial static plan, trigger a re-optimization.

**Re-optimization Algorithm:**

The existing **`destroy_and_repair_large_orders` function** provides an excellent foundation. We can adapt this into a more general re-optimization strategy:

1.  **Identify Scope:** Select a subset of routes to re-optimize. This could be all routes in a specific geographic area, or all routes that have been modified since the last optimization.
2.  **Destroy:** Instead of just removing large unassigned orders, the "destroy" phase will remove a percentage (e.g., 10-20%) of orders from the selected routes, prioritizing those on routes with high costs or constraint violations. These orders are moved to a temporary "unassigned" pool.
3.  **Repair:** Use the existing **L1 Heuristic (`first_level.py`)** to re-insert the unassigned orders. The `l1_heuristic` can be run with a tight iteration limit (`M1`, `M2` parameters) to ensure it completes quickly. This will re-shuffle orders between the selected vehicles to find a better local optimum.

### 3.2. Real-time Vehicle State and Simulation

The dynamic system's decisions depend on having an accurate, real-time understanding of each vehicle's state.

**Vehicle State Data Model:**

A new table, `VehicleState`, is required in the database:

*   `vehicle_id` (Primary Key, Foreign Key to `Vehicles`)
*   `last_seen_timestamp`
*   `current_lat`, `current_lon`
*   `current_speed_kmh`
*   `status` (Enum: `IDLE`, `DRIVING_TO_PICKUP`, `AT_PICKUP`, `DRIVING_TO_DELIVERY`, `AT_DELIVERY`, `ON_BREAK`, `OFF_DUTY`)
*   `current_weight_load`, `current_volume_load`, `current_pallet_load`
*   `active_route_id` (Foreign Key to `Routes`)
*   `current_task_id` (Foreign Key to `Tasks`)
*   `hos_state_json` (A JSON blob of the `DriverState` object from `epdt_data_structures.py`)

**Simulation Logic:**

The Vehicle Simulation Service will run a continuous loop:

1.  For each vehicle with an `active_route_id`:
2.  Fetch the vehicle's current state and its route plan (a sequence of tasks with locations).
3.  Determine the vehicle's current leg of the journey (e.g., driving from task A to task B).
4.  Use the **`route_provider.py`** to get the OSRM geometry (the polyline) for that leg.
5.  Calculate the vehicle's expected position along that polyline based on the time elapsed since it departed the last task and its average speed. The vehicle-specific speeds in `route_provider` will be crucial here.
6.  Update the `VehicleState` in the database with the new position.
7.  Publish a `VehiclePositionUpdated` event to the message bus, which the dashboard will consume to update the map.
8.  Check if the vehicle has arrived at its next task. If so, update its status (e.g., to `AT_PICKUP`), publish a `TaskCompleted` event, and begin simulating the service time.

### 3.3. Calendarization and Multi-Day Planning

The request for calendarization implies handling orders scheduled for future days. The existing data structures in `epdt_data_structures.py` already have a `day` attribute on the `Task` object, which is an excellent starting point.

**Workflow:**

1.  **Advance Planning:** At the end of each day (e.g., at 8 PM), the system can run a full, static optimization for all orders scheduled for the *next* day. This creates a high-quality baseline plan. The `l1_heuristic` is used here in its standard batch mode.
2.  **Storing Future Plans:** The resulting `Solution` object is stored in the database, associated with the future date.
3.  **Day of Execution:** On the morning of execution, the plan for that day is loaded. The routes are assigned to drivers, and the system switches into **dynamic mode**.
4.  **Handling Multi-Day Routes:** The `hos_simulation.py` and `second_level.py` already contain logic for multi-day routes, including calculating prospective costs (`_calculate_prospective_cost`) and handling mandatory daily rests. This logic will be critical. When a route spans multiple days, the state of the vehicle (location, load, driver HoS) at the end of Day 1 becomes the `initial_state` for that vehicle on Day 2. The `is_feasible` check in `second_level.py` correctly handles this `initial_state`.

---

## 4. Implementation Roadmap

A phased approach is recommended to manage complexity and deliver value incrementally.

**Phase 1: Core Infrastructure (The "Static-Live" System)**

*   **Goal:** Build the foundation for the dynamic system.
*   **Tasks:**
    1.  Set up the database with the new schema (`VehicleState`, etc.).
    2.  Create the basic dashboard UI with a map to display vehicle locations from the database.
    3.  Implement the Vehicle Simulation Service. At this stage, it will simply execute the static plan generated by the current solver, updating vehicle positions along the route.
    4.  Implement the Order API and a simple form on the dashboard to add new orders to the database (they will not be dispatched yet).
*   **Outcome:** A system that can execute and visualize a static plan in "real-time".

**Phase 2: Dynamic Insertion Heuristic**

*   **Goal:** Implement the ability to handle new orders dynamically.
*   **Tasks:**
    1.  Create the Dynamic Solver Engine service.
    2.  Implement the event-driven flow: The Order API publishes a `NewOrderReceived` event. The solver engine consumes this event.
    3.  Implement the fast insertion heuristic as described in section 3.1.1, using the existing L2 heuristic to evaluate insertion costs.
    4.  The solver, upon finding the best insertion, updates the `Route` in the database.
    5.  The Vehicle Simulation Service will automatically pick up the route change and adjust the vehicle's simulated path.
*   **Outcome:** A system that can dynamically accept and assign new orders to vehicles already on the road.

**Phase 3: Advanced Re-optimization and Dashboard Interactivity**

*   **Goal:** Improve solution quality over time and enhance user control.
*   **Tasks:**
    1.  Implement the re-optimization trigger and the `destroy_and_repair` logic for dynamic contexts.
    2.  Enhance the dashboard to show real-time vehicle status, HoS timers, and load capacities.
    3.  Use WebSockets to push all updates (vehicle positions, route changes, status updates) to the dashboard for a truly live experience.
    4.  Add manual override capabilities for the dispatcher (e.g., manually re-assigning an order).

**Phase 4: Calendarization and Reporting**

*   **Goal:** Enable forward-looking planning and business intelligence.
*   **Tasks:**
    1.  Implement the end-of-day batch process for planning the next day's routes.
    2.  Develop the calendar view on the dashboard to show future plans.
    3.  Create a reporting module to analyze historical performance, including metrics like on-time delivery rates, cost per delivery, and vehicle utilization.

---

## 5. Conclusion

Transforming the current static VRP solver into a dynamic, real-time fleet management system is a significant but achievable undertaking. By building upon the powerful existing heuristic components (`L1`, `L2`, `HoS Simulation`) and wrapping them in an event-driven architecture, we can create a highly effective and profitable system. The key is to shift the paradigm from a single, monolithic solve to a continuous process of fast, incremental updates, punctuated by periodic, intelligent re-optimization. This approach will provide the required agility to thrive in the demanding world of modern logistics, directly impacting the company's bottom line through reduced costs and improved customer service.

---

## 6. References

*   Pillac, V., Gendreau, M., Guéret, C., & Medeiros, A. L. (2013). A review of dynamic vehicle routing problems. *European Journal of Operational Research, 225*(1), 1-11.
*   Ritzinger, U., Puchinger, J., & Hartl, R. F. (2016). A survey on dynamic and stochastic vehicle routing problems. *International Journal of Production Research, 54*(1), 215-231.
*   Amazon Web Services. (n.d.). *What is an Event-Driven Architecture?* Retrieved from [https://aws.amazon.com/event-driven-architecture/](https://aws.amazon.com/event-driven-architecture/)
*   Campbell, A. M., & Savelsbergh, M. (2004). A decomposition approach for the inventory-routing problem. *Transportation Science, 38*(4), 488-502. (Note: While for IRP, the decomposition and dynamic update concepts are relevant).
