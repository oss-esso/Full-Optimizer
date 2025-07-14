# TODO List for Advanced EPDT Implementation

This document outlines the necessary steps to integrate the advanced, realistic constraints described in Chapter 3 of the thesis into the EPDT algorithm.

## 1. Data Model Enhancements

**Objective:** Update the core data structures to support multi-day planning and advanced constraints.

- [ ] **`Task` Data Structure:**
    - **Action:** Add a `day` attribute to the `Task` class (e.g., in `epdt_data_structures.py`). This will be an integer representing the day relative to the planning horizon (-1 for yesterday, 0 for today, 1 for tomorrow).
    - **Reason:** This is essential for the route simulation logic to correctly sequence tasks chronologically.

- [ ] **`Vehicle` Data Structure:**
    - **Action:** Add a `lifo_required` boolean attribute to the `Vehicle` class.
    - **Reason:** To identify vehicles that do not have side doors and must follow a Last-In, First-Out loading policy.
    - **Action:** Add an `initial_state` attribute to the `Vehicle` class. This should store the vehicle's position and any pending tasks from the previous day at the start of the simulation.
    - **Reason:** To handle open routes that continue from the previous day's state.

- [ ] **`Order` Data Structure:**
    - **Action:** Add a `priority` attribute (e.g., 'mandatory', 'urgent', 'normal').
    - **Reason:** To allow the scoring functions to apply the correct penalties for unassigned orders.

## 2. Second-Level Heuristic (Intra-Route) Modifications (`algo/second_level.py`)

**Objective:** Enhance the route feasibility and cost calculation functions to handle the advanced constraints.

- [ ] **Implement Multi-Day Route Simulation:**
    - **Action:** Modify the chronological simulation logic within `is_feasible` and `calculate_z2_score`.
    - **Details:**
        1.  The simulation must start from the vehicle's `initial_state`.
        2.  It must process tasks in strict chronological order: yesterday's tasks, then today's, then tomorrow's.
        3.  The Hours-of-Service (`_check_hos`) simulation must be initialized with the driver's state from the end of the previous day.

- [ ] **Implement LIFO Loading Constraint:**
    - **Action:** Add a new check within the `is_feasible` function.
    - **Logic:**
        1.  If `route.vehicle.lifo_required` is `True`, simulate the loading process.
        2.  Use a stack (`cargo_stack`) to track the `order_id` of loaded items.
        3.  When a pickup task is encountered, push its `order_id` onto the stack.
        4.  When a delivery task is encountered, check if its `order_id` matches the top of the stack. If not, the route is infeasible. If it matches, pop from the stack.
        5.  The route is only feasible if the stack is empty at the end.

- [ ] **Implement Prospective Cost Calculation:**
    - **Action:** Enhance the `calculate_z2_score` function.
    - **Logic:** The `A(r)` (prospective cost) component should be calculated for routes that have "tomorrow" tasks. This can be estimated as the travel time/distance from the location of the last "today" task to the locations of all "tomorrow" tasks, respecting their internal sequence.

## 3. First-Level Heuristic (Inter-Route) Modifications (`algo/first_level.py`)

**Objective:** Ensure the main search algorithm correctly uses the enhanced second-level logic.

- [ ] **Update `calculate_z1_score`:**
    - **Action:** Modify the main scoring function to correctly penalize unassigned orders based on their new `priority` attribute.
    - **Logic:**
        - Apply a very high penalty for unassigned "mandatory" orders.
        - Apply the `Lo` penalty for unassigned "urgent" orders.
        - Apply no penalty for unassigned "normal" orders.

- [ ] **Handle Open Routes in Initializers:**
    - **Action:** Modify `best_insertion_initializer` and `round_robin_insertion_with_priority_initializer`.
    - **Logic:** The initial solution construction must respect the initial state of vehicles. Any pending "yesterday" tasks must be considered fixed and part of the initial routes before any new orders are inserted.

## 4. Test Runner (`run_scenario_test.py`)

**Objective:** Create a test script to validate the implementation of the advanced features.

- [ ] **Create `run_scenario_test.py`:**
    - **Action:** Develop a new Python script dedicated to running a full scenario test.
    - **Steps:**
        1.  **Load Data:** Use `create_furgoni_scenario` from `src/moda_scenarios.py` to generate the `VRPInstance`.
        2.  **Adapt Data:** Use the `data_adapter.py` module to convert the `VRPInstance` into the `Order` and `Vehicle` lists required by the EPDT algorithm.
        3.  **Configure:** Set up the `params` dictionary to control the heuristic (e.g., iterations, tabu tenure, enabling/disabling enhancements).
        4.  **Execute:** Call `l1_heuristic` with the prepared data and parameters.
        5.  **Report:** Create a `print_solution_summary` function to display the results, including the final score, vehicle routes, and a list of unassigned orders.

---
### Findings from Chapters 3 and 4 of "tesi_definitiva_Nicola_Gastaldon.pdf"

**Problem Description (Chapter 3):**
The thesis focuses on the Express Pickup and Delivery in freight Trucking problem (EPDT), a Multi-Attribute Vehicle Routing Problem (MAVRP) inspired by Trans-Cel's scenario. This problem combines express courier requirements (urban contexts) with vehicle/route characteristics of medium-to-long haul trips. Key attributes include:
- **Heterogeneous Fleet:** Vehicles have different capacities (weight, volume) and loading tools.
- **Multi-Pickup/Multi-Delivery Orders:** Orders can involve multiple pickups and deliveries, with all pickups for an order needing to precede all deliveries by the same vehicle.
- **Multi-Day Planning:** Tasks can span across "yesterday," "today," and "tomorrow" relative to the planning horizon.
- **Time Windows:** Both hard and soft time windows are considered for tasks, with penalties for soft time window violations.
- **Driver Regulations:** European Hours of Service Regulations (max drive/work time, mandatory breaks) are incorporated.
- **Objective:** Maximize net profit, which is revenue from satisfied orders minus costs (route costs, prospective tomorrow route costs, missed urgent orders, soft time window violations).
- **Order Priority:** Orders can be mandatory (must be fulfilled), urgent (penalty if not fulfilled), or normal (no penalty if rejected).
- **Initial State:** Vehicles have an initial state including position and pending tasks from the previous day.

**Solving Strategy (Chapter 4):**
A meta-heuristic algorithm is proposed to solve EPDT, designed for fast execution and integration into Chainment's algorithmic engine. The solution method is a two-level heuristic:
- **First Level:** A Tabu Search algorithm hybridized with Variable Neighborhood Descent is used to explore order-to-vehicle assignments.
- **Second Level:** A Local Search algorithm determines the sequence of customers within each vehicle's route and evaluates the routes.
- **Efficiency Enhancements:** The algorithm incorporates granular exploration, fast solution evaluation procedures, and parallel implementations.
- **Mathematical Formulation:** EPDT can be formulated as a Mixed-Integer Linear Programming (MILP) problem.
- **Optimality Bounds:** A Column Generation algorithm is used to provide optimality bounds by solving the continuous relaxation of the MILP. The Pricing Problem within this framework is an Elementary Shortest Path Problem with Resource Constraints (ESPPRC), solved using a label correcting algorithm adapted for EPDT's specific multi-pickup/multi-delivery attributes.
- **Dynamic and Stochastic Settings:** The thesis also explores data-driven approaches (e.g., clustering, accessibility measures) to handle dynamic requests and uncertainty in the problem.
- **Integration:** The algorithm is integrated into the Chainment platform, a cloud-based decision support system for freight transportation.

---
### Findings from Chapters 6, 7, and 8 of "tesi_definitiva_Nicola_Gastaldon.pdf"

**Chapter 6: Bounding through a Column Generation Algorithm**
- **Purpose:** To assess the performance of the proposed EPDT algorithm by providing optimality bounds.
- **Method:** Uses an Integer Programming model based on set covering formulation.
- **Components:**
    - **Master Problem (MP):** A set covering model that accounts for open routes.
    - **Pricing Problem (PP):** An Elementary Shortest Path Problem with Resource Constraints (ESPPRC).
- **Solution:** The ESPPRC is solved using a label correcting algorithm, adapted for multi-pickup and multi-delivery attributes.
- **Contribution:** Development of a specific label correcting algorithm for ESPPRC that handles EPDT attributes.
- **Outcome:** Provides a bound to optimal solutions of EPDT, used to assess the effectiveness of the Tabu Variable Neighborhood Descent.

**Chapter 7: Toward new Data-Driven approaches for Dynamic and Stochastic VRPs**
- **Context:** EPDT problem in dynamic and stochastic settings, where requests are issued during the planning horizon and stochastic components affect the problem.
- **Goal:** Exploit historical data to improve decision-making in dynamic and stochastic VRPs.
- **Approaches:**
    - **Representative Orders:** Introduction of artificial orders (centroids of clusters from historical data) to aggregate space-time information on demand, guiding the optimization algorithm towards more stable solutions.
        - Clustering techniques (K-Means based) are used to create these representative orders.
        - Reliability factor (linear combination of sparsity and frequency) is introduced to assess the quality of clusters.
    - **Accessibility Approach:** Incorporates the concept of accessibility (how easy it is to reach future orders from a given space-time point) into the objective function.
        - A new component is added to the second-level score function to account for cumulative accessibility.
- **Simplified Problem Analysis:** Analytical insights into waiting times for a simplified version of the problem (single task, single vehicle, single random order) suggest that the optimal waiting time can be determined by minimizing the average arrival time.

**Chapter 8: Integration in a decision support system**
- **Platform:** Chainment, a cloud-based platform developed by Trans-Cel, integrates various management tools through a data-sharing system.
- **Purpose:** To aid operations managers in difficult decisional tasks related to vehicle routing and scheduling.
- **Modules:**
    - **Orders Portal:** Supports inserting and managing orders, collects data, and provides views for carriers and customers. Includes an interactive negotiation system and a profit estimator tool.
    - **Driver App:** Handles vehicle tracking, provides daily routes to drivers, and transmits feedback (operations completion, delays, maintenance issues) to the central system.
    - **Planning Module:** Determines daily operations assignments to vehicles and routes. Allows for user interaction (drag-and-drop interface, overriding suggestions) and re-optimization based on real-time data.
    - **Demand Forecast Tool:** Analyzes past demand and predicts future orders, using historical data to identify likely future requests.
    - **Algorithmic Engine:** The core of Chainment, implementing the EPDT routing optimizer (from Chapter 5) and predictive models.
- **Integration Aspects:**
    - The routing optimizer adapts to platform requirements (initial partial/complete plans, infeasible plans, user modifications, real-time data).
    - Predictive models estimate service time (based on load weight/volume) and price ranges for orders (using machine learning on historical data).
    - Uses third-party APIs like TomTom Telematics and Google Maps for real-time data and mapping.
    - Implemented using C++ for the core optimization engine and Python for high-level functionalities and prediction tools.
