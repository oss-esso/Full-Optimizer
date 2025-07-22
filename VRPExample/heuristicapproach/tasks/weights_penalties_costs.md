# Weights, Penalties, and Costs in the EPDT Algorithm

This document outlines the various weights, penalties, and costs used within the EPDT (Enhanced Parallel Diversified Tabu) algorithm, as implemented in the `algo` folder. The values are cross-referenced with Chapters 3 and 5 of the provided thesis (`tesi_definitiva_Nicola_Gastaldon.pdf`).

## 1. Objective Function (Z1 Score)

The primary objective function, `calculate_z1_score` in `first_level.py`, aims to maximize the overall profit of a solution. This is not a direct cost minimization, but rather a maximization of a score that balances revenue and various costs and penalties.

The components of the Z1 score are:

*   **`G(s)` (Net Profit):** This is the base of the score, calculated as the sum of revenues from all assigned orders minus the operational costs of the routes.
    *   **`Po` (Revenue):** The revenue for completing order `o`. This is a positive value that increases the score.
    *   **`C(r)` (Route Cost):** The operational cost of a route `r`. This is a negative value that decreases the score. It is composed of:
        *   **`CF` (Fixed Cost):** A fixed cost for deploying a vehicle.
        *   **`Cv` (Variable Cost):** A cost per distance unit, which depends on factors like fuel consumption and maintenance.
    *   **`A(r)` (Prospective Cost):** The estimated cost of a route's "tomorrow" tasks. This is a penalty for routes that have tasks scheduled for the next day, representing the uncertainty and future cost of those tasks.
    *   **`W(r)` (Time Window Violation Cost):** A penalty for violating soft time windows.

*   **`Lo` (Penalty for Missed Urgent Orders):** A penalty applied for each urgent order that is not assigned to a route.

*   **`wk` (Weight for Soft Constraints):** These are weights applied to various soft constraint violations, allowing for fine-tuning of the objective function based on operational priorities. The specific soft constraints and their corresponding indicators are:
    *   **`ID(s)`:** Penalty for not assigning preferred subsets of orders to the same vehicle.
    *   **`IE(s)`:** Penalty for having fewer vehicles at the depot than the preferred minimum.
    *   **`IF(r)`:** Penalty for not assigning an order to its preferred vehicle.
    *   **`IH(r)`:** Penalty for a route not ending at its preferred end-of-day position.
    *   **`IJ(r)`:** Penalty for a route exceeding its preferred maximum duration.

*   **`M` (Penalty for Missing Mandatory Orders):** A large penalty (`B(s)`) is applied for any mandatory order that is not assigned to a route, making such solutions highly undesirable.

## 2. Second-Level Heuristic (Z2 Score)

The second-level heuristic, `calculate_z2_score` in `second_level.py`, evaluates the quality of a single route (`r`). This score is a component of the main Z1 score and is composed of:

*   **`C(r)` (Route Cost):** As defined above.
*   **`A(r)` (Prospective Cost):** As defined above.
*   **`W(r)` (Time Window Violation Cost):** As defined above.
*   **Penalties for Soft Constraints:** The route-specific components of the soft constraint penalties from the Z1 score (`IF(r)`, `IH(r)`, `IJ(r)`).

## 3. QUBO Formulation

The QUBO formulation in `qubo_formulation.py` uses penalties to enforce constraints in a quadratic unconstrained binary optimization model.

*   **`P_task` (Task Servicing Penalty):** A large positive coefficient to penalize violations of the task-servicing constraint (i.e., a task not being serviced by exactly one route or rejected).
*   **`P_fleet` (Fleet Size Penalty):** A large positive coefficient to penalize violations of the fleet size constraint.

## 4. Comparison with Thesis

The weights, penalties, and costs identified in the code are consistent with the problem definition in **Chapter 3** of the thesis. The objective function described in section 3.4 of the thesis, which aims to maximize the net profit, is directly reflected in the `calculate_z1_score` function. The hard and soft constraints listed in the thesis are all accounted for in the Z1 and Z2 scoring functions, either as hard feasibility checks or as weighted penalties.

The implementation of the two-level heuristic in **Chapter 5** of the thesis is also consistent with the code. The first-level heuristic explores order-to-vehicle assignments, and the second-level heuristic evaluates the quality of the resulting routes, which is exactly what is implemented in `first_level.py` and `second_level.py`. The score function described in section 5.2 of the thesis is a direct match for the `calculate_z1_score` function in the code.
