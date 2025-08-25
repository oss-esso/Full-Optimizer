# Optimizer Improvement Plan for 100% Order Assignment

## 1. Objective

The primary goal is to modify the optimizer to achieve a 100% order assignment rate. This will be accomplished by addressing the root causes of assignment failures, primarily focusing on handling oversized orders and freeing up vehicle capacity intelligently.

## 2. Analysis of Current Performance

The main blocker to achieving 100% assignment is that several orders are too large (in terms of weight, volume, or pallet count) to fit into any single vehicle in the fleet. No amount of heuristic tuning can solve this fundamental capacity constraint.

Secondary issues that prevent a high assignment rate *before* the final "force assignment" step include:
*   **High Rate of HoS Violations:** Many routes are created and then invalidated by post-processing HoS checks.
*   **Strict Heuristics:** The insertion logic is too rigid and rejects solutions that might be slightly infeasible but still good.

## 3. Instructions for the Agent

To achieve 100% assignment, the following strategies must be implemented. The primary strategy is order splitting, which is critical. The supporting strategies will ensure the optimizer can handle the new, smaller orders efficiently and produce a high-quality solution.

### **Workflow for Testing**

Due to your limited context window, you must redirect the test output to a file for analysis.

1.  **Run the test:**
    ```shell
    python tests/comprehensive_integration_test.py > test_output.txt
    ```
2.  **Analyze the results:**
    Read the `test_output.txt` file to check the "Order Assignment Summary". The goal is to achieve a 100% assignment rate.

---

### **Primary Strategy: Enable and Enhance Order Splitting**

This is the most critical step to solve the problem of oversized orders.

1.  **Enable Order Splitting:** In `tests/comprehensive_integration_test.py`, find the `configure_algorithm_parameters` function and set the `'separate_orders'` parameter to `True`. This will activate the existing order splitting logic.
2.  **Enhance Splitting Logic (Optional but Recommended):** The current `split_large_order` function can be improved. A more advanced implementation could create non-uniform splits (e.g., one large part, one small part) to better match the specific available capacities of the vehicle fleet.

### **Supporting Strategies**

These are crucial for improving the overall quality and robustness of the solution.

#### **Step 1: Make Heuristics More Tolerant (Soft Constraints)**

The optimizer should be able to explore "good enough" solutions that might have minor violations.

1.  **Analyze Feasibility Checks:** Review the `is_feasible` and `is_feasible_for_insertion` functions in `algo/second_level.py`.
2.  **Implement Penalty System:** Modify these functions to treat capacity and time window constraints as "soft". Instead of returning `False` for a slight violation, the function should return `True` along with a penalty cost. This cost should then be incorporated into the `calculate_z2_score` function, guiding the search towards more feasible solutions without completely forbidding exploration.

#### **Step 2: Integrate HoS Checks into the Heuristics**

The optimizer must stop creating invalid routes.

1.  **Analyze HoS Logic:** Familiarize yourself with the `HoSEngine` in `algo/hos_simulation.py`.
2.  **Modify Heuristics:** In `algo/first_level.py` and `algo/second_level.py`, before a move (like an insertion or swap) is accepted, call the `HoSEngine` to check if the potential new route is compliant. Moves that lead to HoS violations should be heavily penalized or rejected.

#### **Step 3: Leverage Destroy and Repair**

This strategy is excellent for finding space for difficult-to-place orders (including newly split ones).

1.  **Ensure Activation:** Make sure the `enable_destroy_and_repair` parameter is set to `True` in the configuration.
2.  **Enhance Logic:** The `destroy_and_repair_large_orders` function in `algo/destroy_and_repair.py` could be improved to be smarter about which orders it removes from a route to make space. It should prioritize removing smaller, non-urgent orders that are easier to re-insert elsewhere.

#### **Step 4: Implement a Vehicle Reallocation Strategy**

This is an advanced strategy to handle cases where an order fails to be assigned because all capable vehicles are already occupied, but potentially with routes that smaller vehicles could handle.

1.  **Analyze `destroy_and_repair.py`:** Familiarize yourself with the logic in `destroy_and_repair_large_orders`. This new strategy will be an enhancement to it.
2.  **Implement the Reallocation Logic:**
    *   When a difficult order (`order_A`) cannot be placed, identify the set of vehicles that have the capacity for it (`capable_vehicles`).
    *   For each `vehicle_1` in `capable_vehicles` that is already assigned a route (`route_1`):
        *   Search for a different, smaller, or idle vehicle (`vehicle_2`) in the fleet that can feasibly handle `route_1`.
        *   If a suitable `vehicle_2` is found, create a new candidate solution where `route_1` is moved from `vehicle_1` to `vehicle_2`.
        *   With `vehicle_1` now free, attempt to insert `order_A` into it.
        *   Evaluate the total cost change of this complex move. If it's beneficial, this becomes a valid neighbor in the search.
    *   This logic can be integrated into the `destroy_and_repair` function as a more powerful "destroy" mechanism, where instead of just removing orders, you are reallocating an entire route.
