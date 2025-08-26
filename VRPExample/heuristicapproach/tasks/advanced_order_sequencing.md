
# Task: Advanced Intra-Order Task Sequencing

**Objective:** Replace the current "aggressive splitting" logic for oversized orders with a more intelligent, profit-driven sequencing mechanism for complex orders (those with multiple pickup and/or delivery tasks).

## 1. Background

The current system handles large orders that don't fit into a single vehicle by splitting them into smaller, independent sub-orders. This approach, while ensuring assignment, is suboptimal as it increases vehicle mileage and administrative overhead. It fails to consider that a single vehicle might be able to handle a large, multi-part order if the pickups and deliveries are sequenced intelligently to manage the vehicle's load dynamically.

This task is to implement a **Complex Order Management** system that finds the most profitable sequence of tasks within a single complex order, respecting all physical and logical constraints.

## 2. Target Module

The primary location for this new logic will be within the **L2 Heuristic** in `algo/second_level.py`. 

Currently, the `l2_heuristic` and its helper `_generate_initial_task_sequence` insert tasks in a simple, clustered (all pickups, then all deliveries) or interleaved fashion. This new logic will replace that for complex orders.

## 3. Implementation Steps

### Step 1: Identify Complex Orders

In `algo/second_level.py`, within the `l2_heuristic`, the first step is to identify if an order is "complex." 

- An order is considered complex if it has more than one pickup task or more than one delivery task.
  ```python
  # In l2_heuristic(route, order, ...)
  is_complex_order = len(order.get_pickups()) > 1 or len(order.get_deliveries()) > 1
  if is_complex_order:
      # Execute new advanced sequencing logic
      return find_best_sequence_for_complex_order(route, order)
  else:
      # Proceed with existing simple insertion logic
      # ...
  ```

### Step 2: Generate All Valid Task Permutations

For a complex order, the core challenge is to find the best permutation of its pickup (P) and delivery (D) tasks. A permutation is only valid if it respects precedence constraints (i.e., a delivery cannot occur before its corresponding pickup).

- **Input:** A list of task objects, e.g., `[P1, P2, D1, D2]`.
- **Output:** A list of all valid sequences, e.g., `[[P1, P2, D1, D2], [P1, D1, P2, D2], [P2, P1, D1, D2], ...]`. 
- **Constraint:** A sequence is invalid if a `D_i` appears before its corresponding `P_i`.

**Example:**
- **Order Tasks:** `P1` (Pickup A), `P2` (Pickup B), `D1` (Delivery A), `D2` (Delivery B)
- **Valid Sequence:** `[P1, P2, D1, D2]` (Pick up both, then deliver both)
- **Valid Sequence:** `[P1, D1, P2, D2]` (Pick up A, Deliver A, Pick up B, Deliver B)
- **Invalid Sequence:** `[D1, P1, P2, D2]` (Cannot deliver A before picking it up)

This can be implemented using a recursive or backtracking algorithm that explores valid next tasks at each step.

### Step 3: Evaluate Each Valid Permutation

For each valid task sequence generated in Step 2, the system must evaluate its feasibility and profitability when inserted into the current vehicle's route.

This involves iterating through all possible insertion points in the existing route and, for each point, performing a simulation:

1.  **Create a temporary route** with the task sequence inserted.
2.  **Perform a real-time load simulation:**
    - Initialize `current_load` based on the vehicle's load at the insertion point.
    - For each task `t` in the inserted sequence:
        - Update `current_load += t.demand` (demand is positive for pickups, negative for deliveries).
        - **Check Load Constraints:** If `current_load < 0` or `current_load > vehicle.capacity`, this sequence is **infeasible** at this insertion point. Discard and move to the next.
3.  **Calculate Profit:** If the sequence is feasible, calculate its profit.
    - **Cost:** The marginal cost added to the route, primarily from the extra travel time (`calculate_z2_score` can be used here).
    - **Revenue:** The revenue associated with the order.
    - **Profit = Revenue - Cost**

### Step 4: Select the Best Sequence

After evaluating all valid permutations at all possible insertion points, select the sequence and insertion point that results in the **highest positive profit** while maintaining overall route feasibility (`is_feasible` check).

If no sequence results in a feasible route, the insertion fails, and the function should return `None`.

### Step 5: Deprecate Aggressive Splitting

Once this new logic is implemented and tested, the old `split_large_order` function in `tests/comprehensive_integration_test.py` and any related calling logic should be removed. This new, more intelligent sequencing handles large, complex orders without needing to split them.

## 4. Acceptance Criteria

- The system correctly identifies complex orders (more than 2 tasks).
- For complex orders, the system generates and evaluates only valid (precedence-respecting) task permutations.
- The real-time load tracking correctly identifies and rejects sequences that violate vehicle capacity (both underflow and overflow).
- The final sequence chosen maximizes profit for the given order and vehicle route.
- The old order splitting mechanism is no longer called for handling large orders; this new logic serves as its replacement.
