
# Analysis of Persisting Constraint Violations

## 1. Objective

The EPDT optimizer has achieved a high assignment rate (only 3 unassigned orders), but the generated solution still contains critical violations related to vehicle capabilities and physical constraints (pallets), as detailed in `tests/violations_intermediate.txt`. 

This document provides a root cause analysis of these violations and a clear plan to fix them, ensuring the final solution is not only complete but also 100% valid.

## 2. Executive Summary

The root cause of the persisting violations is a fundamental disconnect between the optimizer's objective function and the complete set of real-world constraints. The core search algorithm (`l1_heuristic`) makes decisions based on a cost score (`calculate_z2_score`) that **does not account for vehicle capabilities or pallet capacity**. 

Essentially, the optimizer is never told that it's "bad" to put an order requiring a loader onto a vehicle that doesn't have one, or to exceed a vehicle's pallet limit. These checks are performed too late in the process (e.g., in reporting or driver assignment) and are not part of the core route-building logic.

The solution is to integrate these critical checks directly into the main feasibility (`is_feasible`) and scoring (`calculate_z2_score`) functions in `algo/second_level.py`. By making the optimizer aware of these constraints during its decision-making process, it will naturally learn to produce valid, compliant routes.

## 3. Analysis of Capability Violations

- **Symptom:** The violation report shows 10 capability violations (e.g., "Missing LOADER capability", "Missing LOW_TEMP capability"). This means orders with specific requirements are being assigned to vehicles that cannot service them.

- **Root Cause:** The function responsible for checking route feasibility, `is_feasible()` in `algo/second_level.py`, **does not perform any capability checks**. It validates capacity, time windows, and HoS, but is completely blind to whether the vehicle has the required equipment (loader, low temp, etc.) for the orders placed on its route.

- **Evidence:** The `is_feasible` function in `algo/second_level.py` lacks any logic that compares `task.required_capabilities` with `vehicle.capabilities`.

- **Recommendation:** The capability check must be a **hard constraint** integrated directly into the main feasibility function.

- **Action Plan:** Modify `algo/second_level.py` to include a capability check within `is_feasible()`.

  **Proposed Code for `algo/second_level.py`:**
  ```python
  # Inside the is_feasible function in algo/second_level.py

  # ... after existing checks ...

  # H10: Vehicle Capability Check (NEW)
  # This is a new hard constraint to ensure the vehicle can service all orders.
  all_required_capabilities = set()
  for task in route.tasks:
      if hasattr(task, 'requires_loader') and task.requires_loader:
          all_required_capabilities.add('LOADER')
      if hasattr(task, 'requires_low_temp') and task.requires_low_temp:
          all_required_capabilities.add('LOW_TEMP')
      # Add any other capabilities from the task
      if hasattr(task, 'required_capabilities') and task.required_capabilities:
          all_required_capabilities.update(task.required_capabilities)

  vehicle_capabilities = route.vehicle.capabilities if hasattr(route.vehicle, 'capabilities') else set()
  
  if not vehicle_capabilities.issuperset(all_required_capabilities):
      missing_caps = all_required_capabilities - vehicle_capabilities
      reason = f"Vehicle {route.vehicle.id} is missing capabilities: {', '.join(missing_caps)}"
      if debug_feasibility:
          print(f"            DEBUG FEASIBILITY: {reason}")
      if return_reason:
          return False, reason
      return False

  # If all checks pass
  if return_reason:
      return True, "Feasible"
  return True
  ```

## 4. Analysis of Pallet Constraint Violations

- **Symptom:** The violation report shows 10 "Pallet Constraint Violations," indicating that the total number of pallets on a vehicle exceeds its physical capacity.

- **Root Cause:** Similar to the capability issue, the pallet constraint is not being properly enforced during the optimization process. While the `is_feasible` function *does* contain logic to check for pallet capacity, the optimizer's scoring function, `calculate_z2_score`, **does not penalize pallet violations**. Therefore, if a route that violates the pallet constraint has a slightly better travel time, the optimizer will choose it because it is not being told that the pallet violation is costly.

- **Recommendation:** Pallet capacity is a hard physical limit and violations must be treated as infinitely costly to the optimizer.

- **Action Plan:** Add a high-penalty term for pallet violations to the `calculate_z2_score` function in `algo/second_level.py`. This ensures that any route exceeding pallet capacity is given a very poor score, effectively removing it from consideration.

  **Proposed Code for `algo/second_level.py`:**
  ```python
  # Inside the calculate_z2_score function in algo/second_level.py

  # ... after existing cost calculations ...

  # --- NEW: Pallet Violation Penalty ---
  pallet_violation_penalty = 0.0
  max_pallets = route.vehicle.pallet_capacity
  if max_pallets is not None:
      current_pallets = 0
      for task in sorted_tasks:
          # Note: task.pallets is negative for deliveries
          current_pallets += getattr(task, 'pallets', 0)
          if current_pallets > max_pallets:
              # Apply a very high penalty for any violation
              pallet_violation_penalty += 1000000  # Large penalty
  
  # Add the new penalty to the total cost
  total_cost = (travel_cost + time_window_penalty + prospective_cost + 
                driver_cost + vehicle_assignment_penalty + end_position_penalty +
                soft_time_window_penalty + weight_violation_penalty + hos_violation_penalty +
                pallet_violation_penalty) # Add the new penalty here

  # Cache the score
  route._z2_score = total_cost
  return total_cost
  ```

## 5. Conclusion

The existing constraint violations are not due to complex edge cases but rather to missing checks and penalties in the core logic that guides the optimization. The solver is achieving a high assignment rate by creating routes that are physically or operationally impossible.

By implementing the two fixes proposed above:
1.  **Adding a hard capability check** to `is_feasible()`.
2.  **Adding a high penalty for pallet violations** to `calculate_z2_score()`.

The optimizer will be forced to respect these fundamental constraints, leading to solutions that are not only complete but also valid and executable in the real world.
