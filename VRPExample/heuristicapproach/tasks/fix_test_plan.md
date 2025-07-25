# Plan to Diagnose and Fix Heuristic Solver Issues

This document outlines a multi-step strategy to identify and resolve the critical bugs in the EPDT heuristic solver, addressing the issues of incorrect scores, poor assignment rates, and excessive runtime.

## Phase 1: Locate the Root Cause

1.  **Analyze Score Discrepancy:**
    *   **Action:** Read `algo/first_level.py` and `tests/run_scenario_test.py`.
    *   **Goal:** Compare the score calculation within the `l1_heuristic` main loop with the final score calculation in `print_solution_summary`. The discrepancy likely stems from one of them using an incorrect formula (e.g., double-counting penalties, using raw costs instead of net profit).

2.  **Investigate Initial Assignment Failures:**
    *   **Action:** Read `algo/first_level.py` (specifically the `best_insertion_initializer`) and `algo/second_level.py` (`is_feasible` function).
    *   **Goal:** The warnings "Could not assign order..." indicate that `is_feasible` is returning `False` for initial insertions. This is the first point of failure. Add logging to the `is_feasible` function to pinpoint which specific constraint (capacity, time window, HoS) is causing the rejection for the first few unassigned orders.

3.  **Review the `is_feasible` and HoS Logic:**
    *   **Action:** Read `algo/second_level.py` in detail.
    *   **Goal:** The Hours of Service (`_simulate_hos_advanced`) logic is highly complex and a likely source of bugs. A flaw here could incorrectly invalidate routes that should be feasible, leading to the mass assignment failures seen in the initializer. Scrutinize the state variables and the logic for breaks and rests.

## Phase 2: Fix the Issues

1.  **Correct the Scoring Function:**
    *   **Action:** Based on the findings from Phase 1, modify the score calculation in `algo/first_level.py` to be consistent with the final summary.
    *   **Hypothesis:** The main loop's score is likely not correctly accounting for revenues, treating all costs and penalties as negative, leading to the large negative value. The fix is to ensure it calculates a proper net profit (Revenues - Costs - Penalties).

2.  **Fix Feasibility and HoS Constraints:**
    *   **Action:** Based on the logging from the investigation, correct the faulty constraint check in `algo/second_level.py`.
    *   **Hypothesis:** The HoS simulation is likely too restrictive or contains a bug that doesn't correctly handle multi-day routes, causing it to fail valid routes. The fix will involve correcting the simulation logic to accurately reflect the rules.

3.  **Improve Initializer Strategy:**
    *   **Action:** Modify the `best_insertion_initializer` in `algo/first_level.py`.
    *   **Goal:** Instead of immediately rejecting an order, the initializer could be modified to try assigning it to a new, empty vehicle if all existing routes are infeasible. This would improve the initial assignment rate.

## Phase 3: Correct Task Sequencing Logic (Based on Thesis Chapter 5)

**Objective:** Enforce the strict "all pickups before all deliveries" precedence constraint for the entire route, as defined in the EPDT problem description (Chapter 3 and illustrated in Chapter 5 of the thesis).

*   **Problem:** The second-level heuristic is producing routes that incorrectly interleave pickup and delivery tasks. This indicates the feasibility check (`is_feasible`) does not correctly enforce the global route-level precedence constraint.
*   **Source of Requirement:** The thesis (Chapter 3, Section 3.3.4 and Figure 3.2) mandates that for a route to be feasible, all pickup operations for all assigned orders must be completed before any delivery operations for any assigned order can begin.
*   **Action:** Modify the `is_feasible` function in `algo/second_level.py` to correctly implement this constraint.
*   **Implementation Details:**
    1.  Introduce a boolean flag, `delivery_phase_started`, initialized to `False` at the beginning of the function.
    2.  Iterate through the route's tasks in their chronological sequence.
    3.  If a delivery task is encountered, set `delivery_phase_started = True`.
    4.  If a pickup task is encountered **after** `delivery_phase_started` has been set to `True`, the route is infeasible. The function must immediately return `False`.
    5.  This check should be performed before the more computationally expensive HoS simulations to fail invalid routes quickly.