## 7. QUBO Formulation for Quantum Annealing ✅ **COMPLETED**

**Objective:** To explore solving the EPDT problem on quantum annealing hardware, the MILP formulation must be converted into a Quadratic Unconstrained Binary Optimization (QUBO) model.

### 7.1. Overview ✅
- [x] ✅ **IMPLEMENTED:** The conversion process involves taking the constraints from the MILP and representing them as quadratic penalty terms in the objective function. This transforms the problem into a single polynomial to be minimized.
- [x] ✅ **IMPLEMENTED:** The recommended starting point is the **set partitioning** version of the problem, where each task must be served by **exactly one** route.

### 7.2. QUBO Objective Function ✅
- [x] ✅ **IMPLEMENTED:** The total QUBO objective function, or Hamiltonian (`H`), is the sum of the original MILP objective and the new penalty terms.
- [x] ✅ **IMPLEMENTED:** **`H = H_objective + H_penalty_tasks + H_penalty_fleet`**

### 7.3. Components of the QUBO Formulation ✅
- [x] ✅ **IMPLEMENTED:** **Objective Term (`H_objective`):**
    - **Action:** Directly translate the MILP objective function.
    - **Formula:** `H_objective = Σ_v Σ_r (cr * yr) + Σ_o (qo * xo)`
    - **Note:** `yr` and `xo` are the binary variables representing route selection and order rejection, respectively.
    - **File:** `algo/qubo_formulation.py` - `_formulate_objective_term()`

- [x] ✅ **IMPLEMENTED:** **Task Servicing Penalty (`H_penalty_tasks`):**
    - **Action:** For each task `i`, add a quadratic penalty that is minimized (equals zero) only when the task is serviced by exactly one route or is rejected.
    - **Formula:** `H_penalty_tasks = Σ_i P_task * ( (Σ_r air * yr + xo(i)) - 1 )^2`
    - **`P_task`:** A large positive coefficient to penalize violations of the task-servicing constraint.
    - **File:** `algo/qubo_formulation.py` - `_formulate_task_penalties()`

- [x] ✅ **IMPLEMENTED:** **Fleet Size Penalty (`H_penalty_fleet`):**
    - **Action:** Convert the fleet size inequality (`Σ_r yr <= |V|`) into an equality using binary slack variables, and then penalize its violation.
    - **Step 1: Introduce Slack Variables.** Define an integer slack variable `s` such that `Σ_v Σ_r yr + s = |V|`.
    - **Step 2: Binary Expansion.** Represent `s` using a set of binary variables `sk`: `s = Σ_k (2^k * sk)`. The number of slack variables needed depends on the maximum possible fleet size.
    - **Step 3: Formulate Penalty.** Create a quadratic penalty term.
    - **Formula:** `H_penalty_fleet = P_fleet * ( (Σ_v Σ_r yr + Σ_k (2^k * sk)) - |V| )^2`
    - **`P_fleet`:** A large positive coefficient to penalize violations of the fleet size constraint.
    - **File:** `algo/qubo_formulation.py` - `_formulate_fleet_penalties()`

### 7.4. Implementation Steps ✅
- [x] ✅ **IMPLEMENTED:** **Expand and Simplify:**
    - **Action:** Algebraically expand all the squared terms in the penalty functions. This will produce linear (diagonal) and quadratic (off-diagonal) terms for the QUBO matrix.
    - **Example:** `(a + b - 1)^2 = a^2 + b^2 + 1 + 2ab - 2a - 2b`. Since `a` and `b` are binary, `a^2 = a` and `b^2 = b`. The expression simplifies to `2ab - a - b + 1`.
    - **File:** `algo/qubo_formulation.py` - `_formulate_task_penalties()` and `_formulate_fleet_penalties()`

- [x] ✅ **IMPLEMENTED:** **Determine Penalty Coefficients:**
    - **Action:** Choose appropriate values for `P_task` and `P_fleet`.
    - **Guideline:** The penalty coefficients must be larger than any possible change in the original objective function to ensure that feasible solutions are always preferred over infeasible ones. A common rule of thumb is to set the penalty to be greater than the maximum possible value of the objective function.
    - **File:** `algo/qubo_formulation.py` - `_calculate_penalty_coefficients()`

- [x] ✅ **IMPLEMENTED:** **Construct QUBO Matrix:**
    - **Action:** Create the final QUBO matrix `Q`, where `Q_ii` will contain the linear coefficients of the binary variables and `Q_ij` will contain the quadratic coefficients.
    - **Goal:** The final model will be in the form `min(x'Qx)`, where `x` is the vector of all binary variables (`yr`, `xo`, `sk`). This matrix can then be submitted to a quantum annealer or a classical QUBO solver.
    - **File:** `algo/qubo_formulation.py` - `_construct_qubo_matrix()`

### 7.5. Implementation Files Created ✅
- [x] ✅ **`algo/qubo_formulation.py`** - Core QUBO formulation and solver integration
- [x] ✅ **`algo/qubo_integration.py`** - Hybrid Column Generation + QUBO workflow
- [x] ✅ **`algo/qubo_test.py`** - Comprehensive testing suite
- [x] ✅ **Integration with test runner** - Command line options for QUBO testing

### 7.6. Quantum Solver Support ✅
- [x] ✅ **Neal Simulated Annealing** - Classical QUBO solver (always available)
- [x] ✅ **D-Wave Quantum Annealer** - Quantum hardware support (when available)
- [x] ✅ **Exact Solver** - Small problem exact solutions
- [x] ✅ **Hybrid CG+QUBO** - Best of both classical and quantum approaches

### 7.7. Usage Examples ✅
```powershell
# Test QUBO formulation
python "testsun_scenario_test.py" --test-qubo

# Benchmark quantum vs classical
python "testsun_scenario_test.py" --test-quantum-benchmark

# Hybrid Column Generation + QUBO
python "testsun_scenario_test.py" --test-enhanced
```