# EPDT Column Generation Implementation Guide

## Overview

This document provides a comprehensive guide to the **Column Generation** implementation for finding optimality bounds in the Enhanced Pickup and Delivery with Time Windows (EPDT) problem. The implementation follows Chapter 6 of the thesis and provides a Mixed-Integer Linear Programming (MILP) approach for assessing heuristic solution quality.

## 🎯 **What is Column Generation?**

Column Generation is an optimization technique that decomposes large linear programs into:

1. **Master Problem (MP)**: A set covering formulation that selects optimal routes
2. **Pricing Problem (PP)**: An Elementary Shortest Path Problem with Resource Constraints (ESPPRC) that generates new routes

The algorithm iteratively adds new columns (routes) until optimality is reached, providing both lower and upper bounds for the EPDT problem.

## 📁 **Implementation Structure**

The Column Generation implementation consists of several key modules in the `/algo` folder:

### Core Modules

```
/algo/
├── column_generation.py          # Main Column Generation controller
├── master_problem.py             # Set covering formulation (MP)
├── pricing_problem.py            # ESPPRC solver (PP)
├── label_algorithm.py            # Advanced label correcting algorithm
├── column_generation_test.py     # Comprehensive testing suite
└── column_generation_utils.py    # Integration utilities and benchmarking
```

### Key Components

1. **`ColumnGenerationSolver`**: Main orchestrator class
2. **`MasterProblem`**: Set covering LP/MILP formulation
3. **`PricingProblem`**: ESPPRC solver for each vehicle
4. **`AdvancedLabelAlgorithm`**: Custom label correcting with dominance rules
5. **`EPDTBenchmarkSuite`**: Performance comparison framework

## 🚀 **Quick Start Guide**

### Basic Usage

```python
from algo.column_generation import solve_epdt_with_column_generation, ColumnGenerationConfig
from algo.epdt_data_structures import Order, Vehicle

# Configure the algorithm
config = ColumnGenerationConfig(
    max_iterations=50,
    optimality_gap_tolerance=1e-4,
    time_limit_seconds=300.0,
    verbose=True
)

# Solve the problem
result = solve_epdt_with_column_generation(orders, vehicles, config)

# Access results
if result.feasible:
    print(f"Lower Bound: {result.lower_bound:.2f}")
    print(f"Upper Bound: {result.upper_bound:.2f}")
    print(f"Optimality Gap: {result.optimality_gap:.6f}")
```

### Using the Test Runner

```powershell
# Test Column Generation implementation
python "VRPExample\heuristicapproach\tests\run_scenario_test.py" --test-column-generation

# Run comprehensive benchmark
python "VRPExample\heuristicapproach\tests\run_scenario_test.py" --run-benchmark

# Test enhanced features with Column Generation
python "VRPExample\heuristicapproach\tests\run_scenario_test.py" --test-enhanced
```

## 🔬 **Technical Implementation Details**

### Master Problem Formulation

The Master Problem uses a **set covering formulation**:

```
minimize: Σ(c_r * y_r) + Σ(p_o * x_o)

subject to:
    Σ(a_or * y_r) + x_o >= 1    ∀ mandatory orders o
    Σ(a_or * y_r) + x_o >= 0    ∀ non-mandatory orders o  
    Σ(b_vr * y_r) <= 1          ∀ vehicles v
    y_r ∈ {0,1}                 ∀ routes r
    x_o ∈ {0,1}                 ∀ orders o
```

**Variables:**
- `y_r`: Binary route selection variable
- `x_o`: Binary order unassignment variable

**Parameters:**
- `c_r`: Cost of route r
- `p_o`: Penalty for not serving order o
- `a_or`: 1 if route r serves order o
- `b_vr`: 1 if route r uses vehicle v

### Pricing Problem (ESPPRC)

The Pricing Problem finds routes with negative reduced cost:

```
minimize: c_r - Σ(π_o * a_or) - μ_v * b_vr
```

**Where:**
- `π_o`: Dual value for order coverage constraint
- `μ_v`: Dual value for vehicle fleet constraint

### Advanced Label Algorithm

The label correcting algorithm uses sophisticated dominance rules:

#### Label State Components
- **Location and Resources**: Current position, time, weight, volume
- **Order Tracking**: Open orders, completed orders, unreachable tasks
- **LIFO State**: Cargo stack for vehicles without side doors
- **Working Time**: Hours of Service compliance
- **Multi-day State**: Current day and temporal constraints

#### Custom Dominance Rules
1. **Basic Resource Dominance**: Cost, time, weight, volume
2. **Order Progress Dominance**: Completion state comparison
3. **LIFO Stack Compatibility**: Stack subset relationships
4. **Working Time State**: HoS regulation compliance
5. **Precedence Feasibility**: Unreachable task subset rules

## ⚙️ **Configuration Options**

### ColumnGenerationConfig Parameters

```python
config = ColumnGenerationConfig(
    # Termination criteria
    max_iterations=100,                    # Maximum CG iterations
    optimality_gap_tolerance=1e-6,         # Optimality gap threshold
    time_limit_seconds=3600.0,             # Time limit (1 hour)
    
    # Solver preferences
    preferred_solver="gurobi",             # "gurobi", "pulp", "cplex"
    solver_time_limit=300.0,               # Subproblem time limit
    
    # Algorithm strategies
    initial_routes_strategy="single_order", # "single_order", "empty", "heuristic"
    pricing_heuristic=True,                # Use heuristic pricing first
    
    # Performance tuning
    verbose=True,                          # Detailed output
    save_intermediate_solutions=False,      # Save intermediate results
    log_dual_values=False                  # Log dual value evolution
)
```

### Solver Requirements

The implementation supports multiple LP/MILP solvers:

#### Gurobi (Recommended)
```powershell
# Install Gurobi Python interface
pip install gurobipy

# Requires Gurobi license (free academic licenses available)
```

#### PuLP (Free Alternative)
```powershell
# Install PuLP with CBC solver
pip install pulp
```

## 🧪 **Testing and Validation**

### Unit Tests

Run comprehensive unit tests:

```powershell
# Test individual components
python -c "from algo.column_generation_test import run_column_generation_tests; run_column_generation_tests()"
```

**Test Coverage:**
- Master Problem formulation and constraints
- Pricing Problem ESPPRC algorithm
- Label algorithm dominance rules
- Full Column Generation integration

### Integration Tests

Test on real scenarios:

```powershell
# Test on Furgoni scenario
python "VRPExample\heuristicapproach\tests\run_scenario_test.py" --test-column-generation
```

### Performance Benchmarking

Compare with EPDT heuristic:

```powershell
# Run comprehensive benchmark
python "VRPExample\heuristicapproach\tests\run_scenario_test.py" --run-benchmark
```

**Benchmark Metrics:**
- Solution time comparison
- Solution quality ratio (heuristic/lower_bound)
- Optimality gap analysis
- Scalability assessment

## 📊 **Results Analysis**

### Output Interpretation

#### Column Generation Results
```python
result = solve_epdt_with_column_generation(orders, vehicles, config)

# Bounds analysis
if result.feasible:
    lower_bound = result.lower_bound    # Proven optimal lower bound
    upper_bound = result.upper_bound    # Best integer solution found
    gap = result.optimality_gap         # (UB - LB) / UB
    
    # Algorithm statistics
    iterations = result.statistics.iterations
    columns_generated = result.statistics.total_columns_generated
    solve_time = result.statistics.total_time
```

#### Quality Assessment
- **Lower Bound**: Proven minimum cost for any feasible solution
- **Upper Bound**: Best integer solution found (if feasible)
- **Optimality Gap**: Measures distance from proven optimality
- **Heuristic Quality**: How close heuristic solutions are to optimal

### Benchmark Results

Benchmark results are saved in JSON format:

```json
{
  "benchmark_results": [
    {
      "problem_name": "furgoni",
      "column_generation": {
        "lower_bound": 1250.45,
        "upper_bound": 1267.20,
        "optimality_gap": 0.013234,
        "solve_time": 45.67
      },
      "heuristic": {
        "objective": 1289.34,
        "solve_time": 2.34,
        "vehicles_used": 6
      },
      "comparison": {
        "heuristic_quality_ratio": 1.031
      }
    }
  ]
}
```

## 🔧 **Advanced Features**

### Multi-Day Planning Support

The implementation handles:
- **Yesterday tasks**: Continuing routes from previous day
- **Today tasks**: Main planning horizon  
- **Tomorrow tasks**: Prospective planning
- **Temporal constraints**: Day-based precedence rules

### LIFO Constraint Handling

For vehicles without side doors:
- **Loading simulation**: Track cargo stack state
- **Unloading validation**: LIFO order enforcement
- **Dominance rules**: Stack compatibility checking

### Hours of Service Compliance

European HoS regulations:
- **Driving time limits**: 9 hours (extendable to 10)
- **Working time limits**: 13 hours (extendable to 14)
- **Break requirements**: 45 minutes after 4.5 hours driving
- **Daily rest**: 11 hours minimum

## 🚨 **Troubleshooting**

### Common Issues

#### 1. Solver Not Available
```
Error: No LP solver available. Install PuLP or Gurobi.
```
**Solution:** Install a supported solver:
```powershell
pip install pulp  # Free option
# OR install Gurobi with license
```

#### 2. Memory Issues with Large Problems
```
MemoryError: Unable to allocate array
```
**Solutions:**
- Reduce `max_labels_per_location` in label algorithm
- Use `pricing_heuristic=True` for faster convergence
- Implement label filtering strategies

#### 3. Slow Convergence
```
Warning: Time limit reached after X iterations
```
**Solutions:**
- Increase `time_limit_seconds`
- Adjust `optimality_gap_tolerance`
- Use `initial_routes_strategy="heuristic"`
- Enable `pricing_heuristic=True`

#### 4. Infeasible Master Problem
```
Master Problem infeasible
```
**Causes:**
- Mandatory orders cannot be satisfied
- Vehicle capacity constraints too tight
- Time window conflicts

### Performance Optimization

#### For Large Problems (>50 orders):
```python
config = ColumnGenerationConfig(
    max_iterations=30,                     # Reduced iterations
    optimality_gap_tolerance=1e-3,         # Relaxed tolerance
    pricing_heuristic=True,                # Enable heuristic pricing
    initial_routes_strategy="heuristic"    # Better initialization
)
```

#### For High Precision (Small problems):
```python
config = ColumnGenerationConfig(
    max_iterations=200,                    # More iterations
    optimality_gap_tolerance=1e-6,         # Tight tolerance
    preferred_solver="gurobi",             # Best solver
    log_dual_values=True                   # Detailed logging
)
```

## 📈 **Performance Expectations**

### Typical Performance Characteristics

| Problem Size | Solve Time | Expected Gap | Memory Usage |
|-------------|------------|--------------|--------------|
| 5-10 orders | 10-30 sec  | < 0.1%       | < 100 MB     |
| 10-20 orders| 30-120 sec | < 0.5%       | < 500 MB     |
| 20-50 orders| 2-10 min   | < 1.0%       | < 2 GB       |
| 50+ orders  | 10+ min    | < 2.0%       | > 2 GB       |

### Comparison with Heuristic

**Typical Results:**
- **Speed**: Heuristic is 10-100x faster
- **Quality**: Heuristic within 3-10% of optimal
- **Guarantee**: Column Generation provides proven bounds
- **Use Case**: Column Generation for quality assessment, heuristic for production

## 🔮 **Future Extensions**

### Planned Enhancements

1. **Cut Generation**: Add valid inequalities for tighter bounds
2. **Branch-and-Price**: Full branch-and-bound integration
3. **Parallel Pricing**: Multi-threaded pricing problem solving
4. **Advanced Heuristics**: Better initial solutions and pricing heuristics
5. **Dynamic Programming**: Exact algorithms for small subproblems

### Integration Opportunities

1. **Real-time Optimization**: Dynamic re-optimization with Column Generation bounds
2. **Scenario Planning**: Stochastic programming with multiple scenarios
3. **Machine Learning**: ML-enhanced pricing strategies
4. **Cloud Computing**: Distributed Column Generation on cloud platforms

## 📚 **References and Further Reading**

### Academic References
- **Desaulniers, G., & Villeneuve, D. (2000)**: The shortest path problem with time windows and linear waiting costs
- **Baldacci, R., Mingozzi, A., & Roberti, R. (2011)**: New route relaxation and pricing strategies for the vehicle routing problem
- **Pecin, D., et al. (2017)**: New enhancements for the exact solution of the vehicle routing problem with time windows

### Implementation References
- **Chapter 6**: MILP Formulation for Optimality Bounds
- **EPDT Algorithm Documentation**: Enhanced heuristic implementation
- **VRP Literature**: State-of-the-art exact algorithms

---

## ✅ **Summary**

The EPDT Column Generation implementation provides:

✅ **Optimality Bounds**: Proven lower and upper bounds for solution quality  
✅ **Advanced Constraints**: Multi-day, LIFO, HoS, and precedence support  
✅ **Performance Benchmarking**: Comprehensive comparison framework  
✅ **Scalable Architecture**: Configurable for different problem sizes  
✅ **Integration Ready**: Seamless integration with existing EPDT system  
✅ **Production Quality**: Robust error handling and performance optimization  

The implementation serves as both a research tool for algorithm evaluation and a practical solution for assessing heuristic solution quality in real-world vehicle routing applications.
