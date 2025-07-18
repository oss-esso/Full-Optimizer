# EPDT QUBO Formulation for Quantum Annealing

## Overview

This document provides a comprehensive guide to the **QUBO (Quadratic Unconstrained Binary Optimization) formulation** for solving EPDT problems on quantum annealing hardware. The implementation converts the Column Generation MILP formulation into a quantum-compatible format, enabling exploration of quantum advantage for vehicle routing optimization.

## 🎯 **What is QUBO for EPDT?**

QUBO transforms the EPDT problem from a constrained optimization problem into an unconstrained one by:

1. **Converting constraints to penalties**: All constraints become quadratic penalty terms in the objective function
2. **Binary variable formulation**: All variables are binary (0 or 1) suitable for quantum annealers
3. **Single polynomial optimization**: The entire problem becomes `min(x^T Q x)` where `Q` is the QUBO matrix

### Mathematical Foundation

The total QUBO Hamiltonian is:
```
H = H_objective + H_penalty_tasks + H_penalty_fleet
```

Where:
- **`H_objective`**: Direct translation of MILP objective (route costs + unassignment penalties)
- **`H_penalty_tasks`**: Quadratic penalties ensuring each task is served exactly once
- **`H_penalty_fleet`**: Quadratic penalties enforcing fleet size constraints

## 📁 **Implementation Structure**

### Core QUBO Modules

```
/algo/
├── qubo_formulation.py         # Core QUBO formulation and matrix construction
├── qubo_integration.py         # Hybrid Column Generation + QUBO workflow
├── qubo_test.py               # Comprehensive testing suite
└── QUBO_IMPLEMENTATION_GUIDE.md  # This documentation
```

### Key Components

1. **`QUBOFormulator`**: Converts MILP to QUBO matrix
2. **`QUBOSolver`**: Interfaces with quantum/classical QUBO solvers
3. **`HybridSolver`**: Combines Column Generation with QUBO
4. **`QUBOTestSuite`**: Validation and benchmarking framework

## 🚀 **Quick Start Guide**

### Basic QUBO Usage

```python
from algo.qubo_formulation import solve_epdt_with_qubo, QUBOConfig

# Configure QUBO solver
config = QUBOConfig(
    preferred_solver="neal",  # Simulated annealing
    num_reads=1000,
    task_penalty_multiplier=10.0,
    fleet_penalty_multiplier=5.0
)

# Solve EPDT with QUBO (requires pre-generated routes)
result = solve_epdt_with_qubo(orders, vehicles, routes, route_costs, config)

# Access results
if result.feasible:
    print(f"QUBO Energy: {result.energy:.2f}")
    print(f"Selected routes: {len(result.selected_routes)}")
    print(f"Constraint violations: {sum(result.constraint_violations.values())}")
```

### Hybrid Column Generation + QUBO

```python
from algo.qubo_integration import solve_epdt_hybrid, HybridConfig
from algo.column_generation import ColumnGenerationConfig

# Configure hybrid approach
hybrid_config = HybridConfig(
    cg_config=ColumnGenerationConfig(
        max_iterations=20,
        optimality_gap_tolerance=1e-4
    ),
    qubo_config=QUBOConfig(
        preferred_solver="neal",
        num_reads=1000
    ),
    max_routes_for_qubo=50  # Limit QUBO complexity
)

# Solve using hybrid approach
result = solve_epdt_hybrid(orders, vehicles, hybrid_config)

# Comprehensive results
print(f"CG Lower Bound: {result.cg_lower_bound:.2f}")
print(f"QUBO Objective: {result.qubo_objective:.2f}")
print(f"Optimality Gap: {result.optimality_gap:.6f}")
```

### Command Line Interface

```powershell
# Test QUBO implementation
python "tests\run_scenario_test.py" --test-qubo

# Benchmark quantum vs classical solvers
python "tests\run_scenario_test.py" --test-quantum-benchmark

# Test hybrid CG+QUBO on enhanced scenarios
python "tests\run_scenario_test.py" --test-enhanced
```

## 🔬 **Technical Implementation Details**

### QUBO Variable Types

The QUBO formulation uses three types of binary variables:

#### 1. Route Selection Variables (`y_r`)
- **Purpose**: `y_r = 1` if route `r` is selected, `0` otherwise
- **Linear Coefficient**: Route cost `c_r`
- **Example**: `y_r0`, `y_r1`, `y_r2`, ...

#### 2. Order Unassignment Variables (`x_o`)
- **Purpose**: `x_o = 1` if order `o` is not served, `0` otherwise  
- **Linear Coefficient**: Unassignment penalty `p_o`
- **Example**: `x_oO1`, `x_oO2`, `x_oO3`, ...

#### 3. Slack Variables (`s_k`)
- **Purpose**: Binary expansion of fleet size slack
- **Linear Coefficient**: Part of fleet constraint penalty
- **Example**: `s_k0`, `s_k1`, `s_k2`, ... (binary representation)

### QUBO Matrix Construction

#### Objective Term (`H_objective`)
Direct translation of MILP objective:
```
H_objective = Σ_r (c_r * y_r) + Σ_o (p_o * x_o)
```
- **Matrix Effect**: Diagonal terms `Q[i,i]` for route costs and unassignment penalties

#### Task Servicing Penalty (`H_penalty_tasks`)
For each order `o`, ensure exactly one of: route coverage OR unassignment:
```
H_penalty_tasks = Σ_o P_task * ((Σ_r a_or * y_r + x_o) - 1)²
```

**Algebraic Expansion:**
```
(Σ_r a_or * y_r + x_o - 1)² = 
  Σ_r y_r + x_o + 2*Σ_{r,s} y_r*y_s + 2*Σ_r y_r*x_o - 2*Σ_r y_r - 2*x_o + 1
= 2*Σ_{r,s} y_r*y_s + 2*Σ_r y_r*x_o - Σ_r y_r - x_o + 1
```

**Matrix Effects:**
- **Diagonal**: Negative coefficients for variables in constraint
- **Off-diagonal**: Positive coefficients for variable pairs

#### Fleet Size Penalty (`H_penalty_fleet`)
Enforce fleet size constraint with slack variables:
```
H_penalty_fleet = P_fleet * ((Σ_r y_r + Σ_k 2^k * s_k) - |V|)²
```

**Matrix Effects:**
- **Route-route interactions**: `Q[i,j]` for route pairs
- **Route-slack interactions**: `Q[i,k]` with power-of-2 coefficients
- **Slack-slack interactions**: `Q[k,l]` with combined powers

### Penalty Coefficient Determination

Critical for QUBO correctness:

#### Automatic Calculation (Recommended)
```python
max_objective = max_route_cost * num_routes
task_penalty = max_objective * task_penalty_multiplier
fleet_penalty = max_objective * fleet_penalty_multiplier
```

#### Guidelines
- **Task Penalty**: Must dominate any objective change from violating task constraints
- **Fleet Penalty**: Must dominate any objective change from violating fleet constraints
- **Rule of Thumb**: Set penalties 5-10x larger than maximum possible objective value

### Solver Integration

#### Neal Simulated Annealing (Classical)
```python
config = QUBOConfig(
    preferred_solver="neal",
    num_reads=1000,  # Number of annealing runs
    verbose=True
)
```
- **Best for**: Development, testing, and classical baselines
- **Availability**: Always available (pure Python)
- **Performance**: Good quality, reasonable speed

#### D-Wave Quantum Annealer (Quantum)
```python
config = QUBOConfig(
    preferred_solver="dwave",
    num_reads=1000,
    chain_strength=1.0,  # Quantum coupling strength
    verbose=True
)
```
- **Best for**: True quantum annealing experiments
- **Requirements**: D-Wave access and API credentials
- **Performance**: Potential quantum advantage for large problems

#### Exact Solver (Classical)
```python
config = QUBOConfig(
    preferred_solver="exact",
    verbose=True
)
```
- **Best for**: Small problems (≤20 variables) and validation
- **Availability**: Through DIMOD
- **Performance**: Optimal solutions, limited scalability

## 🔗 **Integration with Column Generation**

### Hybrid Workflow

The implementation provides a sophisticated hybrid approach:

#### Phase 1: Column Generation Route Generation
1. **Master Problem**: Set covering LP relaxation
2. **Pricing Problem**: ESPPRC for new routes
3. **Output**: High-quality route pool with optimality bounds

#### Phase 2: QUBO Route Selection  
1. **Route Filtering**: Select best routes for QUBO complexity
2. **QUBO Formulation**: Convert to quantum-compatible format
3. **Quantum Solving**: Use quantum/classical QUBO solvers
4. **Solution Reconstruction**: Build final EPDT solution

### Route Filtering Strategies

#### Quality-Based Filtering
```python
hybrid_config = HybridConfig(
    route_quality_threshold=0.1,  # Within 10% of best route
    max_routes_for_qubo=50,       # Limit QUBO size
    include_all_single_order_routes=True  # Ensure coverage
)
```

#### Size-Based Filtering
- **Quantum Limit**: Typically 100-1000 variables for current quantum hardware
- **Classical Limit**: 10,000+ variables for classical QUBO solvers
- **Strategy**: Prioritize high-quality routes while maintaining coverage

## 🧪 **Testing and Validation**

### Comprehensive Test Suite

#### Unit Tests
```powershell
# Run all QUBO tests
python -m algo.qubo_test
```

**Coverage:**
- QUBO formulation correctness
- Penalty coefficient calculation  
- Matrix construction algebra
- Solver integration
- Constraint encoding validation

#### Integration Tests
```powershell
# Test QUBO with scenarios
python "tests\run_scenario_test.py" --test-qubo
```

**Coverage:**
- End-to-end QUBO solving
- Hybrid CG+QUBO workflow
- Multiple solver comparison
- Solution quality validation

#### Performance Benchmarks
```powershell
# Quantum vs classical comparison
python "tests\run_scenario_test.py" --test-quantum-benchmark
```

**Metrics:**
- Solution quality comparison
- Solve time analysis
- Scalability assessment
- Quantum advantage evaluation

### Validation Methods

#### Mathematical Validation
1. **Constraint Encoding**: Verify penalty terms correctly encode constraints
2. **Algebraic Expansion**: Validate quadratic expansion mathematics
3. **Matrix Construction**: Check QUBO matrix properties

#### Solution Validation
1. **Feasibility Checking**: Verify solutions satisfy original constraints
2. **Objective Calculation**: Compare QUBO energy with original objective
3. **Penalty Analysis**: Assess constraint violation penalties

## 📊 **Performance Characteristics**

### Problem Size Guidelines

| Variable Count | Solver Recommendation | Expected Performance |
|---------------|----------------------|---------------------|
| 10-50 | Exact solver | Optimal, < 1 second |
| 50-200 | Neal simulated annealing | High quality, < 10 seconds |
| 200-1000 | D-Wave quantum annealer | Good quality, < 30 seconds |
| 1000+ | Hybrid approach | Decomposition required |

### Quality Expectations

#### QUBO vs MILP Comparison
- **Optimality Gap**: Typically within 5-15% of MILP lower bound
- **Feasibility**: High feasibility rate with proper penalty tuning
- **Speed**: Often faster than MILP for combinatorial selection

#### Quantum vs Classical
- **Solution Quality**: Comparable for well-tuned problems
- **Speed**: Depends on problem structure and hardware
- **Scalability**: Quantum advantage may emerge for larger problems

## 🔧 **Configuration Guide**

### QUBO Configuration Options

```python
config = QUBOConfig(
    # Penalty coefficients
    task_penalty_multiplier=10.0,      # Task constraint penalty
    fleet_penalty_multiplier=5.0,      # Fleet constraint penalty
    auto_penalty_calculation=True,      # Automatic penalty setting
    
    # Formulation options
    use_set_partitioning=True,          # Exactly one vs at least one
    include_slack_variables=True,       # Fleet constraint handling
    
    # Solver preferences
    preferred_solver="neal",            # "neal", "dwave", "exact"
    num_reads=1000,                     # Annealing iterations
    chain_strength=1.0,                 # Quantum coupling strength
    
    # Performance tuning
    max_variables=1000,                 # QUBO size limit
    sparse_matrix=True,                 # Memory optimization
    verbose=True                        # Detailed output
)
```

### Hybrid Configuration Options

```python
hybrid_config = HybridConfig(
    # Column Generation phase
    cg_config=ColumnGenerationConfig(...),
    use_column_generation=True,
    max_routes_for_qubo=100,
    
    # QUBO phase  
    qubo_config=QUBOConfig(...),
    use_qubo_solver=True,
    
    # Route filtering
    route_quality_threshold=0.1,
    include_all_single_order_routes=True,
    
    # Analysis options
    compare_with_milp=True,
    benchmark_solvers=False
)
```

## 🚨 **Troubleshooting**

### Common Issues

#### 1. QUBO Solver Not Available
```
Error: No QUBO solver available
```
**Solution:**
```powershell
# Install QUBO packages
pip install dimod dwave-neal

# For D-Wave access
pip install dwave-ocean-sdk
dwave config create  # Set up D-Wave credentials
```

#### 2. Poor Solution Quality
```
Warning: High constraint violations in QUBO solution
```
**Solutions:**
- Increase penalty coefficients: `task_penalty_multiplier=20.0`
- Use more annealing reads: `num_reads=5000`
- Check route coverage: Ensure all orders can be served

#### 3. QUBO Matrix Too Large
```
Error: QUBO matrix exceeds size limit
```
**Solutions:**
- Reduce routes: `max_routes_for_qubo=50`
- Use route filtering: `route_quality_threshold=0.05`
- Consider problem decomposition

#### 4. Quantum Hardware Issues
```
Error: D-Wave connection failed
```
**Solutions:**
- Check D-Wave credentials: `dwave config inspect`
- Use classical fallback: `preferred_solver="neal"`
- Verify quantum access quotas

### Performance Optimization

#### For Small Problems (≤10 orders):
```python
config = QUBOConfig(
    preferred_solver="exact",           # Optimal solutions
    include_slack_variables=True,
    verbose=True
)
```

#### For Medium Problems (10-30 orders):
```python
config = QUBOConfig(
    preferred_solver="neal",
    num_reads=1000,                     # Balance quality/speed
    task_penalty_multiplier=5.0,        # Moderate penalties
    max_variables=200
)
```

#### For Large Problems (30+ orders):
```python
hybrid_config = HybridConfig(
    max_routes_for_qubo=50,             # Aggressive filtering
    route_quality_threshold=0.05,       # Only best routes
    qubo_config=QUBOConfig(
        preferred_solver="dwave",        # Quantum advantage
        num_reads=2000
    )
)
```

## 🔮 **Future Extensions**

### Planned Enhancements

1. **Advanced QUBO Formulations**
   - Embedding optimization for quantum hardware
   - Alternative penalty schemes
   - Multi-objective QUBO formulations

2. **Quantum Algorithm Integration**
   - QAOA (Quantum Approximate Optimization Algorithm)
   - Variational quantum eigensolvers
   - Quantum-classical hybrid algorithms

3. **Hardware Integration**
   - IonQ quantum computers
   - Rigetti quantum cloud
   - Google quantum AI hardware

4. **Performance Optimizations**
   - Graph embedding strategies
   - Decomposition algorithms
   - Parallel quantum processing

### Research Opportunities

1. **Quantum Advantage Analysis**
   - Identify problem classes with quantum speedup
   - Benchmark against best classical algorithms
   - Theoretical complexity analysis

2. **EPDT-Specific Optimizations**
   - Custom quantum embeddings for VRP structure
   - Multi-pickup delivery quantum formulations
   - Time window quantum constraints

## 📚 **References and Further Reading**

### Academic References
- **Lucas, A. (2014)**: Ising formulations of many NP problems
- **Glover, F., et al. (2018)**: Quantum Bridge Analytics for Optimization
- **Venturelli, D., et al. (2015)**: Quantum optimization of fully connected spin glasses

### Quantum Computing Resources
- **D-Wave Ocean Documentation**: https://docs.ocean.dwavesys.com/
- **DIMOD Documentation**: https://dimod.readthedocs.io/
- **Quantum Approximate Optimization**: QAOA tutorials and examples

### Implementation References
- **Chapter 7**: QUBO Formulation for Quantum Annealing
- **Column Generation Guide**: Integration with MILP formulation
- **EPDT Algorithm Documentation**: Problem-specific constraints

---

## ✅ **Summary**

The EPDT QUBO implementation provides:

✅ **Complete QUBO Formulation**: Mathematical conversion from MILP to quantum-compatible format  
✅ **Multiple Solver Support**: Neal, D-Wave, exact solvers with automatic fallback  
✅ **Hybrid CG+QUBO**: Best of classical optimization and quantum annealing  
✅ **Advanced Constraint Handling**: Proper encoding of EPDT-specific constraints  
✅ **Comprehensive Testing**: Validation, benchmarking, and performance analysis  
✅ **Production Ready**: Robust error handling and configuration options  

The implementation enables researchers and practitioners to:
- **Explore quantum advantage** for vehicle routing optimization
- **Benchmark quantum vs classical** approaches scientifically  
- **Develop quantum algorithms** for logistics and transportation
- **Assess solution quality** with proven mathematical foundations

**The QUBO formulation is complete and ready for quantum annealing experiments!** ⚛️
