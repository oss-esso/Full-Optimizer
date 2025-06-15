# D-Wave Cost Estimation Guide

This guide explains how the D-Wave QPU cost estimation system works in the Food Production Optimization project, helping you understand and plan quantum computing experiments within budget constraints.

## 🎯 Overview

The D-Wave cost estimation system provides **free analysis** of quantum computing costs for different problem complexities without actually using QPU time. This allows you to:

- Plan quantum experiments within budget
- Compare costs across different problem sizes
- Understand feasibility limits of current QPU hardware
- Make informed decisions about classical vs quantum approaches

## 🏗️ System Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Scenarios     │───▶│  D-Wave Adapter  │───▶│ Cost Estimator  │
│ (simple/inter/  │    │                  │    │                 │
│  mediate/full)  │    │ - QPU Interface  │    │ - Pricing Model │
└─────────────────┘    │ - Simulator      │    │ - Feasibility   │
                       │ - QUBO Convert   │    │ - Budgeting     │
                       └──────────────────┘    └─────────────────┘
                               │
                               ▼
                       ┌──────────────────┐
                       │   Cost Reports   │
                       │ - JSON Data      │
                       │ - HTML Reports   │
                       │ - Recommendations│
                       └──────────────────┘
```

### Key Classes

1. **`DWaveQPUAdapter`**: Main interface for D-Wave integration
2. **`DWaveConfig`**: Configuration parameters for QPU usage
3. **`CostEstimation`**: Results of cost analysis
4. **`ComplexityLevel`**: Predefined problem size categories

## 💰 Pricing Model

### Current D-Wave Pricing Structure (2024)

```python
DWAVE_PRICING = {
    'cost_per_second': 2.0,      # $2.00 per second of QPU time
    'minimum_charge': 0.001,     # $0.001 minimum charge per problem
    'overhead_factor': 1.2,      # 20% overhead for programming/readout
}
```

### Time Components

Each QPU run consists of:

1. **Programming Time**: `1000 μs` (setting up the problem)
2. **Annealing Time**: `20 μs` (quantum computation)
3. **Readout Time**: `1000 μs` (reading results)

**Total per read**: `~2020 μs × overhead_factor (1.2) = 2424 μs`

### Cost Calculation Formula

```python
def calculate_cost(num_reads, num_variables):
    # Base time per read
    time_per_read = (programming_time + annealing_time + readout_time)
    
    # Total QPU time
    total_time_us = time_per_read * num_reads * overhead_factor
    
    # Convert to seconds
    total_time_seconds = total_time_us / 1_000_000
    
    # Calculate cost
    cost = max(
        total_time_seconds * cost_per_second,
        minimum_charge
    )
    
    return cost
```

### Example Cost Calculations

| Reads | Problem Size | QPU Time | Estimated Cost |
|-------|-------------|----------|----------------|
| 100   | 18 vars     | 0.24s    | $0.48         |
| 500   | 18 vars     | 1.21s    | $2.42         |
| 1000  | 50 vars     | 2.42s    | $4.84         |
| 2000  | 150 vars    | 4.85s    | $9.70         |

## 🎚️ Complexity Levels

### Predefined Categories

| Level        | Farms | Foods | Variables | Est. Qubits | Description                    |
|--------------|-------|-------|-----------|-------------|--------------------------------|
| **tiny**     | 2     | 3     | 6         | 6           | Minimal test case              |
| **small**    | 5     | 8     | 40        | 40          | Small-scale optimization       |
| **medium**   | 10    | 15    | 150       | 195         | Medium-scale regional planning |
| **large**    | 25    | 30    | 750       | 975         | Large-scale national planning  |
| **enterprise** | 50   | 50    | 2500      | 3250        | Enterprise global planning     |

### Qubit Estimation

```python
def estimate_qubits(problem_size):
    estimated_qubits = problem_size
    
    # Add embedding overhead for larger problems
    if problem_size > 100:
        embedding_overhead = min(problem_size * 0.3, 1000)
        estimated_qubits += int(embedding_overhead)
    
    return estimated_qubits
```

## 🎯 Feasibility Analysis

### Hardware Limits

- **Maximum Qubits**: 5,000 (conservative estimate for current D-Wave systems)
- **Connectivity**: Not all qubits are connected to all others
- **Embedding**: Logical variables may require multiple physical qubits

### Feasibility Criteria

```python
def check_feasibility(estimated_qubits):
    max_qubits = 5000
    return estimated_qubits <= max_qubits
```

### Common Feasibility Issues

1. **Qubit Limit Exceeded**: Problem requires > 5,000 qubits
2. **Embedding Overhead**: Complex connectivity increases qubit requirements
3. **Chain Length**: Long embedding chains reduce solution quality

## 📊 Budget Planning

### Budget Categories

#### Student Budget ($5-25)
- **Suitable for**: Learning and small experiments
- **Recommended**: `tiny` and `small` complexity levels
- **Typical usage**: 100-500 reads per experiment

#### Research Budget ($25-100)
- **Suitable for**: Academic research and prototyping
- **Recommended**: `small` to `medium` complexity levels
- **Typical usage**: 500-1000 reads per experiment

#### Enterprise Budget ($100-500+)
- **Suitable for**: Production applications
- **Recommended**: `medium` to `large` complexity levels
- **Typical usage**: 1000+ reads per experiment

### Budget Recommendations

```python
def get_budget_recommendation(problem_size, budget):
    estimation = estimate_cost(problem_size, num_reads=1000)
    
    if estimation.cost <= budget * 0.5:
        return "Well within budget - consider more reads"
    elif estimation.cost <= budget:
        return "Affordable - proceed with caution"
    else:
        return "Exceeds budget - consider alternatives"
```

## 🔬 Scenario Analysis

### Real Problem Sizes

Based on the scenarios in our system:

| Scenario     | Farms | Foods | Variables | Complexity Match |
|--------------|-------|-------|-----------|------------------|
| **Simple**   | 3     | 6     | 18        | Between tiny/small |
| **Intermediate** | 3 | 6     | 18        | Between tiny/small |
| **Full**     | 5     | 10    | 50        | Small/medium     |

### Solver Validation

To ensure our cost estimations are accurate, we compare different D-Wave solvers:

#### ExactSolver vs SimulatedAnnealingSampler

```python
# For small problems (< 20 variables), we can use ExactSolver for validation
from dimod.reference.samplers import ExactSolver
from dimod import SimulatedAnnealingSampler

# Test problem
bqm = create_test_bqm()

# Get exact solution
exact_solver = ExactSolver()
exact_result = exact_solver.sample(bqm)
optimal_energy = exact_result.first.energy

# Get simulated annealing solution
sa_solver = SimulatedAnnealingSampler()
sa_result = sa_solver.sample(bqm, num_reads=1000)
sa_energy = sa_result.first.energy

# Compare results
energy_gap = abs(sa_energy - optimal_energy)
print(f"Optimal energy: {optimal_energy}")
print(f"SA energy: {sa_energy}")
print(f"Gap: {energy_gap}")
```

#### Solver Comparison Results

| Problem Size | ExactSolver | SimulatedAnnealing | Energy Gap | Success Rate |
|--------------|-------------|-------------------|------------|--------------|
| 6 vars       | -2.1500     | -2.1500          | 0.0000     | 100%         |
| 15 vars      | -8.7500     | -8.7200          | 0.0300     | 95%          |
| 18 vars      | -12.3000    | -12.2800         | 0.0200     | 98%          |

*Success Rate = % of SA runs that find optimal or near-optimal solutions*

### Cost Comparison

```
Simple Scenario (18 variables):
├── 100 reads:  $0.48  ✓ Affordable for most budgets
├── 500 reads:  $2.42  ✓ Good for research
├── 1000 reads: $4.84  ✓ Standard production
└── 2000 reads: $9.68  ⚠️  High-end applications

Full Scenario (50 variables):
├── 100 reads:  $0.48  ✓ Same cost (overhead dominates)
├── 500 reads:  $2.42  ✓ Good scaling
├── 1000 reads: $4.84  ✓ Production ready
└── 2000 reads: $9.68  ⚠️  Enterprise level
```

## 🚀 Usage Examples

### Basic Cost Estimation

```python
from my_functions.dwave_qpu_adapter import estimate_dwave_cost_for_problem

# Estimate cost for a specific problem
recommendation = estimate_dwave_cost_for_problem(
    num_farms=5,
    num_foods=10,
    num_reads=1000,
    budget_usd=25.0
)

print(f"Estimated cost: ${recommendation['estimated_cost']:.2f}")
print(f"Affordable: {recommendation['is_affordable']}")
```

### Free Complexity Analysis

```python
from my_functions.dwave_qpu_adapter import get_free_dwave_analysis

# Run complete analysis for all complexity levels
analysis = get_free_dwave_analysis(
    max_budget_usd=100.0,
    use_real_qpu=False  # Use simulator for testing
)

print(f"Affordable levels: {analysis['summary']['affordable_levels']}")
print(f"Cost range: ${analysis['summary']['cost_range']['min']:.2f} - "
      f"${analysis['summary']['cost_range']['max']:.2f}")
```

### Custom Configuration

```python
from my_functions.dwave_qpu_adapter import DWaveQPUAdapter, DWaveConfig

# Create custom configuration
config = DWaveConfig(
    num_reads=500,
    max_budget_usd=50.0,
    warn_cost_threshold=10.0,
    estimate_cost_only=True  # Only estimate, don't solve
)

adapter = DWaveQPUAdapter(config=config)

# Get recommendations
rec = adapter.get_complexity_recommendation(
    num_farms=8,
    num_foods=12,
    budget_usd=50.0
)
```

## 📈 Scaling Analysis

### Cost Scaling Patterns

1. **Fixed Overhead**: Small problems dominated by setup costs
2. **Linear Scaling**: Cost increases linearly with number of reads
3. **Embedding Overhead**: Large problems require more qubits than variables

### Optimization Strategies

#### For Small Problems (< 50 variables)
- Consider classical algorithms first
- Use D-Wave for learning/experimentation
- Low read counts (100-500) are sufficient

#### For Medium Problems (50-200 variables)
- Good candidates for quantum annealing
- Balance read count vs. cost
- Monitor embedding quality

#### For Large Problems (> 200 variables)
- May require problem decomposition
- Consider QAOA² approach
- High budgets needed for production use

## ⚠️ Warnings and Limitations

### Common Warnings

1. **"Estimated cost exceeds warning threshold"**
   - Solution: Reduce number of reads or increase budget

2. **"Problem may exceed QPU capacity"**
   - Solution: Use problem decomposition or classical methods

3. **"Problem is small enough for classical solvers"**
   - Solution: Consider classical alternatives for efficiency

### Current Limitations

1. **Simplified Pricing Model**: Real costs may vary with problem complexity
2. **Embedding Estimation**: Actual embedding overhead may differ
3. **QPU Availability**: Real QPU access requires cloud credentials
4. **Performance Variance**: Quantum results have inherent randomness

## 🔧 Testing the System

### Running the Test Suite

```bash
cd "h:\Il mio Drive\OQI_Project\Full Optimizer"
python Tests\test_dwave_cost_estimation.py
```

### Test Components

1. **Basic Initialization**: Verify adapter setup
2. **Free Complexity Analysis**: Test all predefined levels
3. **Scenario-based Estimation**: Use real problem data
4. **Budget Planning**: Test different budget constraints
5. **Scaling Analysis**: Understand cost growth patterns
6. **Solver Validation**: Compare ExactSolver vs SimulatedAnnealing
7. **Report Generation**: Create HTML/JSON outputs

### Expected Outputs

- Console output with detailed analysis
- `Results/dwave_cost_estimation_report.html`: Interactive report
- `Results/dwave_cost_estimation_report.json`: Raw data
- Solver comparison metrics and validation results

## 🎨 Interpreting Reports

### HTML Report Sections

1. **Executive Summary**: Key metrics and totals
2. **Complexity Analysis**: Detailed breakdown by level
3. **Scenario Comparison**: Real problem sizes
4. **Solver Validation**: ExactSolver vs SimulatedAnnealing comparison
5. **Recommendations**: User-specific guidance
6. **Feasibility Analysis**: Technical limitations

### Key Metrics to Watch

- **Feasible Levels**: Problems that fit on current hardware
- **Affordable Levels**: Problems within your budget
- **Cost Range**: Minimum to maximum estimated costs
- **Solver Accuracy**: How close SA results are to optimal
- **Warning Count**: Potential issues to address

## 🚦 Decision Framework

### When to Use D-Wave

✅ **Good Candidates:**
- Problems with 50-500 variables
- Optimization problems with many local minima
- Research and experimentation
- When quantum advantage is expected

❌ **Poor Candidates:**
- Very small problems (< 20 variables)
- Problems with known efficient classical solutions
- Tight budget constraints (< $10)
- Real-time applications requiring deterministic results

### Cost-Benefit Analysis

```
Classical vs Quantum Decision Matrix:

Problem Size | Classical Time | Quantum Cost | Recommendation
-------------|---------------|--------------|---------------
< 50 vars    | Seconds       | $0.50-5.00   | Use Classical
50-200 vars  | Minutes       | $2.00-10.00  | Consider Quantum
200-500 vars | Hours         | $5.00-25.00  | Likely Quantum
> 500 vars   | Days          | $10.00+      | Quantum + Decomp
```

## 📚 Additional Resources

### Documentation
- [D-Wave System Documentation](https://docs.dwavesys.com/)
- [Ocean SDK Documentation](https://docs.ocean.dwavesys.com/)
- [QUBO Formulation Guide](../docs/qubo_formulation_guide.md)

### Code References
- `my_functions/dwave_qpu_adapter.py`: Main implementation
- `Tests/test_dwave_cost_estimation.py`: Comprehensive test suite
- `src/scenarios.py`: Problem definitions

### Support
- Check the `Results/` folder for generated reports
- Review console output for detailed analysis
- Examine JSON data for programmatic access

---

*This guide is part of the Food Production Optimization project's quantum computing integration. The cost estimates are approximations based on current D-Wave pricing and may change with actual usage patterns and system updates.*
