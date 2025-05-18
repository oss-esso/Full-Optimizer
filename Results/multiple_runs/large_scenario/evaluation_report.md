# Optimization Methods Evaluation Report

## Solution Quality Metrics

### Objective Values
#### PuLP
- Mean: 309.4058 ± 23.1090
- Range: [262.1810, 358.2130]
- Success Rate: 100.0%

#### Benders
- Mean: 307.4457 ± 23.9205
- Range: [265.5188, 349.4253]
- Success Rate: 0.0%

#### Quantum-Enhanced
- Mean: 213.3619 ± 38.0816
- Range: [138.4102, 255.9821]
- Success Rate: 0.0%

#### Quantum-Enhanced-Merge
- Mean: 223.0409 ± 70.4190
- Range: [0.0000, 283.1924]
- Success Rate: 0.0%

### Solution Differences (compared to PuLP)
#### Benders
- Gap: nan ± nan
- Max Violation: nan ± nan
- Infinity Norm X: nan ± nan
- One Norm Y: nan ± nan

#### Quantum-Enhanced
- Gap: nan ± nan
- Max Violation: nan ± nan
- Infinity Norm X: nan ± nan
- One Norm Y: nan ± nan

#### Quantum-Enhanced-Merge
- Gap: nan ± nan
- Max Violation: nan ± nan
- Infinity Norm X: nan ± nan
- One Norm Y: nan ± nan

## Performance Metrics

### Runtime
#### PuLP
- Mean: 0.14 ± 0.01 seconds

#### Benders
- Mean: 8.68 ± 1.46 seconds

#### Quantum-Enhanced
- Mean: 56.24 ± 10.63 seconds

#### Quantum-Enhanced-Merge
- Mean: 49.53 ± 11.34 seconds

### Resource Utilization
#### PuLP
- Memory Peak: 0.50 ± 0.54 MB

#### Benders
- Memory Peak: 3.11 ± 3.32 MB

#### Quantum-Enhanced
- Memory Peak: 212.51 ± 102.10 MB

#### Quantum-Enhanced-Merge
- Memory Peak: 257.93 ± 49.06 MB

## Quantum Optimization Details

### Quantum-Enhanced QAOA² Details

- QAOA² Decomposition used in 0 out of 25 runs (0.0%)

### Quantum-Enhanced-Merge QAOA² Details

- QAOA² Decomposition used in 0 out of 25 runs (0.0%)
