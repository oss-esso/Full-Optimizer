# Optimization Methods Evaluation Report

## Solution Quality Metrics

### Objective Values
#### PuLP
- Mean: 105.0812 ± 15.0674
- Range: [86.6516, 127.3313]
- Success Rate: 100.0%

#### Benders
- Mean: 109.2669 ± 18.0189
- Range: [89.0097, 139.4527]
- Success Rate: 55.0%

#### Quantum-Enhanced
- Mean: 98.8809 ± 14.6066
- Range: [66.2366, 123.0124]
- Success Rate: 30.0%

#### Quantum-Enhanced-Merge
- Mean: 100.8892 ± 8.8148
- Range: [89.6082, 123.3505]
- Success Rate: 70.0%

### Solution Differences (compared to PuLP)
#### Benders
- Gap: 0.1756 ± 0.1672
- Max Violation: 70.7050 ± 26.4960
- Infinity Norm X: 70.7050 ± 26.4960
- One Norm Y: 512.5180 ± 185.6219

#### Quantum-Enhanced
- Gap: 0.1894 ± 0.1093
- Max Violation: 89.4779 ± 13.3791
- Infinity Norm X: 89.4779 ± 13.3791
- One Norm Y: 657.5837 ± 77.8103

#### Quantum-Enhanced-Merge
- Gap: 0.0933 ± 0.1040
- Max Violation: 85.8713 ± 16.9649
- Infinity Norm X: 85.8713 ± 16.9649
- One Norm Y: 617.6703 ± 87.2529

## Performance Metrics

### Runtime
#### PuLP
- Mean: 0.21 ± 0.05 seconds

#### Benders
- Mean: 5.11 ± 1.95 seconds

#### Quantum-Enhanced
- Mean: 21.30 ± 3.07 seconds

#### Quantum-Enhanced-Merge
- Mean: 21.86 ± 3.20 seconds

### Resource Utilization
#### PuLP
- Memory Peak: 4.56 ± 3.38 MB

#### Benders
- Memory Peak: 8.87 ± 1.99 MB

#### Quantum-Enhanced
- Memory Peak: 233.56 ± 21.25 MB

#### Quantum-Enhanced-Merge
- Memory Peak: 232.62 ± 22.13 MB

## Quantum Optimization Details

### Quantum-Enhanced QAOA² Details

- QAOA² Decomposition used in 0 out of 20 runs (0.0%)

### Quantum-Enhanced-Merge QAOA² Details

- QAOA² Decomposition used in 0 out of 20 runs (0.0%)
