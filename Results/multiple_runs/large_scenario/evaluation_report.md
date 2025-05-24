# Optimization Methods Evaluation Report

## Solution Quality Metrics

### Objective Values
#### PuLP
- Mean: 102.0066 ± 15.0029
- Range: [86.5997, 125.4586]
- Success Rate: 100.0%

#### Benders
- Mean: 106.1221 ± 17.3152
- Range: [88.8973, 139.2665]
- Success Rate: 72.0%

#### Quantum-Enhanced
- Mean: 101.4607 ± 10.5876
- Range: [83.2558, 125.6600]
- Success Rate: 60.0%

#### Quantum-Enhanced-Merge
- Mean: 102.6467 ± 8.5472
- Range: [90.4643, 124.7508]
- Success Rate: 56.0%

#### RQAOA
- Mean: 100.0789 ± 9.0421
- Range: [86.5570, 120.9181]
- Success Rate: 56.0%

### Solution Differences (compared to PuLP)
#### Benders
- Gap: 0.1232 ± 0.1315
- Max Violation: 71.2986 ± 23.4323
- Infinity Norm X: 71.2986 ± 23.4323
- One Norm Y: 525.4075 ± 182.8808

#### Quantum-Enhanced
- Gap: 0.1418 ± 0.1132
- Max Violation: 83.1340 ± 16.8549
- Infinity Norm X: 83.1340 ± 16.8549
- One Norm Y: 624.4159 ± 109.1696

#### Quantum-Enhanced-Merge
- Gap: 0.1312 ± 0.0970
- Max Violation: 77.7719 ± 21.1353
- Infinity Norm X: 77.7719 ± 21.1353
- One Norm Y: 604.2282 ± 117.2819

#### RQAOA
- Gap: 0.1405 ± 0.1027
- Max Violation: 88.3653 ± 10.5850
- Infinity Norm X: 88.3653 ± 10.5850
- One Norm Y: 658.0564 ± 61.9453

## Performance Metrics

### Runtime
#### PuLP
- Mean: 0.46 ± 0.06 seconds

#### Benders
- Mean: 9.92 ± 8.48 seconds

#### Quantum-Enhanced
- Mean: 16.17 ± 2.32 seconds

#### Quantum-Enhanced-Merge
- Mean: 15.30 ± 2.30 seconds

#### RQAOA
- Mean: 2.93 ± 0.75 seconds

### Resource Utilization
#### PuLP
- Memory Peak: 3.07 ± 3.47 MB

#### Benders
- Memory Peak: 4.29 ± 4.73 MB

#### Quantum-Enhanced
- Memory Peak: 165.94 ± 32.45 MB

#### Quantum-Enhanced-Merge
- Memory Peak: 164.38 ± 32.40 MB

#### RQAOA
- Memory Peak: 4.23 ± 4.61 MB

## Quantum Optimization Details

### Quantum-Enhanced QAOA² Details

- QAOA² Decomposition used in 0 out of 25 runs (0.0%)

### Quantum-Enhanced-Merge QAOA² Details

- QAOA² Decomposition used in 0 out of 25 runs (0.0%)
