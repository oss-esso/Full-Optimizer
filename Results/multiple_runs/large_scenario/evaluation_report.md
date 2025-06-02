# Optimization Methods Evaluation Report

## Solution Quality Metrics

### Objective Values
#### PuLP
- Mean: 104.1029 ± 15.5292
- Range: [81.7823, 127.3084]
- Success Rate: 100.0%

#### Benders
- Mean: 110.0672 ± 18.6728
- Range: [88.7790, 139.8920]
- Success Rate: 52.0%

#### Quantum-Enhanced
- Mean: 99.1879 ± 11.8479
- Range: [73.1310, 129.0366]
- Success Rate: 50.0%

#### Quantum-Enhanced-Merge
- Mean: 103.8761 ± 10.2904
- Range: [90.7291, 128.0637]
- Success Rate: 46.0%

#### RQAOA
- Mean: 98.8919 ± 8.8433
- Range: [84.0915, 119.9844]
- Success Rate: 64.0%

### Solution Differences (compared to PuLP)
#### Benders
- Gap: 0.1877 ± 0.1754
- Max Violation: 77.5019 ± 21.0985
- Infinity Norm X: 77.5019 ± 21.0985
- One Norm Y: 587.5361 ± 162.7429

#### Quantum-Enhanced
- Gap: 0.1542 ± 0.1128
- Max Violation: 88.0472 ± 13.4414
- Infinity Norm X: 88.0472 ± 13.4414
- One Norm Y: 646.4756 ± 76.0216

#### Quantum-Enhanced-Merge
- Gap: 0.1492 ± 0.0906
- Max Violation: 80.0219 ± 17.3901
- Infinity Norm X: 80.0219 ± 17.3901
- One Norm Y: 626.3944 ± 99.2844

#### RQAOA
- Gap: 0.1152 ± 0.0934
- Max Violation: 88.3935 ± 10.7671
- Infinity Norm X: 88.3935 ± 10.7671
- One Norm Y: 643.0022 ± 77.1000

## Performance Metrics

### Runtime
#### PuLP
- Mean: 0.65 ± 0.24 seconds

#### Benders
- Mean: 8.19 ± 4.70 seconds

#### Quantum-Enhanced
- Mean: 18.99 ± 3.55 seconds

#### Quantum-Enhanced-Merge
- Mean: 16.63 ± 3.03 seconds

#### RQAOA
- Mean: 4.01 ± 1.23 seconds

### Resource Utilization
#### PuLP
- Memory Peak: 1.54 ± 2.88 MB

#### Benders
- Memory Peak: 2.13 ± 3.83 MB

#### Quantum-Enhanced
- Memory Peak: 124.22 ± 56.60 MB

#### Quantum-Enhanced-Merge
- Memory Peak: 133.35 ± 49.51 MB

#### RQAOA
- Memory Peak: 2.12 ± 3.88 MB

## Quantum Optimization Details

### Quantum-Enhanced QAOA² Details

- QAOA² Decomposition used in 0 out of 50 runs (0.0%)

### Quantum-Enhanced-Merge QAOA² Details

- QAOA² Decomposition used in 0 out of 50 runs (0.0%)
