# Optimization Methods Evaluation Report

## Solution Quality Metrics

### Objective Values
#### PuLP
- Mean: 68.6250 ± 0.0000
- Range: [68.6250, 68.6250]
- Success Rate: 0.0%

#### Benders
- Mean: 67.1250 ± 0.0000
- Range: [67.1250, 67.1250]
- Success Rate: 0.0%

#### Quantum-Inspired
- Mean: 54.6750 ± 0.0000
- Range: [54.6750, 54.6750]
- Success Rate: 0.0%

#### Quantum-Enhanced
- Mean: 61.1250 ± 0.0000
- Range: [61.1250, 61.1250]
- Success Rate: 100.0%

### Solution Differences (compared to PuLP)
#### Benders
- Gap: 0.0219 ± 0.0000
- Max Violation: 5.0000 ± 0.0000
- Infinity Norm X: 5.0000 ± 0.0000
- One Norm Y: 36.0000 ± 0.0000

#### Quantum-Inspired
- Gap: 0.2033 ± 0.0000
- Max Violation: 95.0000 ± 0.0000
- Infinity Norm X: 95.0000 ± 0.0000
- One Norm Y: 338.0000 ± 0.0000

#### Quantum-Enhanced
- Gap: 0.1093 ± 0.0000
- Max Violation: 10.0000 ± 0.0000
- Infinity Norm X: 10.0000 ± 0.0000
- One Norm Y: 72.0000 ± 0.0000

## Performance Metrics

### Runtime
#### PuLP
- Mean: 0.11 ± 0.03 seconds

#### Benders
- Mean: 2.96 ± 0.35 seconds

#### Quantum-Inspired
- Mean: 9.83 ± 1.20 seconds

#### Quantum-Enhanced
- Mean: 9.46 ± 1.97 seconds

### Resource Utilization
#### PuLP
- Memory Peak: 0.07 ± 0.12 MB

#### Benders
- Memory Peak: 0.48 ± 0.87 MB

#### Quantum-Inspired
- Memory Peak: 0.79 ± 1.37 MB

#### Quantum-Enhanced
- Memory Peak: 118.50 ± 81.50 MB
