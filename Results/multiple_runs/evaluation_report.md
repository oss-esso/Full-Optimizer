# Optimization Methods Evaluation Report

## Solution Quality Metrics

### Objective Values
#### PuLP
- Mean: 68.6250 ± 0.0000
- Range: [68.6250, 68.6250]
- Success Rate: 100.0%

#### Benders
- Mean: 67.1250 ± 0.0000
- Range: [67.1250, 67.1250]
- Success Rate: 100.0%

#### Quantum-Inspired
- Mean: 54.6750 ± 0.0000
- Range: [54.6750, 54.6750]
- Success Rate: 0.0%

#### Quantum-Enhanced
- Mean: 51.3310 ± 9.4270
- Range: [24.7000, 65.3250]
- Success Rate: 24.0%

#### Quantum-Enhanced-Merge
- Mean: 61.0780 ± 4.7310
- Range: [42.6000, 63.9750]
- Success Rate: 76.0%

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
- Gap: 0.2520 ± 0.1374
- Max Violation: 74.9800 ± 24.7440
- Infinity Norm X: 74.9800 ± 24.7440
- One Norm Y: 250.7200 ± 101.4119

#### Quantum-Enhanced-Merge
- Gap: 0.1100 ± 0.0689
- Max Violation: 34.0000 ± 30.9489
- Infinity Norm X: 34.0000 ± 30.9489
- One Norm Y: 121.8800 ± 69.1233

## Performance Metrics

### Runtime
#### PuLP
- Mean: 0.10 ± 0.03 seconds

#### Benders
- Mean: 2.93 ± 0.33 seconds

#### Quantum-Inspired
- Mean: 9.53 ± 1.02 seconds

#### Quantum-Enhanced
- Mean: 9.13 ± 1.41 seconds

#### Quantum-Enhanced-Merge
- Mean: 10.33 ± 3.14 seconds

### Resource Utilization
#### PuLP
- Memory Peak: 0.07 ± 0.14 MB

#### Benders
- Memory Peak: 0.47 ± 0.85 MB

#### Quantum-Inspired
- Memory Peak: 0.81 ± 1.40 MB

#### Quantum-Enhanced
- Memory Peak: 128.08 ± 56.95 MB

#### Quantum-Enhanced-Merge
- Memory Peak: 130.81 ± 56.60 MB
