# Optimization Methods Evaluation Report

## Solution Quality Metrics

### Objective Values
#### PuLP
- Mean: 137.0795 ± 12.2069
- Range: [113.8828, 161.4862]
- Success Rate: 100.0%

#### Benders
- Mean: 136.8667 ± 11.4847
- Range: [113.0464, 157.2102]
- Success Rate: 0.0%

#### Quantum-Enhanced
- Mean: 100.4136 ± 14.8699
- Range: [73.5106, 127.3789]
- Success Rate: 0.0%

#### Quantum-Enhanced-Merge
- Mean: 110.3757 ± 9.9999
- Range: [84.0319, 129.9514]
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
- Mean: 0.11 ± 0.02 seconds

#### Benders
- Mean: 4.15 ± 1.26 seconds

#### Quantum-Enhanced
- Mean: 19.62 ± 5.35 seconds

#### Quantum-Enhanced-Merge
- Mean: 18.84 ± 5.20 seconds

### Resource Utilization
#### PuLP
- Memory Peak: 0.26 ± 0.26 MB

#### Benders
- Memory Peak: 1.54 ± 1.75 MB

#### Quantum-Enhanced
- Memory Peak: 177.60 ± 46.71 MB

#### Quantum-Enhanced-Merge
- Memory Peak: 176.69 ± 46.72 MB
