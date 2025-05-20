# Optimization Methods Evaluation Report

## Solution Quality Metrics

### Objective Values
#### PuLP
- Mean: 104.7325 ± 0.0000
- Range: [104.7325, 104.7325]
- Success Rate: 100.0%

#### Benders
- Mean: 115.7770 ± 0.0000
- Range: [115.7770, 115.7770]
- Success Rate: 100.0%

#### Quantum-Enhanced
- Mean: 108.9480 ± 8.2373
- Range: [84.9950, 122.5100]
- Success Rate: 88.0%

#### Quantum-Enhanced-Merge
- Mean: 116.1603 ± 5.1858
- Range: [102.5900, 122.7850]
- Success Rate: 68.0%

### Solution Differences (compared to PuLP)
#### Benders
- Gap: 0.1055 ± 0.0000
- Max Violation: 36.0000 ± 0.0000
- Infinity Norm X: 36.0000 ± 0.0000
- One Norm Y: 177.2000 ± 0.0000

#### Quantum-Enhanced
- Gap: 0.0705 ± 0.0526
- Max Violation: 63.4600 ± 15.3987
- Infinity Norm X: 63.4600 ± 15.3987
- One Norm Y: 283.5800 ± 53.3062

#### Quantum-Enhanced-Merge
- Gap: 0.1099 ± 0.0476
- Max Violation: 49.0600 ± 7.2412
- Infinity Norm X: 49.0600 ± 7.2412
- One Norm Y: 234.0200 ± 31.2145

## Performance Metrics

### Runtime
#### PuLP
- Mean: 0.25 ± 0.09 seconds

#### Benders
- Mean: 9.39 ± 1.65 seconds

#### Quantum-Enhanced
- Mean: 5.09 ± 2.33 seconds

#### Quantum-Enhanced-Merge
- Mean: 4.93 ± 2.36 seconds

### Resource Utilization
#### PuLP
- Memory Peak: 0.20 ± 0.25 MB

#### Benders
- Memory Peak: 0.90 ± 1.07 MB

#### Quantum-Enhanced
- Memory Peak: 69.17 ± 44.28 MB

#### Quantum-Enhanced-Merge
- Memory Peak: 68.61 ± 44.67 MB
