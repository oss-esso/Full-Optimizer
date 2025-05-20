# Optimization Methods Evaluation Report

## Solution Quality Metrics

### Objective Values
#### PuLP
- Mean: 58.8273 ± 2.1482
- Range: [55.6949, 61.5749]
- Success Rate: 100.0%

#### Benders
- Mean: 67.4878 ± 0.0412
- Range: [67.2471, 67.5000]
- Success Rate: 50.0%

#### Quantum-Enhanced
- Mean: 55.1460 ± 8.6992
- Range: [17.1000, 67.6000]
- Success Rate: 80.0%

#### Quantum-Enhanced-Merge
- Mean: 59.2825 ± 6.0666
- Range: [42.1000, 67.3500]
- Success Rate: 90.0%

### Solution Differences (compared to PuLP)
#### Benders
- Gap: 0.1487 ± 0.0418
- Max Violation: 20.3294 ± 2.7919
- Infinity Norm X: 20.3294 ± 2.7919
- One Norm Y: 81.7244 ± 10.9631

#### Quantum-Enhanced
- Gap: 0.1155 ± 0.1164
- Max Violation: 60.9607 ± 26.2697
- Infinity Norm X: 60.9607 ± 26.2697
- One Norm Y: 224.7374 ± 99.2752

#### Quantum-Enhanced-Merge
- Gap: 0.0889 ± 0.0667
- Max Violation: 46.9688 ± 30.7984
- Infinity Norm X: 46.9688 ± 30.7984
- One Norm Y: 160.4228 ± 82.1001

## Performance Metrics

### Runtime
#### PuLP
- Mean: 0.14 ± 0.03 seconds

#### Benders
- Mean: 4.73 ± 1.23 seconds

#### Quantum-Enhanced
- Mean: 14.95 ± 3.37 seconds

#### Quantum-Enhanced-Merge
- Mean: 14.39 ± 3.85 seconds

### Resource Utilization
#### PuLP
- Memory Peak: 0.17 ± 0.24 MB

#### Benders
- Memory Peak: 0.87 ± 1.02 MB

#### Quantum-Enhanced
- Memory Peak: 134.42 ± 73.79 MB

#### Quantum-Enhanced-Merge
- Memory Peak: 130.55 ± 85.45 MB
