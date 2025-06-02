# Optimization Methods Evaluation Report

## Solution Quality Metrics

### Objective Values
#### PuLP
- Mean: 59.1295 ± 2.1915
- Range: [55.5494, 61.5720]
- Success Rate: 100.0%

#### Benders
- Mean: 67.4859 ± 0.0347
- Range: [67.3698, 67.5000]
- Success Rate: 62.0%

#### Quantum-Enhanced
- Mean: 55.4190 ± 9.5029
- Range: [17.1750, 66.1750]
- Success Rate: 78.0%

#### Quantum-Enhanced-Merge
- Mean: 59.4970 ± 5.3218
- Range: [39.3500, 63.9750]
- Success Rate: 96.0%

#### RQAOA
- Mean: 44.4775 ± 8.7637
- Range: [22.2000, 58.6250]
- Success Rate: 30.0%

### Solution Differences (compared to PuLP)
#### Benders
- Gap: 0.1429 ± 0.0429
- Max Violation: 20.5030 ± 2.3046
- Infinity Norm X: 20.5030 ± 2.3046
- One Norm Y: 84.3602 ± 8.8546

#### Quantum-Enhanced
- Gap: 0.1082 ± 0.1354
- Max Violation: 60.6828 ± 22.4852
- Infinity Norm X: 60.6828 ± 22.4852
- One Norm Y: 216.3103 ± 90.6768

#### Quantum-Enhanced-Merge
- Gap: 0.0741 ± 0.0668
- Max Violation: 46.2915 ± 27.1628
- Infinity Norm X: 46.2915 ± 27.1628
- One Norm Y: 166.5009 ± 73.5134

#### RQAOA
- Gap: 0.2486 ± 0.1502
- Max Violation: 78.6313 ± 14.5679
- Infinity Norm X: 78.6313 ± 14.5679
- One Norm Y: 317.4312 ± 75.3272

## Performance Metrics

### Runtime
#### PuLP
- Mean: 0.15 ± 0.03 seconds

#### Benders
- Mean: 2.87 ± 0.39 seconds

#### Quantum-Enhanced
- Mean: 8.42 ± 1.03 seconds

#### Quantum-Enhanced-Merge
- Mean: 7.91 ± 1.15 seconds

#### RQAOA
- Mean: 2.02 ± 0.32 seconds

### Resource Utilization
#### PuLP
- Memory Peak: 0.10 ± 0.17 MB

#### Benders
- Memory Peak: 0.49 ± 0.87 MB

#### Quantum-Enhanced
- Memory Peak: 136.03 ± 23.24 MB

#### Quantum-Enhanced-Merge
- Memory Peak: 136.48 ± 23.23 MB

#### RQAOA
- Memory Peak: 0.56 ± 0.86 MB
