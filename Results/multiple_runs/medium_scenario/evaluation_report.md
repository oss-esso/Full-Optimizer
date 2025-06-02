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
- Mean: 110.3370 ± 8.3324
- Range: [88.2550, 122.5000]
- Success Rate: 82.0%

#### Quantum-Enhanced-Merge
- Mean: 116.3915 ± 4.6989
- Range: [106.7850, 122.8100]
- Success Rate: 68.0%

#### RQAOA
- Mean: 105.0898 ± 5.4582
- Range: [92.8050, 119.1550]
- Success Rate: 100.0%

### Solution Differences (compared to PuLP)
#### Benders
- Gap: 0.1055 ± 0.0000
- Max Violation: 36.0000 ± 0.0000
- Infinity Norm X: 36.0000 ± 0.0000
- One Norm Y: 177.2000 ± 0.0000

#### Quantum-Enhanced
- Gap: 0.0802 ± 0.0518
- Max Violation: 58.3600 ± 13.8133
- Infinity Norm X: 58.3600 ± 13.8133
- One Norm Y: 270.4800 ± 47.5048

#### Quantum-Enhanced-Merge
- Gap: 0.1113 ± 0.0449
- Max Violation: 50.3600 ± 8.0945
- Infinity Norm X: 50.3600 ± 8.0945
- One Norm Y: 234.8400 ± 28.1268

#### RQAOA
- Gap: 0.0421 ± 0.0303
- Max Violation: 70.6600 ± 13.7949
- Infinity Norm X: 70.6600 ± 13.7949
- One Norm Y: 312.1200 ± 35.8692

## Performance Metrics

### Runtime
#### PuLP
- Mean: 0.16 ± 0.05 seconds

#### Benders
- Mean: 5.55 ± 0.57 seconds

#### Quantum-Enhanced
- Mean: 2.48 ± 0.62 seconds

#### Quantum-Enhanced-Merge
- Mean: 2.24 ± 0.66 seconds

#### RQAOA
- Mean: 1.94 ± 0.28 seconds

### Resource Utilization
#### PuLP
- Memory Peak: 0.10 ± 0.18 MB

#### Benders
- Memory Peak: 0.51 ± 0.91 MB

#### Quantum-Enhanced
- Memory Peak: 45.27 ± 23.11 MB

#### Quantum-Enhanced-Merge
- Memory Peak: 45.21 ± 23.07 MB

#### RQAOA
- Memory Peak: 0.61 ± 0.86 MB
