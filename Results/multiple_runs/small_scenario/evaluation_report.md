# Optimization Methods Evaluation Report

## Solution Quality Metrics

### Objective Values
#### PuLP
- Mean: 58.6026 ± 2.1129
- Range: [56.3079, 61.3439]
- Success Rate: 100.0%

#### Benders
- Mean: 67.4929 ± 0.0226
- Range: [67.4286, 67.5000]
- Success Rate: 40.0%

#### Quantum-Inspired
- Mean: 60.5600 ± 3.8794
- Range: [54.9250, 68.3250]
- Success Rate: 90.0%

#### Quantum-Enhanced
- Mean: 47.5225 ± 13.8623
- Range: [22.2000, 60.5500]
- Success Rate: 70.0%

#### Quantum-Enhanced-Merge
- Mean: 59.7425 ± 3.4449
- Range: [54.2500, 63.9750]
- Success Rate: 100.0%

#### RQAOA
- Mean: 46.9625 ± 7.7717
- Range: [35.0750, 58.4500]
- Success Rate: 40.0%

### Solution Differences (compared to PuLP)
#### Benders
- Gap: 0.1530 ± 0.0411
- Max Violation: 19.8322 ± 2.5592
- Infinity Norm X: 19.8322 ± 2.5592
- One Norm Y: 76.3624 ± 7.9008

#### Quantum-Inspired
- Gap: 0.0553 ± 0.0654
- Max Violation: 53.4172 ± 24.5842
- Infinity Norm X: 53.4172 ± 24.5842
- One Norm Y: 192.6259 ± 80.9491

#### Quantum-Enhanced
- Gap: 0.2089 ± 0.2175
- Max Violation: 83.7061 ± 11.5852
- Infinity Norm X: 83.7061 ± 11.5852
- One Norm Y: 295.0157 ± 96.7583

#### Quantum-Enhanced-Merge
- Gap: 0.0465 ± 0.0480
- Max Violation: 56.0872 ± 31.7906
- Infinity Norm X: 56.0872 ± 31.7906
- One Norm Y: 169.0319 ± 74.5851

#### RQAOA
- Gap: 0.2006 ± 0.1384
- Max Violation: 84.0655 ± 13.8927
- Infinity Norm X: 84.0655 ± 13.8927
- One Norm Y: 322.0646 ± 65.3618

## Performance Metrics

### Runtime
#### PuLP
- Mean: 0.13 ± 0.02 seconds

#### Benders
- Mean: 2.71 ± 0.24 seconds

#### Quantum-Inspired
- Mean: 0.22 ± 0.03 seconds

#### Quantum-Enhanced
- Mean: 10.40 ± 0.18 seconds

#### Quantum-Enhanced-Merge
- Mean: 9.64 ± 0.12 seconds

#### RQAOA
- Mean: 2.00 ± 0.23 seconds

### Resource Utilization
#### PuLP
- Memory Peak: 0.30 ± 0.19 MB

#### Benders
- Memory Peak: 1.91 ± 0.08 MB

#### Quantum-Inspired
- Memory Peak: 1.73 ± 0.26 MB

#### Quantum-Enhanced
- Memory Peak: 178.75 ± 0.62 MB

#### Quantum-Enhanced-Merge
- Memory Peak: 179.32 ± 1.18 MB

#### RQAOA
- Memory Peak: 1.92 ± 0.31 MB
