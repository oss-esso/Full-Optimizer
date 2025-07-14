# Multi-Solver Integration Tasks

## Relevant Files

- `test_overnight.py` – Main test script to be enhanced with multiple solvers
- `tsp_multiple_days1.py` – First TSP solver to integrate (OR-Tools based with time_details dependency)
- `tsp_multiple_days2.py` – Second TSP solver to integrate (JSON-based configuration)
- `vrp_multiday_sequential.py` – Baseline solver for comparison and logic reference
- `route_database.py` – OSRM routing cache implementation for data extraction
- `vrp_scenarios.py` – Scenario creation and vehicle speed ratio definitions
- `changelog.md` – Documentation of all changes made

### Notes

- Testing: Run with `python test_overnight.py` after integration
- Validation: Compare solver outputs within ±15% tolerance
- Constraint Preservation: Ensure all solvers respect scenario time/capacity limits

## Tasks

- [ ] **1.0 Data Transformation Implementation**
  - [ ] 1.1 Create `time_details.py` stub module for tsp_multiple_days1.py compatibility
  - [ ] 1.2 Implement scenario-to-matrix converter using CachedOSRMDistanceCalculator logic
  - [ ] 1.3 Extract service time and vehicle constraint data from furgoni scenario
  - [ ] 1.4 Create JSON generator for tsp_multiple_days2.py with all required fields
  - [ ] 1.5 Implement time window and working hours conversion functions
  - [ ] 1.6 Add truck speed ratio application for realistic travel times

- [ ] **2.0 Solver Integration**
  - [ ] 2.1 Modify test_overnight.py to support multiple solver execution modes
  - [ ] 2.2 Create solver wrapper functions with standardized input/output interfaces
  - [ ] 2.3 Implement tsp_multiple_days1.py integration with parameter injection
  - [ ] 2.4 Implement tsp_multiple_days2.py integration with JSON scenario generation
  - [ ] 2.5 Add error handling and fallback mechanisms for solver failures
  - [ ] 2.6 Create solver execution orchestration with sequential runs

- [ ] **3.0 Standardized Reporting**
  - [ ] 3.1 Extract baseline reporting format from existing test_overnight.py output
  - [ ] 3.2 Create report normalization functions for tsp_multiple_days1.py results
  - [ ] 3.3 Create report normalization functions for tsp_multiple_days2.py results
  - [ ] 3.4 Implement consistent metric calculation (distance, cost, time, vehicles used)
  - [ ] 3.5 Add constraint violation tracking across all solvers
  - [ ] 3.6 Create unified console output format with solver identification

- [ ] **4.0 Plotting and Visualization**
  - [ ] 4.1 Extract plot styling and format from baseline vrp_multiday_sequential.py
  - [ ] 4.2 Create plot data converter for tsp_multiple_days1.py solution format
  - [ ] 4.3 Create plot data converter for tsp_multiple_days2.py solution format
  - [ ] 4.4 Implement consistent route visualization with vehicle colors and overnight stops
  - [ ] 4.5 Add solver identification to plot titles and legends
  - [ ] 4.6 Generate separate plot files for each solver with timestamp naming

- [ ] **5.0 Solution Validation**
  - [ ] 5.1 Define ±15% tolerance validation criteria and justification
  - [ ] 5.2 Implement objective value extraction from all three solver outputs
  - [ ] 5.3 Create comparison matrix showing baseline vs. alternative solver results
  - [ ] 5.4 Add constraint feasibility validation across all solutions
  - [ ] 5.5 Implement performance metrics comparison (runtime, solution quality)
  - [ ] 5.6 Generate validation report with pass/fail status for each solver
